"""
MCAR simulation runner for generating only imputed datasets (no pooling/analysis).

Features
- Loads the initial simulated dataset from `Simulate/Data/Inputs/initial_simulated_data.csv`.
- Applies MCAR masking at specified rates.
- Tests imputation with CART and RF only (separate runs), saving:
  - All completed datasets (per-imputation CSVs)
  - Mask used (CSV)
  - Chain statistics per imputation (mean/var per variable x iteration)
  - Metadata with parameters, seeds, and durations
- Skips recomputation if the output folder already exists for a configuration.

Notes
- `Z` is coerced to categorical to ensure classification is used for binary.
- No pooling or downstream analysis is performed here by design.
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path so `imputation` package is importable
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from imputation.MICE import MICE


# ---------------------------------------------------------------------------
# Configuration (you can tweak these)
# ---------------------------------------------------------------------------

# Input data (built by generate_initial_data.py)
INPUT_CSV = os.path.join(SCRIPT_DIR, "Data", "Inputs", "initial_simulated_data.csv")

# Output root
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "Data", "Imputations")

# Scenarios (MCAR only)
MCAR_RATES = [0.2]  # vary this list if needed, e.g., [0.1, 0.2, 0.3]

# Core MICE parameters
M_LIST = [5, 20]         # number of imputations per configuration
MAXIT_LIST = [10]     # MICE iterations per imputation
INITIAL_LIST = ["sample"]  # or ["sample"]
VISIT_SEQ_LIST = ["random"]  # or ["random"]

# Method-specific grids (what to vary among imputation)
CART_MIN_SAMPLES_LEAF_LIST = [5, 20]
# Additional CART parameters to try
CART_MAX_DEPTH_LIST: List[Optional[int]] = [None]
CART_CCP_ALPHA_LIST = [0, 1e-3]

RF_N_ESTIMATORS_LIST = [50, 300]
# Additional RF parameters to try
from typing import Optional
RF_MIN_SAMPLES_LEAF_LIST = [5, 20]
RF_MAX_DEPTH_LIST: List[Optional[int]] = [None]

# Replicates per scenario/config (distinct MCAR masks)
REPLICATES = 120

# Reproducibility
BASE_SEED = 2025


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def derive_seed(*parts: int) -> int:
    """Derive a deterministic seed from parts."""
    seed = BASE_SEED
    for p in parts:
        seed = (seed * 10_000_019 + int(p)) % 2_147_483_647
    return seed


def make_mcar_mask(df: pd.DataFrame, rate: float, columns: List[str], seed: int) -> pd.DataFrame:
    """Return a DataFrame of booleans with True where a value should be masked (MCAR)."""
    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in columns:
        mask[col] = rng.random(len(df)) < rate
    return mask


def apply_mask(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    masked = df.copy(deep=True)
    masked[mask] = np.nan
    return masked


def coerce_binary_to_category(df: pd.DataFrame, binary_columns: List[str]) -> pd.DataFrame:
    out = df.copy(deep=True)
    for col in binary_columns:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def build_full_predictor_matrix(columns: List[str]) -> pd.DataFrame:
    """Build a predictor matrix that uses all other columns for each target (off-diagonal = 1)."""
    pm = pd.DataFrame(1, index=columns, columns=columns, dtype=int)
    np.fill_diagonal(pm.values, 0)
    return pm


def save_chain_stats(output_dir: str, chain_mean: Dict[str, np.ndarray], chain_var: Dict[str, np.ndarray], imp_idx: int) -> None:
    """Save chain mean/var for a given imputation index (1-based)."""
    # Each entry is shape (maxit, n_imputations)
    idx = max(0, imp_idx - 1)
    mean_df = pd.DataFrame({col: vals[:, idx] for col, vals in chain_mean.items()})
    var_df = pd.DataFrame({col: vals[:, idx] for col, vals in chain_var.items()})
    mean_df.index.name = "iteration"
    var_df.index.name = "iteration"
    mean_path = os.path.join(output_dir, f"chain_mean_imp_{imp_idx:02d}.csv")
    var_path = os.path.join(output_dir, f"chain_var_imp_{imp_idx:02d}.csv")
    mean_df.to_csv(mean_path)
    var_df.to_csv(var_path)


def run_multi_imputation(
    data_with_nas: pd.DataFrame,
    method_name: str,
    n_imputations: int,
    maxit: int,
    method_params: Dict[str, object],
    initial: str,
    visit_sequence: str,
    seed: int,
    predictor_matrix: Optional[pd.DataFrame] = None,
) -> Tuple[List[pd.DataFrame], Dict[str, np.ndarray], Dict[str, np.ndarray], float, List[float]]:
    """
    Run multi-imputation (n_imputations=m) in one call. Returns list of completed datasets,
    chain stats, total duration, and per-chain durations parsed from logs.
    """
    # Set a replicate-level seed for reproducibility across chains
    np.random.seed(seed)

    mice = MICE(data_with_nas)
    # Workaround for attribute access before assignment in MICE.impute
    if not hasattr(mice, "imputation_params"):
        mice.imputation_params = {}

    # Map same method to all columns
    method_map = {col: method_name for col in data_with_nas.columns}

    # Prefix method-specific params so MICE routes them correctly
    # Do not set per-chain random_state here to allow stochastic variation between chains
    prefixed_params = {f"{method_name}_{k}": v for k, v in method_params.items()}

    # Capture logs from MICE to extract per-chain durations
    mice_logger = logging.getLogger("imputation.MICE")
    captured: List[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _CaptureHandler(level=logging.INFO)
    mice_logger.addHandler(handler)
    try:
        start = time.time()
        mice.impute(
            n_imputations=n_imputations,
            maxit=maxit,
            predictor_matrix=predictor_matrix,
            method=method_map,
            initial=initial,
            visit_sequence=visit_sequence,
            **prefixed_params,
        )
        total_duration = time.time() - start
    finally:
        mice_logger.removeHandler(handler)

    # Parse messages like: "Completed imputation chain 1 in 0.08 seconds"
    # or "Completed imputation chain 1/5 in 0.08 seconds"
    per_chain: List[float] = []
    pattern = re.compile(r"Completed imputation chain\s+\d+(?:/\d+)?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
    for msg in captured:
        m = pattern.search(msg)
        if m:
            try:
                per_chain.append(float(m.group(1)))
            except Exception:
                pass

    return mice.imputed_datasets, mice.chain_mean, mice.chain_var, total_duration, per_chain


def already_done(dir_path: str, m: int) -> bool:
    """Decide whether to skip this configuration. Simple folder presence with m outputs check."""
    if not os.path.isdir(dir_path):
        return False
    # Check presence of m completed datasets
    expected = [os.path.join(dir_path, f"imp_{i:02d}.csv") for i in range(1, m + 1)]
    return all(os.path.exists(p) for p in expected)


def main() -> None:
    # Load full data
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    full = pd.read_csv(INPUT_CSV)

    # Ensure binary Z is categorical for classification models
    binary_cols = ["Z"]
    full = coerce_binary_to_category(full, binary_cols)

    # Columns to mask under MCAR
    target_columns = list(full.columns)

    # Calculate total number of configurations for progress tracking
    total_configs = (len(MCAR_RATES) * len(M_LIST) * len(MAXIT_LIST) * 
                    len(INITIAL_LIST) * len(VISIT_SEQ_LIST) * 
                    (len(CART_MIN_SAMPLES_LEAF_LIST) * len(CART_MAX_DEPTH_LIST) * len(CART_CCP_ALPHA_LIST) +
                     len(RF_N_ESTIMATORS_LIST) * len(RF_MIN_SAMPLES_LEAF_LIST) * len(RF_MAX_DEPTH_LIST)))
    
    print(f"Starting simulation with {total_configs} configurations across {len(MCAR_RATES)} MCAR scenario(s)")
    print(f"Each configuration will run {REPLICATES} replicates")
    print(f"Total expected runs: {total_configs * REPLICATES}")
    
    config_progress = tqdm(total=total_configs, desc="Overall Progress", unit="config")
    
    for rate in tqdm(MCAR_RATES, desc="MCAR Rates", leave=False):
        scenario_name = f"MCAR_{int(rate*100)}"
        scenario_root = os.path.join(OUTPUT_ROOT, scenario_name)
        ensure_dir(scenario_root)

        for m in M_LIST:
            for maxit in MAXIT_LIST:
                for initial in INITIAL_LIST:
                    for visit_sequence in VISIT_SEQ_LIST:
                        # Two methods: CART and RF
                        # CART grid
                        for min_leaf in CART_MIN_SAMPLES_LEAF_LIST:
                            for max_depth in CART_MAX_DEPTH_LIST:
                                for ccp_alpha in CART_CCP_ALPHA_LIST:
                                    method = "cart"
                                    depth_tag = "none" if max_depth is None else str(max_depth)
                                    method_cfg = f"leaf{min_leaf}_depth{depth_tag}_alpha{ccp_alpha}"
                                    cfg_root = os.path.join(
                                        scenario_root,
                                        f"{method}__m{m}_it{maxit}_init-{initial}_vs-{visit_sequence}_{method_cfg}"
                                    )

                                    rep_progress = tqdm(range(1, REPLICATES + 1), 
                                                       desc=f"CART {method_cfg}", 
                                                       leave=False, 
                                                       unit="rep")
                                    for rep in rep_progress:
                                        rep_dir = os.path.join(cfg_root, f"rep_{rep:02d}")
                                        if already_done(rep_dir, m):
                                            # Skip if already computed
                                            rep_progress.set_postfix(status="skipped")
                                            continue
                                        ensure_dir(rep_dir)
                                        rep_progress.set_postfix(status="running")

                                        # Generate and save mask for this replicate
                                        mask_seed = derive_seed(int(rate * 100), rep)
                                        mask = make_mcar_mask(full, rate, target_columns, seed=mask_seed)
                                        mask_path = os.path.join(rep_dir, "mask.csv")
                                        mask.astype(int).to_csv(mask_path, index=False)

                                        # Apply mask
                                        masked = apply_mask(full, mask)

                                        # Run multi-imputation in a single call
                                        imp_paths: List[str] = []
                                        meta = {
                                            "input_csv": INPUT_CSV,
                                            "scenario": scenario_name,
                                            "method": method,
                                            "method_params": {
                                                "min_samples_leaf": min_leaf,
                                                "max_depth": max_depth,
                                                "ccp_alpha": ccp_alpha,
                                            },
                                            "m": m,
                                            "maxit": maxit,
                                            "initial": initial,
                                            "visit_sequence": visit_sequence,
                                            "rate": rate,
                                            "replicate": rep,
                                            "base_seed": BASE_SEED,
                                            "seed": mask_seed,
                                            "start_time": datetime.now().isoformat(timespec="seconds"),
                                        }
                                        pm = build_full_predictor_matrix(list(masked.columns))
                                        datasets, chain_mean, chain_var, total_dur, per_chain = run_multi_imputation(
                                            masked,
                                            method_name=method,
                                            n_imputations=m,
                                            maxit=maxit,
                                            method_params={
                                                "min_samples_leaf": min_leaf,
                                                "max_depth": max_depth,
                                                "ccp_alpha": ccp_alpha,
                                            },
                                            initial=initial,
                                            visit_sequence=visit_sequence,
                                            seed=mask_seed,
                                            predictor_matrix=pm,
                                        )

                                        # Save completed datasets and chain stats per chain
                                        for i, completed_df in enumerate(datasets, start=1):
                                            out_path = os.path.join(rep_dir, f"imp_{i:02d}.csv")
                                            completed_df.to_csv(out_path, index=False,  float_format="%.6g")
                                            imp_paths.append(out_path)
                                            save_chain_stats(rep_dir, chain_mean, chain_var, i)

                                        meta["durations_sec"] = per_chain
                                        meta["total_duration_sec"] = float(total_dur)
                                        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
                                        meta["outputs"] = imp_paths

                                        # Save metadata
                                        with open(os.path.join(rep_dir, "metadata.json"), "w") as f:
                                            json.dump(meta, f, indent=2)
                                        
                                        rep_progress.set_postfix(status="completed")
                                    
                                    # Update overall progress after completing this CART configuration
                                    config_progress.update(1)

                    
        for m in M_LIST:
            for maxit in MAXIT_LIST:
                for initial in INITIAL_LIST:
                    for visit_sequence in VISIT_SEQ_LIST:
                        # RF grid
                        for n_estimators in RF_N_ESTIMATORS_LIST:
                            for min_leaf in RF_MIN_SAMPLES_LEAF_LIST:
                                for max_depth in RF_MAX_DEPTH_LIST:
                                    method = "rf"
                                    depth_tag = "none" if max_depth is None else str(max_depth)
                                    method_cfg = f"ntree{n_estimators}_leaf{min_leaf}_depth{depth_tag}"
                                    cfg_root = os.path.join(
                                        scenario_root,
                                        f"{method}__m{m}_it{maxit}_init-{initial}_vs-{visit_sequence}_{method_cfg}"
                                    )

                                    rep_progress = tqdm(range(1, REPLICATES + 1), 
                                                       desc=f"RF {method_cfg}", 
                                                       leave=False, 
                                                       unit="rep")
                                    for rep in rep_progress:
                                        rep_dir = os.path.join(cfg_root, f"rep_{rep:02d}")
                                        if already_done(rep_dir, m):
                                            # Skip if already computed
                                            rep_progress.set_postfix(status="skipped")
                                            continue
                                        ensure_dir(rep_dir)
                                        rep_progress.set_postfix(status="running")

                                        # Generate and save mask for this replicate
                                        mask_seed = derive_seed(int(rate * 100), rep)
                                        mask = make_mcar_mask(full, rate, target_columns, seed=mask_seed)
                                        mask_path = os.path.join(rep_dir, "mask.csv")
                                        mask.astype(int).to_csv(mask_path, index=False)

                                        # Apply mask
                                        masked = apply_mask(full, mask)

                                        # Run multi-imputation in a single call
                                        imp_paths: List[str] = []
                                        meta = {
                                            "input_csv": INPUT_CSV,
                                            "scenario": scenario_name,
                                            "method": method,
                                            "method_params": {
                                                "n_estimators": n_estimators,
                                                "min_samples_leaf": min_leaf,
                                                "max_depth": max_depth,
                                            },
                                            "m": m,
                                            "maxit": maxit,
                                            "initial": initial,
                                            "visit_sequence": visit_sequence,
                                            "rate": rate,
                                            "replicate": rep,
                                            "base_seed": BASE_SEED,
                                            "seed": mask_seed,
                                            "start_time": datetime.now().isoformat(timespec="seconds"),
                                        }
                                        pm = build_full_predictor_matrix(list(masked.columns))
                                        datasets, chain_mean, chain_var, total_dur, per_chain = run_multi_imputation(
                                            masked,
                                            method_name=method,
                                            n_imputations=m,
                                            maxit=maxit,
                                            method_params={
                                                "n_estimators": n_estimators,
                                                "min_samples_leaf": min_leaf,
                                                "max_depth": max_depth,
                                            },
                                            initial=initial,
                                            visit_sequence=visit_sequence,
                                            seed=mask_seed,
                                            predictor_matrix=pm,
                                        )

                                        # Save completed datasets and chain stats per chain
                                        for i, completed_df in enumerate(datasets, start=1):
                                            out_path = os.path.join(rep_dir, f"imp_{i:02d}.csv")
                                            completed_df.to_csv(out_path, index=False, float_format="%.6g")
                                            imp_paths.append(out_path)
                                            save_chain_stats(rep_dir, chain_mean, chain_var, i)

                                        meta["durations_sec"] = per_chain
                                        meta["total_duration_sec"] = float(total_dur)
                                        meta["end_time"] = datetime.now().isoformat(timespec="seconds")
                                        meta["outputs"] = imp_paths

                                        # Save metadata
                                        with open(os.path.join(rep_dir, "metadata.json"), "w") as f:
                                            json.dump(meta, f, indent=2)
                                        
                                        rep_progress.set_postfix(status="completed")
                                    
                                    # Update overall progress after completing this RF configuration
                                    config_progress.update(1)
    # Save the metadata for all configurations
    # Close the overall progress bar
    config_progress.close()
    print("Simulation completed!")


if __name__ == "__main__":
    main()


