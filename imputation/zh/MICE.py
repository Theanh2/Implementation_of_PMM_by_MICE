import warnings
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Union, Optional, List
from datetime import datetime
from logging.handlers import RotatingFileHandler
import os

# Configure a single root logger for the entire project.
# Any logger created in other modules (e.g., cart.py, plotting/diagnostics.py)
# will inherit this configuration.
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Prevent adding handlers multiple times
if not root_logger.handlers:
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Define the log filename based on the current date.
    log_filename = f"mice_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # Create console handler with an INFO log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create rotating file handler which logs even debug messages.
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    console_format = logging.Formatter('%(levelname)s - %(message)s')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    file_handler.setFormatter(file_format)

    # Add the handlers to the root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# Get a logger for the current module.
# It will automatically inherit handlers from the root logger.
logger = logging.getLogger(__name__)


from .validators import (
    validate_dataframe,
    validate_columns,
    check_n_imputations,
    check_maxit,
    check_method,
    check_initial_method,
    validate_predictor_matrix,
    check_visit_sequence,
)
from .constants import (
    ImputationMethod,
    InitialMethod,
    SUPPORTED_METHODS,
    DEFAULT_METHOD,
    SUPPORTED_INITIAL_METHODS,
    DEFAULT_INITIAL_METHOD,
    VisitSequence,
)

# Import concrete imputation functions
from .utils import get_imputer_func
# External helpers
from .mice_result import MICEresult

# pm and visit sequenc -- if there are columns not in pm, but in visit sequence, add them to pm
class MICE:
    """
    Multiple Imputation by Chained Equations (MICE) class.
    
    This class implements the MICE algorithm for handling missing data through
    multiple imputations using chained equations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data with missing values. Must be a pandas DataFrame.
        
    Attributes
    ----------
    data : pd.DataFrame
        The validated and cleaned input data
    id_obs : Dict[str, np.ndarray]
        Dictionary mapping column names to indices of observed values
    id_mis : Dict[str, np.ndarray]
        Dictionary mapping column names to indices of missing values
    """
    
    def __init__(self, data):
        """
        Initialize the MICE object.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with missing values. Must be a pandas DataFrame.
            
        Raises
        ------
        ValueError
            If data is not a pandas DataFrame or contains duplicate column names
        """
        logger.info("Initializing MICE object")
        logger.debug(f"Input data shape: {data.shape}")

        self.data = validate_dataframe(data)
        self.data = validate_columns(self.data)
        
        self.id_obs = {}
        self.id_mis = {}
        missing_stats = {}
        
        for col in self.data.columns:
            notna = self.data[col].notna()
            self.id_obs[col] = notna
            self.id_mis[col] = ~notna
            missing_stats[col] = {
                'missing_count': (~notna).sum(),
                'missing_percentage': (~notna).mean() * 100
            }
            
        logger.debug("Missing value statistics:")
        for col, stats in missing_stats.items():
            logger.debug(f"  {col}: {stats['missing_count']} values ({stats['missing_percentage']:.2f}%) missing")
            
        # Container for pooled results
        self.result = None  # Will hold the pooled `MICEresult` instance
        self.run_output_dir = None

        # Required by statsmodels result wrappers
        self.nobs = self.data.shape[0]
        logger.info("MICE object initialized successfully")

    def impute(
        self,
        n_imputations: int = 5,
        maxit: int = 10,
        predictor_matrix: Optional[pd.DataFrame] = None,
        initial: str = DEFAULT_INITIAL_METHOD,
        method: Optional[Union[str, Dict[str, str]]] = None,
        visit_sequence: Union[str, List[str]] = "monotone",
        **kwargs
    ) -> None:
        """
        Perform multiple imputation by chained equations.
        
        Parameters
        ----------
        n_imputations : int, default=5
            Number of imputations to perform
            
        maxit : int, default=10
            Maximum number of iterations for each imputation cycle.
            Must be a positive integer.
            
        predictor_matrix : pd.DataFrame, optional
            Binary matrix indicating which variables should be used as predictors
            for each target variable. Should have column names as both index and columns.
            A 1 indicates that the column variable is used as predictor for the index variable.
            If None, a predictor matrix is estimated using `_quickpred`.
            
        initial : str, default=DEFAULT_INITIAL_METHOD
            Initial imputation method. Must be one of SUPPORTED_INITIAL_METHODS.
            
        method : Union[str, Dict[str, str]], optional
            Imputation method(s) to use:
            - str: use the same method for all columns
            - Dict[str, str]: dictionary mapping column names to their methods
            - None: use default method for all columns
            Must be one of SUPPORTED_METHODS.
            
        visit_sequence : Union[str, List[str]], default="monotone"
            Sequence in which variables should be visited during imputation:
            - str: "monotone" for monotone missing data pattern
            - List[str]: list of column names specifying the order to visit variables
            
        **kwargs : dict
            Additional keyword arguments.
            - `output_dir` (str, optional): Directory to save outputs for this run.
              If not provided, a timestamped folder is created in `output_figures`.
            
            Parameters for specific imputation methods can also be passed. These should
            be prefixed with the method name and an underscore, e.g., `pmm_donors=5` to pass
            `donors=5` to the `pmm` imputer.
            
            When `predictor_matrix` is not specified, the following can be passed for `_quickpred`:
            - `min_cor` (float, default=0.1): Minimum correlation for a predictor.
            - `min_puc` (float, default=0.0): Minimum proportion of usable cases.
            - `include` (list, optional): Columns to always include as predictors.
            - `exclude` (list, optional): Columns to always exclude as predictors.
            - `correlation_method` (str, default="pearson"): Correlation method used to
              compute the correlation matrix inside `_quickpred`.
        """
        logger.info("Starting imputation process")
        logger.debug(f"Parameters: n_imputations={n_imputations}, maxit={maxit}, "
                    f"initial={initial}, method={method}, visit_sequence={visit_sequence}")

        start_time = time.time()

        # Set up output directory for the run
        output_dir = kwargs.pop('output_dir', None)
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_output_dir = os.path.join('output_figures', f'run_{timestamp}')
        else:
            self.run_output_dir = output_dir
        
        os.makedirs(self.run_output_dir, exist_ok=True)
        logger.info(f"Outputs for this run will be saved in: {self.run_output_dir}")

        check_n_imputations(n_imputations)
        check_maxit(maxit)
        check_initial_method(initial)
        
        if predictor_matrix is None:
            min_cor = kwargs.pop('min_cor', 0.1)
            min_puc = kwargs.pop('min_puc', 0.0)
            include = kwargs.pop('include', None)
            exclude = kwargs.pop('exclude', None)
            correlation_method = kwargs.pop('correlation_method', 'pearson')
            predictor_matrix = self._quickpred(
                min_cor=min_cor, 
                min_puc=min_puc, 
                include=include, 
                exclude=exclude, 
                method=correlation_method
            )
        else:
            predictor_matrix = validate_predictor_matrix(predictor_matrix, list(self.data.columns), self.data)
            logger.debug("Predictor matrix validated successfully")
        
        if method is not None:
            self.method = check_method(method, list(self.data.columns))
        else:
            self.method = DEFAULT_METHOD
        logger.debug(f"Using imputation methods: {self.method}")

        # Warn if user provided method-specific parameters for methods not used
        if self.imputation_params:
            provided_prefixes = set()
            for key in self.imputation_params.keys():
                if '_' in key:
                    provided_prefixes.add(key.split('_', 1)[0])
            used_methods = set(self.method.values())
            unused_provided = provided_prefixes - used_methods
            if unused_provided:
                logger.warning(
                    "Method-specific parameters were provided for unused methods: %s. "
                    "These parameters will be ignored.",
                    sorted(list(unused_provided))
                )
        
        self.n_imputations = n_imputations
        self.maxit = maxit
        self.predictor_matrix = predictor_matrix
        self.initial = initial
        self.imputation_params = kwargs

        self._set_visit_sequence(visit_sequence)
        logger.debug(f"Visit sequence set to: {self.visit_sequence}")

        # Prepare chain statistics containers
        self.chain_mean = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float)
            for col in self.visit_sequence
        }
        self.chain_var = {
            col: np.full((self.maxit, self.n_imputations), np.nan, dtype=float)
            for col in self.visit_sequence
        }

        self.imputed_datasets = []

        for chain_idx in range(self.n_imputations):
            logger.info(f"Starting imputation chain {chain_idx + 1}/{self.n_imputations}")
            self.imputed_datasets.append(self._impute_once(chain_idx))
            logger.info(f"Completed imputation chain {chain_idx + 1}")
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Imputation completed in {duration:.2f} seconds")
        
        logger.debug("Final imputation statistics:")
        logger.debug(f"  - Number of imputations: {self.n_imputations}")
        logger.debug(f"  - Maximum iterations: {self.maxit}")
        logger.debug(f"  - Initial method: {self.initial}")
        logger.debug(f"  - Method: {self.method}")
        logger.debug(f"  - Visit sequence: {self.visit_sequence}")
        logger.debug(f"  - Predictor matrix provided: {self.predictor_matrix is not None}")

    def _quickpred(
        self, 
        min_cor: float = 0.1, 
        min_puc: float = 0.0, 
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Generate a predictor matrix based on correlation and proportion of usable cases.
        
        This method is inspired by the `quickpred` function from the R `mice` package.
        
        Parameters
        ----------
        min_cor : float, default=0.1
            The minimum absolute correlation required to be included as a predictor.
        min_puc : float, default=0.0
            The minimum proportion of usable cases for correlation calculation.
        include : list of str, optional
            Columns to always include as predictors.
        exclude : list of str, optional
            Columns to always exclude as predictors.
        method : str, default="pearson"
            The correlation method to use ('pearson', 'kendall', 'spearman').
        
        Returns
        -------
        pd.DataFrame
            A square binary matrix indicating predictor relationships.
        """
        logger.info(f"Estimating predictor matrix with min_cor={min_cor}, min_puc={min_puc}, method='{method}'")
        
        predictor_matrix = pd.DataFrame(0, index=self.data.columns, columns=self.data.columns)
        
        # Calculate correlation matrix
        cor_matrix = self.data.corr(method=method)

        for target_col in self.data.columns:
            # Skip targets with no missing values
            if self.id_obs[target_col].all():
                continue

            for predictor_col in self.data.columns:
                if target_col == predictor_col:
                    continue

                # Proportion of usable cases
                puc = self.data[[target_col, predictor_col]].notna().all(axis=1).mean()

                if puc >= min_puc:
                    correlation = cor_matrix.loc[target_col, predictor_col]
                    if abs(correlation) >= min_cor:
                        predictor_matrix.loc[target_col, predictor_col] = 1
        
        # Handle include and exclude lists with validation for unknown columns
        if include:
            unknown_includes = [c for c in include if c not in predictor_matrix.columns]
            if unknown_includes:
                raise ValueError(f"_quickpred include contains unknown columns: {unknown_includes}")
            predictor_matrix.loc[:, include] = 1
        if exclude:
            unknown_excludes = [c for c in exclude if c not in predictor_matrix.columns]
            if unknown_excludes:
                raise ValueError(f"_quickpred exclude contains unknown columns: {unknown_excludes}")
            predictor_matrix.loc[:, exclude] = 0
            
        # Ensure diagonal is zero
        np.fill_diagonal(predictor_matrix.values, 0)
        
        logger.debug(f"Estimated predictor matrix:\n{predictor_matrix}")
        return predictor_matrix

    def pool(self, summ: bool = False):
        """Pool descriptive estimates across ``self.imputed_datasets`` using Rubin's rules.

        What is pooled
        --------------
        - Numeric columns: the sample mean per column.
        - Categorical columns (object/category): the per-level proportions for each column.

        Within-imputation variance
        --------------------------
        - Numeric: ``Var(mean) = s^2 / n`` (with ``ddof=1`` for ``s^2``).
        - Categorical level proportion ``p``: ``Var(p) = p(1-p)/n``.

        Notes
        -----
        - Cross-parameter covariances are ignored and a diagonal covariance matrix is constructed.
        - Degrees of freedom small-sample adjustments are not applied.
        - Categorical level parameter names are formatted as ``<column>[<level>]``.

        Parameters
        ----------
        summ : bool, optional
            If True, return ``self.result.summary()``.
        """
        logger.info("Starting pooling of imputed datasets")

        if not self.imputed_datasets:
            msg = "No imputed datasets found – run `.impute()` first."
            logger.error(msg)
            raise ValueError(msg)

        m = len(self.imputed_datasets)
        logger.debug(f"Pooling {m} imputed datasets")

        if m == 1:
            warnings.warn("Number of multiple imputations m = 1. Pooling will not reflect between-imputation uncertainty.")

        # Sample size (after imputation there should be no missing)
        n = self.imputed_datasets[0].shape[0]
        logger.debug(f"Sample size: {n}")

        # Identify columns by type from the first imputed dataset
        first_df = self.imputed_datasets[0]
        numeric_cols = first_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = first_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if not numeric_cols and not categorical_cols:
            msg = "No numeric or categorical columns available for pooling."
            logger.error(msg)
            raise ValueError(msg)

        if numeric_cols:
            logger.debug(f"Found {len(numeric_cols)} numeric columns for pooling")
        if categorical_cols:
            logger.debug(f"Found {len(categorical_cols)} categorical columns for pooling")

        # Build parameter vectors per imputed dataset
        param_names: List[str] = []
        q_list: List[List[float]] = [[] for _ in range(m)]
        u_list: List[List[float]] = [[] for _ in range(m)]

        # 1) Numeric columns: mean and its within-imputation variance
        for col in numeric_cols:
            param_names.append(col)
            for j, df in enumerate(self.imputed_datasets):
                series = df[col]
                q_ij = float(series.mean())
                # Within-imputation variance of the mean: var / n
                u_ij = float(series.var(ddof=1)) / n if n > 0 else np.nan
                q_list[j].append(q_ij)
                u_list[j].append(u_ij)

        # 2) Categorical columns: per-level proportions and their within-imputation variance p(1-p)/n
        for col in categorical_cols:
            # Determine stable set of levels across imputations
            all_levels = []
            for df in self.imputed_datasets:
                # Using unique preserves order of appearance; we collect then take unique again to retain stability
                all_levels.extend(pd.unique(df[col]))
            # Create ordered, unique levels while preserving first occurrence order
            seen = set()
            levels: List[object] = []
            for lvl in all_levels:
                if lvl not in seen:
                    seen.add(lvl)
                    levels.append(lvl)

            for lvl in levels:
                lvl_name = f"{col}[{str(lvl)}]"
                param_names.append(lvl_name)
                for j, df in enumerate(self.imputed_datasets):
                    # Proportion of rows equal to this level
                    # Using .to_numpy() for speed and robust equality
                    col_vals = df[col].to_numpy()
                    p = float(np.mean(col_vals == lvl)) if n > 0 else np.nan
                    u = p * (1.0 - p) / n if n > 0 else np.nan
                    q_list[j].append(p)
                    u_list[j].append(u)

        # Apply Rubin's rules
        logger.debug("Applying Rubin's rules for pooling")
        q_mat = np.asarray(q_list, dtype=float)
        u_mat = np.asarray(u_list, dtype=float)

        q_bar = np.nanmean(q_mat, axis=0)
        u_bar = np.nanmean(u_mat, axis=0)

        if m > 1:
            b = np.nansum((q_mat - q_bar) ** 2, axis=0) / (m - 1)
        else:
            b = np.zeros_like(q_bar)

        t = u_bar + (1.0 + 1.0 / max(m, 1)) * b

        # Avoid division by zero in FMI
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_miss_info = ((1.0 + 1.0 / max(m, 1)) * b) / t
            frac_miss_info = np.where(np.isfinite(frac_miss_info), frac_miss_info, np.nan)

        # Log pooling statistics
        for i, col in enumerate(param_names):
            logger.debug(f"Pooling statistics for '{col}':")
            logger.debug(f"  - Pooled estimate: {q_bar[i]:.4f}")
            logger.debug(f"  - Total variance: {t[i]:.4f}")
            logger.debug(f"  - Fraction of missing information: {frac_miss_info[i]:.4f}")

        # Build diagonal covariance matrix (ignoring cross-parameter covariances)
        cov_params = np.diag(t)
        logger.debug("Covariance matrix constructed")

        # Make parameter names available for summaries
        self.exog_names = param_names

        # Create results object
        logger.debug("Creating results object")
        self.result = MICEresult(self, q_bar, cov_params)
        self.result.scale = 1.0
        self.result.frac_miss_info = frac_miss_info

        logger.info("Pooling completed successfully")

        if summ:
            logger.debug("Generating summary")
            return self.result.summary()
    
    def plot_chain_stats(self, columns: Optional[List[str]] = None):
        """
        Plot convergence of chain mean and variance
        
        Parameters
        ----------
        columns : list, optional
            List of column names to plot. If None, plots all columns
        """
        from plotting.diagnostics import plot_chain_stats
        
        if self.chain_mean is None or self.chain_var is None:
            logger.warning("No chain statistics to plot. Run imputation first.")
            return
            
        # Filter columns if specified
        if columns is not None:
            # Check that all specified columns exist in chain statistics
            available_cols = list(self.chain_mean.keys())
            columns = [col for col in columns if col in available_cols]
            
            if not columns:
                logger.warning(f"None of the specified columns found in chain statistics. Available columns: {available_cols}")
                return
                
            # Filter chain statistics to only include specified columns
            filtered_chain_mean = {col: self.chain_mean[col] for col in columns}
            filtered_chain_var = {col: self.chain_var[col] for col in columns}
        else:
            filtered_chain_mean = self.chain_mean
            filtered_chain_var = self.chain_var
        
        plot_chain_stats(filtered_chain_mean, filtered_chain_var, columns)


    def _impute_once(self, chain_idx: int):
        """
        Perform one complete imputation cycle.
        
        Returns
        -------
        pd.DataFrame
            A copy of the data with one complete imputation cycle applied
        """
        logger.debug(f"Starting imputation cycle for chain {chain_idx}")
        current_data = self.data.copy(deep=True)
        
        logger.debug("Performing initial imputation")
        self._initial_imputation(current_data)

        for iter_idx in range(self.maxit):
            logger.debug(f"Starting iteration {iter_idx + 1}/{self.maxit} for chain {chain_idx}")
            current_data = self._iterate(current_data, iter_idx, chain_idx)
            logger.debug(f"Completed iteration {iter_idx + 1}")
        
        logger.debug(f"Completed imputation cycle for chain {chain_idx}")
        return current_data
    
    def _iterate(self, data: pd.DataFrame, iter_idx: int, chain_idx: int):
        """
        Perform one iteration of the imputation cycle.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to iterate over

        Returns
        -------
        pd.DataFrame
            A copy of the data with one iteration of the imputation cycle applied
        """
        updated_data = data
        iteration_start_time = time.time()

        for col in self.visit_sequence:
            logger.debug(f"Processing column '{col}' (iteration {iter_idx + 1}, chain {chain_idx})")
            method_name = self.method[col]

            # Determine predictors
            if self.predictor_matrix is not None:
                predictor_flags = self.predictor_matrix.loc[col]
                predictor_cols = predictor_flags[predictor_flags == 1].index.tolist()
                predictor_cols = [c for c in predictor_cols if c != col]
            else:
                predictor_cols = [c for c in updated_data.columns if c != col]

            logger.debug(f"Using {len(predictor_cols)} predictors for column '{col}'")
            predictors = updated_data[predictor_cols]

            # Prepare arrays/masks
            y = updated_data[col].to_numpy()
            id_obs_mask = self.id_obs[col]
            id_mis_mask = self.id_mis[col]
            id_obs = id_obs_mask.to_numpy()
            id_mis = id_mis_mask.to_numpy()

            # Get imputer function and perform imputation
            imputer_func = get_imputer_func(method_name)
            logger.debug(f"Using imputation method '{method_name}' for column '{col}'")

            # Extract method-specific parameters from kwargs
            method_params = {}
            prefix = f"{method_name}_"
            for key, value in self.imputation_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    method_params[param_name] = value
            
            if method_params:
                logger.debug(f"Passing parameters to imputer: {method_params}")

            imputed_values = imputer_func(y=y, id_obs=id_obs, id_mis=id_mis, x=predictors, **method_params)
            logger.debug(f"Successfully imputed {len(imputed_values)} values for column '{col}'")

            # Assign imputed values
            updated_data.loc[id_mis_mask, col] = imputed_values

            # Record chain statistics
            if id_mis.sum() > 0:
                imputed_arr = np.asarray(imputed_values, dtype=float)
                mean_val = np.nanmean(imputed_arr)
                self.chain_mean[col][iter_idx, chain_idx] = mean_val
                
                if imputed_arr.size > 1:
                    var_val = np.nanvar(imputed_arr, ddof=1)
                    self.chain_var[col][iter_idx, chain_idx] = var_val
                    logger.debug(f"Chain statistics for '{col}': mean={mean_val:.4f}, variance={var_val:.4f}")
                else:
                    self.chain_var[col][iter_idx, chain_idx] = np.nan
                    logger.debug(f"Chain statistics for '{col}': mean={mean_val:.4f}, variance=N/A (single value)")

        iteration_time = time.time() - iteration_start_time
        logger.debug(f"Iteration {iter_idx + 1} completed in {iteration_time:.2f} seconds")
        return updated_data
  
    def _initial_imputation(self, data):
        """
        Initialize missing values based on the initial method.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to initialize missing values in
        """
        if self.initial == InitialMethod.SAMPLE.value:
            for col in data.columns:
                if data[col].isna().any():
                    observed_values = data.loc[self.id_obs[col], col].values
                    data.loc[self.id_mis[col], col] = np.random.choice(observed_values, size=self.id_mis[col].sum())
                    
        elif self.initial == InitialMethod.MEANOBS.value:
            for col in data.columns:
                if data[col].isna().any():
                    col_mean = data[col].mean()
                    observed_values = data.loc[self.id_obs[col], col]
                    closest_idx = (observed_values - col_mean).abs().idxmin()
                    closest_value = data.loc[closest_idx, col]
                    data.loc[self.id_mis[col], col] = closest_value

    def _set_visit_sequence(self, visit_sequence):
        """
        Set the visit sequence for imputation based on the input parameter.
        
        Parameters
        ----------
        visit_sequence : Union[str, List[str]]
            Visit sequence specification. Can be:
            - str: "monotone" or "random" for predefined sequences
            - List[str]: list of column names specifying the order to visit variables
        """
        check_visit_sequence(visit_sequence, list(self.data.columns))
        
        if isinstance(visit_sequence, list):
            self.visit_sequence = visit_sequence
        else:
            columns_with_missing = [col for col in self.data.columns if self.data[col].isna().any()]
            
            if visit_sequence == VisitSequence.RANDOM.value:
                self.visit_sequence = list(np.random.permutation(columns_with_missing))
            elif visit_sequence == VisitSequence.MONOTONE.value:
                nmis = np.array([self.id_mis[col].sum() for col in columns_with_missing])
                ii = np.argsort(nmis)
                self.visit_sequence = [columns_with_missing[i] for i in ii]