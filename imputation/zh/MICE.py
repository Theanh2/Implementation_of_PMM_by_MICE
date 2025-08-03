import warnings
import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Union, Optional, List

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create file handler which logs even debug messages
file_handler = logging.FileHandler('mice_imputation.log')
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to the handlers
console_format = logging.Formatter('%(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

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

# Import concrete imputation functions (skip RF for now)
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
            If None, all variables are used as predictors for each target.
            
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
            Additional keyword arguments for future extensions
        """
        logger.info("Starting imputation process")
        logger.debug(f"Parameters: n_imputations={n_imputations}, maxit={maxit}, "
                    f"initial={initial}, method={method}, visit_sequence={visit_sequence}")

        start_time = time.time()

        check_n_imputations(n_imputations)
        check_maxit(maxit)
        check_initial_method(initial)
        
        if predictor_matrix is not None:
            predictor_matrix = validate_predictor_matrix(predictor_matrix, list(self.data.columns), self.data)
            logger.debug("Predictor matrix validated successfully")
        
        if method is not None:
            self.method = check_method(method, list(self.data.columns))
        else:
            self.method = DEFAULT_METHOD
        logger.debug(f"Using imputation methods: {self.method}")
        
        self.n_imputations = n_imputations
        self.maxit = maxit
        self.predictor_matrix = predictor_matrix
        self.initial = initial

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

    def pool(self, summ: bool = False):
        """Pool column means across `self.imputed_datasets` using Rubin's rules.

        Each imputed dataset provides an estimate (the column mean) and a within-
        imputation variance (the variance of that column divided by *n*).  The
        function combines these to obtain pooled estimates and their standard
        errors.  Results are stored in ``self.result`` (an instance of
        :class:`MICEresult`).

        Parameters
        ----------
        summ : bool, optional
            If True, return a textual summary (equivalent to
            ``self.result.summary()``).
        """
        logger.info("Starting pooling of imputed datasets")

        if not self.imputed_datasets:
            msg = "No imputed datasets found – run `.impute()` first."
            logger.error(msg)
            raise ValueError(msg)

        m = len(self.imputed_datasets)
        logger.debug(f"Pooling {m} imputed datasets")

        # Restrict to numeric columns for pooling
        numeric_cols = self.imputed_datasets[0].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            msg = "No numeric columns available for pooling."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug(f"Found {len(numeric_cols)} numeric columns for pooling")

        col_names = numeric_cols
        q_mat = np.empty((m, len(col_names)))
        u_mat = np.empty_like(q_mat)

        n = self.imputed_datasets[0].shape[0]
        logger.debug(f"Sample size: {n}")

        # Calculate means and variances for each imputed dataset
        for j, df in enumerate(self.imputed_datasets):
            logger.debug(f"Processing imputed dataset {j + 1}/{m}")
            q_mat[j, :] = df.mean(axis=0).values  # q_j
            # Within-imputation variance of the mean: var / n
            u_mat[j, :] = df.var(ddof=1, axis=0).values / n

        # Apply Rubin's rules
        logger.debug("Applying Rubin's rules for pooling")
        q_bar = q_mat.mean(axis=0)                        # pooled estimate
        u_bar = u_mat.mean(axis=0)                        # average within
        b = ((q_mat - q_bar) ** 2).sum(axis=0) / (m - 1)  # between
        t = u_bar + (1 + 1/m) * b                        # total variance

        frac_miss_info = ((1 + 1/m) * b) / t

        # Log pooling statistics
        for i, col in enumerate(col_names):
            logger.debug(f"Pooling statistics for '{col}':")
            logger.debug(f"  - Pooled estimate: {q_bar[i]:.4f}")
            logger.debug(f"  - Total variance: {t[i]:.4f}")
            logger.debug(f"  - Fraction of missing information: {frac_miss_info[i]:.4f}")

        # Build covariance matrix
        cov_params = np.diag(t)
        logger.debug("Covariance matrix constructed")

        # Make variable names available for summaries
        self.exog_names = col_names

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
        Visualize per-iteration chain mean and variance of imputed values.

        Parameters
        ----------
        columns : list of str, optional
            List of column names to plot. If None, plots all columns tracked
            in `self.chain_mean`.
        """
        if not getattr(self, 'chain_mean', None) or not getattr(self, 'chain_var', None):
            raise ValueError("No chain statistics available – run `.impute()` first.")

        if columns is None:
            columns = list(self.chain_mean.keys())

        # Import locally to avoid heavy plotting libraries during module import
        import os
        from diagnostics.plots import plot_chain_stats
        
        # Create additionalz directory if it doesn't exist
        os.makedirs('additionalz', exist_ok=True)
        
        # Save the plot to the additionalz directory
        plot_chain_stats(
            self.chain_mean, 
            self.chain_var, 
            columns, 
            save_path='additionalz/chain_stats.png'
        )
        logger.info("Chain statistics plot saved to additionalz/chain_stats.png")


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

            predictor_cols = [c for c in updated_data.columns if c != col]
            predictors = updated_data[predictor_cols]

            # RF currently not implemented
            if method_name == ImputationMethod.RF.value:
                logger.warning(
                    f"Random Forest imputation not implemented yet; skipping column '{col}'"
                )
                continue

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
            ry_mask = self.id_obs[col]
            wy_mask = self.id_mis[col]
            ry = ry_mask.to_numpy()
            wy = wy_mask.to_numpy()

            # Get imputer function and perform imputation
            imputer_func = get_imputer_func(method_name)
            logger.debug(f"Using imputation method '{method_name}' for column '{col}'")

            try:
                imputed_values = imputer_func(y=y, ry=ry, wy=wy, x=predictors)
                logger.debug(f"Successfully imputed {len(imputed_values)} values for column '{col}'")
            except TypeError as e:
                logger.debug(f"Retrying imputation without 'wy' parameter for column '{col}'")
                try:
                    imputed_values = imputer_func(y=y, ry=ry, x=predictors)
                    logger.debug("Imputation successful on retry")
                except Exception as e:
                    logger.error(f"Failed to impute column '{col}': {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"Failed to impute column '{col}': {str(e)}")
                raise

            # Assign imputed values
            updated_data.loc[wy_mask, col] = imputed_values

            # Record chain statistics
            if wy.sum() > 0:
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
                nmis = np.array([len(self.id_mis[col]) for col in columns_with_missing])
                ii = np.argsort(nmis)
                self.visit_sequence = [columns_with_missing[i] for i in ii]

