import warnings
import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List
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

        self.data = validate_dataframe(data)
        self.data = validate_columns(self.data)
        
        self.id_obs = {}
        self.id_mis = {}
        for col in self.data.columns:
            notna = self.data[col].notna()
            self.id_obs[col] = notna
            self.id_mis[col] = ~notna
        # Container for pooled results
        self.result = None  # Will hold the pooled `MICEresult` instance

        # Required by statsmodels result wrappers
        self.nobs = self.data.shape[0]

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
            
        Returns
        -------
        None
            Imputed datasets are stored in self.imputed_datasets attribute
            
        Raises
        ------
        ValueError
            If any parameters are invalid
            
        Notes
        -----
        After calling this method, the imputed datasets can be accessed via
        self.imputed_datasets as a list of pandas DataFrames.
        """

        check_n_imputations(n_imputations)

        check_maxit(maxit)

        check_initial_method(initial)
        
        if predictor_matrix is not None:
            predictor_matrix = validate_predictor_matrix(predictor_matrix, list(self.data.columns), self.data)
        
        if method is not None:
            self.method = check_method(method, list(self.data.columns))
        else:
            self.method = DEFAULT_METHOD
        
        self.n_imputations = n_imputations
        self.maxit = maxit
        self.predictor_matrix = predictor_matrix
        self.initial = initial

        self._set_visit_sequence(visit_sequence)

        # -------------------------------------------------------------
        # Prepare containers for chain statistics. These mimic the
        # `imp$chainMean` and `imp$chainVar` objects available in the
        # original R implementation: dimension order is
        #   [iteration, chain, variable]
        # Only track columns that are actually being imputed (visit_sequence)
        # Both are stored as dictionaries keyed by column name for
        # convenient access while keeping the memory footprint modest.
        # -------------------------------------------------------------
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
            self.imputed_datasets.append(self._impute_once(chain_idx))
        
        print(f"Parameters validated successfully:")
        print(f"  - Number of imputations: {self.n_imputations}")
        print(f"  - Maximum iterations: {self.maxit}")
        print(f"  - Initial method: {self.initial}")
        print(f"  - Method: {self.method}")
        print(f"  - Visit sequence: {self.visit_sequence}")
        print(f"  - Predictor matrix provided: {self.predictor_matrix is not None}")
        print(f"  - Imputed datasets will be stored in self.imputed_datasets")

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

        if not self.imputed_datasets:
            raise ValueError("No imputed datasets found – run `.impute()` first.")

        m = len(self.imputed_datasets)

        # Restrict to numeric columns for pooling
        numeric_cols = self.imputed_datasets[0].select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns available for pooling.")

        col_names = numeric_cols
        q_mat = np.empty((m, len(col_names)))
        u_mat = np.empty_like(q_mat)

        n = self.imputed_datasets[0].shape[0]

        for j, df in enumerate(self.imputed_datasets):
            q_mat[j, :] = df.mean(axis=0).values  # q_j
            # Within-imputation variance of the mean: var / n
            u_mat[j, :] = df.var(ddof=1, axis=0).values / n

        # Rubin's rules
        q_bar = q_mat.mean(axis=0)                        # pooled estimate
        u_bar = u_mat.mean(axis=0)                        # average within
        b = ((q_mat - q_bar) ** 2).sum(axis=0) / (m - 1)  # between
        t = u_bar + (1 + 1/m) * b                        # total variance

        frac_miss_info = ((1 + 1/m) * b) / t

        # Build covariance matrix (diagonal – assumes independence across vars)
        cov_params = np.diag(t)

        # Make variable names available for summaries that expect them
        self.exog_names = col_names

        # Create results object (scale=1 so cov_params == normalized_cov_params)
        self.result = MICEresult(self, q_bar, cov_params)
        self.result.scale = 1.0
        self.result.frac_miss_info = frac_miss_info

        if summ:
            return self.result.summary()

    def _impute_once(self, chain_idx: int):
        """
        Perform one complete imputation cycle.
        
        Returns
        -------
        pd.DataFrame
            A copy of the data with one complete imputation cycle applied
        """
        current_data = self.data.copy(deep=True)
        self._initial_imputation(current_data)

        for iter_idx in range(self.maxit):
            current_data = self._iterate(current_data, iter_idx, chain_idx)
        
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

        for col in self.visit_sequence:
            method_name = self.method[col]

            predictor_cols = [c for c in updated_data.columns if c != col]
            predictors = updated_data[predictor_cols]

            # RF currently not implemented
            if method_name == ImputationMethod.RF.value:
                warnings.warn(
                    "Random Forest imputation is not implemented yet; skipping "
                    f"column '{col}'.",
                    UserWarning,
                )
                continue

            # Determine predictors – use predictor_matrix if provided
            if self.predictor_matrix is not None:
                # predictor_matrix rows index = target, columns = predictors
                predictor_flags = self.predictor_matrix.loc[col]
                predictor_cols = predictor_flags[predictor_flags == 1].index.tolist()
                predictor_cols = [c for c in predictor_cols if c != col]
            else:
                predictor_cols = [c for c in updated_data.columns if c != col]

            predictors = updated_data[predictor_cols]

            # Prepare arrays/masks
            y = updated_data[col].to_numpy()
            ry_mask = self.id_obs[col]
            wy_mask = self.id_mis[col]
            ry = ry_mask.to_numpy()
            wy = wy_mask.to_numpy()

            # Get the appropriate imputer function and perform imputation
            imputer_func = get_imputer_func(method_name)

            try:
                imputed_values = imputer_func(y=y, ry=ry, wy=wy, x=predictors)
            except TypeError:
                # Some imputers (e.g. sample) might not accept wy explicitly
                imputed_values = imputer_func(y=y, ry=ry, x=predictors)


            # Assign imputed values back to the DataFrame
            updated_data.loc[wy_mask, col] = imputed_values

            # -----------------------------------------------------
            # Record chain statistics (mean & variance of newly
            # imputed values for this variable in this iteration &
            # chain). Aligns with behaviour of `mice` in R.
            # -----------------------------------------------------
            if wy.sum() > 0:
                imputed_arr = np.asarray(imputed_values, dtype=float)
                # Mean of imputed values for current variable
                self.chain_mean[col][iter_idx, chain_idx] = np.nanmean(imputed_arr)

                # Variance is undefined for a single value; handle safely
                if imputed_arr.size > 1:
                    self.chain_var[col][iter_idx, chain_idx] = np.nanvar(imputed_arr, ddof=1)
                else:
                    self.chain_var[col][iter_idx, chain_idx] = np.nan
 
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

