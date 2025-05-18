import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable
import random
from imputation.helpers import check_n_imputations, check_data, check_method
from imputation.cart import mice_impute_cart
from imputation.PMM import pmm
from imputation.miles import mice_impute_li
from imputation.midas import mice_impute_midas
from imputation.sample import mice_impute_sample

class MICEimputer:
    # init - only initializing everything (passing parameters, calculation or column order)
    # impute - perform the actual imputation, return a list of imputed datasets
    def __init__(
        self,
        data: pd.DataFrame,
        predictor_matrix: Optional[np.ndarray] = None,
        method: Optional[Union[str, Dict[str, str]]] = None,
        random_state: Optional[int] = None,
        initial: str = "sample",
        method_args: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize the MICE imputer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with missing values
        predictor_matrix : np.ndarray, optional
            Matrix defining relationships between variables
        method : Union[str, Dict[str, str]], optional
            Imputation method(s) to use. Can be:
            - None: use default method ("pmm") for all columns
            - str: use the same method for all columns
            - Dict[str, str]: dictionary mapping column names to their methods
            Supported methods: "pmm", "miles", "midas", "cart", "sample"
        random_state : int, optional
            Random seed for reproducibility
        initial : str, default="sample"
            Initial imputation method ("sample" or "meanobs")
        method_args : Dict[str, Dict], optional
            Additional arguments for each imputation method
        """
        # Check and validate input data
        self.data = check_data(data)
        
        # Store other input parameters
        self.predictor_matrix = predictor_matrix
        self.initial = initial
        self.method_args = method_args or {}
        
        # Set random state
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            
        # Initialize internal attributes
        self.meth = check_method(method, list(self.data.columns))  # Methods for each variable
        self.id_obs = {}  # Indices of observed values
        self.id_mis = {}  # Indices of missing values
        
        # Initialize observed and missing indices
        self._initialize_indices()
        
        # Define imputation order (columns with fewest missing values first)
        self._define_imputation_order()
        
        # Perform initial imputation
        self._initial_imputation()

    def _initialize_indices(self):
        """Initialize indices for observed and missing values for each column."""
        for col in self.data.columns:
            id_obs, id_mis = self._split_indices(self.data[col])
            self.id_obs[col] = id_obs
            self.id_mis[col] = id_mis

    def _define_imputation_order(self):
        """
        Define the order in which variables are imputed in each cycle.
        Variables with the fewest missing values are imputed first.
        """
        # Get number of missing values for each column
        nmis = np.array([len(self.id_mis[col]) for col in self.data.columns])
        
        # Sort columns by number of missing values (ascending)
        # Only include columns that have missing values
        ii = np.argsort(nmis)
        ii = ii[sum(nmis == 0):]  # Skip columns with no missing values
        self._imputation_order = [self.data.columns[i] for i in ii]

    def _split_indices(self, col: pd.Series) -> tuple:
        """
        Split indices into observed and missing values.
        
        Parameters
        ----------
        col : pd.Series
            Column to split indices for
            
        Returns
        -------
        tuple
            (indices of observed values, indices of missing values)
        """
        null = pd.isnull(col)
        id_obs = np.flatnonzero(~null)
        id_mis = np.flatnonzero(null)
        if len(id_obs) == 0:
            raise ValueError(f"Variable {col.name} has no observed values")
        return id_obs, id_mis

    def _initial_imputation(self):
        """
        Perform initial imputation using the specified method.
        """
        if self.initial == "sample":
            for col in self.data.columns:
                robs = self.data.loc[self.id_obs[col], col].values
                for idx in self.id_mis[col]:
                    self.data.at[idx, col] = np.random.choice(robs)
        elif self.initial == "meanobs":
            for col in self.data.columns:
                di = self.data[col] - self.data[col].mean()
                di = np.abs(di)
                ix = di.idxmin()
                self.data.loc[self.id_mis[col], col] = self.data[col].loc[ix]

    def _get_imputer_func(self, method_name: str) -> Callable:
        """
        Get the appropriate imputation function for the given method.
        
        Parameters
        ----------
        method_name : str
            Name of the imputation method
            
        Returns
        -------
        Callable
            The imputation function to use
        """
        imputer_map = {
            "cart": mice_impute_cart,
            "pmm": pmm,
            "miles": mice_impute_li,
            "midas": mice_impute_midas,
            "sample": mice_impute_sample
        }
        
        if method_name not in imputer_map:
            raise ValueError(f"Unsupported imputation method: {method_name}")
            
        return imputer_map[method_name]

    def _get_predictors_for(self, col: str) -> List[str]:
        """
        Get the list of predictor columns for a given target column based on the predictor matrix.
        
        Parameters
        ----------
        col : str
            Target column name
            
        Returns
        -------
        List[str]
            List of predictor column names
        """
        if self.predictor_matrix is None:
            # If no predictor matrix is provided, use all other columns as predictors
            return [c for c in self.data.columns if c != col]
            
        # Get the index of the target column
        col_idx = list(self.data.columns).index(col)
        
        # Get predictor columns based on the predictor matrix
        predictor_indices = np.where(self.predictor_matrix[col_idx, :] == 1)[0]
        return [self.data.columns[i] for i in predictor_indices]

    def impute(self, n_imputations: int = 5) -> List[pd.DataFrame]:
        """
        TODO:
        - convergence criterions instead of fixed number of iterations
        - if random sample, don't do over multiple iterations
        
        Perform multiple imputations.
        
        Parameters
        ----------
        n_imputations : int, default=5
            Number of imputations to perform
            
        Returns
        -------
        List[pd.DataFrame]
            List of imputed datasets
        """
        # Check if n_imputations is valid
        check_n_imputations(n_imputations)
        
        imputed_datasets = []

        for i in range(n_imputations):
            current_imputation = self.data.copy(deep=True)  # fresh copy per imputation

            # Perform multiple iterations for each imputation
            for iteration in range(5):  # TODO: replace with convergence criterion
                for col in self._imputation_order:
                    method_name = self.meth[col]
                    imputer_func = self._get_imputer_func(method_name)
                    
                    y = current_imputation[col]
                    # Use pre-calculated indices
                    ry = np.zeros(len(y), dtype=bool)
                    ry[self.id_obs[col]] = True
                    wy = np.zeros(len(y), dtype=bool)
                    wy[self.id_mis[col]] = True

                    # Select predictors for this column using the predictor matrix
                    predictors = self._get_predictors_for(col)
                    x = current_imputation[predictors]

                    if x.isnull().values.any():
                        raise ValueError(f"Predictors for {col} contain missing values. Check initialization or predictor matrix.")

                    # Get method-specific arguments
                    method_kwargs = self.method_args.get(col, {}).copy()
                    
                    # Add random state for reproducibility
                    if self.random_state is not None:
                        method_kwargs['random_state'] = self.random_state + i

                    # Call the selected imputation function
                    imputed_values = imputer_func(
                        y=y,
                        ry=ry,
                        x=x,
                        wy=wy,
                        **method_kwargs
                    )

                    # Apply imputations to missing values
                    current_imputation.loc[wy, col] = imputed_values

            imputed_datasets.append(current_imputation)

        return imputed_datasets
