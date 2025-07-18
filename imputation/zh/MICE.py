import pandas as pd
import warnings
import numpy as np
from typing import Dict, Union, Optional, List
from .validators import validate_dataframe, validate_columns, check_n_imputations, check_maxit, check_method, check_initial_method, validate_predictor_matrix, check_visit_sequence
from .constants import ImputationMethod, InitialMethod, SUPPORTED_METHODS, DEFAULT_METHOD, SUPPORTED_INITIAL_METHODS, DEFAULT_INITIAL_METHOD, VisitSequence

# pm and visit sequenc
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

        self.imputed_datasets = []
        # prepare imputations - methods for columns

        for i in range(self.n_imputations):
            self.imputed_datasets.append(self._impute_once())
        
        print(f"Parameters validated successfully:")
        print(f"  - Number of imputations: {self.n_imputations}")
        print(f"  - Maximum iterations: {self.maxit}")
        print(f"  - Initial method: {self.initial}")
        print(f"  - Method: {self.method}")
        print(f"  - Visit sequence: {self.visit_sequence}")
        print(f"  - Predictor matrix provided: {self.predictor_matrix is not None}")
        print(f"  - Method args provided: {len(self.method_args) > 0}")
        print(f"  - Imputed datasets will be stored in self.imputed_datasets")
    
    def _impute_once(self):
        """
        Perform one complete imputation cycle.
        
        Returns
        -------
        pd.DataFrame
            A copy of the data with one complete imputation cycle applied
        """
        current_data = self.data.copy(deep=True)
        self._initial_imputation(current_data)

        for i in range(self.maxit):
            current_data = self._iterate(current_data)
        
        return current_data
    
    def _iterate(self, data):
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
        pass
    
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

    def pool(self):
        pass