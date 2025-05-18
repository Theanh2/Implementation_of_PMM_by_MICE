import pandas as pd
import warnings
from typing import Dict, Union, List

def check_n_imputations(n_imputations: int) -> None:
    """
    Check if the number of imputations is valid and provide a warning if it's high.
    
    Parameters
    ----------
    n_imputations : int
        Number of imputations to perform
        
    Raises
    ------
    ValueError
        If n_imputations is not a positive integer
    """
    if not isinstance(n_imputations, int):
        raise ValueError("n_imputations must be an integer")
    
    if n_imputations <= 0:
        raise ValueError("n_imputations must be positive")
        
    if n_imputations > 100:
        print(f"Warning: {n_imputations} imputations is a large number. This might take a while to compute.")

def check_method(method: Union[str, Dict[str, str]], columns: List[str]) -> Dict[str, str]:
    """
    Check and process the method parameter for MICE imputation.
    
    Parameters
    ----------
    method : Union[str, Dict[str, str]]
        Method specification. Can be:
        - str: use the same method for all columns
        - Dict[str, str]: dictionary mapping column names to their methods
    columns : List[str]
        List of column names in the data
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping each column to its imputation method
        
    Raises
    ------
    ValueError
        If method is invalid or references non-existent columns
    """
    supported_methods = ["pmm", "miles", "midas", "cart", "sample"]
    
    # If method is a string, validate and use for all columns
    if isinstance(method, str):
        if method not in supported_methods:
            raise ValueError(f"Unsupported method: {method}. Supported methods are: {supported_methods}")
        return {col: method for col in columns}
    
    # If method is a dictionary, validate each entry
    if isinstance(method, dict):
        # Check if all specified columns exist
        invalid_cols = [col for col in method.keys() if col not in columns]
        if invalid_cols:
            raise ValueError(f"Columns not found in data: {invalid_cols}")
        
        # Check if all methods are supported
        invalid_methods = {col: m for col, m in method.items() if m not in supported_methods}
        if invalid_methods:
            raise ValueError(f"Unsupported methods: {invalid_methods}. Supported methods are: {supported_methods}")
        
        # Create result dict with default method for unspecified columns
        # TODO: make default method dependent on column type or just handle it otherwise, 
        # e.g. not let method for a column be not specified
        result = {col: "sample" for col in columns}  # Default method
        result.update(method)  # Override with specified methods
        return result
    
    raise ValueError("method must be either a string or a dictionary")

def check_data(data) -> pd.DataFrame:
    """
    Check and validate input data for MICE imputation.
    
    Parameters
    ----------
    data : Any
        Input data to be checked and converted to DataFrame
        
    Returns
    -------
    pd.DataFrame
        Validated and cleaned DataFrame
        
    Raises
    ------
    ValueError
        If data cannot be converted to DataFrame or has duplicate column names
    """
    # Try to convert to DataFrame if it's not already one
    try:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Input data cannot be converted to DataFrame: {str(e)}")
    
    # Check for duplicate column names
    duplicate_cols = data.columns[data.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Found duplicate column names: {duplicate_cols}. Please make sure that the column names are unique.")
        raise ValueError("DataFrame contains duplicate column names")
    
    # Check for fully empty rows
    n_rows_before = len(data)
    data = data.dropna(how='all')
    n_rows_after = len(data)
    n_dropped = n_rows_before - n_rows_after
    
    if n_dropped > 0:
        print(f"Dropped {n_dropped} fully empty rows")
    
    # Check for columns with no values
    empty_cols = data.columns[data.isna().all()].tolist()
    if empty_cols:
        warnings.warn(f"Found columns with no values: {empty_cols}. These columns will be dropped as they cannot be imputed.")
        print(f"Dropping {len(empty_cols)} columns with no values: {empty_cols}")
        data = data.drop(columns=empty_cols)
    
    return data
