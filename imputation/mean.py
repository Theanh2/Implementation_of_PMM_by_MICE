import numpy as np
import pandas as pd
from typing import Union, Optional

def mean(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values using the arithmetic mean of observed data.
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as other imputation methods.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (not used in this method, but kept for consistency)
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    **kwargs : dict
        Additional arguments (not used in this method)
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
    Notes
    -----
    This is the simplest imputation method that:
    1. Calculates the arithmetic mean of all observed values
    2. Replaces all missing values with this mean
    
    WARNING: Imputing the mean of a variable is almost never appropriate
    for serious analysis. This method:
    - Reduces variance in the data
    - Distorts relationships between variables
    - Underestimates standard errors
    - Should only be used as a baseline comparison
    
    See Little and Rubin (2002, p. 61-62) or Van Buuren (2012, p. 10-11)
    for detailed discussion of why mean imputation is problematic.
    
    This method ignores the predictor variables (x) and only uses the observed
    values of the target variable for imputation.
    """
    # Convert inputs to numpy arrays for consistency
    y = np.asarray(y)
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs
    
    # Get observed values
    y_obs = y[id_obs]
    
    # Check if we have any observed values
    if len(y_obs) == 0:
        raise ValueError("No observed values available for mean imputation")
    
    # Calculate mean of observed values
    mean_value = np.mean(y_obs)
    
    # Number of missing values to impute
    n_mis = np.sum(id_mis)
    
    # Create imputed values (all the same mean)
    imputed_values = np.full(n_mis, mean_value)
    
    return imputed_values 