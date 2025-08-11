import numpy as np
import pandas as pd
from typing import Union, Optional

def sample(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values by random sampling from observed values.
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as PMM, midas, and CART imputation methods.
    
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
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments (not used in this method)
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
    Notes
    -----
    This is the simplest imputation method that:
    1. Takes all observed values in the target variable
    2. Randomly samples from them to fill in missing values
    3. No modeling is involved, just random sampling with replacement
    
    This method ignores the predictor variables (x) and only uses the observed
    values of the target variable for imputation.
    
    Edge cases handled (matching R implementation):
    - If no observed values: returns random normal values
    - If only one observed value: duplicates it to allow sampling
    """
    # Convert inputs to numpy arrays for consistency
    y = np.asarray(y)
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs
    
    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get observed values
    y_obs = y[id_obs]
    
    # Handle edge cases (matching R implementation)
    if len(y_obs) < 1:
        # If no observed values, return random normal values
        n_mis = np.sum(id_mis)
        imputed_values = np.random.normal(0, 1, n_mis)
    elif len(y_obs) == 1:
        # If only one observed value, duplicate it to allow sampling
        y_obs = np.array([y_obs[0], y_obs[0]])
        n_mis = np.sum(id_mis)
        imputed_values = np.random.choice(y_obs, size=n_mis, replace=True)
    else:
        # Normal case: sample from observed values
        n_mis = np.sum(id_mis)
        imputed_values = np.random.choice(y_obs, size=n_mis, replace=True)
    
    return imputed_values 