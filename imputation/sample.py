import numpy as np
import pandas as pd
from typing import Union, Optional

def mice_impute_sample(
    y: Union[pd.Series, np.ndarray],
    ry: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    wy: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values by random sampling from observed values.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    ry : np.ndarray
        Boolean mask of observed values in y
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (not used in this method, but kept for consistency)
    wy : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses !ry
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional arguments (not used in this method)
        
    Returns
    -------
    np.ndarray
        Imputed values for the missing entries
        
    Notes
    -----
    This is the simplest imputation method that:
    1. Takes all observed values
    2. Randomly samples from them to fill in missing values
    No modeling is involved, just random sampling with replacement.
    """
    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert input to numpy array if it's a pandas object
    if isinstance(y, pd.Series):
        y = y.values
    
    # Set default wy if not provided
    if wy is None:
        wy = ~ry
    
    # Get observed values
    y_obs = y[ry]
    
    # Number of missing values to impute
    n_mis = np.sum(wy)
    
    # Randomly sample from observed values
    imputed_values = np.random.choice(y_obs, size=n_mis, replace=True)
    
    return imputed_values 