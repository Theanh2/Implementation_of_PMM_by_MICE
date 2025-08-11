import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, Optional
import logging
logger = logging.getLogger(__name__)

def cart(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 1e-4,
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values using Classification and Regression Trees (CART).
    
    This function is designed to be compatible with the MICE framework.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (must be fully observed)
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    ccp_alpha : float, default=1e-4
        Complexity parameter for pruning
    random_state : int, optional
        Random seed for reproducibility; also passed to the sklearn tree
    **kwargs : dict
        Additional parameters passed to the tree model
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
    Notes
    -----
    The procedure is as follows:
    1. Fit a classification or regression tree by recursive partitioning
    2. For each missing value, find the terminal node it would end up in
    3. Make a random draw among the members in that node, and take the observed
       value from that draw as the imputation
    """
    logger.debug("Starting CART imputation.")
    
    # Pre-process x to handle categorical predictors
    if isinstance(x, pd.DataFrame) and (x.select_dtypes(include=['object', 'category']).shape[1] > 0):
        logger.debug("One-hot encoding categorical predictors for column %s.", x.select_dtypes(include=['object', 'category']).columns[0])
        # One-hot encode categorical features, which is necessary for scikit-learn.
        # This mimics R's ability to handle factors.
        x = pd.get_dummies(x, drop_first=True)
    
    # Convert inputs to numpy arrays for consistency, y is handled later
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs

    # Set random state if provided for reproducibility
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure minimum samples per leaf is at least 1
    min_samples_leaf = max(1, min_samples_leaf)
    
    # Add intercept if no predictors (matching R behavior)
    if x.shape[1] == 0:
        x = np.ones((len(x), 1))
    
    # Split data into observed and missing
    x_obs = x[id_obs].copy()
    x_mis = x[id_mis].copy()
    y_obs = y[id_obs]
    
    # Check if we have any missing values to impute
    if len(x_mis) == 0:
        # No missing values to impute, return empty array
        return np.array([])
    
    # Check if we have enough observed data to fit the model
    if len(y_obs) < 2:
        logger.warning("Not enough observed data to fit a tree. Using fallback imputation.")
        # Not enough observed data, use mean/sample for imputation
        is_numeric = pd.api.types.is_numeric_dtype(y_obs)
        if is_numeric:
            # Numeric case - use mean
            mean_val = np.mean(y_obs)
            imputed_values = np.full(np.sum(id_mis), mean_val)
        else:
            # Categorical case - use most frequent
            from collections import Counter
            # np.asarray is needed in case y_obs is a pandas Series
            most_frequent = Counter(np.asarray(y_obs)).most_common(1)[0][0]
            imputed_values = np.full(np.sum(id_mis), most_frequent)
        
        return imputed_values
    
    # Handle numeric and categorical variables differently
    is_numeric = pd.api.types.is_numeric_dtype(y_obs)
    if is_numeric:
        logger.debug("Performing regression tree imputation.")
        # Regression case
        tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
            **kwargs
        )
        
        # Fit the tree
        tree.fit(x_obs, y_obs)
        
        # Get leaf nodes for observed data
        leaf_nodes = tree.apply(x_obs)
        
        # Get leaf nodes for missing data
        mis_leaf_nodes = tree.apply(x_mis)
        
        # For each missing value, sample from the same leaf node
        imputed_values = np.zeros(np.sum(id_mis))
        y_obs_arr = np.asarray(y_obs)
        for i, leaf in enumerate(mis_leaf_nodes):
            # Get all observed values in the same leaf
            leaf_values = y_obs_arr[leaf_nodes == leaf]
            # Randomly sample one value
            imputed_values[i] = np.random.choice(leaf_values)
            
    else:
        logger.debug("Performing classification tree imputation.")
        # Classification case - following R implementation more closely
        
        # Check if all observed values are in one category (matching R behavior)
        unique_cats, _ = np.unique(y_obs, return_counts=True)
        if len(unique_cats) == 1:
            return np.repeat(unique_cats[0], np.sum(id_mis))
        
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            random_state=random_state,
            **kwargs
        )
        
        # Fit the tree
        tree.fit(x_obs, y_obs)
        
        # Get class probabilities for missing data
        class_probs = tree.predict_proba(x_mis)
        
        # Sample from the predicted class probabilities
        imputed_values = np.array([
            np.random.choice(tree.classes_, p=probs)
            for probs in class_probs
        ])
    
    logger.debug(f"CART imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values
