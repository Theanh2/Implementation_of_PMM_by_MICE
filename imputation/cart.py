import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, Optional, Tuple

def mice_impute_cart(
    y: pd.Series,
    ry: np.ndarray,
    x: pd.DataFrame,
    wy: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 1e-4,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values using Classification and Regression Trees (CART).
    
    Parameters
    ----------
    y : pd.Series
        Target variable with missing values
    ry : np.ndarray
        Boolean mask of observed values in y
    x : pd.DataFrame
        Predictor variables
    wy : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses !ry
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    ccp_alpha : float, default=1e-4
        Complexity parameter for pruning
    **kwargs : dict
        Additional parameters passed to the tree model
        
    Returns
    -------
    np.ndarray
        Imputed values for the missing entries
        
    Notes
    -----
    The procedure is as follows:
    1. Fit a classification or regression tree by recursive partitioning
    2. For each missing value, find the terminal node it would end up in
    3. Make a random draw among the members in that node, and take the observed
       value from that draw as the imputation
    """
    # Set default wy if not provided
    if wy is None:
        wy = ~ry
    
    # Ensure minimum samples per leaf is at least 1
    min_samples_leaf = max(1, min_samples_leaf)
    
    # Add intercept if no predictors
    if x.shape[1] == 0:
        x = pd.DataFrame({'int': np.ones(len(x))})
    
    # Split data into observed and missing
    x_obs = x[ry].copy()
    x_mis = x[wy].copy()
    y_obs = y[ry].copy()
    
    # Handle numeric and categorical variables differently
    if not pd.api.types.is_categorical_dtype(y_obs) and not pd.api.types.is_object_dtype(y_obs):
        # Regression case
        tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            **kwargs
        )
        
        # Fit the tree
        tree.fit(x_obs, y_obs)
        
        # Get leaf nodes for observed data
        leaf_nodes = tree.apply(x_obs)
        
        # Get leaf nodes for missing data
        mis_leaf_nodes = tree.apply(x_mis)
        
        # For each missing value, sample from the same leaf node
        imputed_values = np.zeros(np.sum(wy))
        for i, leaf in enumerate(mis_leaf_nodes):
            # Get all observed values in the same leaf
            leaf_values = y_obs[leaf_nodes == leaf]
            # Randomly sample one value
            imputed_values[i] = np.random.choice(leaf_values)
            
    else:
        # Classification case
        # Check if all observed values are in one category
        unique_cats = pd.unique(y_obs)
        if len(unique_cats) == 1:
            return np.repeat(unique_cats[0], np.sum(wy))
        
        # Remove any unused categories
        y_obs = pd.Categorical(y_obs).remove_unused_categories()
        
        tree = DecisionTreeClassifier(
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
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
    
    return imputed_values
