import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from typing import Union, Optional, Tuple

def mice_impute_cart(
    y: Union[pd.Series, np.ndarray],
    ry: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    wy: Optional[np.ndarray] = None,
    min_samples_leaf: int = 5,
    ccp_alpha: float = 1e-4,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values using Classification and Regression Trees (CART).
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as PMM and midas imputation methods.
    
    Parameters
    ----------
    y : Union[pd.Series, np.ndarray]
        Target variable with missing values
    ry : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (must be fully observed)
    wy : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~ry
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node (equivalent to R's minbucket)
    ccp_alpha : float, default=1e-4
        Complexity parameter for pruning (equivalent to R's cp)
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
    
    This implementation closely follows the R mice package's cart imputation method.
    """
    # Convert inputs to numpy arrays for consistency
    y = np.asarray(y)
    x = np.asarray(x)
    ry = np.asarray(ry, dtype=bool)
    
    # Set default wy if not provided
    if wy is None:
        wy = ~ry
    
    # Ensure minimum samples per leaf is at least 1
    min_samples_leaf = max(1, min_samples_leaf)
    
    # Add intercept if no predictors (matching R behavior)
    if x.shape[1] == 0:
        x = np.ones((len(x), 1))
    
    # Split data into observed and missing
    x_obs = x[ry].copy()
    x_mis = x[wy].copy()
    y_obs = y[ry].copy()
    
    # Check if we have any missing values to impute
    if len(x_mis) == 0:
        # No missing values to impute, return empty array
        return np.array([])
    
    # Check if we have enough observed data to fit the model
    if len(y_obs) < 2:
        # Not enough observed data, use mean/sample for imputation
        if not pd.api.types.is_categorical_dtype(y_obs) and not pd.api.types.is_object_dtype(y_obs):
            # Numeric case - use mean
            mean_val = np.mean(y_obs)
            imputed_values = np.full(np.sum(wy), mean_val)
        else:
            # Categorical case - use most frequent
            from collections import Counter
            most_frequent = Counter(y_obs).most_common(1)[0][0]
            imputed_values = np.full(np.sum(wy), most_frequent)
        
        return imputed_values
    
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
        # Classification case - following R implementation more closely
        
        # Check if all observed values are in one category (matching R behavior)
        unique_cats, counts = np.unique(y_obs, return_counts=True)
        if len(unique_cats) == 1:
            return np.repeat(unique_cats[0], np.sum(wy))
        
        # Check if any category has all observed values (R's cat.has.all.obs logic)
        if np.any(counts == np.sum(ry)):
            dominant_cat = unique_cats[counts == np.sum(ry)][0]
            return np.repeat(dominant_cat, np.sum(wy))
        
        # Remove any unused categories (equivalent to R's droplevels)
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

# Alias for compatibility with MICE framework
cart = mice_impute_cart
