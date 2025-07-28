import numpy as np
import pandas as pd
from typing import Union, Optional, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import warnings

def mice_impute_rf(
    y: Union[pd.Series, np.ndarray],
    ry: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    wy: Optional[np.ndarray] = None,
    n_estimators: int = 10,
    rf_package: str = "sklearn",
    random_state: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Impute missing values using Random Forests.
    
    This function is designed to be compatible with the MICE framework,
    following the same interface as PMM, midas, CART, and sample imputation methods.
    
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
    n_estimators : int, default=10
        Number of trees in the forest (equivalent to R's ntree)
    rf_package : str, default="sklearn"
        Backend for random forest implementation. Currently supports "sklearn"
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters passed to the random forest model
        
    Returns
    -------
    np.ndarray
        Imputed values for missing positions only (matching R implementation).
        
    Notes
    -----
    The procedure is as follows:
    1. Fit a random forest using observed data
    2. For each missing value, find the terminal nodes across all trees
    3. For each tree, find eligible donors (observed values in the same terminal node)
    4. Randomly sample one donor from each tree
    5. Take the final imputed value as a random sample from the tree predictions
    
    This implementation follows the algorithm described in Doove et al. (2014).
    """
    # Convert inputs to numpy arrays for consistency
    y = np.asarray(y)
    x = np.asarray(x)
    ry = np.asarray(ry, dtype=bool)
    
    # Set default wy if not provided
    if wy is None:
        wy = ~ry
    
    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure minimum number of trees
    n_estimators = max(1, n_estimators)
    
    # Number of missing values to impute
    n_mis = np.sum(wy)
    
    # Split data into observed and missing
    x_obs = x[ry].copy()
    x_mis = x[wy].copy()
    y_obs = y[ry].copy()
    
    # Check if we have any missing values to impute
    if len(x_mis) == 0:
        # No missing values to impute, return empty array
        return np.array([])
    
    # Check if we have enough data
    if len(y_obs) < 2:
        raise ValueError("Need at least 2 observed values for random forest imputation")
    
    # Handle numeric and categorical variables differently
    if not pd.api.types.is_categorical_dtype(y_obs) and not pd.api.types.is_object_dtype(y_obs):
        # Regression case
        imputed_values = _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs)
    else:
        # Classification case
        imputed_values = _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs)
    
    return imputed_values

def _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs):
    """Helper function for regression random forest imputation."""
    n_mis = x_mis.shape[0]
    tree_predictions = []
    
    # Fit individual trees and get predictions
    for i in range(n_estimators):
        # Set random state for each tree
        tree_random_state = random_state + i if random_state is not None else None
        
        # Fit single tree
        tree = DecisionTreeRegressor(
            random_state=tree_random_state,
            **kwargs
        )
        tree.fit(x_obs, y_obs)
        
        # Get leaf nodes for observed and missing data
        leaf_nodes_obs = tree.apply(x_obs)
        leaf_nodes_mis = tree.apply(x_mis)
        
        # For each missing value, find eligible donors
        tree_pred = np.zeros(n_mis)
        for j, leaf in enumerate(leaf_nodes_mis):
            # Find observed values in the same leaf
            donors = y_obs[leaf_nodes_obs == leaf]
            if len(donors) > 0:
                # Randomly sample one donor
                tree_pred[j] = np.random.choice(donors)
            else:
                # Fallback: use mean of all observed values
                tree_pred[j] = np.mean(y_obs)
        
        tree_predictions.append(tree_pred)
    
    # Convert to array
    tree_predictions = np.array(tree_predictions).T  # Shape: (n_mis, n_estimators)
    
    # For each missing value, randomly sample from tree predictions
    imputed_values = np.array([
        np.random.choice(tree_pred) for tree_pred in tree_predictions
    ])
    
    return imputed_values

def _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs):
    """Helper function for classification random forest imputation."""
    n_mis = x_mis.shape[0]
    tree_predictions = []
    
    # Check if all observed values are in one category
    unique_cats = pd.unique(y_obs)
    if len(unique_cats) == 1:
        return np.repeat(unique_cats[0], n_mis)
    
    # Remove any unused categories
    y_obs = pd.Categorical(y_obs).remove_unused_categories()
    
    # Fit individual trees and get predictions
    for i in range(n_estimators):
        # Set random state for each tree
        tree_random_state = random_state + i if random_state is not None else None
        
        # Fit single tree
        tree = DecisionTreeClassifier(
            random_state=tree_random_state,
            **kwargs
        )
        tree.fit(x_obs, y_obs)
        
        # Get leaf nodes for observed and missing data
        leaf_nodes_obs = tree.apply(x_obs)
        leaf_nodes_mis = tree.apply(x_mis)
        
        # For each missing value, find eligible donors
        tree_pred = np.zeros(n_mis, dtype=object)
        for j, leaf in enumerate(leaf_nodes_mis):
            # Find observed values in the same leaf
            donors = y_obs[leaf_nodes_obs == leaf]
            if len(donors) > 0:
                # Randomly sample one donor
                tree_pred[j] = np.random.choice(donors)
            else:
                # Fallback: use most frequent class
                tree_pred[j] = pd.Series(y_obs).mode().iloc[0]
        
        tree_predictions.append(tree_pred)
    
    # Convert to array
    tree_predictions = np.array(tree_predictions).T  # Shape: (n_mis, n_estimators)
    
    # For each missing value, randomly sample from tree predictions
    imputed_values = np.array([
        np.random.choice(tree_pred) for tree_pred in tree_predictions
    ])
    
    return imputed_values

# Alias for compatibility with MICE framework
rf = mice_impute_rf 