import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

# Get a logger for the current module.
# This will be a child of the 'imputation' logger configured in MICE.py
logger = logging.getLogger(__name__)

def rf(
    y: Union[pd.Series, np.ndarray],
    id_obs: np.ndarray,
    x: Union[pd.DataFrame, np.ndarray],
    id_mis: Optional[np.ndarray] = None,
    n_estimators: int = 10,
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
    id_obs : np.ndarray
        Boolean mask of observed values in y (True for observed, False for missing)
    x : Union[pd.DataFrame, np.ndarray]
        Predictor variables (must be fully observed)
    id_mis : np.ndarray, optional
        Boolean mask of missing values to impute. If None, uses ~id_obs
    n_estimators : int, default=10
        Number of trees in the forest (equivalent to R's ntree)
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
    
    This implementation follows the algorithm described in Doove et al. (2014)
    and closely mirrors the R mice implementation.
    """
    logger.debug("Starting Random Forest imputation.")
    
    # Pre-process x to handle categorical predictors
    if isinstance(x, pd.DataFrame) and (x.select_dtypes(include=['object', 'category']).shape[1] > 0):
        logger.debug("One-hot encoding categorical predictors.")
        # One-hot encode categorical features, which is necessary for scikit-learn.
        # This mimics R's ability to handle factors.
        x = pd.get_dummies(x, drop_first=True)
    
    # Convert inputs to numpy arrays for consistency, y is handled later
    x = np.asarray(x)
    id_obs = np.asarray(id_obs, dtype=bool)
    
    # Set default id_mis if not provided
    if id_mis is None:
        id_mis = ~id_obs
    
    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Ensure minimum number of trees
    n_estimators = max(1, n_estimators)
    
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
        logger.warning("Not enough observed data to fit a random forest. Using fallback imputation.")
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
        logger.debug("Performing regression random forest imputation.")
        # Regression case
        imputed_values = _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs)
    else:
        logger.debug("Performing classification random forest imputation.")
        # Classification case
        imputed_values = _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs)
    
    logger.debug(f"Random Forest imputation complete. Imputed {len(imputed_values)} values.")
    return imputed_values

def _rf_regression_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs):
    """Helper function for regression random forest imputation using RandomForestRegressor."""
    n_mis = x_mis.shape[0]
    
    # Fit the random forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )
    rf.fit(x_obs, y_obs)
    
    # Get terminal nodes for all trees at once
    # For observed data
    nodes_obs = np.array([tree.apply(x_obs) for tree in rf.estimators_]).T  # Shape: (n_obs, n_estimators)
    # For missing data  
    nodes_mis = np.array([tree.apply(x_mis) for tree in rf.estimators_]).T  # Shape: (n_mis, n_estimators)
    
    # For each missing value, collect donors from each tree
    tree_predictions = []
    
    for i in range(n_mis):  # For each missing observation
        tree_pred = []
        for j in range(n_estimators):  # For each tree
            # Find observed values in the same terminal node as this missing value
            same_node_mask = nodes_obs[:, j] == nodes_mis[i, j]
            donors = y_obs[same_node_mask]
            
            if len(donors) > 0:
                # Randomly sample one donor
                tree_pred.append(np.random.choice(donors))
            else:
                # Fallback: use mean of all observed values
                tree_pred.append(np.mean(y_obs))
        
        tree_predictions.append(tree_pred)
    
    # For each missing value, randomly sample from its tree predictions
    imputed_values = np.array([
        np.random.choice(predictions) for predictions in tree_predictions
    ])
    
    return imputed_values

def _rf_classification_impute(x_obs, x_mis, y_obs, n_estimators, random_state, **kwargs):
    """Helper function for classification random forest imputation using RandomForestClassifier."""
    n_mis = x_mis.shape[0]
    
    # Check if all observed values are in one category (matching R behavior)
    unique_cats = np.unique(y_obs)
    if len(unique_cats) == 1:
        return np.repeat(unique_cats[0], n_mis)
    
    # Fit the random forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs
    )
    rf.fit(x_obs, y_obs)
    
    # Get terminal nodes for all trees at once
    # For observed data
    nodes_obs = np.array([tree.apply(x_obs) for tree in rf.estimators_]).T  # Shape: (n_obs, n_estimators)
    # For missing data
    nodes_mis = np.array([tree.apply(x_mis) for tree in rf.estimators_]).T  # Shape: (n_mis, n_estimators)
    
    # For each missing value, collect donors from each tree
    tree_predictions = []
    
    for i in range(n_mis):  # For each missing observation
        tree_pred = []
        for j in range(n_estimators):  # For each tree
            # Find observed values in the same terminal node as this missing value
            same_node_mask = nodes_obs[:, j] == nodes_mis[i, j]
            donors = y_obs[same_node_mask]
            
            if len(donors) > 0:
                # Randomly sample one donor
                tree_pred.append(np.random.choice(donors))
            else:
                # Fallback: use most frequent class
                from collections import Counter
                most_frequent = Counter(np.asarray(y_obs)).most_common(1)[0][0]
                tree_pred.append(most_frequent)
        
        tree_predictions.append(tree_pred)
    
    # For each missing value, randomly sample from its tree predictions
    imputed_values = np.array([
        np.random.choice(predictions) for predictions in tree_predictions
    ])
    
    return imputed_values 