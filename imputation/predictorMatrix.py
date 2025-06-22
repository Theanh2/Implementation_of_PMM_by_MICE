import numpy as np
import pandas as pd
def quickpred(data, mincor=0.1, minpuc=0, include="", exclude="", method="pearson"):
    """
    Generates a predictor matrix indicating which variables to use as predictors for imputation.
    
    Works only if all columns are numeric. Categorical columns are not handled yet.
    In R's mice package, categorical variables use internal codes; 
    here, you preprocess categorical variables by encoding them numerically.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Data frame with incomplete data.
    mincor : float, optional
        Minimum threshold for the absolute correlation to consider a variable as predictor (default 0.1).
    minpuc : float, optional
        Minimum threshold for the proportion of usable cases (default 0).
    include : list or iterable of str, optional
        Variables to always include as predictors, regardless of other criteria.
    exclude : list or iterable of str, optional
        Variables to exclude as predictors.
    method : {'pearson', 'kendall', 'spearman'}, optional
        Correlation method passed to pandas `.corr()` (default 'pearson').
    
    Returns
    -------
    pandas.DataFrame
        Predictor matrix (0/1) with variables as both index and columns.
        A 1 indicates that the column variable is used as predictor for the index variable.
    
    Notes
    -----
    - Diagonal elements are zeroed out.
    - Variables with no missing values are excluded as predictors.
    - If a variable appears in both `include` and `exclude`, it will be included.
    """
    predictormatrix = pd.DataFrame(0, index=data.columns, columns=data.columns, dtype=int)
    r = data.notna()

    # Correlation matrices
    #pairwise correlation and replace NA with 0
    v = np.abs(pd.DataFrame(data).corr(method=method,numeric_only=True).fillna(0).to_numpy())
    #pairwise correlation and replace NA with 0
    u = np.abs(pd.DataFrame(data).corrwith(pd.DataFrame(r.astype(float)), method=method,numeric_only=True).fillna(0).to_numpy())

    maxc = np.maximum(v, u)
    predictormatrix[:] = (maxc > mincor).astype(int)

    # Exclude predictors below minpuc threshold
    if minpuc != 0:
        p = md_pairs(data)
        puc = p['mr'] / (p['mr'] + p['mm'])
        puc = puc.replace([np.inf, -np.inf], 0).fillna(0)
        predictormatrix[puc < minpuc] = 0

    # Exclude variables in 'exclude'
    if exclude:
        for col in data.columns:
            if col in exclude:
                predictormatrix[col].values[:] = 0


    # # Include variables in 'include'
    if include:
        for col in data.columns:
            if col in include:
                predictormatrix[col].values[:] = 1

    #Diagonal = 0
    np.fill_diagonal(predictormatrix.values, 0)

    #column no missing values set to 0
    complete_cases = data.isna().sum(axis=0) == 0
    predictormatrix.loc[complete_cases, :] = 0

    return predictormatrix
def md_pairs(data):
    """
    Calculates the number of observation pairs for variable pairs in the dataset.

    Mimics md.pairs from mice package, providing counts of observed-missing patterns:
    - rr: number of cases observed for both variables
    - rm: number of cases observed for the first variable and missing for the second
    - mr: number of cases missing for the first variable and observed for the second
    - mm: number of cases missing for both variables

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing missing values (NaNs).

    Returns
    -------
    dict
        Dictionary with keys 'rr', 'rm', 'mr', and 'mm', each mapping to
        a square matrix (numpy.ndarray) of shape (num_vars, num_vars).
    """
    r = data.notna().astype(int)
    m = data.isna().astype(int)
    rr = np.matmul(r.T, r)
    mm = np.matmul(m.T,m)
    mr = np.matmul(m.T,r)
    rm = np.matmul(r.T,m)
    return {'rr': rr, 'rm': rm, 'mr': mr, 'mm': mm}

