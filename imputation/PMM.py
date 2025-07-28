import pandas as pd
from .sampler import *
from .Utils import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cross_decomposition import CCA
def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
                    quantify = True, ridge = 1e-5, matcher = "NN", **kwargs):
    """
       Predictive Mean Matching (PMM) imputation.

       This function imputes missing values in a variable `y` using predictive mean matching.
       The method is based on Rubin's (1987) Bayesian linear regression and mimics the behavior
       of the R `mice` package's PMM imputation method.

       Parameters
       ----------
       y : array-like (1D), shape (n_samples,)
           Target variable to be imputed. Can be numeric or categorical.

       ry : array-like of bool, shape (n_samples,)
           Logical array indicating which elements of `y` are observed (True) or missing (False).

       x : array-like (2D), shape (n_samples, n_features)
           Numeric design matrix of predictors. Must have no missing values.

       wy : array-like of bool, shape (n_samples,), optional
           Logical array indicating which values should be imputed.
           If None, wy is set to the complement of `ry`.

       donors : int, default=5
           Number of donors to draw from the observed cases when imputing missing values.

       matchtype : int, default=1
           Type of matching:
           - 0: Predicted value of y_obs vs predicted value of y_mis
           - 1: Predicted value of y_obs vs drawn value of y_mis (default)
           - 2: Drawn value of y_obs vs drawn value of y_mis

       quantify : bool, default=True
           If True and `y` is categorical, factor levels are replaced by the first canonical variate (via CCA).
           If False, categorical values are replaced by integer codes (less accurate).

       ridge : float, default=1e-5
           Ridge regularization parameter used in `norm_draw()` to stabilize estimation.
           Increase for multicollinear data, decrease to reduce bias.

       matcher : str, default="NN"
           Matching method. Currently only "NN" (nearest neighbor) is supported.

       **kwargs : dict
           Additional arguments passed to `norm_draw()`, such as `ls_meth`.

       Returns
       -------
       y_imp : np.ndarray
           Imputed values for missing positions only (matching R implementation).
           Returns object array if `y` was categorical, else float array.

       Notes
       -----
       Based on:
       - Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*.
       - Van Buuren, S. & Groothuis-Oudshoorn, K. (2011). `mice` R package.

       Examples
       --------
       >>> y = np.array([7, np.nan, 9, 10, 11])
       >>> ry = ~np.isnan(y)
       >>> x = np.array([[1, 2], [3, 4], [5, 7], [7, 8], [9, 10]])
       >>> pmm(y=y, ry=ry, x=x, donors=3)
    """
    if wy is None:
        wy = ~ry

    # Add a column of ones to the matrix x
    x = np.c_[np.ones(x.shape[0]), x]
    ynum = y
    # Quantify categories for categorical data y
    if y.dtype == "object":
        if quantify:
            # quantify function returns the numeric transformation of the factor
            #Experimental has different output than R
            #id to retransform cca to categories back
            ynum, id = quantify_cca(y, ry, x)
        else:
            ynum, id = pd.factorize(y)

    # Parameter estimation
    p = norm_draw(ynum, ry, x, ridge=ridge, **kwargs)

    #dotproduct x @ parameter = predicted values
    if matchtype == 0:
        yhatobs = np.dot(x[ry, :], p["coef"])
        yhatmis = np.dot(x[wy, :], p["coef"])
    elif matchtype == 1:
        yhatobs = np.dot(x[ry, :], p["coef"])
        yhatmis = np.dot(x[wy, :], p["beta"])
    elif matchtype == 2:
        yhatobs = np.dot(x[ry, :], p["beta"])
        yhatmis = np.dot(x[wy, :], p["beta"])

    idx = matcherid(d = yhatobs, t = yhatmis, matcher = "NN", k = donors)
    
    # Get the observed values that were selected as donors
    donor_values = ynum[ry][idx]
    
    # Handle categorical data retransformation if needed
    if y.dtype == "object":
        if quantify:
            #retransform cca numericals to categories
            donor_values_obj = donor_values.astype(object)
            for col in id.columns:
                val = id.at[0, col]
                mask = np.isclose(donor_values, val)  # Use donor values here
                donor_values_obj[mask] = col
            donor_values = donor_values_obj
    
    return donor_values
def quantify_cca(y, ry, x):
    """
    Factorize a categorical variable y into numeric values via optimal scaling
    using Canonical Correlation Analysis (CCA) with predictors x.

    Parameters
    ----------
    y : array-like, categorical variable with missing values
    ry : boolean array-like, mask indicating observed (True) and missing (False) in y
    x : array-like or DataFrame, predictors without missing values corresponding to y

    Returns
    -------
    ynum : numpy.ndarray
        Numeric transformation of y with missing positions as np.nan.
    id : pandas.DataFrame
        DataFrame representing the canonical components for the observed y.

    Notes
    -----
    This method encodes y as one-hot vectors, then applies CCA to find
    numeric representations that maximize correlation with predictors x.
    """
    # Subset y and x based on ry
    xd = np.array(x)[ry]
    yf = np.array(y)[ry]
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    yf = encoder.fit_transform(yf.reshape(-1, 1))

    #canonical correlation analysis to find "correlation"
    cca = CCA(scale=False, n_components=min(xd.shape[1], yf.shape[1]))
    # yf design matrix, xd data
    cca.fit(X=yf, y=xd)
    # yf design matrix, xd data
    xd_c, yf_c = cca.transform(X=yf, y=xd)
    scaler = StandardScaler()
    y_t = scaler.fit_transform(yf_c[:, 1].reshape(-1, 1)).flatten()
    ynum = np.array([np.nan] * len(y), dtype=np.float64)
    ynum[ry] = y_t
    id = pd.DataFrame([y_t], columns=y[ry].values)
    return ynum, id

