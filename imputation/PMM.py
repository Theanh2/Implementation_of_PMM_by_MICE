import pandas as pd
from imputation.sampler import *
from imputation.Utils import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cross_decomposition import CCA


def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
                    quantify = True, ridge = 1e-5, matcher = "NN", **kwargs):
    """
    :param y: Array: Vector to be imputed
    :param ry: Logical: vector of length(y). ry distinguishes the observed TRUE and missing values FALSE in y.
    :param x: Array: Numeric design matrix with length(y) rows with predictors for y. Matrix x may have no missing values.
    :param wy: opposite of ry
    :param donors: Numeric: size of donor pool
    :param matchtype: 0, 1 or 2: Type of matching distance. The default type 1 matching calculates the distance between the predicted
        value of yobs and the drawn values of ymis (called type-1 matching). Other choices are type 0 matching (distance between predicted values)
        and type 2 matching (distance between drawn values).
    :param quantify: Logical. If TRUE, factor levels are replaced by the first canonical variate before fitting the imputation model.
        If false, the procedure reverts to the old behaviour and takes the integer codes
        Relevant only of y is categorical. !!!WIP
    :param ridge: The ridge penalty used in norm.draw() to prevent problems with multicollinearity.
        The default is ridge = 1e-05, which means that 0.01 percent of the diagonal is added to the cross-product.
        Larger ridges may result in more biased estimates. For highly noisy data (e.g. many junk variables),
        set ridge = 1e-06 or even lower to reduce bias. For highly collinear data, set ridge = 1e-04 or higher. !cite
    :param kwargs:
    :return: Vector with imputed data

    Example:
    Numerical:
    y = np.array([7, np.nan, 9,10,11])
    ry = ~np.isnan(y)
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    p = pmm(x = x,ry = ry,y = y, matcher = "NN", donors = 3)

    Categorical:
    y = pd.Series(["age", np.nan, "what", "w", "b"])
    ry = ~pd.isna(y)
    x = np.array([[1, 2], [3, 4], [5, 6], [9, 8], [9, 10]])
    p = pmm(x = x,ry = ry,y = y, matcher = "NN", donors = 3)
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
    #replace y (missing value array) with imputed value from observed values
    ynum[wy] = yhatobs[idx]

    if y.dtype == "object":
        if quantify:
            #retransform cca numericals to categories
            ynumobj = ynum.astype(object)
            for col in id.columns:
                val = id.at[0, col]
                mask = np.isclose(ynum, val)  # Use original numeric arr here
                ynumobj[mask] = col
        ynum = ynumobj
    return ynum

def quantify_cca(y, ry, x):
    """
    factorization of categorical variables via optimal scaling

    :param y: categorical variable
    :param ry: bool vector indicating missing values
    :param x: data
    :return:
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

