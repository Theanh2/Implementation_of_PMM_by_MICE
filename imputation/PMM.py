import numpy as np
import pandas as pd
from imputation.sampler import *
from imputation.Utils import *
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cross_decomposition import CCA, PLSSVD


def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
                    quantify = True, trim = 1, ridge = 1e-5, matcher = "NN", **kwargs):
    """
    :param y: Vector to be imputed
    :param ry: Logical vector of length(y) indicating the subset y[ry] of elements in y to which the imputation model is fitted.
        ry generally distinguishes the observed TRUE and missing values FALSE in y.
    :param x: Numeric design matrix with length(y) rows with predictors for y. Matrix x may have no missing values.
    :param wy: opposite of ry. indicates locations in y for which imputations are created
    :param donors: The size of the donor pool among which a draw is made. The default is donors = 5L.
        Setting donors = 1L always selects the closest match, but is not recommended.
        Values between 3L and 10L provide the best results in most cases (Morris et al, 2015).
    :param matchtype: Type of matching distance. The default choice matchtype = 1L calculates the distance between the predicted
        value of yobs and the drawn values of ymis (called type-1 matching). Other choices are matchtype = 0L
        (distance between predicted values) and matchtype = 2L (distance between drawn values).
    :param quantify: Logical. If TRUE, factor levels are replaced by the first canonical variate before fitting the imputation model.
        If false, the procedure reverts to the old behaviour and takes the integer codes
        (which may lack a sensible interpretation) Relevant only of y is a factor.
    :param trim: Scalar integer. Minimum number of observations required in a category in order to be considered as a
        potential donor value. Relevant only of y is a factor.
    :param ridge: The ridge penalty used in norm.draw() to prevent problems with multicollinearity.
        The default is ridge = 1e-05, which means that 0.01 percent of the diagonal is added to the cross-product.
        Larger ridges may result in more biased estimates. For highly noisy data (e.g. many junk variables),
        set ridge = 1e-06 or even lower to reduce bias. For highly collinear data, set ridge = 1e-04 or higher.
    :param kwargs:
    :return: Vector with imputed data, same type as y, and of length sum(wy)

    Example:
    y = np.array([7, np.nan, 9,10,11])
    ry = ~np.isnan(y)
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    p = pmm(x = x,ry = ry,y = y, matcher = "NN", donors = 3)
    """
    if wy is None:
        wy = ~ry

    # Add a column of ones to the matrix x
    x = np.c_[np.ones(x.shape[0]), x]
    #
    # Quantify categories for factors (for categorical data y)
    ynum = y
    if isinstance(y, pd.Categorical):
        if quantify:
            # quantify function returns the numeric transformation of the factor
            ynum = quantify(y, ry, x)
        else:
            ynum = pd.factorize(y)

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
    y[wy] = yhatobs[idx]
    return y



def quantify(y, ry, x):
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
    ynum = y
    ynum[ry] = scaler.fit_transform(yf_c[:,1].reshape(-1, 1))
    return ynum


##TEST
# y = pd.Series(["age", np.nan, "what", "w", "b", "b"])
# ry = ~pd.isna(y)
# x = np.array([[1, 2], [3, 4], [5, 6], [5, 4], [5, 6], [9, 6]])
#
# yf = np.array(y)[ry]
# xd = np.array(x)[ry]
# encoder = OneHotEncoder(sparse_output= False, drop = None)
# yf = encoder.fit_transform(yf.reshape(-1, 1))
#
# cca = cca(scale = False, n_components = min(xd.shape[1], yf.shape[1]))
# #yf design matrix, xd data
# cca.fit(X = yf, y = xd)
#
# xd_c, yf_c = cca.transform(X = yf, y = xd)
# ynum = y
# #replaces values with scaled coeffs
# ynum[ry] = StandardScaler().fit_transform(yf_c[:, [1]]).flatten()

#
# y = np.array([7, np.nan, 9, 10, 11])
# ry = ~np.isnan(y)
# x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [11, 10]])
# p = pmm(x=x, ry=ry, y=y, matcher="NN", donors=3)
# print(p)