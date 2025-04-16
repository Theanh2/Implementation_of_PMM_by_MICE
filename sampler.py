import numpy as np
import pandas as pd
from scipy.stats import chi2

def sym(x):
    #mimics sym function of R mice internal sym() function
    (x + x.T) / 2

def norm_draw(y, ry, x, rank_adjust=True, **kwargs):
    """
    #From R Mice impute.norm norm.draw() function
    #Alogorithm: https://www.rdocumentation.org/packages/mice/versions/3.17.0/topics/mice.impute.norm (Rubin(1987, p. 167)
    # Draws values of beta and sigma by Bayesian linear regression
    # Uses least squares parameters from ...
    :param y: Incomplete data vector of length n
    :param ry: Vector of missing data pattern
    :param x: Matrix n x p of complete covariates
    :param rank_adjust: rank.adjust Argument that specifies whether NA in the coefficients need to be set to zero.
    Only relevant when ls.meth = "qr" AND the predictor matrix is rank-deficient.
    :param kwargs: Other names arguments
    :return: list containing components coef (least squares estimate), beta (drawn regression weights)
    and sigma (drawn value of the residual standard deviation.
    """
    #estimice
    #p = estimice()

    #               sqrt(   sum((p$r)   ^2) /   rchisq(1, p$df)) #one random deviate with p$df
    #sqrt because we need sigma
    sigma_star = np.sqrt(np.sum(p["r"] ** 2) / chi2.rvs(p["df"], size=1))

    #p$c + (t(chol(sym(p$v))) %*% rnorm(ncol(x))) * sigma.star
    #sym ensures matrix is symmetrical, must be positive definite
    #np.linalg.cholesky returns lower triangular matrix
    chol_v = np.linalg.cholesky(sym(p["v"]))
    #coef + lower triangular matrix from Cholesky Decomposition * random n draws from standard normal * sigma
    beta_star = p["c"] + (chol_v.T @ np.random.normal(size=x.shape[1])) * sigma_star

    #return list
    parm = {
        "coef": p["c"],
        "beta": beta_star,
        "sigma": sigma_star,
        "estimation": p["ls_meth"]
    }
    #Replaces NaN with 0 if rank_adjust = True
    if np.any(np.isnan(parm["coef"])) and rank_adjust:
        parm["coef"] = np.nan_to_num(parm["coef"], nan=0.0)
        parm["beta"] = np.nan_to_num(parm["beta"], nan=0.0)

    return parm


def estimice(x, y, ls_meth="qr", ridge=1e-5):
    """
    This function computes least squares estimates, variance/covariance matrices,
    residuals and degrees of freedom according to ridge regression, QR decomposition
    or Singular Value Decomposition.
    :param x: Matrix n x p of complete covariates
    :param y: Incomplete data vector of length n
    :param ls_meth: least squares method, default QR decomposition, qr, ridge or svd
    :param ridge: size of ridge The default value ridge = 1e-05 represents a compromise between stability
    and unbiasedness
    :return: A list containing components c (least squares estimate), r (residuals),
    v (variance/covariance matrix) and df (degrees of freedom).
    """

    #degrees of freedom length of y - number of columns of x, min 1 df
    df = max(len(y) - x.shape[1], 1)
    #QR Decomposition
    if ls_meth == "qr":
        try:
        # # Linear regression via QR decomposition (c = coefficients, f = fitted values, r = residuals)
        #
            qr = np.linalg.qr(x)
            c = np.linalg.solve(qr.R, (qr.Q).T @ y)
            f = x @ c.T
            r = y - f

            # Try inverse of RR
            rr = (qr.R).T @ qr.R
            v = np.linalg.solve(rr, np.eye(rr.shape[1]))

            # missing ridge penalty and check for multicollinearity
            # catch error in v and calculate v + ridge penalty
            return {
                "c": c.flatten(),  # transpose to match R's shape
                "r": r.flatten(),
                "v": v,
                "df": df,
                "ls_meth": ls_meth
            }

        except Exception as e:
            raise RuntimeError(f"QR method failed: {e}")
#     #Ridge Regression
    elif ls_meth == "ridge":
        xx = x.T @ x
        pen = ridge * np.eye(xx.shape[0]) * xx
        v = np.linalg.solve(xx + pen, np.eye(xx.shape[1]))
        c = y.T @ x @ v
        r = y - x @ c.T
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }
#     #Singular Value Decomposition
    elif ls_meth == "svd":
        svd = np.linalg.svd(x, full_matrices=False)
        c = svd.Vh @ (((svd.U).T @ y) / svd.S)
        f = x @ c
        r = f - y
        v = svd.Vh @ np.eye((svd.S).shape[0]) @ svd.S ** 2 @ (svd.Vh).T

        # missing ridge penalty and check for multicollinearity
        # catch error in v and calculate v + ridge penalty
        return {
            "c": c.flatten(),
            "r": r.flatten(),
            "v": v,
            "df": df,
            "ls_meth": ls_meth
        }

#TESTS
# def testimice(x = None, y = None,ls_meth="qr", ridge=1e-5):
#     x = np.array([[1, 2],
#                   [3, 4],
#                   [5, 6]])
#     y = np.array([7, 8, 9])
#     svd = np.linalg.svd(x, full_matrices=False)
#     c = svd.Vh @ (((svd.U).T @ y) / svd.S)
#     f = x @ c
#     r = f - y
#     v = svd.Vh @ np.eye((svd.S).shape[0]) @ svd.S**2 @ (svd.Vh).T
#
# testimice()
#
print(estimice(x = np.array([[1, 2],[3, 4],[5, 6]]),y = np.array([7, 8, 9]),ls_meth="qr"))
print(estimice(x = np.array([[1, 2],[3, 4],[5, 6]]),y = np.array([7, 8, 9]),ls_meth="ridge"))
print(estimice(x = np.array([[1, 2],[3, 4],[5, 6]]),y = np.array([7, 8, 9]),ls_meth="svd"))