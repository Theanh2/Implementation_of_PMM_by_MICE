import numpy as np
import pandas as pd
#Auxiliary
def bootfunc_plain(n):
    #random n int with size n from 1:n
    random = np.random.choice(n, size=n, replace=True)+1
    #returns histogram of drawn ints
    table, _ = np.histogram(random, bins=np.arange(1, n + 2))
    return table


def minmax(x, domin=True, domax=True):
    #max float and min float
    maxx = np.sqrt(np.finfo(float).max)
    minx = np.sqrt(np.finfo(float).eps)
    if domin:
        x = np.minimum(x, maxx)
    if domax:
        x = np.maximum(x, minx)
    return x

def compute_beta(x, m):
    A = x[:m**2].reshape((m, m))
    b = x[m**2:]
    return np.linalg.solve(A, b)

def midas(y, ry, x, ridge = 1e-5, midas_kappa = None, outout = True):
    """
    Gaffert, P., Meinfelder, F., Bosch V. (2018) Towards an MI-proper Predictive Mean Matching
    :param y: Array: Vector to be imputed
    :param ry: Logical: vector of length(y). ry distinguishes the observed TRUE and missing values FALSE in y.
    :param x: Design matrix with length(y) rows and p columns containing complete covariates.
    :param ridge: The ridge penalty used in norm.draw() to prevent problems with multicollinearity.
        The default is ridge = 1e-05, which means that 0.01 percent of the diagonal is added to the cross-product.
        Larger ridges may result in more biased estimates. For highly noisy data (e.g. many junk variables),
        set ridge = 1e-06 or even lower to reduce bias. For highly collinear data, set ridge = 1e-04 or higher. !cite
    :param midas_kappa: Scalar. If None optimal kappa gets selected automatically. Siddique and Belin 2008 find midas.kappa = 3 to be sensible.
    :param outout: Logical. Default TRUE one model is estimated for each donor (leave-one-out principle). !outout False not implemented
    For speedup choose outout = FALSE, which estimates one model for all observations leading to in-sample predictions for the donors and out-of-sample predictions for the recipients.
    Mind the inappropriateness.
    :return: Vector with imputed data

    Example:
    y = np.array([7, np.nan, 9,10,11])
    ry = ~np.isnan(y)
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 13], [11, 10]])
    print(midas(y,ry,x))
    """
    wy = ~ry
    #machine epsilon
    sminx = np.finfo(float).eps ** (1 / 4)

    x = np.asarray(x, dtype=float)
    x = np.c_[np.ones(x.shape[0]), x]
    y = np.asarray(y, dtype=float)
    nobs = np.sum(ry)
    n = len(ry)
    m = x.shape[1]

    yobs = y[ry]
    xobs = x[ry, :]
    xmis = x[wy, :]
    #P Step
    omega = bootfunc_plain(nobs)

    CX = omega.reshape(-1, 1) * xobs
    XCX = xobs.T @ CX
##
    if ridge > 0:
        dia = np.diag(XCX)
        dia = dia * np.concatenate(([1], np.repeat(1 + ridge, m - 1)))
        np.fill_diagonal(XCX, dia)

    diag0 = np.where(np.diag(XCX) == 0)[0]
    if len(diag0) > 0:
        XCX[diag0, diag0] = max(sminx, ridge)

    Xy = CX.T @ yobs

    #CX = observed data * bootstrap frequencies
    #XCX = observed data * CX
    beta = np.linalg.solve(XCX, Xy)
    yhat_obs = xobs @ beta

    if midas_kappa is None:
        mean_y = np.dot(yobs, omega) / nobs
        eps = yobs - yhat_obs
        r2 = 1 - (np.dot(omega, eps ** 2) / np.dot(omega, (yobs - mean_y) ** 2))
        #section 5.3.1
        #min function is used correction gets active for r2>.999 only because division by 0
        #if r2 cannot be determined (eg zero variance in yhat), use 3 as suggested by Siddique / Belin
        #if taking delta as in the paper there are numerical errors needing to be fixed
        if r2 < 1:
            midas_kappa = min((50 * r2 / (1 - r2))** (3 / 8),100)
        if np.isnan(midas_kappa):
            midas_kappa = 3

    if outout:
        XXarray_pre = np.array([np.outer(xobs[i], xobs[i]).flatten() * omega[i] for i in range(nobs)]).T
        ridgeind = np.arange(1, m) * (m + 1)
        if ridge > 0:
            XXarray_pre[ridgeind, :] *= (1 + ridge)

        XXarray = XCX.ravel()[:, None] - XXarray_pre
        diag0 = np.where(np.diag(XXarray) == 0)[0]
        if len(diag0) > 0:
            XXarray[diag0, diag0] = max(sminx, ridge)
        Xyarray = Xy.ravel()[:, None] - (xobs * yobs[:, None] * omega[:, None]).T

        ##solve(a = matrix(head(x, m^2), m), b = tail(x, m)) for each column
        stacked_array = np.vstack((XXarray, Xyarray))
        BETAarray = np.apply_along_axis(compute_beta, axis=0, arr=stacked_array, m=m)

        # y
        YHATdon = np.sum(xobs * BETAarray.T, axis=1)
        YHATrec = xmis @ BETAarray

        # distance matrix
        dist_mat = YHATdon - YHATrec
    else:
        yhat_mis = xmis @ beta
        dist_mat = (yhat_obs[:, np.newaxis] - np.tile(yhat_mis, (nobs, 1))).T

    delta_mat = 1 / (np.abs(dist_mat) ** midas_kappa)
    print(dist_mat)
    print(midas_kappa)
    delta_mat = minmax(delta_mat)

    probs = delta_mat * omega
    csums = minmax(np.nansum(probs, axis=1))
    probs /= csums

    index = np.random.choice(nobs, size=1, replace=False, p=probs.flatten())
    y[wy] = y[index]

    #PLF correction implemented needs to be saved globally over iterations
    #mean(1 / rowSums((t(delta.mat) / csums)^2))
    #consists
    row_sums = np.sum((delta_mat / csums)**2, axis=1)
    neff = np.mean(1 / row_sums)
    return y


