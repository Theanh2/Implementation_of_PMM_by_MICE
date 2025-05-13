import numpy as np

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

def mice_impute_midastouch(y, ry, x, ridge = 1e-5, midas_kappa = None, outout = True):
    """
    R documentation van burren: change later on #https://rdrr.io/cran/mice/src/R/mice.impute.midastouch.R
    1. Draw a bootstrap sample from the donor pool.
    2. Estimate a beta matrix on the bootstrap sample by the leave one out principle.
    3. Compute type II predicted values for yobs (nobs x 1) and ymis (nmis x nobs).
    4. Calculate the distance between all yobs and the corresponding ymis.
    5. Convert the distances in drawing probabilities.
    6. For each recipient draw a donor from the entire pool while considering the probabilities from the model.
    7. Take its observed value in y as the imputation.
    :param y: Numeric vector with incomplete data
    :param ry: Response pattern of y
    :param x: Design matrix with length(y) rows and p columns containing complete covariates.
    :param ridge: The ridge penalty applied to prevent problems with multicollinearity.
    The default is ridge = 1e-05, which means that 0.001 percent of the diagonal is added to the cross-product.
    Larger ridges may result in more biased estimates.
    For highly noisy data (e.g. many junk variables), set ridge = 1e-06 or even lower to reduce bias.
    For highly collinear data, set ridge = 1e-04 or higher.
    :param midas_kappa: Scalar. If NULL (default) then the optimal kappa gets selected automatically. Alternatively, the user may specify a scalar.
    Siddique and Belin 2008 find midas.kappa = 3 to be sensible.
    :param outout: Logical. Default TRUE one model is estimated for each donor (leave-one-out principle).
    For speedup choose outout = FALSE, which estimates one model for all observations leading to in-sample predictions for the donors and out-of-sample predictions for the recipients.
    Mind the inappropriateness, though.
    :return: Numeric vector of length sum(wy) with imputations

    Example:
    y = np.array([7, np.nan, 9,10,11])
    ry = ~np.isnan(y)
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 13], [11, 10]])
    print(mice_impute_midastouch(y,ry,x))
    """
    wy = ~ry
    #machine epsilon
    sminx = np.finfo(float).eps ** (1 / 4)

    x = np.asarray(x, dtype=float)
    x = np.c_[np.ones(x.shape[0]), x]
    y = np.asarray(y, dtype=float)
    nobs = np.sum(ry)
    nmis = np.sum(wy)
    n = len(ry)
    m = x.shape[1]

    yobs = y[ry]
    xobs = x[ry, :]
    xmis = x[wy, :]

    omega = bootfunc_plain(nobs)
    #for testing omega = np.array([2, 1, 1, 0])

    CX = omega.reshape(-1, 1) * xobs
    XCX = xobs.T @ CX

    diag0 = np.where(np.diag(XCX) == 0)[0]
    if len(diag0) > 0:
        XCX[diag0, diag0] = max(sminx, ridge)

    Xy = CX.T @ yobs
    #numpy.linalg.LinAlgError: Singular matrix when XCX is not inversible maybe throw error and return?
    #CX = observed data * bootstrap frequencies
    #XCX = observed data * CX
    beta = np.linalg.solve(XCX, Xy)
    yhat_obs = xobs @ beta

    if midas_kappa is None:
        mean_y = np.dot(yobs, omega) / nobs
        eps = yobs - yhat_obs
        r2 = 1 - (np.dot(omega, eps ** 2) / np.dot(omega, (yobs - mean_y) ** 2))
        #section 5.3.1
        # min function is used correction gets active for r2>.999 only because division by 0
        # if r2 cannot be determined (eg zero variance in yhat), use 3 as suggested by Siddique / Belin
        #if taking delta as in the paper there are numerical errors needing to be fixed
        if r2 < 1:
            midas_kappa = (50 * r2 / (1 - r2))** (3 / 8)
        else:
            midas_kappa = 100

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


    #
    # else:
    #     yhat_mis = Xmis @ beta
    #     dist_mat = yhat_obs[:, None] - yhat_mis
    #Test: divide by 0 here -> probs do not sum to 1
    delta_mat = 1 / (np.abs(dist_mat) ** midas_kappa)

    print(dist_mat, midas_kappa)
    delta_mat = minmax(delta_mat)

    probs = delta_mat * omega
    csums = minmax(np.nansum(probs, axis=1))
    probs /= csums



    index = np.random.choice(nobs, size=1, replace=False, p=probs.flatten())
    yimp = y[ry][index]

    #PLF correction implemented needs to be saved globally over iterations
    #mean(1 / rowSums((t(delta.mat) / csums)^2))
    #consists
    row_sums = np.sum((delta_mat / csums)**2, axis=1)
    neff = np.mean(1 / row_sums)
    return yimp

y = np.array([7, np.nan, 9,10,11])
ry = ~np.isnan(y)
x = np.array([[1, 2], [3, 4], [5, 6], [7, 13], [11, 10]])
mice_impute_midastouch(y,ry,x)