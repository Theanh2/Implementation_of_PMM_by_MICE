#by chatgpt
#https://rdrr.io/cran/mice/src/R/mice.impute.midastouch.R
import numpy as np

#Auxiliary
def bootfunc_plain(n):
    random = np.random.choice(n, size=n, replace=True)
    weights, _ = np.histogram(random, bins=np.arange(1, n + 2))
    return weights


def minmax(x, domin=True, domax=True):
    maxx = np.sqrt(np.finfo(float).max)
    minx = np.sqrt(np.finfo(float).eps)
    if domin:
        x = np.minimum(x, maxx)
    if domax:
        x = np.maximum(x, minx)
    return x


def mice_impute_midastouch(y, ry, x, ridge=1e-5, midas_kappa=None, outout=None, neff=None, debug=None):
    sminx = np.finfo(float).eps ** (1 / 4)

    x = np.asarray(x, dtype=float)
    X = np.hstack((np.ones((x.shape[0], 1)), x))
    y = np.asarray(y, dtype=float)

    nobs = np.sum(ry)
    nmis = np.sum(~ry)
    n = len(ry)
    obsind = np.where(ry)[0]
    misind = np.where(~ry)[0]
    m = X.shape[1]

    yobs = y[obsind]
    Xobs = X[obsind, :]
    Xmis = X[misind, :]

    omega = bootfunc_plain(nobs)
    CX = omega[:, None] * Xobs
    XCX = Xobs.T @ CX

    if ridge > 0:
        np.fill_diagonal(XCX, np.diag(XCX) * (1 + np.array([0] + [ridge] * (m - 1))))

    diag0 = np.where(np.diag(XCX) == 0)[0]
    if len(diag0) > 0:
        XCX[diag0, diag0] = max(sminx, ridge)

    Xy = CX.T @ yobs
    beta = np.linalg.solve(XCX, Xy)
    yhat_obs = Xobs @ beta

    if midas_kappa is None:
        mean_y = np.dot(yobs, omega) / nobs
        eps = yobs - yhat_obs
        r2 = 1 - (np.dot(omega, eps ** 2) / np.dot(omega, (yobs - mean_y) ** 2))
        midas_kappa = min((50 * r2 / (1 - r2)) ** (3 / 8), 100) if r2 < 0.999 else 3

    if outout is None:
        outout = nobs <= 250

    if outout:
        XXarray_pre = np.array([np.outer(Xobs[i], Xobs[i]) * omega[i] for i in range(nobs)]).T
        ridgeind = np.arange(1, m) * (m + 1)
        if ridge > 0:
            XXarray_pre[ridgeind, :] *= (1 + ridge)
        XXarray = XCX.ravel()[:, None] - XXarray_pre
        XXarray[ridgeind, :] = np.maximum(XXarray[ridgeind, :], sminx)
        Xyarray = Xy.ravel()[:, None] - (Xobs * yobs[:, None] * omega[:, None]).T
        BETAarray = np.linalg.solve(XXarray[:m * m].reshape(m, m, -1), Xyarray[-m:])
        YHATdon = np.sum(Xobs[:, :, None] * BETAarray, axis=1)
        YHATrec = Xmis @ BETAarray
        dist_mat = YHATdon - YHATrec.T
    else:
        yhat_mis = Xmis @ beta
        dist_mat = yhat_obs[:, None] - yhat_mis

    delta_mat = 1 / (np.abs(dist_mat) ** midas_kappa)
    delta_mat = minmax(delta_mat)
    probs = delta_mat * omega[:, None]
    csums = minmax(np.sum(probs, axis=0))
    probs /= csums

    index = np.array([np.random.choice(nobs, p=probs[:, j]) for j in range(nmis)])
    yimp = y[obsind][index]

    return yimp
