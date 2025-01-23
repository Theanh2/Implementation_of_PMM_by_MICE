import numpy as np
from scipy.spatial import cKDTree


def bootfunc(n):
    random_values = np.random.rand(n - 1)
    sorted_values = np.sort(np.concatenate(([0], random_values, [1])))
    weights = np.diff(sorted_values) * n
    return weights


def removecons(X):
    return np.where(np.apply_along_axis(lambda col: np.unique(col).size > 1, axis=0, arr=X))[0]


def rescale_quartiles(X, weight=None):
    def wiqr(x, w=None):
        ox = np.argsort(x)
        if w is not None:
            ssw = np.cumsum(w[ox])
            limits = np.array([0.25, 0.75]) * ssw[-1]
            index = np.searchsorted(ssw, limits)
        else:
            index = np.round(np.array([0.25, 0.75]) * len(x)).astype(int)
        return np.diff(np.sort(x)[index])

    iqrvec = np.apply_along_axis(lambda col: wiqr(col, weight), axis=0, arr=X)
    iqrvec[iqrvec == 0] = 1
    return iqrvec


def modeltable(Xdon_full, Xrec=None, k=5):
    tree = cKDTree(Xdon_full)
    if Xrec is None:
        dist, idx = tree.query(Xdon_full, k=k + 1)
        idx, dist = idx[:, 1:], dist[:, 1:]  # Remove self-matches
    else:
        dist, idx = tree.query(Xrec, k=k)

    maxdist = np.repeat(dist[:, -1], k)
    maxdist[maxdist == 0] = 1
    weights = (1 - (dist.flatten() / maxdist) ** 3) ** 3 + 0.01
    return np.column_stack((np.repeat(np.arange(len(Xdon_full)), k), idx.flatten(), dist.flatten(), weights))


def ipm_create(id_array, k):
    unique_ids = np.unique(id_array)
    start = np.arange(0, len(id_array), k)
    stop = start + k
    return np.column_stack((unique_ids, start, stop))


def adjust_weights(id_array, dist, k):
    maxdist = np.repeat(dist[::k], k)
    maxdist[maxdist == 0] = 1
    weights = (1 - (dist / maxdist) ** 3) ** 3 + 0.01
    return weights


def mice_impute_li(y, ry, x, boot=True, kgran=6, rgran=3, rescale=1, midastouch=True):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n, donind, recind = len(y), np.where(ry)[0], np.where(~ry)[0]
    if len(donind) == 1:
        return np.full(len(recind), y[donind[0]])

    X = x[:, removecons(x)]
    bw = bootfunc(n) if boot else None

    if rescale > 0:
        if rescale == 2 and X.shape[1] > 1:
            beta = np.abs(np.linalg.lstsq(X[donind], ry[donind], rcond=None)[0])
            X *= beta / np.max(beta)
        else:
            X /= rescale_quartiles(X, bw)

    k = np.linspace(5, len(donind) - 1, kgran, dtype=int)[-1]
    model_tab = modeltable(X[donind], X[recind], k)
    matchind = model_tab[:, 1][recind] if k == 1 else model_tab[:, 1]
    return y[matchind]