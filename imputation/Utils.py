from sklearn.neighbors import KDTree
import numpy as np
import random

def matcherid(d, t, matcher = "NN", k = 10, radius = 3):
    """
    :param d: Numeric vector with values from observed cases. np.array
    :param t: Numeric vector with values from missing cases. np.array
    :param matcher: matching method: n closest neighbors, every donor based on threshold distance, distance aided
    :param donors: k donors (for n closest neighbors)
    :return: returns index of chosen match

    Example:
    d = np.array([-5, 6, 0, 10, 12])
    t = np.array([-6])
    matcherid(d, t, matcher = "NN", k = 3)
    matcherid(d, t, matcher="fixedNN", radius = 5)
    """
    if matcher == "NN": #random from n closest Donors
        idx = []
        tree = KDTree(d.reshape(-1, 1), leaf_size = 40)
        #returns index k NN indices choose 1 on random
        dist, ind = tree.query(t.reshape(-1, 1), k = k)
        for list in ind:
            idx.append(random.choice(list))
        #returns indices of one random nearest neighbor for each t
        return idx
    elif matcher == "fixedNN": #fixed radius nearest neighbour
        idx = []
        tree = KDTree(d.reshape(-1, 1), leaf_size = 40)
        #returns index k NN indices choose 1 on random
        ind = tree.query_radius(t.reshape(-1, 1), radius)
        for list in ind:
            idx.append(random.choice(list))
        #returns indices of one random nearest neighbor
        return idx
    else:
        raise ValueError("unknown matcher")

def split_dataframe(df, n):
    """Split DataFrame into n roughly equal parts."""
    k, m = divmod(len(df), n)
    parts = [
        df.iloc[i * k + min(i, m):(i + 1) * k + min(i + 1, m)].reset_index(drop=True)
        for i in range(n)
    ]
    return parts

def logit(p):
    return np.log(p / (1 - p))
def expit(x):
    return 1 / (1 + np.exp(-x))