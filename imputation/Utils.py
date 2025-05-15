from sklearn.neighbors import KDTree
import numpy as np
import random

def matcherid(d, t, matcher = "NN", k = 10, radius = 3):
    """
    Return vector of n0 positions in d.
    :param d: Numeric vector with values from observed cases. np.array
    :param t: Numeric vector with values from missing cases. np.array
    :param matcher: matching method: n closest neighbors, every donor based on threshold distance, distance aided
    :param donors: k donors (for n closest neighbors)
    :return:

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
    # elif matcher == "distance": #based on distance of donor to imputed value (midas)
    # matcher
    #
    #
    else:
        raise ValueError("unknown matcher")
