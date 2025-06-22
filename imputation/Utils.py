from sklearn.neighbors import KDTree
import numpy as np
import random
def matcherid(d, t, matcher = "NN", k = 10, radius = 3):
    """
    Find donor indices matching missing values based on specified matching method.

    Parameters
    ----------
    d : np.array
        Numeric vector of observed values (donor pool).
    t : np.array
        Numeric vector of missing values to be matched.
    matcher : str, optional
        Matching method to use:
        - "NN": Randomly selects one from the k nearest neighbors (default).
        - "fixedNN": Randomly selects one donor within a fixed radius.
    k : int, optional
        Number of nearest neighbors to consider (only for "NN" matcher).
    radius : float, optional
        Radius threshold for fixedNN matcher (only for "fixedNN" matcher).

    Returns
    -------
    list of int
        List of indices corresponding to chosen donors in d for each element in t.

    Raises
    ------
    ValueError
        If an unknown matcher method is specified.

    Examples
    --------
    >>> d = np.array([-5, 6, 0, 10, 12])
    >>> t = np.array([-6])
    >>> matcherid(d, t, matcher="NN", k=3)
    [0]
    >>> matcherid(d, t, matcher="fixedNN", radius=5)
    [0]
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
    """
    Split a DataFrame into n roughly equal parts.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to split.
    n : int
        Number of parts to split the DataFrame into.

    Returns
    -------
    list of pandas.DataFrame
        List containing n DataFrames, each a part of the original DataFrame.
    """
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

