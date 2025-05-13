import pandas as pd
import numpy as np

class mice:
    def __init__(self, data = None, m = 5, predictorMatrix = None, seed = None, parallel = False):
        self._check_m(m)
        self._check_pm(predictorMatrix)
        self.data = data
        self.parallel = parallel
        if seed is not None:
            random.seed(seed)

        #id_obs = ry, id _mis = wy saved in dict for each column mask it later
        self.id_obs = {}
        self.id_mis = {}
        for col in self.data.columns:
            id_obs, id_mis = self._split_indices(self.data[col])
            self.id_obs[col] = id_obs
            self.id_mis[col] = id_mis

        self._initial_imputation()

    def _check_pm(self, predictorMatrix):
        self.predictorMatrix = predictorMatrix

    def _check_m(self, m):
        #takes int of m for user error
        if m < 1:
            raise Exception("Number of imputations is lower than 1")
        m = int(m)
        return m

    #def _initial(self):
        #Initial imputation returning ry of each variable

    def _split_indices(self, col):
        #saves the indices of observed and missing values before initial imputation
        null = pd.isnull(col)
        id_obs = np.flatnonzero(~null)
        id_mis = np.flatnonzero(null)
        if len(id_obs) == 0:
            raise ValueError("variable to be imputed has no observed values")
        return id_obs, id_mis

    def _initial_imputation(self):
        """
        Use a PMM-like procedure for initial imputed values.

        For each variable, missing values are imputed as the observed
        value that is closest to the mean over all observed values.
        """
        imp_values = {}
        for col in self.data.columns:
            di = self.data[col] - self.data[col].mean()
            di = np.abs(di)
            ix = di.idxmin()
            imp_values[col] = self.data[col].loc[ix]
        self.data = self.data.fillna(imp_values)

    #What I need for pmm:
    #def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
    #quantify = True, trim = 1, ridge = 1e-5, matcher = "NN", **kwargs):


