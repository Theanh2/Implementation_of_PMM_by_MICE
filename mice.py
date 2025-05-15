import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class mice:
    def __init__(self, data = None, m = 5, predictorMatrix = None, seed = None, parallel = False, initial = "meanobs"):
        #dict for method for each variable
        self.meth = {}

        #check m and pm
        self._check_m(m)
        self._check_pm(predictorMatrix)

        #drop empty rows and copy
        self.data = data.dropna(how='all').reset_index(drop=True)

        #sets multiprocessing (not implemented yet)
        self.parallel = parallel

        #sets seed
        if seed is not None:
            random.seed(seed)

        #id_obs = ry, id _mis = wy saved in dict for each column mask it later
        self.id_obs = {}
        self.id_mis = {}
        for col in self.data.columns:
            id_obs, id_mis = self._split_indices(self.data[col])
            self.id_obs[col] = id_obs
            self.id_mis[col] = id_mis

        # The order in which variables are imputed in each cycle.
        # Impute variables with the fewest missing values first.
        # statsmodels
        vnames = list(data.columns)
        nmis = [len(self.id_mis[v]) for v in vnames]
        nmis = np.asarray(nmis)
        ii = np.argsort(nmis)
        ii = ii[sum(nmis == 0):]
        self._cycle_order = [vnames[i] for i in ii]

        #initial imputation Y0
        self._initial_imputation(initial)

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

    def _initial_imputation(self, initial):
        """
        Use a PMM-like procedure for initial imputed values.

        For each variable, missing values are imputed as the observed
        value that is closest to the mean over all observed values.
        Replacements are inplace. reference pointer to input data
        """
        imp_values = {}
        if initial == "meanobs":
            for col in self.data.columns:
                di = self.data[col] - self.data[col].mean()
                di = np.abs(di)
                ix = di.idxmin()
                imp_values[col] = self.data[col].loc[ix]
            self.data.fillna(imp_values, inplace=True)
        elif initial == "random":
            for col in self.data.columns:
                robs = self.data.loc[self.id_obs[col], col].values
                for idx in self.id_mis[col]:
                    self.data.at[idx, col] = np.random.choice(robs)

    def check_d(self, d):
        """
        check if methods passed are supported
        :param d: dictionary of methods
        :return raise exception if invalid method
        """
        supported = ["pmm", "miles", "midas", "cart"]

        if not isinstance(d, dict):
            raise ValueError("d not dict")

        for x in d.values():
            if x not in supported:
                methods = f"Imputation Method: {x} is not supported"
                raise ValueError(methods)

    def set_methods(self, d):
        """
        :param d: dictionary of methods
        sets imputation methods for each variable. If not set uses default
        :return:
        """
        self.check_d(d)

        for col in self.data.columns:
            if col in d:
                self.meth[col] = d[col]
            else:
                if isinstance(col, pd.Categorical):
                    self.meth[col] = "pmm"
                if is_numeric_dtype(self.data[col]):
                    self.meth[col] = "pmm"



        #need predictormatrix for


    def fit(self):
        #if method == "pmm":
            print("...")

    #What I need for pmm:
    #def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
    #quantify = True, trim = 1, ridge = 1e-5, matcher = "NN", **kwargs):


