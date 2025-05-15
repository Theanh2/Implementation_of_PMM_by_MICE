import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from imputation.PMM import pmm
from imputation.midas import midas
from imputation.predictorMatrix import quickpred

class mice:
    def __init__(self, data = None, m = 5, predictorMatrix = None, seed = None, initial = "meanobs"):
        #drop empty rows and copy
        self.data = data.dropna(how='all').reset_index(drop=True)
        self.history = {}
        #dict for method for each variable
        self.meth = {}

        #check m and pm
        self._check_m(m)
        self._check_pm(predictorMatrix)
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
        if predictorMatrix is None:
            self.predictorMatrix = quickpred(self.data, mincor= 0.1, minpuc = 0.1)
        #self.predictorMatrix = predictorMatrix

    def _check_m(self, m):
        #takes int of m for user error
        if m < 1:
            raise Exception("Number of imputations is lower than 1")
        self.m = int(m)
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
        supported = ["pmm", "midas"]

        if not isinstance(d, dict):
            raise ValueError("d not dict")

        for x in d.values():
            if x not in supported:
                methods = f"Imputation Method: {x} is not supported"
                raise ValueError(methods)

    def set_methods(self, d):
        """
        :param d: dictionary of methods
        sets imputation methods for each variable. If not set uses default based on variable type
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


    def fit(self, HMI = False, cv = 0.05, alpha = 0.05, **kwargs):
        """

        Parameters
        ----------
        HMI: bool uses HowManyImputations from Hippel with default 5 pilot imputations. Overwrites m from mice() call

        Returns
        -------

        """
        supp_meth = {"pmm": pmm, "midas": midas}
        for i in range(self.m):
            for col, method in self.meth.items():
                #pass into function call
                #y needs to be masked
                #x needs to be subset by predictormatrix
                y = self.data[col]
                y[self.id_mis[col]] = np.nan
                ry = ~np.isnan(y)
                xid = self.predictorMatrix[col]
                x = self.data[xid[xid == 1].index]
                #from pandas to numpy
                y = np.array(y)
                print(y)
                ry = np.array(ry)
                x = np.array(x)
                self.data[col] = supp_meth[method](y = y, ry = ry, x = x, **kwargs)
            #Fix updating and save for history
            temp_data = self.data
            self.history.update({i: temp_data})

        print(self.history)




    #What I need for pmm:
    #def pmm(y, ry, x, wy = None, donors = 5, matchtype = 1,
    #quantify = True, trim = 1, ridge = 1e-5, matcher = "NN", **kwargs):

#intended use:
#1. mice()
#2. optional: mice.set_imputer()
#3. mice.fit()
