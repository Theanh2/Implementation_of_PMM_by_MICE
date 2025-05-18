import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from imputation.PMM import pmm
from imputation.midas import midas
from imputation.predictorMatrix import quickpred
import statsmodels.formula.api as smf
from statsmodels.base.model import LikelihoodModelResults
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

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
        self.amodel = {}
        self.model_results = []

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


    def fit(self, fml, history = False, HMI = False, alpha = 0.05, cv = 0.05, pilot = 5, **kwargs):
        """

        Parameters
        ----------
        :param fml: analysis model formula in patsy format. Takes any patsy, enables transforming variables
            Note: Patsy does not handle dots in variable names thros Errors
        :param HMI: HowManyImputations by Hippel (2020) https://doi.org/10.48550/arXiv.1608.05406, pass alpha and cv to HMI() function
        :param history: bool, if True saves all iterations in a dict, if False only passes metrics for summary



        Returns
        -------

        """
        #Global variables: makes code easier no passing around **kwargs
        self.hist_bool = history
        self.fml = fml

        self.HMI_bool = HMI
        if not HMI:
            for i in tqdm(range(self.m)):
                self._analysis(fml, **kwargs)
                # Save iterations if history True
                if self.hist_bool:
                    self.history[i] = self.data.copy()
            self.pool(summ = True)
                # Analysis model
        if self.HMI_bool:
            self.HMI(pilot, alpha, cv)

        self.exog_names = self.amodel.exog_names
        self.endog_names = self.amodel.endog_names
        return self.results

    def _analysis(self,fml, **kwargs):
        supp_meth = {"pmm": pmm, "midas": midas}
        for col, method in self.meth.items():
            # pass into function call
            # y needs to be masked
            # x needs to be subset by predictormatrix
            y = self.data[col]
            y[self.id_mis[col]] = np.nan
            ry = ~np.isnan(y)
            xid = self.predictorMatrix[col]
            x = self.data[xid[xid == 1].index]
            # from pandas to numpy
            y = np.array(y)
            ry = np.array(ry)
            x = np.array(x)
            self.data[col] = supp_meth[method](y=y, ry=ry, x=x, **kwargs)

            #analysis model specification
            self.amodel = smf.ols(fml, data=self.data)
            self.model_results.append(self.amodel.fit())

    def pool(self, summ = False):
        """
        Pools
        Returns: summary of fit
        -------

        """
        params_list = []
        cov_within = 0.
        scale_list = []
        for results in self.model_results:
            results_uw = results._results
            params_list.append(results_uw.params)
            cov_within += results_uw.cov_params()
            scale_list.append(results.scale)
            # The estimated parameters for the MICE analysis

        params_list = np.asarray(params_list)
        scale_list = np.asarray(scale_list)

        params = params_list.mean(0)
        # The average of the within-imputation covariances
        cov_within /= len(self.model_results)
        # The between-imputation covariance
        cov_between = np.cov(params_list.T)

        # The estimated covariance matrix for the MICE analysis
        f = 1 + 1 / float(len(self.model_results))
        cov_params = cov_within + f * cov_between
        # Fraction of missing information
        self.fmi = f * np.diag(cov_between) / np.diag(cov_params)
        scale = np.mean(scale_list)

        if summ:
            # Set up a results instance
            self.results = MICEResults(self, params, cov_params / scale)
            self.results.scale = scale
            self.results.frac_miss_info = self.fmi
            # self.results.exog_names = self.exog_names
            # self.results.endog_names = self.endog_names


    def HMI(self, pilot, alpha, cv):
        """
        Runs pilot imputation to get fraction of missing information and then runs the remaining iterations so ensure
            point estimates and standard deviation for replicability
            https://arxiv.org/pdf/1608.05406 chapter 1.2
        Parameters
        ----------
        pilot: Initial pilot imputation to calculate fraction of missing information
        cv: desired coefficient of variation
        alpha: Confidence Interval
        kwargs

        Returns
        -------

        """
        for i in tqdm(range(pilot), desc = "pilot "):
            self._analysis(self.fml)
            # Save iterations if history True
            if self.hist_bool:
                self.history[i] = self.data.copy()
            #returns fmi of pilot imputations for calculating second stage
        self.pool()

        #standard normal quantile
        z = norm.ppf(1 - alpha / 2)
        #confidence interval for delta mis by applying the inverse-logit transformation
        #to the endpoints
        fmiu = self.expit(self.logit(max(self.fmi)) + z * np.sqrt(2 / pilot))

        #upper bound of confidence interval rounded with int
        self.m2 = int(np.ceil(1 + 0.5 * (fmiu / cv)**2))

        for i in tqdm(range(self.m2-2), desc = "stage2"):
            self._analysis(self.fml)
            # Save iterations if history True
            if self.hist_bool:
                self.history[i+ pilot] = self.data.copy()
            #returns fmi of pilot imputations for calculating second stage
        self.pool(summ = True)
    #short util function to not import
    def logit(self,p):
        return np.log(p / (1 - p))

    def expit(self,x):
        return 1 / (1 + np.exp(-x))




class MICEResults(LikelihoodModelResults):

    def __init__(self, model, params, normalized_cov_params):

        super().__init__(model, params, normalized_cov_params)

    def summary(self, title=None, alpha=.05):
        """
        Summarize the results of running MICE.

        Parameters
        ----------
        title : str, optional
            Title for the top table. If not None, then this replaces
            the default title
        alpha : float
            Significance level for the confidence intervals

        Returns
        -------
        smry : Summary instance
            This holds the summary tables and text, which can be
            printed or converted to various output formats.
        """

        from statsmodels.iolib import summary2

        smry = summary2.Summary()
        float_format = "%8.3f"

        info = {}
        info["Method:"] = "MICE"
        info["Model:"] = "OLS"
        info["Dependent variable:"] = self.model.endog_names
        info["Sample size:"] = "%d" % self.model.data.shape[0]
        info["Scale"] = "%.2f" % self.scale
        info["Num. iterations"] = "%d" % (len(self.model.model_results) / 3)

        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        param["FMI"] = self.frac_miss_info

        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)

        return smry