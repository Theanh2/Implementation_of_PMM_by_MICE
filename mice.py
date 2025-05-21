import pandas as pd
from pandas.api.types import is_numeric_dtype
from imputation.PMM import pmm
from imputation.midas import midas
from imputation.checks import _check_m, _check_pm, _check_d
from imputation.Utils import logit, expit, split_dataframe
import statsmodels.formula.api as smf
from statsmodels.base.model import LikelihoodModelResults
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
from functools import partial
class mice:
    def __init__(self, data = None, m = 5, maxit = 5, predictorMatrix = None, initial = "meanobs"):
        #drop empty rows and copy
        self.data = data.dropna(how='all').reset_index(drop=True)
        self.history = {}
        self.maxit = maxit
        #dict for method for each variable
        self.meth = {}
        #check m and pm
        self.m =_check_m(m)
        self.predictorMatrix = _check_pm(self.data, predictorMatrix)
        #analysis model
        self.amodel = {}
        self.model_results = []
        self.supported_meth = {"pmm": pmm, "midas": midas}
        #id_obs = ry, id _mis = wy saved in dict for each column mask it later
        self.id_obs = {}
        self.id_mis = {}
        for col in self.data.columns:
            id_obs, id_mis = self._split_indices(self.data[col])
            self.id_obs[col] = id_obs
            self.id_mis[col] = id_mis
        self.initial = initial
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
        Uses either closest observed value to the mean or a sample of the observed value as initial values in iteration 0
        """
        imp_values = {}
        if initial == "meanobs":
            for col in self.data.columns:
                di = self.data[col] - self.data[col].mean()
                di = np.abs(di)
                ix = di.idxmin()
                imp_values[col] = self.data[col].loc[ix]
            initdata = self.data.fillna(imp_values)
        elif initial == "sample":
            #deep copy to not overwrite pointer to self.data
            initdata = self.data.copy()
            for col in self.data.columns:
                robs = self.data.loc[self.id_obs[col], col].values
                for idx in self.id_mis[col]:
                    initdata.at[idx, col] = np.random.choice(robs)
        else:
            raise ValueError("Only meanobs or sample supported")
        return initdata
    def set_methods(self, d):
        """
        Assign method to columns
        :param d: dictionary of methods
        sets imputation methods for each variable. If not set uses default based on variable type
        :return:
        """
        _check_d(d, self.supported_meth)
        for col in self.data.columns:
            if col in d:
                self.meth[col] = d[col]
            else:
                if isinstance(col, pd.Categorical):
                    self.meth[col] = "pmm"
                if is_numeric_dtype(self.data[col]):
                    self.meth[col] = "pmm"
    def fit(self, fml, history = True, HMI = False, alpha = 0.05, cv = 0.05, pilot = 5, **kwargs):
        """
        Parameters
        ----------
        :param fml: analysis model formula in patsy format. Takes any patsy formula, enables transforming variables
            Note: Patsy does not handle dots in variable names throws Errors
        :param HMI: HowManyImputations by Hippel (2020) https://doi.org/10.48550/arXiv.1608.05406, pass alpha and cv to HMI() function
        :param history: bool, if True saves all iterations in a dict, if False only passes metrics for summary
        Returns
        -------

        """
        self.hist_bool = history
        self.fml = fml
        self.HMI_bool = HMI

        if not HMI:
            ###Parallelize this block
            for i in tqdm(range(self.m), desc = "M Multiple Imputations"): #runs m times each time returning final dataset and coef
                iterdata = self.iterate()
                if self.hist_bool:
                    self.history[i] = iterdata
            ###
            self.pool()
        if self.HMI_bool:
            self.HMI(pilot, alpha, cv, self.fml)

        #analysis model variables, for summary
        self.exog_names = self.amodel.exog_names
        self.endog_names = self.amodel.endog_names
        return self.results
    def _analysis(self, iterdata, **kwargs):
        for col, method in self.meth.items():
            # pass into function call
            # y needs to be masked
            # x needs to be subset by predictormatrix
            y = iterdata[col]
            y[self.id_mis[col]] = np.nan
            ry = ~np.isnan(y)
            xid = self.predictorMatrix[col]
            x = iterdata[xid[xid == 1].index]
            # from pandas to numpy
            y = np.array(y)
            ry = np.array(ry)
            x = np.array(x)
            iterdata[col] = self.supported_meth[method](y=y, ry=ry, x=x, **kwargs)
            #analysis model specification
        return iterdata
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

        #if summ:
        # Set up a results instance
        self.results = MICEResults(self, params, cov_params / scale)
        self.results.scale = scale
        self.results.frac_miss_info = self.fmi
    def HMI(self, pilot, alpha, cv, fml):
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
        for i in tqdm(range(pilot), desc="pilot "):
            iterdata = self.iterate()
            if self.hist_bool:
                self.history[i] = iterdata
        self.pool()

        #standard normal quantile
        z = norm.ppf(1 - alpha / 2)
        #confidence interval for delta mis by applying the inverse-logit transformation
        #to the endpoints
        fmiu = expit(logit(max(self.fmi)) + z * np.sqrt(2 / pilot))

        #upper bound of confidence interval rounded with int
        self.m2 = int(np.ceil(1 + 0.5 * (fmiu / cv)**2))

        for i in tqdm(range(self.m2-pilot), desc="stage2"):
            iterdata = self.iterate()
            if self.hist_bool:
                self.history[i + pilot] = iterdata
        self.pool()
    def iterate(self):
        iterdata = self._initial_imputation(self.initial)
        for j in range(self.maxit):
            iterdata = self._analysis(iterdata=iterdata)
        self.amodel = smf.ols(self.fml, data=iterdata)
        self.model_results.append(self.amodel.fit())
        return iterdata
    def complete(self):
        """
        Runs imputation step once.
        Returns: returns a single completed dataset
        -------

        """
        iterdata = self._initial_imputation(self.initial)
        iterdata = self._analysis(iterdata=iterdata)
        return iterdata
    def convergence_plot(self, fml, x = "mean"):
        """
        Parameters
        ----------
        fml: Analysis model formula
        x: mean or sd
        Returns matplotlib plot
        -------
        """
        for i in tqdm(range(self.m), desc="M"):
            iterdata = self._initial_imputation(self.initial)
            for j in range(self.maxit):
                iterdata = self._analysis(iterdata=iterdata)
                self.amodel = smf.ols(fml, data=iterdata)
                self.model_results.append(self.amodel.fit())

        n = self.m
        out = []
        if x == "mean":
            for mod in self.model_results:
                out.append(mod.params)
            out = pd.DataFrame(out)
            out = split_dataframe(out, n)
        elif x == "sd":
            for mod in self.model_results:
                out.append(mod.bse)
            out = pd.DataFrame(out)
            out = split_dataframe(out, n)
        #variable name + count
        columns = out[0].columns
        num_vars = len(columns)

        fig, axes = plt.subplots(int(num_vars/2),2, figsize=(6 * num_vars, 4), sharex=False)
        colors = plt.cm.get_cmap('tab10', self.m)
        # Plot each variable
        for col_idx, col in enumerate(columns):
            ax = axes[col_idx]
            for m, df in enumerate(out):
                ax.plot(df.index, df[col], marker='o',markersize = 2, color=colors(m))
            ax.set_title(f'{col}')
            ax.set_xlabel('Iterations')
            ax.set_ylabel(col)
            ax.grid(True)
        plt.tight_layout()
        plt.show()
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
        info["Dependent variable:"] = self.model.endog_names
        info["Sample size:"] = "%d" % self.model.data.shape[0]
        info["Scale"] = "%.2f" % self.scale
        info["M"] = "%d" % (len(self.model.model_results))

        smry.add_dict(info, align='l', float_format=float_format)

        param = summary2.summary_params(self, alpha=alpha)
        param["FMI"] = self.frac_miss_info

        smry.add_df(param, float_format=float_format)
        smry.add_title(title=title, results=self)

        return smry
