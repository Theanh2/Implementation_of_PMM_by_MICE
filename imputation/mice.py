import pandas as pd
from pandas.api.types import is_numeric_dtype
from imputation.PMM import pmm
from imputation.midas import midas
from imputation.cart import cart
from imputation.sample import sample
from imputation.rf import rf
from imputation.mean import mean
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
from statsmodels.iolib import summary2
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
        self.supported_meth = {"pmm": pmm, "midas": midas, "cart": cart, "sample": sample, "rf": rf, "mean": mean}
        #id_obs = ry, id _mis = wy saved in dict for each column mask it later
        self.id_obs = {}
        self.id_mis = {}
        for col in self.data.columns:
            id_obs, id_mis = self._split_indices(self.data[col])
            self.id_obs[col] = id_obs
            self.id_mis[col] = id_mis
        self.initial = initial
    def _split_indices(self, col):
        """
        Splits a column into indices of observed and missing values.

        Identifies the positions of non-missing and missing values in the input
        column before imputation. Raises an error if no observed values exist.

        :param col: The data column to check for missing values.
        :type col: pandas.Series or array-like

        :return: A tuple containing two numpy arrays:
            - Indices of observed (non-missing) values
            - Indices of missing values
        :rtype: tuple of numpy.ndarray

        :raises ValueError: If the column has no observed (non-missing) values.
        """
        null = pd.isnull(col)
        id_obs = np.flatnonzero(~null)
        id_mis = np.flatnonzero(null)
        if len(id_obs) == 0:
            raise ValueError("variable to be imputed has no observed values")
        return id_obs, id_mis
    def _initial_imputation(self, initial):
        """
        Performs initial imputation on missing data before starting the iterative process.

        Depending on the `initial` parameter, missing values are imputed either with
        the observed value closest to the mean of the column, or with a random sample
        drawn from the observed values.

        :param initial: Method for initial imputation. Supported options:
                        - "meanobs": Impute with observed value closest to the mean.
                        - "sample": Impute with a random observed value sampled with replacement.
        :type initial: str

        :return: DataFrame with missing values initially imputed.
        :rtype: pandas.DataFrame

        :raises ValueError: If the `initial` parameter is not "meanobs" or "sample".
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
        Assigns imputation methods to columns in the dataset.

        For each column, the method specified in the dictionary `d` is assigned.
        If a column is not specified in `d`, a default method is assigned based on the
        variable type: `"pmm"` for categorical or numeric columns.

        :param d: Dictionary mapping column names to imputation methods.
        :type d: dict

        :return: None

        :raises ValueError: If any method in `d` is not supported (checked by `_check_d`).
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
            Fits the imputation model and performs analysis using the specified formula.

            :param fml: Analysis model formula in Patsy syntax.
                        Supports variable transformations but does not allow dots in variable names
                        (which may cause Patsy errors).
            :type fml: str

            :param history: If True, saves all iterations of the imputation in a dictionary.
                            If False, only final metrics are kept.
            :type history: bool, optional (default=True)

            :param HMI: Whether to use HowManyImputations (Hippel, 2020) for pooling results.
                        If True, alpha and cv parameters are passed to the HMI method.
            :type HMI: bool, optional (default=False)

            :param alpha: Significance level used in HMI pooling.
            :type alpha: float, optional (default=0.05)

            :param cv: Coefficient of variation threshold for HMI pooling.
            :type cv: float, optional (default=0.05)

            :param pilot: Number of pilot imputations for HMI.
            :type pilot: int, optional (default=5)

            :param kwargs: Additional keyword arguments (currently unused).

            :return: Results of the imputation and analysis.
            :rtype: depends on self.results
            """
        self.hist_bool = history
        self.fml = fml
        self.HMI_bool = HMI

        if not HMI:
            ###Parallelize this block, or easier to parallelize simulations
            for i in tqdm(range(self.m), desc = "M Multiple Imputations", disable = False): #runs m times each time returning final dataset and coef
                iterdata = self.iterate()
                if self.hist_bool:
                    self.history[i] = iterdata
            self.pool()
        if self.HMI_bool:
            self.HMI(pilot, alpha, cv, self.fml)

        self.exog_names = self.amodel.exog_names
        self.endog_names = self.amodel.endog_names
        return self.results
    def _analysis(self, iterdata, **kwargs):
        """
        Performs imputation on the given dataset iteration using specified methods for each variable.

        For each column, the method:
        - Masks the missing values in the target variable.
        - Selects predictor variables based on the predictor matrix.
        - Applies the imputation method assigned to the column.
        - Updates the imputed values in the dataset.

        Parameters
        ----------
        :param iterdata: pandas DataFrame representing the current iteration of data with missing values.
        :type iterdata: pd.DataFrame

        :param kwargs: Additional keyword arguments to pass to the imputation methods.

        Returns
        -------
        :return: DataFrame with imputed values updated for each column.
        :rtype: pd.DataFrame
        """
        for col, method in self.meth.items():
            # Skip columns that don't have missing values
            if len(self.id_mis[col]) == 0:
                continue
                
            # pass into function call
            # y needs to be masked
            # x needs to be subset by predictormatrix
            y = iterdata[col]
            # ensure target missing positions are masked
            y.iloc[self.id_mis[col]] = np.nan
            # robust observed/missing masks across dtypes
            id_obs = ~pd.isna(y).values
            id_mis = ~id_obs
            xid = self.predictorMatrix[col]
            if (xid == 0).all():
                continue
            x = iterdata[xid[xid == 1].index]
            # call imputer with consistent argument names
            iterdata[col] = self.supported_meth[method](
                y=y,
                id_obs=id_obs,
                id_mis=id_mis,
                x=x,
                **kwargs,
            )
        return iterdata
    def pool(self, summ = False):
        """
        Pools parameter estimates and covariance matrices from multiple imputations
        to produce overall inference estimates following Rubin's rules.

        Aggregates results across multiple imputed datasets by combining within-imputation
        variance and between-imputation variance to estimate overall parameter uncertainty.

        :param summ: bool, optional
            If True, returns a summary of the pooled fit (default is False).

        :return: None
            Stores pooled results in `self.results` as a MICEResults object.
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
            Performs How-Many-Imputations (HMI) procedure to determine the required number
            of imputations based on fraction of missing information (FMI) for stable
            point estimates and standard errors.

            This method first runs a pilot set of imputations to estimate the FMI,
            then calculates the number of additional imputations needed to achieve
            a target coefficient of variation (cv) for the standard error estimates,
            following Hippel (2020) [https://arxiv.org/pdf/1608.05406].

            :param pilot: int
                Number of pilot imputations to run for initial FMI estimation.
            :param alpha: float
                Significance level for confidence interval calculation (e.g., 0.05 for 95% CI).
            :param cv: float
                Desired coefficient of variation for the standard error estimates.
            :param fml: str
                Analysis model formula (in patsy format) to fit during imputation.

            :return: None
            """
        for i in tqdm(range(pilot), desc="pilot ", disable = False):
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

        for i in tqdm(range(self.m2-pilot), desc="stage2", disable = False):
            iterdata = self.iterate()
            if self.hist_bool:
                self.history[i + pilot] = iterdata
        self.pool()
    def iterate(self):
        """
        Performs the iterative imputation procedure.

        Starts with an initial imputation, then iteratively updates the imputations
        for a number of cycles (self.maxit) by applying the imputation model
        for each variable. After the final iteration, fits the analysis model to the
        imputed dataset and stores the fitted model results.

        Parameters
        ----------
        None

        Returns
        -------
        iterdata : pandas.DataFrame
            The imputed dataset after the final iteration.
        """
        iterdata = self._initial_imputation(self.initial)
        for j in range(self.maxit):
            iterdata = self._analysis(iterdata=iterdata)
        self.amodel = smf.ols(self.fml, data=iterdata)
        self.model_results.append(self.amodel.fit())
        return iterdata
    def complete(self):
        """
        Performs a single-step imputation to produce a completed dataset.

        Runs the initial imputation and then one iteration of the imputation analysis
        step to fill in missing values.

        Parameters
        ----------
        None

        Returns
        -------
        iterdata : pandas.DataFrame
            A completed dataset with imputed values after one imputation step.
        """
        iterdata = self._initial_imputation(self.initial)
        iterdata = self._analysis(iterdata=iterdata)
        return iterdata
    def convergence_plot(self, fml, x = "mean"):
        """
        Generates convergence plots of parameter estimates over iterations for multiple imputations.

        Runs multiple imputations and fits the analysis model for each iteration,
        then plots either the mean or standard error of model parameters across cycles.

        Parameters
        ----------
        fml : str
            Analysis model formula in patsy syntax.
        x : str, optional
            Metric to plot: "mean" for parameter estimates or "sd" for standard errors (default is "mean").

        Returns
        -------
        None
            Displays matplotlib plots of convergence diagnostics.
        """
        for i in tqdm(range(self.m), desc="M", disable = True):
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
            ax.set_xlabel('Cycles')
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
