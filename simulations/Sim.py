import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, poisson
from scipy.stats import ks_2samp
from tqdm import tqdm
from imputation.PMM import *
from imputation.midas import *
from imputation.mice import *
from imputation.predictorMatrix import quickpred
import time
import os
import time
def data_norm(n, locY, scaleY, rho):
    """
    Simulates `n` normally distributed observations for response variable Y and a covariate X.

    The covariate X is generated with a specified correlation `rho` to Y using Cholesky decomposition.

    :param n: Number of observations.
    :param locY: Mean (location) of Y.
    :param scaleY: Standard deviation (scale) of Y.
    :param rho: Correlation coefficient between Y and X (between -1 and 1).

    :return: A tuple containing:
        - Y (np.ndarray): Simulated response variable.
        - X (np.ndarray): Simulated covariate with correlation `rho` to Y.
        - corr (float): Empirical correlation between Y and X.
    """
    Y = np.random.normal(loc=locY, scale=scaleY, size=n)
    e = np.random.normal(size=n)
    X = rho * ((Y - np.mean(Y)) / np.std(Y)) + np.sqrt(1 - rho ** 2) * e
    corr = np.corrcoef(X, Y)[0, 1]
    return Y, X, corr
def data_semi(n, locY, scaleY, rho, pmass):
    """
    Simulates a semi-continuous response variable Y and a correlated covariate X.

    - Y is drawn from a normal distribution with mean `locY` and standard deviation `scaleY`.
    - Y is transformed via `Y^4 / max(Y^3)` to induce right skewness.
    - A point mass at zero is introduced by randomly setting values to zero with probability `pmass`.
    - Covariate X is generated to have correlation `rho` with Y using Cholesky decomposition.

    :param n: Number of observations to simulate.
    :param locY: Mean (location) of the normally distributed base Y.
    :param scaleY: Standard deviation (scale) of Y.
    :param rho: Desired Pearson correlation coefficient between Y and X (range -1 to 1).
    :param pmass: Probability of setting a value in Y to zero (point mass at zero).

    :return: A tuple containing:
        - Y (np.ndarray): Semi-continuous response variable with right skewness and point mass.
        - X (np.ndarray): Covariate with specified correlation to Y.
        - corr (float): Empirical correlation between X and Y.
    """
    Y = np.random.normal(loc=locY, scale=scaleY, size=n)
    zeroes = np.random.binomial(1, pmass, size=n)
    Y = (Y**4)/np.max(Y**3)
    Y = np.where(zeroes == 1, 0, Y)
    e = np.random.normal(size=n)
    X = rho * ((Y - np.mean(Y)) / np.std(Y)) + np.sqrt(1 - rho ** 2) * e
    corr = np.corrcoef(X, Y)[0, 1]
    return Y, X, corr
def data_pois(n, lambda_poisson, rho, max_iter=30, tol=0.01):
    """
     Simulates a Poisson-distributed response variable Y and a continuous covariate X
     with a specified correlation using a Gaussian copula approach.

     - Y is generated from a Poisson distribution with mean `lambda_poisson`.
     - X is a standard normal variable correlated with Y using a Gaussian copula.
     - Iteratively adjusts the latent correlation to match the target `rho` within a tolerance `tol`.

     :param n: Number of observations to simulate.
     :param lambda_poisson: Mean (λ) of the Poisson distribution.
     :param rho: Desired Pearson correlation between Y and X.
     :param max_iter: Maximum number of iterations to adjust latent correlation.
     :param tol: Tolerance threshold for the difference between target and achieved correlation.

     :return: A tuple containing:
         - Y (np.ndarray): Poisson-distributed response variable.
         - X (np.ndarray): Covariate with approximately the specified correlation to Y.
         - corr (float): Achieved empirical correlation between X and Y.
     """
    latent_rho = rho

    for i in range(max_iter):
        mean = [0, 0]
        cov = [[1, latent_rho], [latent_rho, 1]]
        z = np.random.multivariate_normal(mean, cov, size=n)
        u = norm.cdf(z[:, 0])
        Y = poisson.ppf(u, mu=lambda_poisson).astype(int)
        X = z[:, 1]
        actual_rho = np.corrcoef(X, Y)[0, 1]
        if abs(actual_rho - rho) < tol:
            break
        latent_rho = latent_rho * rho / (actual_rho + 1e-6)
        latent_rho = np.clip(latent_rho, -0.99, 0.99)

    return Y, X, actual_rho
def expit(x):
    return 1 / (1 + np.exp(-x))
def logit(p):
    p = np.asarray(p)
    return np.log(p / (1 - p))
def corr_mean(n, sim):
    """
    Tests whether the target correlation is achieved across multiple simulated datasets.

    Simulates `n` datasets, each with 5000 observations, using the specified distribution.
    The distributions correspond to those used in the thesis: semicontinuous, normal, or Poisson.
    The function returns the average empirical correlation across all simulations.

    :param n: Number of datasets to generate.
    :param sim: Distribution type; must be one of "semi", "norm", or "pois".

    :return: Average empirical correlation across the `n` simulated datasets.
    :rtype: float
    """
    corr_list = []
    for i in range(n):
        if sim == "semi":
            y,x,corr = data_semi(5000,5,1,0.5,0.5)
            corr_list.append(corr)
        elif sim == "norm":
            y,x,corr = data_norm(5000, 5, 1, 0.5)
            corr_list.append(corr)
        elif sim == "pois":
            y,x,corr = data_pois(n=5000, lambda_poisson=4, rho=0.5)
            corr_list.append(corr)
    return np.mean(corr_list)
def MCAR(ymis, miss):
    """
    Introduces Missing Completely at Random (MCAR) missingness into a numeric array.

    Randomly sets a proportion of values in `ymis` to NaN based on the specified
    missingness probability `miss`.

    :param ymis: Array-like numeric input Y.
    :param miss: Proportion of values to be set as missing (float between 0 and 1).

    :return: A NumPy array with MCAR-induced missing values.
    :rtype: np.ndarray
    """
    ymis = np.asarray(ymis, dtype=float)
    mask = np.random.rand(len(ymis)) < miss
    ycopy= ymis.copy()
    ycopy[mask] = np.nan
    return ycopy
def MAR(ymis, xdep, miss, tail):
    """
    Introduces Missing At Random (MAR) missingness into a numeric array,
    dependent on a covariate.

    The missingness in `ymis` is introduced based on the values of `xdep`. The
    probability of missingness is determined using a logistic model, where higher or
    lower values of `xdep` (depending on the `tail`) lead to increased missingness.

    :param ymis: Array-like numeric input (e.g., outcome variable Y).
    :param xdep: Covariate (X) that missingness depends on.
    :param miss: Target overall missingness probability (float between 0 and 1).
    :param tail: Direction of dependency; "left" (more missing for low X) or
                 "right" (more missing for high X).

    :return: A NumPy array with MAR-induced missing values based on `xdep`.
    :rtype: np.ndarray
    """
    xdep = np.asarray(xdep)
    ymis = np.asarray(ymis, dtype=float)
    c = logit(miss)
    if tail == "left":
        zs = c + (np.mean(xdep)-xdep)/np.std(xdep)
    elif tail == "right":
        zs = c - (np.mean(xdep)-xdep) / np.std(xdep)
    pmis = expit(zs)
    mask = np.random.binomial(n=1, p=pmis, size = len(ymis)).astype(bool)
    ycopy = ymis.copy()
    ycopy[mask] = np.nan
    return ycopy
def miss_mean(Y, miss, n, md, X = None, tail = None):
    """
    Estimates the average proportion of missing values introduced under a specified
    missingness mechanism.

    Repeats the missingness process `n` times on the variable `Y` using either
    Missing Completely At Random (MCAR) or Missing At Random (MAR) mechanism.
    For MAR, a covariate `X` and a direction `tail` must be provided.

    :param Y: Array-like outcome variable.
    :param miss: Target missingness probability (float between 0 and 1).
    :param n: Number of repetitions to simulate missingness.
    :param md: Missingness mechanism; must be "MCAR" or "MAR".
    :param X: Covariate X (required if `md` is "MAR").
    :param tail: Direction of missingness for MAR; "left" or "right".

    :return: Average proportion of missing values across `n` repetitions.
    :rtype: float
    """
    miss_list = []
    if md == "MCAR":
        for i in range(n):
            miss_list.append(np.sum(np.isnan(MCAR(Y, miss))))
    elif md == "MAR":
        for i in range(n):
            miss_list.append(np.sum(np.isnan(MAR(Y, X, miss, tail))))
    out = sum(miss_list)/(len(Y)*n)
    return out
def plot_density(Y,X):
    """
    Plots the histogram-based density of two variables, Y and X.

    Overlays histograms of Y and X using 100 bins each for visual comparison.

    :param Y: Array-like variable (e.g., response variable).
    :param X: Array-like variable (e.g., covariate).
    :return: None. Displays the plot.
    :rtype: None
    """
    sns.histplot(X, label='X', color='blue', kde=False, bins=100)
    sns.histplot(Y, label='Y', color='orange', kde=False, bins=100)
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
def plot_missingness_tails(Y, X):
    """
    Visualizes the distribution of the covariate X, conditional on whether
    the corresponding Y values are observed or missing.

    This plot is useful for illustrating Missing At Random (MAR)
    mechanisms where missingness in Y depends on X.

    :param Y: Array-like response variable containing missing values.
    :param X: Array-like covariate variable used to model missingness.
    :return: None. Displays the plot.
    :rtype: None
    """
    X = pd.Series(X)
    Y = pd.Series(Y)
    missing_mask = Y.isna()
    plt.figure(figsize=(8, 5))
    sns.kdeplot(X[~missing_mask], label='Y Observed', fill=True)
    sns.kdeplot(X[missing_mask], label='Y Missing', fill=True, color='red')
    plt.title("Distribution of X by Y Missingness (MAR Visualization)")
    plt.xlabel("X (Observed Variable)")
    plt.legend()
    plt.grid(True)
    plt.show()
def Simulate(dist,n, mp, miss,m,k,hmi,pilot,method, tail = None, pmass = None):
    """
    One simulation run.

    :param dist: Distribution of Y.
    :param n: Number of observations.
    :param mp: Missingness mechanism; "MCAR" or "MAR".
    :param miss: Missingness probability.
    :param m: Number of multiple imputation iterations.
    :param k: Donor size for PMM.
    :param hmi: Whether to use HowManyImputations (True/False).
    :param pilot: Number of pilot imputations for HMI.
    :param method: Imputation method: "pmm" or "midas".
    :param tail: Tail direction for MAR; "left" or "right".
    :param pmass: Point mass probability.

    :return: Tuple of (cBias, Width, MSE, Coverage, sdBias, it, actual_missing, sdY)
    """
    if dist == "norm":
        Y, X, _ = data_norm(n = n, locY=5, scaleY=1, rho=0.5)
    if dist == "semi":
        Y, X, _ = data_semi(n = n,locY = 5,scaleY = 1, rho = 0.5, pmass= pmass)
    if dist == "pois":
        Y, X, _ = data_pois(n=n, lambda_poisson= 4, rho=0.5)
    meanY = np.mean(Y)
    sdY = np.std(Y)

    #Missingness
    if mp == "MCAR":
        Y = MCAR(ymis = Y, miss = miss)

    if mp == "MAR":
        Y = MAR(ymis= Y, xdep = X, miss = miss, tail = tail)

    #actual missing, because of binomial draw
    actual_missing = np.isnan(Y).sum()/len(Y)

    #pass as dataframe
    simdf = pd.DataFrame({"Y": Y, "X": X})

    #predictormatrix skip X in Mice
    pm = pd.DataFrame({
        'Y': {'Y': 0, 'X': 1},
        'X': {'Y': 0, 'X': 0}
    })

    #Run Mice
    obj = mice(data=simdf, m=m, initial="sample",predictorMatrix = pm, maxit=5)
    if method == "pmm":
        obj.set_methods(d={"Y": "pmm"})
    elif method == "midas":
        obj.set_methods(d={"Y": "midas"})
    result = obj.fit(fml="Y ~ X", donors=k, history=False, HMI=hmi, pilot=pilot)
    sdBias = result.bse[0]-sdY
    Coverage = bool(result.conf_int()[0][0] <= meanY <= result.conf_int()[0][1])
    Width = result.conf_int()[0][1] - result.conf_int()[0][0]
    cBias = result.params[0]-meanY
    MSE_list = []
    for i in result.model.model_results:
        MSE_list.append(np.mean(i.resid**2))
    MSE = np.mean(MSE_list)
    it = len(result.model.model_results)
    return cBias, Width, MSE, Coverage, sdBias, it, actual_missing, sdY
def _set_file(path):
    if not os.path.exists(path):
        out = pd.DataFrame({"Bias Coef (mean)": [],
                            "Bias Coef (sd)":[] ,
                            "Width (mean)": [],
                            "Width (sd)":[] ,
                            "MSE (mean)": [],
                            "MSE (sd)": [],
                            "Coverage":[] ,
                            "it (mean)": [],
                            "it (sd)": [],
                            "miss": [],
                            "actual_missing (mean)": [],
                            "actual_missing (sd)": [],
                            "dist":[] ,
                            "mp":[] ,
                            "m": [],
                            "k": [],
                            "tail":[] ,
                            "hmi":[] ,
                            "pmass": [],
                            "std": [],
                             "method": []
        })
        out.to_csv(path, mode="a", index=False, header=True)
def repeat_sim(dist,n, mp, miss,m,k,hmi,pilot,method, tail = None, pmass = None):
    """
        Repeats the simulation experiment 500 times and stores aggregated performance metrics.

        For each run, the function calls `Simulate()` with the given parameters and
        collects evaluation metrics. Results are aggregated
        and written to a CSV file named according to the simulation parameters.

        :param dist: Distribution type of Y ("norm", "semi", or "pois").
        :param n: Number of observations in each simulation.
        :param mp: Missingness mechanism ("MCAR" or "MAR").
        :param miss: Probability of missingness (float between 0 and 1).
        :param m: Number of multiple imputations.
        :param k: Donor pool size for PMM.
        :param hmi: Boolean; whether to use How-Many-Imputations (HMI) approach.
        :param pilot: Number of pilot imputations for HMI.
        :param method: Imputation method ("pmm" or "midas").
        :param tail: Direction of missingness for MAR ("left" or "right"). Required if mp == "MAR".
        :param pmass: Probability of inducing a point mass at zero (for semi-continuous Y).

        :return: None. Saves results to a CSV file in the project directory.
        :rtype: None

        Example:
            >>> repeat_sim(
            ...     dist="norm",
            ...     n=500,
            ...     mp="MCAR",
            ...     miss=0.6,
            ...     m=5,
            ...     k=5,
            ...     hmi=False,
            ...     pilot=5,
            ...     method="pmm",
            ...     tail="left",
            ...     pmass=0.2
            ... )
        """
    cBias_List = []
    Width_List = []
    MSE_List = []
    Coverage_List = []
    sdBias_List = []
    it_List = []
    actual_missing_List = []
    sdY_List = []
    for i in tqdm(range(500)):
        cBias, Width, MSE, Coverage, sdBias, it, actual_missing, sdY = Simulate(dist = dist,
                 n = n,
                 mp = mp,
                 miss = miss,
                 m = m,
                 k = k,
                 hmi = hmi,
                 pilot = pilot,
                 method = method,
                 tail = tail,
                 pmass = pmass)
        cBias_List.append(cBias)
        Width_List.append(Width)
        MSE_List.append(MSE)
        Coverage_List.append(Coverage)
        sdBias_List.append(sdBias)
        it_List.append(it)
        actual_missing_List.append(actual_missing)
        sdY_List.append(sdY)


    cBias_mean = np.mean(cBias_List)
    cBias_sd = np.std(cBias_List)
    Width_mean = np.mean(Width_List)
    Width_sd = np.std(Width_List)
    MSE_mean = np.mean(MSE_List)
    MSE_sd = np.std(MSE_List)
    Coverage = np.sum(Coverage_List)/500
    sdBias_mean = np.mean(sdBias_List)
    it_mean = np.mean(it_List)
    it_sd = np.std(it_List)
    actual_missing_mean = np.mean(actual_missing_List)
    actual_missing_sd = np.std(actual_missing_List)
    sdY_mean = np.mean(sdY_List)


    out = pd.DataFrame({"Bias Coef (mean)": [cBias_mean],
                            "Bias Coef (sd)":[cBias_sd] ,
                            "Width (mean)": [Width_mean],
                            "Width (sd)":[Width_sd] ,
                            "MSE (mean)": [MSE_mean],
                            "MSE (sd)": [MSE_sd],
                            "Coverage":[Coverage] ,
                            "it (mean)": [it_mean],
                            "it (sd)": [it_sd],
                            "miss": [miss],
                            "actual_missing (mean)": [actual_missing_mean],
                            "actual_missing (sd)": [actual_missing_sd],
                            "dist":[dist] ,
                            "mp":[mp] ,
                            "m": [m],
                            "k": [k],
                            "tail":[tail] ,
                            "hmi":[hmi] ,
                            "pmass": [pmass],
                            "std": [sdBias_mean+sdY_mean],
                             "method": [method]
        })
    path  = os.path.join(dist + mp + tail + str(int(miss*100))+ method + ".csv")
    _set_file(path)
    out.to_csv(path, mode="a", index=False, header=False)


