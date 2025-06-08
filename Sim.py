import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, poisson
from scipy.stats import ks_2samp
from tqdm import tqdm
from imputation.PMM import *
from imputation.midas import *
from mice import *
from imputation.predictorMatrix import quickpred
import time
import os
import time
def data_norm(n, locY, scaleY, rho):
    """
    data_norm(5000,5,1,0.5)
    Parameters
    ----------
    n
    gamma
    locY
    scaleY
    rho

    Returns
    -------

    """
    Y = np.random.normal(loc=locY, scale=scaleY, size=n)
    e = np.random.normal(size=n)
    X = rho * ((Y - np.mean(Y)) / np.std(Y)) + np.sqrt(1 - rho ** 2) * e
    corr = np.corrcoef(X, Y)[0, 1]
    return Y, X, corr
def data_semi(n, locY, scaleY, rho, pmass):
    """
    data_semi(5000,5,1,0,5,0.2)
    Parameters
    ----------
    n
    gamma
    locY
    scaleY
    rho
    pmass

    Returns
    -------

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
    Uses a Gaussian copula approach with optional iterative tuning.
    example:
    data_pois(n=10000, lambda_poisson=5, rho=0.5)
    """

    # Start latent correlation guess as target correlation
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
    """
    Compute the logit (log-odds) of a probability p.
    p must be in (0, 1).
    """
    p = np.asarray(p)
    return np.log(p / (1 - p))
def corr_mean(n, sim):
    """
    corr_mean(1000, "pois")
    Parameters
    ----------
    n
    sim

    Returns
    -------

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
            y,x,corr = data_pois(n=10000, lambda_poisson=5, rho=0.5)
            corr_list.append(corr)
    print(np.mean(corr_list))
def MCAR(ymis, miss):
    ymis = np.asarray(ymis, dtype=float)
    mask = np.random.rand(len(ymis)) < miss
    ycopy= ymis.copy()
    ycopy[mask] = np.nan
    return ycopy
def MAR(ymis, xdep, miss, tail):
    xdep = np.asarray(xdep)
    ymis = np.asarray(ymis, dtype=float)
    c = logit(miss)
    if tail == "right":
        zs = c + (np.mean(xdep)-xdep)/np.std(xdep)
    elif tail == "left":
        zs = c - (np.mean(xdep)-xdep) / np.std(xdep)
    pmis = expit(zs)
    mask = np.random.binomial(n=1, p=pmis, size = len(ymis)).astype(bool)
    ycopy = ymis.copy()
    ycopy[mask] = np.nan
    return ycopy
def miss_mean(Y, miss, n, md, X = None, tail = None):
    miss_list = []
    if md == "MCAR":
        for i in range(n):
            miss_list.append(np.sum(np.isnan(MCAR(Y, miss))))
    elif md == "MAR":
        for i in range(n):
            miss_list.append(np.sum(np.isnan(MAR(Y, X, miss, tail))))
    out = sum(miss_list)/(len(Y)*n)
    #print(miss_list)
    print(out)
def plot_density(Y,X):

    sns.histplot(X, label='X', color='blue', kde=False, bins=100)
    sns.histplot(Y, label='Y', color='orange', kde=False, bins=100)

    # Add legend and title
    plt.legend()
    #plt.title('Histogram of X and Y')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

def plot_missingness_tails(Y):
    Y = pd.Series(Y)
    missing_mask = Y.isna()
    plt.figure(figsize=(8, 5))
    sns.kdeplot(Y[~missing_mask], label='SIM', fill=True)
    sns.kdeplot(Y[missing_mask], label='NA', fill=True, color='red')
    plt.title("Distribution of Y: Observed vs. Missing")
    plt.xlabel("Y value")
    plt.show()
def Simulate(dist,n, mp, miss,m,k,hmi,pilot,method, tail = None, pmass = None):
    if dist == "norm":
        Y, X, _ = data_norm(n = n, locY=5, scaleY=1, rho=0.5)
    if dist == "semi":
        Y, X, _ = data_semi(n = n,locY = 5,scaleY = 1, rho = 0.5, pmass= pmass)
    if dist == "pois":
        Y, X, _ = data_pois(n=n, lambda_poisson= 4, rho=0.5)

    #actual Values
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
def _set_file(name):
    path  = os.path.join('/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/out', name + ".csv")
    print(path)
    if not os.path.exists(path):
        out = pd.DataFrame({"Bias Coef (mean)": [],
                            "Bias Coef (sd)":[] ,
                            "Width (mean)": [],
                            "Width (sd)":[] ,
                            "MSE (mean)": [],
                            "MSE (sd)": [],
                            "Bias sd (mean)": [],
                            "Bias sd (sd)": [],
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
                            "sdY": [],
                            "pmass": []})
        out.to_csv(path, mode="a", index=False, header=True)
def repeat_sim(dist,n, mp, miss,m,k,hmi,pilot,method, tail = None, pmass = None):
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
    sdBias_sd = np.std(sdBias_List)
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
                        "Bias sd (mean)": [sdBias_mean],
                        "Bias sd (sd)": [sdBias_sd],
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
                        "sdY": [sdY_mean],
                        "pmass": [pmass]}
                       )
    path  = os.path.join('/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/out', dist + mp + tail + str(int(miss*100)) + ".csv")
    out.to_csv(path, mode="a", index=False, header=False)

start_time = time.time()

m_values = [5,10,25,40]
miss_values = [0.1,0.2,0.4]

#Loop through all combinations of k and m
ex = 0
for miss in miss_values:
    for m in m_values:
        repeat_sim(
            dist="norm",
            n=3000,
            mp="MCAR",
            miss=miss,
            m=m,
            k=5,
            hmi=False,
            pilot=10,
            method="midas",
            tail="right",
            pmass=0.25
        )
    ex += 1
    print("exit in", ex, "/ 3")



#Simulate("norm",n = 3000,mp= "MAR", miss = 0.2,m = 2 ,k = 5,hmi= False, pilot = 10 ,method = "midas", tail = "right", pmass = 0.25)

end_time = time.time()
print("Computation time:", end_time - start_time, "seconds")