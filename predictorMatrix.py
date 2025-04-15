#Mimics quickpred() function from mice package in R
import numpy as np
import pandas as pd

##test data only numerical values from nhanes
df = pd.read_csv("/Users/theanh/Desktop/nhanes.csv")
df = df.iloc[0:20,[4,6,12,13]]

for col in df.columns:
    df['null'] = np.random.choice([0, 1], size=df.shape[0], p=[1-0.5, 0.5])
    df.loc[df['null'] == 1, col] = np.nan

df.drop(columns=['null'], inplace=True)
#---
def all(data):
    """
    creates predictormatrix with using all variables
    :param data: data
    :return: predictormatrix
    """
    predictormatrix = pd.DataFrame(1, index=data.columns, columns=data.columns, dtype=int)
    np.fill_diagonal(predictormatrix.values, 0)
    return predictormatrix




def quickpred(data, mincor=0.1, minpuc=0, include="", exclude="", method="pearson"):
    """

    :param data: data frame with incomplete data
    :param mincor: specifying the minimum threshold against which the absolute correlation in the data is compared
    :param minpuc: specifying the minimum threshold for the proportion of usable cases
    :param include: include variable, if variable is in include and exclude it includes it
    :param exclude: exclude variable
    :param method: uses pandas corr function: pearson, kendall or spearman
    :return:
    """
    predictormatrix = pd.DataFrame(0, index=data.columns, columns=data.columns, dtype=int)
    r = data.notna()

    # Correlation matrices
    #pairwise correlation and replace NA with 0
    v = np.abs(pd.DataFrame(data).corr(method=method,numeric_only=True).fillna(0).to_numpy())
    #pairwise correlation and replace NA with 0
    u = np.abs(pd.DataFrame(data).corrwith(pd.DataFrame(r.astype(float)), method=method,numeric_only=True).fillna(0).to_numpy())

    maxc = np.maximum(v, u)
    predictormatrix[:] = (maxc > mincor).astype(int)

    # Exclude predictors below minpuc threshold
    if minpuc != 0:
        p = md_pairs(data)
        puc = p['mr'] / (p['mr'] + p['mm'])
        puc = puc.replace([np.inf, -np.inf], 0).fillna(0)
        predictormatrix[puc < minpuc] = 0

    # Exclude variables in 'exclude'
    if exclude:
        for col in data.columns:
            if col in exclude:
                predictormatrix[col].values[:] = 0


    # # Include variables in 'include'
    if include:
        for col in data.columns:
            if col in include:

                predictormatrix[col].values[:] = 1

    #Diagonal = 0
    np.fill_diagonal(predictormatrix.values, 0)

    #column no missing values set to 0
    complete_cases = data.isna().sum(axis=0) == 0
    predictormatrix.loc[complete_cases, :] = 0

    return predictormatrix

#
def md_pairs(data):
    """
    mimics md.pairs from mice
    calculates number of observations per variable pair.
    r = response
    m = missing
    :param data:
    :return: rr, rm, mr, mm
    """

    r = data.notna().astype(int)
    m = data.isna().astype(int)
    rr = np.matmul(r.T, r)
    mm = np.matmul(m.T,m)
    mr = np.matmul(m.T,r)
    rm = np.matmul(r.T,m)

    return {'rr': rr, 'rm': rm, 'mr': mr, 'mm': mm}

quickpred(df, mincor= 0.1, exclude = "AgeMonths", include = "Age", minpuc = 0.1)