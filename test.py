from mice import *
from imputation.predictorMatrix import quickpred
import random
import pandas as pd
import numpy as np
random.seed(10)
testdf = pd.read_csv("/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/Implementation of PMM by MICE/data/fdgs.csv")
testdf = testdf.iloc[1:10, [2,5,6,7]]
testdf.columns = testdf.columns.str.replace("."," ", regex=False)
for col in testdf.columns:
    testdf.loc[testdf.sample(frac=0.5).index, col] = np.nan


pm = quickpred(testdf, mincor= 0.1, minpuc = 0.1)
obj = mice(data = testdf, m = 5, initial = "sample", maxit=10)
obj.set_methods(d = {"age": "pmm"})
x = obj.fit(fml = "age ~ wgt", donors = 5, history = True, HMI = True, pilot = 10)
#obj.convergence_plot(fml = "age ~ wgt")
print(x.summary())
print(len(x.model.model_results))
