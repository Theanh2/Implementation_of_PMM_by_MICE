from imputation.predictorMatrix import quickpred
from mice import *
import random
random.seed(10)
testdf = pd.read_csv("/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/Implementation of PMM by MICE/data/fdgs.csv")

testdf = testdf.iloc[0:10, [2,5,6]]

for col in testdf.columns:
    testdf.loc[testdf.sample(frac=0.5).index, col] = np.nan

pm = quickpred(testdf, mincor= 0.1, minpuc = 0.1)
obj = mice(data = testdf, m=2, initial = "random")
obj.set_methods(d = {"age": "pmm"})
obj.fit(donors = 5)

#ry = id_obs
#x = regular data subset by predictormatrix = 1
#y = variable to be imputed replaced ry to na.nan
#obj.pmm(y,ry,x)