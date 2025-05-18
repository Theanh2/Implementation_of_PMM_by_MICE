import statsmodels.imputation.mice

from imputation.predictorMatrix import quickpred
from statsmodels.imputation.mice import MICEData
from statsmodels.regression.linear_model import OLS
from mice import *
import random
random.seed(10)
testdf = pd.read_csv("/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/Implementation of PMM by MICE/data/fdgs.csv")

testdf = testdf.iloc[0:10, [2,5]]

for col in testdf.columns:
    testdf.loc[testdf.sample(frac=0.5).index, col] = np.nan

imp = statsmodels.imputation.mice.MICEData(testdf)
fml = 'age ~ wgt'
x = statsmodels.imputation.mice.MICE(fml, statsmodels.regression.linear_model.OLS, imp)
results = x.fit(5, 10)
print(x.results_list[0]._results.params)
print(results.summary())
print(results)