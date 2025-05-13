import pandas as pd
import numpy as np
from predictorMatrix import quickpred
from mice import *

df = pd.read_csv("/Users/theanh/Library/Mobile Documents/com~apple~CloudDocs/Stats/Master/Thesis/Implementation of PMM by MICE/data/fdgs.csv")

df = df.iloc[0:100, [2,5,6]]

for col in df.columns:
    df.loc[df.sample(frac=0.3).index, col] = np.nan

pm = quickpred(df, mincor= 0.1, minpuc = 0.1)

micobj = mice(df)