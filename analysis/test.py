# imports
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set(context="talk", rc={'figure.figsize': (11.7, 8.27)})

# helper functions used throughout this project
print_header = lambda msg: print(f"{msg}\n{'-' * len(msg)}")

# seed to follow along
np.random.seed(654654)

# generate 1500 data points
N = np.arange(1500)

# helper function for this data
vary = lambda v: np.random.choice(np.arange(v))

# create correlated, random variables
a = 2
b = 1 / 2
eps = np.array([norm(0, vary(50)).rvs() for n in N])
y = (a + b * N + eps) / 100
x = (N + norm(10, vary(250)).rvs(len(N))) / 100

# 20% missing in x, 30% missing in y
x[binom(1, 0.2).rvs(len(N)) == 1] = np.nan
y[binom(1, 0.3).rvs(len(N)) == 1] = np.nan

# collect results in a dataframe
data_miss = pd.DataFrame({"y": y, "x": x})
sns.scatterplot(x="x", y="y", data=data_miss)
plt.show()