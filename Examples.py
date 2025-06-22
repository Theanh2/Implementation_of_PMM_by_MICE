import pandas as pd
import numpy as np
from simulations.Sim import data_norm, MCAR
from imputation.mice import mice

Y, X, _ = data_norm(n = 1000, locY=5, scaleY=1, rho=0.5)
Y = MCAR(ymis = Y, miss = 0.2)
simdf = pd.DataFrame({"Y": Y, "X": X})

pm = pd.DataFrame({
    'Y': {'Y': 0, 'X': 1},
    'X': {'Y': 0, 'X': 0}
})

miceobj = mice(data = simdf, m = 50, predictorMatrix = pm , initial = "sample", maxit = 5)
miceobj.set_methods(d = {"Y": "pmm"})
result = miceobj.fit(fml="Y ~ X", donors=3, history=False)
print(result.summary())
