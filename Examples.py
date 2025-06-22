from imputation.PMM import pmm
from imputation.midas import midas
import pandas as pd
import numpy as np
from simulations.Sim import repeat_sim

y = np.array([7, np.nan, 9, 10, 11])
ry = ~np.isnan(y)
x = np.array([[1, 2], [3, 4], [5, 7], [7, 8], [9, 10]])
print("pmm")
print(pmm(y=y, ry=ry, x=x, donors=3))
print("midas")
print(midas(y, ry, x))

print("Multiple Imputation")
repeat_sim(
        dist="norm",
        n=500,
        mp="MCAR",
        miss=0.6,
        m=5,
        k=5,
        hmi=False,
        pilot=5,
        method="pmm",
        tail="left",
        pmass=0.2
    )