#mice object
import random

def mice(data = None,
         m = 5,
         predictorMatrix = None,
         formulas = None,
         blocks = None,
         seed = 555):

    if seed is not None:
        random.seed(seed)


mice(iris, predictorMatrix = None,formulas = None, blocks = None)


def check_blocks():
    print("check_blocks")


def check_m(m):
    if isinstance(m, (list, tuple)):
        m = m[0]  # Take the first element

    if not isinstance(m, (int, float)):
        raise ValueError("Argument m not numeric")

    m = int(m)

    if m < 1:
        raise ValueError("Number of imputations (m) lower than 1.")
    print("check_m")
    return m


