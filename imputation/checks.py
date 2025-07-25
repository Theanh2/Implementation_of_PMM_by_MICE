from .predictorMatrix import quickpred
def _check_pm(data, predictorMatrix):
    if predictorMatrix is None:
        return quickpred(data, mincor=0.1, minpuc=0.1)
    if predictorMatrix is not None:
        return predictorMatrix
def _check_m(m):
    if m < 1:
        raise Exception("Number of imputations is lower than 1")
    m = int(m)
    return m
def _check_d(d, supported):
    if not isinstance(d, dict):
        raise ValueError("d not dict")
    for x in d.values():
        if x not in supported:
            methods = f"Imputation Method: {x} is not supported"
            raise ValueError(methods)