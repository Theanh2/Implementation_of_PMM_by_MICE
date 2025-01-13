import math

#howManyImputations
#A Two-stage Calculation Using a Quadratic Rule (https://pubmed.ncbi.nlm.nih.gov/39211325/)
#https://github.com/josherrickson/howManyImputations/blob/main/R/how_many_imputations.R
#fmi = fraction of missing information
#cv = Desired precision of standard errors. Default to .05. If the data
#       were re-imputed, the estimated standard errors would differ by no more than this amount.
#alpha = Significance level for choice of "conservative" FMI.
#return = The number of required imputations to obtain the cv level of precision.


def HMI(model, cv = 0.05, alpha = 0.05):
    model = pool(model)
    fmi = max(fmi)
    z = qnorm(1-alpha/2)
    fmiu = plogis(qlogis(fmi) + z*sqrt(2/model$m))
    ceiling(1 + 1 / 2 * (fmiu / cv) ^ 2)