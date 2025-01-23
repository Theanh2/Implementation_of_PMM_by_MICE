import numpy as np
import scipy
import math

#HMI: howManyImputations
#A Two-stage Calculation Using a Quadratic Rule (https://pubmed.ncbi.nlm.nih.gov/39211325/)
#https://github.com/josherrickson/howManyImputations/blob/main/R/how_many_imputations.R
#fmi = fraction of missing information
#cv = Desired precision of standard errors. Default to .05. If the data
#       were re-imputed, the estimated standard errors would differ by no more than this amount.
#alpha = Significance level for choice of "conservative" FMI.
#return = The number of required imputations to obtain the cv level of precision.

def HMI(model, cv = 0.05, alpha = 0.05):
    #pass model correctly
    model = pool(model)
    fmi = max(fmi)
    z = qnorm(1-alpha/2)
    fmiu = cdf(ppf(fmi) + z*sqrt(2/model))
    math.ceil(1 + 1 / 2 * (fmiu / cv) ^ 2)

#R mice::pool
#dfcom: A positive number representing the degrees of freedom in the complete-data analysis.
#rule:  A string indicating the pooling rule
def pool(object, dfcom = NULL, rule = NULL):
    #...

def pool_syn(object, dfcom = NULL):
    #combines estimates by Reiter's partially synthetic data pooling rules
    #This combination rule assumes that the data that is synthesised is completely observed

#!!! Need utils function to get dfcom + cov + vcov

