from scipy.optimize import curve_fit
from .Function import gaussian, bimodal


def fit_gaussian(x, y, expected = (10,0,10)):
    pG1,cov=curve_fit(gaussian,x,y,expected)
    return [pG1], [cov]

def fit_double_gaussian(x, y, expected = (10,-1,1,10,0,10)): 
    params,cov=curve_fit(bimodal,x,y,expected)
    pG1 = abs(params[0:3])
    pG2 = abs(params[3:])
    return [pG1, pG2], [cov]