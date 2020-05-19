from scipy import asarray as ar,exp


def gaussian(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def bimodal(x,a1,x1,sigma1,a2,x2,sigm2):
    return gaussian(x,a1,x1,sigma1)+gaussian(x,a2,x2,sigm2)
