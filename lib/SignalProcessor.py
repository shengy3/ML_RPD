import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter

def filter_signal(signal, n = 13):
    #https://stackoverflow.com/questions/37598986/reducing-noise-on-data
    # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    return lfilter(b,a,signal)


## For locate signal purpose
def differentiate(signal, window):
    derivative = np.zeros(len(signal)-2*window)
    for i in range(window, len(derivative) - window):
        derivative[i] = (np.sum(signal[i:i+window])-np.sum(signal[i-window:i]))
    derivative = derivative - np.mean(derivative)
    derivative = np.pad(derivative, (0, window), 'constant', constant_values = 0)
    return derivative


# Locate signal
def locate_signal(signal, bins = 30, threshold = 7):
    signal = abs(signal)
    dif = differentiate(signal, bins)
    hist, bins = np.histogram(dif, bins='auto')
    std = np.sqrt(np.mean(hist**2))
    start = [i for i, val in enumerate(dif) if val > threshold * std]
    end = [i for i, val in enumerate(dif) if val < -threshold * std]

    if len(start) > 0 and len(end) > 0:

        return start[0], end[-1], dif
    else:
        return None, None, None
    
