##from numarray import *
from numpy import *
import numpy as np

def UnweightedAvg(meanList,errorList):
    mean=sum(meanList)/(len(meanList)+0.0)
    error=0.0
    for e in errorList:
        error=error+e*e
    error=sqrt(error)/len(errorList)
    return (mean,error)


def WeightedAvg (means, errors):
    zeroErrors = False
    for i in errors:
        if i == 0.0:
            zeroErrors = True
    
    if (not zeroErrors):
        weights = map (lambda x: 1.0/(x*x), errors)
        norm = 1.0/sum(weights)
        weights = map(lambda x: x*norm, weights)
        avg = 0.0
        error2 = 0.0
        for i in range (0,len(means)):
            avg = avg + means[i]*weights[i]
            error2 = error2 + weights[i]**2*errors[i]*errors[i]
        return (avg, math.sqrt(error2))
    else:
        return (sum(means)/len(means), 0.0)


def MeanErrorString (mean, error):
     if (mean!=0.0):
          meanDigits = math.floor(math.log(abs(mean))/math.log(10))
     else:
          meanDigits=2
     if (error!=0.0):
          rightDigits = -math.floor(math.log(error)/math.log(10))+1
     else:
          rightDigits=2
     if (rightDigits < 0):
          rightDigits = 0
     formatstr = '%1.' + '%d' % rightDigits + 'f'
     meanstr  = formatstr % mean
     errorstr = formatstr % error
     return (meanstr, errorstr)


def c(i, x, mean, var):
    """
    Compute the normalized autocorrelation for lag i.
    If var is zero, return a very large number (effectively infinity).
    """
    N = len(x)
    if var == 0:
        return 1e100
    # Use numpy's slicing and np.sum for clarity.
    return np.sum((x[:N-i] - mean) * (x[i:] - mean)) / ((N - i) * var)

def Stats(x):
    """
    Compute the mean, variance, error (standard error on the mean using
    the effective sample size), and the integrated autocorrelation time.
    
    Parameters:
      x : numpy array (or array-like) of data.
      
    Returns:
      mean : average of x
      var  : variance of x
      error: standard error on the mean taking autocorrelations into account
      tau  : integrated autocorrelation time (tau_int)
    """
    # Ensure x is a numpy array with a float dtype for robustness.
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    # Use numpy for mean and variance (population variance)
    mean = np.mean(x)
    var = np.var(x)
    
    # Compute integrated autocorrelation time (tau)
    # tau = 1 + 2 * sum_{i=1}^{T} c(i), stopping when c(i) becomes <= 0.
    tau = 1.0
    for i in range(1, N):
        corr = c(i, x, mean, var)
        if corr <= 0:
            break
        tau += 2.0 * corr
    
    # Prevent tau from being zero (should not happen normally)
    if tau <= 0:
        tau = 1.0
    
    # Effective sample size and standard error:
    Neff = N / tau
    error = math.sqrt(var / Neff)
    
    return (mean, var, error, tau)