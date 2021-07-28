import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar



# nasty way to integrate an exp distribution
def distfunction(tLL, tUL, gamma, ti, nob):
  return np.log(-((np.exp(gamma*ti + gamma*tLL + gamma*tUL)*nob)/
  (-np.exp(gamma*ti + gamma*tLL) + np.exp(gamma*ti + gamma*tUL) -
    np.exp(gamma*tLL + gamma*tUL)*nob)))/gamma


#%% do it!

def create_time_bins(nbins, tLL=0.3, tUL=15):
  # tLL = 0.3; tUL = 15
  dummy = 0.66; gamma = 0.0
  # for bins in range(3,13):
  list_bins = [tLL]; ipdf = []; widths = []
  for k in range(0, nbins):
    ti = list_bins[k]
    list_bins.append( distfunction(tLL, tUL, dummy, ti, nbins)   )
  list_bins.append(tUL)
  list_bins[-2] = np.median([list_bins[-3],tUL])
  ans = np.array(list_bins)
  ans = np.round(100*ans)/100
  return ans
  # print(f"{bins:>2} : {[np.round(list_bins[i]*100)/100for i in range(len(list_bins))]},")







'''

3 knots  -> [0.3,                    0.91,                   1.96,                    9.0,  15.0]
6 knots  -> [0.3,        0.58,       0.91,       1.35,       1.96,       3.01,        7.0,  15.0]
12 knots -> [0.30, 0.43, 0.58, 0.74, 0.91, 1.11, 1.35, 1.63, 1.96, 2.40, 3.01, 4.06, 9.00,  15.0]



np.median([3.01, 15])
np.median([1.96, 15])

#%%

(3.0143238 + 15)*np.exp(-0.75)
18/2

'''
