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
  def __compute(nbins, tLL, tUL):
    dummy = 0.66; gamma = 0.0
    # for bins in range(3,13):
    list_bins = [tLL]; ipdf = []; widths = []
    for k in range(0, nbins):
      ti = list_bins[k]
      list_bins.append( distfunction(tLL, tUL, dummy, ti, nbins)   )
    list_bins.append(tUL)
    list_bins[-2] = np.median([list_bins[-3],tUL])
    ans = np.array(list_bins)
    return np.round(100*ans)/100
  
  # we can also have some custom knots predefined
  if np.isclose(tLL, 0.3, 1e-2) and np.isclose(tUL, 15, 1e-2):
    knots =  {
      "2": [ 0.3, 0.91, 2.0, 15.0],
      # concerning the 3-knot version, we have to be sure with 9ps all tests
      # converge
      "3": [ 0.3, 0.91, 1.96, 9.0, 15.0],
      "4": [ 0.3, 0.74, 1.35, 2.4, 9.0, 15.0],
      "5": [ 0.3, 0.64, 1.07, 1.69, 2.74, 9.0, 15.0],
      "6": [ 0.3, 0.58, 0.91, 1.35, 1.96, 3.01, 7.0, 15.0],
      "7": [ 0.3, 0.53, 0.81, 1.15, 1.58, 2.2, 3.25, 9.0, 15.0],
      "8": [ 0.3, 0.5, 0.74, 1.01, 1.35, 1.79, 2.4, 3.45, 9.0, 15.0],
      "9": [ 0.3, 0.48, 0.68, 0.91, 1.19, 1.53, 1.96, 2.58, 3.63, 9.0, 15.0],
      "10": [ 0.3, 0.46, 0.64, 0.84, 1.07, 1.35, 1.69, 2.12, 2.74, 3.79, 7.0, 15.0 ],
      "11": [ 0.3, 0.44, 0.6, 0.78, 0.98, 1.22, 1.49, 1.83, 2.27, 2.88, 3.93, 7.0, 15.0 ],
      "12": [ 0.3, 0.43, 0.58, 0.74, 0.91, 1.12, 1.35, 1.63, 1.96, 2.4, 3.01, 4.06, 7.0, 15.0 ]
    }
    return np.array( knots[str(nbins)] )
  return __compute(nbins, tLL, tUL)
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
