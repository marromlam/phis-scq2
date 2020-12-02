from scipy.stats import chi2
from scipy import stats, special
import numpy as np

def check_agreement(values, covs, sigmas=True, pvalue=False, chi2=False):
  dof = len(values)
  v = [np.array(v).size for v in values]
  c = [np.array(v).size for c in covs]
  v = len(set(v)) <= 1
  c = len(set(c)) <= 1

  if v and c:
    diff = np.array(values[0])-np.array(values[1])
    c = np.matrix(covs[0])+np.matrix(covs[1])
    chisqr = np.dot(np.dot(np.array(diff),c.getI()), np.array(diff).T)[0][0]
    pval = 1 - stats.chi2.cdf(chisqr,dof)
    sigma = np.sqrt(2)*special.erfcinv(pval)
  else:
    print("ERROR: vectors or covariance matrices of different order")
    return 0
  if pvalue and chi2:
    return np.float(sigma[0]), np.float(pval[0]), np.float(chisqr[0])
  if chi2:
    return np.float(sigma[0]), np.float(chisqr[0])
  if pvalue:
    return np.float(sigma[0]), np.float(pval[0])
  return np.float(sigma[0])


def check_DLL_agreement(logls, pars, sigmas=True, pvalue=False, chi2=False,
                        mdof=0, ndof=False):
  cdof = len(set([ len([p.free for p in set.values()]) for set in pars])) <= 1
  dof = [ sum([p.free for p in set.values()]) for set in pars ][0] - mdof
  print(dof)
  if cdof: # fcn_data(parameters, data)
    DLLs = np.zeros(( len(pars) , len(pars) ))
    for i in range(DLLs.shape[0]):
        row = logls[i,i]
        for j in range(DLLs.shape[0]):
            DLLs[i,j] = logls[i,j] - row
    chisqr = DLLs*DLLs.T/(DLLs+DLLs.T+1e-14)
    pval = 1 - stats.chi2.cdf(chisqr[0,1], dof)# warning
    sigma = np.sqrt(2)*special.erfcinv(pval)
  else:
    print('shit')
    return 0
  if pvalue and chi2:
    return np.float(sigma), np.float(pval), np.float(chisqr)
  if chi2:
    return np.float(sigma), np.float(chisqr)
  if pvalue and ndof:
    return np.float(sigma), np.float(pval), np.float(dof)
  if pvalue:
    return np.float(sigma), np.float(pval)
  return np.float(sigma)
