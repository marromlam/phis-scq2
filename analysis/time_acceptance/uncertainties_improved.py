from uncertainties import *
import numpy as np
from uncertainties import unumpy as unp

__all__ = []

### funcs



def numericJacobian(f, x, vals, f_size = 1):
  J = np.zeros([len(x), f_size, len(vals)])
  for l in range(0,len(vals)):
    if vals[l]!= 0:    h = np.sqrt(np.finfo(float).eps)*vals[l];
    else:           h = 1e-14;
    vals1 = np.copy(vals); vals1[l] += +h
    vals2 = np.copy(vals); vals2[l] += -h;
    f1 = f(x,*vals1).astype(np.float64)
    f2 = f(x,*vals2).astype(np.float64)
    thisJ = ((f(x,*vals1) - f(x,*vals2))/(2*h)).astype(np.float64)
    J[:,0,l] = thisJ # nowadays only scalar f allowed
  return J.T

def propagate_term(der,unc):
  return der**2*unc**2



def wrap_unc(f,x,*args):

  # get parameters and uncertainties
  vals = np.array([args[k].nominal_value  for k in range(0,len(args))])
  uncs = np.array([args[k].std_dev        for k in range(0,len(args))])

  # compute f nominal_value
  f_val = f(x,*vals)
  if hasattr(f(np.array([1]),*vals), "__len__"):
    f_size = len(f(np.array([1]),*vals))
  else:
    f_size = 1

  # get numeric derivatives
  derivatives = numericJacobian(f, x, vals, f_size)
  f_unc = np.zeros([len(x)])

  # compute f std_dev
  for i in range(0,len(uncs)):
    f_unc[:] += propagate_term(derivatives[i],uncs[i])[0]
  f_unc = np.sqrt(f_unc)

  return unp.uarray(f_val,f_unc)


def get_confidence_bands(x,y,sigma=1):
  nom = unp.nominal_values(y)
  std = unp.std_devs(y)
  # uncertainty lines (95% confidence)
  return nom+sigma*std, nom-sigma*std
