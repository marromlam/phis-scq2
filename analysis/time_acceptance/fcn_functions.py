DESCRIPTION = """
    This file contains 3 fcn functions to be minimized under ipanema3 framework
    those functions are, actually functions of badjanak kernels.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['splinexerf', 'saxsbxscxerf', 'splinexerfconstr']


from ipanema import ristra, Parameters
import numpy as np
import badjanak

def splinexerf(params, data, weight = None, prob = None, flatend=False):
  # do lists of coeffs
  c = [k.value for k in params.fetch('(a|c|b)([1-9])?([0-9])(u|b)').values()]

  mu = params[params.find('(mu).*')[0]].value
  sigma = params[params.find('(sigma).*')[0]].value
  gamma = params[params.find('(gamma).*')[0]].value

  if prob is None: # for ploting, mainly
    data = ristra.allocate(data)
    prob = ristra.allocate(np.zeros_like(data.get()))
    badjanak.splinexerf(data, prob, coeffs=c, mu=mu, sigma=sigma, gamma=gamma, flatend=flatend)
    return prob.get()
  else:
    badjanak.splinexerf(data, prob, coeffs=c, mu=mu, sigma=sigma, gamma=gamma, flatend=flatend)
    if weight is not None:
      result = (ristra.log(prob)*weight).get()
    else:
      result = (ristra.log(prob)).get()
    return -2*result


def saxsbxscxerf(params, data, weight=False, prob=None, flatend=False):
  # do lists of coeffs
  a = [k.value for k in params.fetch('(a|aA|bA)([1-9])?([0-9])(u|b)').values()]
  b = [k.value for k in params.fetch('(b|aB|bB)([1-9])?([0-9])(u|b)').values()]
  c = [k.value for k in params.fetch('(c)(A|B)?([1-9])?([0-9])(u|b)').values()]

  # get mu, sigma and gamma for each data set in the simultaneous fit
  mu_a = params[ params.find('(mu)_(a|Aa|Ab)')[0] ].value
  mu_b = params[ params.find('(mu)_(b|Ba|Bb)')[0] ].value
  mu_c = params[ params.find('(mu)_(A|B)?(c)')[0] ].value
  sigma_a = params[ params.find('(sigma)_(a|Aa|Ab)')[0] ].value
  sigma_b = params[ params.find('(sigma)_(b|Ba|Bb)')[0] ].value
  sigma_c = params[ params.find('(sigma)_(A|B)?(c)')[0] ].value
  gamma_a = params[ params.find('(gamma)_(a|Aa|Ab)')[0] ].value
  gamma_b = params[ params.find('(gamma)_(b|Ba|Bb)')[0] ].value
  gamma_c = params[ params.find('(gamma)_(A|B)?(c)')[0] ].value

  if not prob: # for ploting, mainly
    data = list( map(ristra.allocate, data) )
    prob = list( map(ristra.zeros_like, data) )
    badjanak.saxsbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b, coeffs_c=c,
                          mu_a=mu_a, mu_b=mu_b, mu_c=mu_c,
                          sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c,
                          gamma_a=gamma_a, gamma_b=gamma_b, gamma_c=gamma_c, flatend=flatend)
    return [ p.get() for p in prob ]
  else:
    badjanak.saxsbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b, coeffs_c=c,
                          mu_a=mu_a, mu_b=mu_b, mu_c=mu_c,
                          sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c,
                          gamma_a=gamma_a, gamma_b=gamma_b, gamma_c=gamma_c, flatend=flatend)
    if weight:
      result  = np.concatenate(( (ristra.log(prob[0])*weight[0]).get(),
                                 (ristra.log(prob[1])*weight[1]).get(),
                                 (ristra.log(prob[2])*weight[2]).get() ))
    else:
      result  = np.concatenate(( ristra.log(prob[0]).get(),
                                 ristra.log(prob[1]).get(),
                                 ristra.log(prob[2]).get() ))
    return -2*result



def saxsbxerf(params, data, weight=False, prob=None, flatend=False):
  # do lists of coeffs
  a = [k.value for k in params.fetch('(a|aA|bA)([1-9])?([0-9])(u|b)').values()]
  b = [k.value for k in params.fetch('(b|aB|bB)([1-9])?([0-9])(u|b)').values()]

  # get mu, sigma and gamma for each data set in the simultaneous fit
  mu_a = params[ params.find('(mu)_(a|Aa|Ab)')[0] ].value
  mu_b = params[ params.find('(mu)_(b|Ba|Bb)')[0] ].value
  sigma_a = params[ params.find('(sigma)_(a|Aa|Ab)')[0] ].value
  sigma_b = params[ params.find('(sigma)_(b|Ba|Bb)')[0] ].value
  gamma_a = params[ params.find('(gamma)_(a|Aa|Ab)')[0] ].value
  gamma_b = params[ params.find('(gamma)_(b|Ba|Bb)')[0] ].value

  if not prob: # for ploting, mainly
    data = list( map(ristra.allocate, data) )
    prob = list( map(ristra.zeros_like, data) )
    badjanak.sbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b, coeffs_c=c,
                        mu_a=mu_a, mu_b=mu_b, mu_c=mu_c,
                        sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c,
                        gamma_a=gamma_a, gamma_b=gamma_b, gamma_c=gamma_c, flatend=flatend)
    return [ p.get() for p in prob ]
  else:
    badjanak.sbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b,
                        mu_a=mu_a, mu_b=mu_b,
                        sigma_a=sigma_a, sigma_b=sigma_b,
                        gamma_a=gamma_a, gamma_b=gamma_b, flatend=flatend)
    if weight:
      result  = np.concatenate(( (ristra.log(prob[0])*weight[0]).get(),
                                 (ristra.log(prob[1])*weight[1]).get() ))
    else:
      result  = np.concatenate(( ristra.log(prob[0]).get(),
                                 ristra.log(prob[1]).get() ))
    return -2*result


def splinexerfconstr(pars, cats, weight=False, flatend=False):

  chi2 = []
  g = pars['gamma'].value

  for y, dy in cats.items():
    for t, dt in dy.items():
      m = pars['mu_Ac'].value if 'mu_Ac' in pars else pars['mu_c'].value
      s = pars['sigma_Ac'].value if 'sigma_Ac' in pars else pars['sigma_c'].value

      # get coeffs - currently being fitted
      lpars = pars.find(f'c(A)?(\d{{1}})(\d{{1}})?({t[0]})_({y[2:]})')
      c = pars.valuesarray(lpars)

      # compute gaussian constraint - from previous fits
      lpars = dt.params.find(f'c(B)?(\d{{1}})(\d{{1}})?({t[0]})')[1:]
      c0 = np.matrix(c[1:] - dt.params.valuesarray(lpars))       # constraint mu
      cov = dt.params.cov(lpars)                  # constraint covariance matrix
      cnstr  = np.dot(np.dot(c0, np.linalg.inv(cov)), c0.T) 
      cnstr += len(c0)*np.log(2*np.pi) + np.log(np.linalg.det(cov))
      cnstr  = np.float64(cnstr[0][0])/len(dt.lkhd)       # per event constraint
      
      # call device function and compute
      badjanak.splinexerf(dt.time, dt.lkhd, coeffs=c, mu=m, sigma=s, gamma=g, flatend=flatend)
      
      # append to chi2 for each subsample - weight switcher
      if weight:
        chi2.append( ristra.get( (-2*ristra.log(dt.lkhd)+cnstr) * dt.weight ) )
      else:
        chi2.append( ristra.get( (-2*ristra.log(dt.lkhd)+cnstr)             ) )
      
  return np.concatenate(chi2)
