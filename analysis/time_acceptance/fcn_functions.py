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

def splinexerf(params, data, weight = None, prob = None):
  # do lists of coeffs
  c = [ k.value for k in params.fetch('(a|c|b)([1-9])?([0-9])(u|b)').values() ]

  mu = params[params.find('(mu).*')[0]].value
  sigma = params[params.find('(sigma).*')[0]].value
  gamma = params[params.find('(gamma).*')[0]].value

  if not prob: # for ploting, mainly
    data = ristra.allocate(data)
    prob = ristra.allocate(np.zeros_like(data.get()))
    badjanak.splinexerf(data, prob, coeffs=c, mu=mu, sigma=sigma, gamma=gamma)
    return prob.get()
  else:
    badjanak.splinexerf(data, prob, coeffs=c, mu=mu, sigma=sigma, gamma=gamma);
    if weight is not None:
      result = (ristra.log(prob)*weight).get()
    else:
      result = (ristra.log(prob)).get()
    return -2*result


def saxsbxscxerf(params, data, weight=False, prob=None):
  # do lists of coeffs
  a = [ k.value for k in params.fetch('(a|aA|bA)([1-9])?([0-9])(u|b)').values() ]
  b = [ k.value for k in params.fetch('(b|aB|bB)([1-9])?([0-9])(u|b)').values() ]
  c = [ k.value for k in params.fetch('(c)(A|B)?([1-9])?([0-9])(u|b)').values() ]

  # get mu, sigma and gamma for each data set in the simultaneous fit
  mu_a = params[params.find('(mu)_(a|Aa|Ab)')[0] ].value
  mu_b = params[params.find('(mu)_(b|Ba|Bb)')[0] ].value
  mu_c = params[params.find('(mu)_(A|B)?(c)')[0] ].value
  sigma_a = params[params.find('(sigma)_(a|Aa|Ab)')[0] ].value
  sigma_b = params[params.find('(sigma)_(b|Ba|Bb)')[0] ].value
  sigma_c = params[params.find('(sigma)_(A|B)?(c)')[0] ].value
  gamma_a = params[params.find('(gamma)_(a|Aa|Ab)')[0] ].value
  gamma_b = params[params.find('(gamma)_(b|Ba|Bb)')[0] ].value
  gamma_c = params[params.find('(gamma)_(A|B)?(c)')[0] ].value

  if not prob: # for ploting, mainly
    data = list( map(ristra.allocate, data) )
    prob = list( map(ristra.zeros_like, data) )
    badjanak.saxsbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b, coeffs_c=c,
                          mu_a=mu_a, mu_b=mu_b, mu_c=mu_c,
                          sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c,
                          gamma_a=gamma_a, gamma_b=gamma_b, gamma_c=gamma_c)
    return [ p.get() for p in prob ]
  else:
    badjanak.saxsbxscxerf(*data, *prob, coeffs_a=a, coeffs_b=b, coeffs_c=c,
                          mu_a=mu_a, mu_b=mu_b, mu_c=mu_c,
                          sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c,
                          gamma_a=gamma_a, gamma_b=gamma_b, gamma_c=gamma_c)
    if weight:
      result  = np.concatenate(( (ristra.log(prob[0])*weight[0]).get(),
                                 (ristra.log(prob[1])*weight[1]).get(),
                                 (ristra.log(prob[2])*weight[2]).get() ))
    else:
      result  = np.concatenate(( ristra.log(prob[0]).get(),
                                 ristra.log(prob[1]).get(),
                                 ristra.log(prob[2]).get() ))

    return -2*result








def fcn_test(pars, cats, weight=False):

  lkhd = []; weights = []; constraints = []
  for t, cat in cats.items():


    # get mu, sigma (locked) + gamma (now free)
    mu = pars['mu_Ac'].value
    sigma = pars['sigma_Ac'].value,
    gamma = pars['gamma_Ac'].value # <-- what is fitted now

    # get coeffs - currently being fitted
    c = [ k.value for k in pars.fetch(f'cA\d{{1}}({t[0]})').values() ]

    # get coeffs and cov - from previous fits
    c0 = [ k.value for k in cats[t].params.fetch(f'cB([1-9])({t[0]})').values() ] #mus del contraint
    s0 = [ k.stdev for k in cats[t].params.fetch(f'cB([1-9])({t[0]})').values() ] #mus del contraint
    c0 = np.array(c[1:]) - np.array(c0)
    s0 = np.array(s0)
    cov = Parameters.build(cats[t].params, cats[t].params.find(f'cB([1-9])({t[0]})') ).cov() #mus del contraint
    #print("c:", c)
    #print("c0:", c0)
    #print(cov)

    # append to lists the likelihoods, weights and the gaussian constraint
    badjanak.splinexerf(cat.time, cat.lkhd, coeffs=c, mu=mu, sigma=sigma, gamma=gamma)
    lkhd.append( -2*ristra.log(cat.lkhd) )
    weights.append(cat.weight)


    #print(c0)
    #print(c0[:,np.newaxis])
    tmp2 = np.sum(np.power(c0*s0, 2) + np.log(2*np.pi*np.power(s0, 2)))
    c0 = np.matrix(c0)
    tmp = np.dot(np.dot(c0, np.linalg.inv(cov)), c0.T) + len(s0)*np.log(2*np.pi) + np.log(np.linalg.det(cov))

    constraints.append(
      np.float64(tmp2) +
      np.float64( tmp[0][0])/len(cat.lkhd)#[0][0]
    )
    #print(constraints)
  #print()

  if weight:
    result = np.concatenate((( (lkhd[0]+constraints[0]) *weights[0]).get(),
                             ( (lkhd[1]+constraints[0]) *weights[1]).get()))
  else:
    result = np.concatenate((( (lkhd[0]+constraints[0]) ).get(),
                             ( (lkhd[1]+constraints[0]) ).get()))

  return result #+ sum(constraints)



"""



def fcn_test(pars, cats, weight=False):
  gamma = pars['gamma'].value

  lkhd = []; weights = []
  for t, cat in cats.items():
    lc = [ k.value for k in cat.params.fetch('(a|c|b)(B)?\d{1}').values() ]
    badjanak.splinexerf(cat.time, cat.lkhd, lc, mu=cat.params['mu_Ac'].value,
                        sigma=cat.params['sigma_Ac'].value, gamma=gamma)
    lkhd.append(cat.lkhd)
    weights.append(cat.weight)
  if weight:
    result  = np.concatenate(( (ristra.log(lkhd[0])*weights[0]).get(),
                               (ristra.log(lkhd[1])*weights[1]).get() ))
  else:
    result  = np.concatenate(( (ristra.log(lkhd[0])).get(),
                               (ristra.log(lkhd[1])).get() ))

  return -2*result

"""
