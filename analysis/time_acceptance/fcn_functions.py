# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__ = ['mromerol@cern.ch']

__all__ = ['splinexerf','saxsbxscxerf']


from ipanema import ristra
import numpy as np
import badjanak

from utils.plot import mode_tex
from utils.strings import cuts_and



def splinexerf(parameters, data, weight = None, prob = None):
  pars_dict = list(parameters.valuesdict().values())
  if not prob: # for ploting, mainly
    data = ristra.allocate(data)
    prob = ristra.allocate(np.zeros_like(data.get()))
    badjanak.splinexerf(data, prob, *pars_dict)
    return prob.get()
  else:
    badjanak.splinexerf(data, prob, *pars_dict)
    if weight is not None:
      result = (ristra.log(prob)*weight).get()
    else:
      result = (ristra.log(prob)).get()
    return -2*result


def saxsbxscxerf(params, data, weight=False, prob=None):
  pars = params.valuesdict()
  if not prob:
    samples = list( map(ristra.allocate,data) )
    prob = list( map(ristra.zeros_like,samples) )
    badjanak.saxsbxscxerf(*samples, *prob, **pars)
    return [ p.get() for p in prob ]
  else:
    badjanak.saxsbxscxerf(*data, *prob, **pars)
    if weight:
      result  = np.concatenate(( (ristra.log(prob[0])*weight[0]).get(),
                                 (ristra.log(prob[1])*weight[1]).get(),
                                 (ristra.log(prob[2])*weight[2]).get() ))
    else:
      result  = np.concatenate(( ristra.log(prob[0]).get(),
                                 ristra.log(prob[1]).get(),
                                 ristra.log(prob[2]).get() ))

    return -2*result
