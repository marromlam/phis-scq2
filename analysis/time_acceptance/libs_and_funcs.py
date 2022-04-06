# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                    DECAY TIME ACCEPTANCE                                     #
#                                                                              #
#     Author: Marcos Romero                                                    #
#    Created: 04 - dec - 2019                                                  #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################



__all__ = []
__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import platform
import json
import pandas
import importlib

from scipy.interpolate import interp1d

from ipanema import Parameters, fit_report, optimize
from ipanema import histogram
from ipanema import Sample
from ipanema import getDataFile
from ipanema import plotting

# Project paths
path  = os.environ['PHIS_SCQ']
samples_path = os.environ['PHIS_SCQ'] + 'samples/'
dta_path = path + 'decay_time_acceptance/'
out_dta = path + 'output/decay_time_acceptance/'
ppath = out_dta + 'plots/'

# Get config_file
config_file = json.load(open(dta_path+'config/'+args['config']+'.json'))

# Fitting options
FIT_EACH = config_file['fit_each_sample']
FIT_RATIO = config_file['fit_ratio']
FIT_FULL = config_file['fit_full_spline']

# Select triggers to fit
triggers_to_fit = config_file['triggers_to_fit']
if triggers_to_fit == 'both':
  triggers = {'biased':1,'unbiased':0}
elif triggers_to_fit == 'biased':
  triggers = {'biased':1}
elif triggers_to_fit == 'unbiased':
  triggers = {'unbiased':0}

# Select samples
samples = {}
samples[args['MC_Bs2JpsiPhi_sample']] = samples_path+args['MC_Bs2JpsiPhi_sample']+'.json'
samples[args['MC_Bd2JpsiKstar_sample']] = samples_path+args['MC_Bd2JpsiKstar_sample']+'.json'
samples[args['Bd2JpsiKstar_sample']] = samples_path+args['Bd2JpsiKstar_sample']+'.json'

# Platform swicher
# PLATFORM = args['platform']
# if PLATFORM == 'cuda':
#   kernel_path = os.path.join(os.environ['PHIS_SCQ'],'cuda')
#   import pycuda.driver as cuda
#   import pycuda.cumath
#   import pycuda.autoinit
#   import pycuda.gpuarray as cu_array
# elif PLATFORM == 'opencl':
#   kernel_path = os.path.join(os.environ['PHIS_SCQ'],'opencl')
#   import pyopencl as cl
#   import pyopencl.array as cl_array
#   context = cl.create_some_context()
#   queue   = cl.CommandQueue(context)






# get Badjanak model and compile kernels
kernel_path = os.path.join(os.environ['PHIS_SCQ'],'cuda')
sys.path.append(kernel_path)
from Badjanak import *
kernel_config = config_file['kernel_config']
BsJpsiKK = Badjanak(kernel_path,**kernel_config);




getSingleTimeAcc = BsJpsiKK.getSingleTimeAcc
getRatioTimeAcc  = BsJpsiKK.getRatioTimeAcc
getFullTimeAcc   = BsJpsiKK.getFullTimeAcc

from ipanema import histogram, plotting

# Plotting style
sys.path.append(os.environ['PHIS_SCQ']+'tools')
importlib.import_module('phis-scq-style')




################################################################################
#%% Likelihood functions to minimize ###########################################



def lkhd_single_spline(parameters, data, weight = None, prob = None):
  pars_dict = list(parameters.valuesdict().values())
  #print(pars_dict)
  if not prob: # for ploting, mainly
    data = cu_array.to_gpu(data)
    prob = cu_array.to_gpu(np.zeros_like(data.get()))
    getSingleTimeAcc(data, prob, *pars_dict)
    return prob.get()
  else:
    getSingleTimeAcc(data, prob, *pars_dict)
    if weight is not None:
      result = (pycuda.cumath.log(prob)*weight).get()
    else:
      result = (pycuda.cumath.log(prob)).get()
    return -2*result



def lkhd_ratio_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,2):
      samples.append(cu_array.to_gpu(data))
      prob.append( cu_array.to_gpu(np.zeros_like(data)) )
    getRatioTimeAcc(samples, prob, pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    getRatioTimeAcc(data, prob, pars_dict)
    if weight is not None:
      result  = np.concatenate(((pycuda.cumath.log(prob[0])*weight[0]).get(),
                                (pycuda.cumath.log(prob[1])*weight[1]).get()
                              ))
    else:
      result  = np.concatenate((pycuda.cumath.log(prob[0]).get(),
                                pycuda.cumath.log(prob[1]).get()
                              ))
    return -2*result



def lkhd_full_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,3):
      samples.append(cu_array.to_gpu(data))
      prob.append( cu_array.to_gpu(np.zeros_like(data)) )
    getFullTimeAcc(samples, prob, pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    getFullTimeAcc(data, prob, pars_dict)
    if weight is not None:
      result  = np.concatenate(((pycuda.cumath.log(prob[0])*weight[0]).get(),
                                (pycuda.cumath.log(prob[1])*weight[1]).get(),
                                (pycuda.cumath.log(prob[2])*weight[2]).get()
                              ))
    else:
      result  = np.concatenate((pycuda.cumath.log(prob[0]).get(),
                                pycuda.cumath.log(prob[1]).get(),
                                pycuda.cumath.log(prob[2]).get()
                              ))
    return -2*result



################################################################################



################################################################################
#%% Plotting functions #########################################################



def plot_fcn_spline(parameters, data, weight, log=False, name='test.pdf'):
  ref = histogram.hist(data, weights=weight, bins = 100)
  fig, axplot, axpull = plotting.axes_plotpull();
  x = np.linspace(0.3,15,200)
  y = lkhd_single_spline(parameters, x )
  y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
  axplot.plot(x,y)
  axpull.fill_between(ref.bins,
                      histogram.pull_pdf(x,y,ref.bins,ref.counts,ref.errl,ref.errh),
                      0, facecolor="C0")
  axplot.errorbar(ref.bins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
  if log:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  fig.savefig(name)
  plt.close()




def plot_spline_single(params, time, weights, conf_level=1, name='test.pdf', bins=30, label=None):
  """
  Hi Marcos,

  Do you mean the points of the data?
  The binning is obtained using 30 bins that are exponentially distributed
  with a decay constant of 0.4 (this means that an exponential distribution
  with gamma=0.4 would result in equally populated bins).
  For every bin, the integral of an exponential with the respective decay
  width (0.66137,0.65833,..) is calculated and its inverse is used to scale
  the number of entries in this bin.

  Cheers,
  Simon
  """
  list_coeffs = [key for key in params if key[0]=='b']
  coeffs = []
  for par in list_coeffs:
    if params[par].stddev:
      coeffs.append(unc.ufloat(params[par].value,params[par].stddev))
    else:
      coeffs.append(unc.ufloat(params[par].value,0))
  gamma = params[[key for key in params.keys() if key[:5]=='gamma'][0]].value

  # Cook where should I place the bins
  tLL = 0.3; tUL = 15
  def distfunction(tLL, tUL, gamma, ti, nob):
    return np.log(-((np.exp(gamma*ti + gamma*tLL + gamma*tUL)*nob)/
    (-np.exp(gamma*ti + gamma*tLL) + np.exp(gamma*ti + gamma*tUL) -
      np.exp(gamma*tLL + gamma*tUL)*nob)))/gamma
  list_bins = [tLL]; ipdf = []; widths = []
  dummy = 0.4; # this is a general gamma to distribute the bins
  for k in range(0,bins):
    ti = list_bins[k]
    list_bins.append( distfunction(tLL, tUL, dummy, ti, bins)   )
    tf = list_bins[k+1]
    ipdf.append( 1.0/((-np.exp(-(tf*gamma)) + np.exp(-(ti*gamma)))/1.0) )
    widths.append(tf-ti)
  bins = np.array(list_bins); int_pdf = np.array(ipdf)

  # Manipulate the decay-time dependence of the efficiency
  x = np.linspace(0.3,15,200)
  y = unc.wrap_unc(getSpline, x, *coeffs)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)

  # Manipulate data
  ref = histogram.hist(time, bins=bins, weights=weights)
  ref.counts *= int_pdf; ref.errl *= int_pdf; ref.errh *= int_pdf
  counts_spline = interp1d(ref.bins, ref.counts, kind='cubic')
  int_5 = counts_spline(5)
  ref.counts /= int_5; ref.errl /= int_5; ref.errh /= int_5

  # Actual ploting
  fig, axplot = plotting.axes_plot()
  axplot.set_ylim(0.4, 1.5)
  axplot.plot(x,y_nom/y_spl)
  axplot.errorbar(ref.cmbins,ref.counts,
                  yerr=[ref.errl,ref.errh],
                  xerr=[-ref.edges[:-1]+ref.cmbins,-ref.cmbins+ref.edges[1:]],
                  fmt='.', color='k')
  y_upp, y_low = unc.get_confidence_bands(x,y, sigma=conf_level)
  axplot.fill_between(x, y_upp/y_spl, y_low/y_spl, alpha=0.2, edgecolor="none",
                      label='$'+str(k)+'\sigma$ confidence band')
  axplot.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'%s [a.u.]' % label)
  fig.savefig(name)
  plt.close()


################################################################################
