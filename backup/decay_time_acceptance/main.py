# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                    DECAY TIME ACCEPTANCE                                     #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
#                                                                              #
################################################################################

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']

################################################################################

# def dig_tree(file, ident = ""):
#   for k,v in file.items():
#     ident +='  '
#     if str(v)[1:8] == 'TBranch':
#       print(ident+"* {0}".format( k.decode() ))
#     else:
#       print(ident+'* '+k.decode())
#       dig_tree(v,ident+'  ')
#     ident = ident[:-2]



################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import platform
import hjson
import pandas
import importlib
from scipy.interpolate import interp1d

import decay_time_acceptance.uncertainties_improved as unc
from uncertainties import unumpy as unp

from ipanema import ristra
from ipanema import Parameters, fit_report, optimize
from ipanema import histogram
from ipanema import Sample
from ipanema import getDataFile
from ipanema import plotting

def argument_parser(flag=None):
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--MC_Bs2JpsiPhi-sample',
                      default = 'MC_Bs2JpsiPhi_DG0_2016__baseline.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--MC_Bd2JpsiKstar-sample',
                      default = 'MC_Bd2JpsiKstar_2016__baseline.json',
                      help='Bd2JpsiKstar MC sample')
  parser.add_argument('--Bd2JpsiKstar-sample',
                      default = 'Bd2JpsiKstar_2016__baseline.json',
                      help='Bd2JpsiKstar data sample')
  # Configuration file
  parser.add_argument('--config',
                      default = 'baseline',
                      help='Configuration')
  # Platform
  parser.add_argument('--backend',
                      default = 'cuda',
                      help='Choose between cuda or opencl')
  # Report
  parser.add_argument('--report',
                      default = True,
                      help='Save a fit report with the results')
  if flag:
    return vars(parser.parse_args(""))
  else:
    return vars(parser.parse_args())



__builtins__.args = argument_parser(1)


#from decay_time_acceptance.libs_and_funcs import *

import bsjpsikk

################################################################################






################################################################################
#%% Configuration ##############################################################

config_file = hjson.load(open(dta_path+'config/'+args['config']+'.json'))



# phis-scq paths

path  = os.environ['PHIS_SCQ']

samples_path = os.environ['PHIS_SCQ'] + 'samples/'
dta_path = path + 'decay_time_acceptance/'
out_dta = path + 'output/decay_time_acceptance/'
ppath = out_dta + 'plots/'



#%% initialize backend

from ipanema import initialize
initialize(args['backend'],1)



#%% Triggers

# Select triggers to fit
triggers_to_fit = config_file['triggers_to_fit']
if triggers_to_fit == 'both':
  triggers = {'biased':1,'unbiased':0}
elif triggers_to_fit == 'biased':
  triggers = {'biased':1}
elif triggers_to_fit == 'unbiased':
  triggers = {'unbiased':0}



#%% Get bsjpsikk model and configure it

import bsjpsikk
bsjpsikk.use_time_acc = 0,
bsjpsikk.use_time_offset = 0
bsjpsikk.use_time_res = 0
bsjpsikk.use_perftag = 0
bsjpsikk.use_truetag = 1
bsjpsikk.get_kernels()



################################################################################






################################################################################
#%% Likelihood functions to minimize ###########################################

def lkhd_single_spline(parameters, data, weight = None, prob = None):
  pars_dict = list(parameters.valuesdict().values())
  #print(pars_dict)
  if not prob: # for ploting, mainly
    data = ristra.allocate(data)
    prob = ristra.allocate(np.zeros_like(data.get()))
    bsjpsikk.single_spline_time_acceptance(data, prob, *pars_dict)
    return prob.get()
  else:
    bsjpsikk.single_spline_time_acceptance(data, prob, *pars_dict)
    if weight is not None:
      result = (ristra.log(prob)*weight).get()
    else:
      result = (ristra.log(prob)).get()
    return -2*result

def lkhd_ratio_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,2):
      samples.append(ristra.allocate(data))
      prob.append( ristra.allocate(np.zeros_like(data)) )
    getRatioTimeAcc(samples, prob, pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    getRatioTimeAcc(data, prob, pars_dict)
    if weight is not None:
      result  = np.concatenate(((ristra.log(prob[0])*weight[0]).get(),
                                (ristra.log(prob[1])*weight[1]).get()
                              ))
    else:
      result  = np.concatenate((ristra.log(prob[0]).get(),
                                ristra.log(prob[1]).get()
                              ))
    return -2*result

def lkhd_full_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,3):
      samples.append(ristra.allocate(data))
      prob.append( ristra.allocate(np.zeros_like(data)) )
    getFullTimeAcc(samples, prob, pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    getFullTimeAcc(data, prob, pars_dict)
    if weight is not None:
      result  = np.concatenate(((ristra.log(prob[0])*weight[0]).get(),
                                (ristra.log(prob[1])*weight[1]).get(),
                                (ristra.log(prob[2])*weight[2]).get()
                              ))
    else:
      result  = np.concatenate((ristra.log(prob[0]).get(),
                                ristra.log(prob[1]).get(),
                                ristra.log(prob[2]).get()
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
  list_coeffs = [key for key in params if key[1]=='_']
  coeffs = []
  for par in list_coeffs:
    if params[par].stdev:
      coeffs.append(unc.ufloat(params[par].value,params[par].stdev))
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
  y = unc.wrap_unc(bsjpsikk.acceptance_spline, x, *coeffs)
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






################################################################################
#%% Get data into categories ###################################################

# Select samples
samples = {}
samples['BsMC'] = samples_path+args['MC_Bs2JpsiPhi_sample']
samples['BdMC'] = samples_path+args['MC_Bd2JpsiKstar_sample']
samples['BdDT'] = samples_path+args['Bd2JpsiKstar_sample']

# Build categories
cats = {}
for name, sample in zip(samples.keys(),samples.values()):
  if name == 'BsMC':
    label = (r'\text{MC}',r'B_s^0')
  elif name == 'BdMC':
    label = (r'\text{MC}',r'B^0')
  elif name == 'BdDT':
    label = (r'\text{data}',r'B^0')
  for trig, bool in zip(triggers.keys(),triggers.values()):
    _name = name+'_'+trig
    cats[_name] = Sample.from_file(sample, cuts='hlt1b=={}'.format(bool))
    cats[_name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+trig
    cats[_name].allocate(time='time',weight='sWeight*kinWeight',lkhd='0*time')
    cats[_name].label = label
    param_path  = dta_path+'input/params-'
    param_path += cats[_name].name.split('__')[0][:-5]+'-baseline.json'
    cats[_name].assoc_params(Parameters.load(param_path))

################################################################################






################################################################################
#%% Fit all categories #########################################################

fits = {}; FIT_EACH = 1
# Fit each sample
if FIT_EACH:
  for name, cat in zip(cats.keys(),cats.values()):
    print('Fitting %s category...' % name)
    if cat.params:
      fits[cat.name] = optimize(lkhd_single_spline, method="hesse",
                            params=cat.params,
                            kws={'data': cat.time,
                                 'prob': cat.lkhd,
                                 'weight': cat.weight},
                            verbose=True);
      fits[cat.name].params.dump(out_dta+cat.name)
    fits[cat.name].label = r'$\varepsilon_{%s}^{%s}$' % cat.label
    print('\n')
  for name, cat in zip(cats.keys(),cats.values()):
    print('Plotting %s category...' % name)
    filename = ppath+cat.name+'_fit.pdf'
    plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                    name = filename )
    filename = ppath+cat.name+'_fit_log.pdf'
    plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                    name = filename, log= True )
    filename = ppath+cat.name+'_spline.pdf'
    plot_spline_single(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                       name = filename, label=fits[cat.name].label,
                       conf_level=1, bins=30 )



# Fit the ratio BsMC/BdMC
if FIT_RATIO:
  for trig in ['_unbiased', '_biased']:
    name = cats['MC_Bd2JpsiKstar'+trig].name.replace('Bd2JpsiKstar','ratioBsBd')
    print('Fitting %s category...' % name)
    fits[name] = optimize(lkhd_ratio_spline, method="minuit-hesse",
                    params=Parameters.load(dta_path+'input/params-ratio-baseline.json'),
                    kws={'data':  [cats['MC_Bs2JpsiPhi'+trig].time_d,
                                   cats['MC_Bd2JpsiKstar'+trig].time_d],
                        'prob':   [cats['MC_Bs2JpsiPhi'+trig].lkhd_d,
                                   cats['MC_Bd2JpsiKstar'+trig].lkhd_d],
                        'weight': [cats['MC_Bs2JpsiPhi'+trig].weight_d,
                                   cats['MC_Bd2JpsiKstar'+trig].weight_d]},
                    verbose=True);
    fits[name].params.dump(out_dta+name)
    print('\n')



# Full fit to get decay-time acceptance
if FIT_FULL:
  for trig in ['_unbiased', '_biased']:
    name = cats['Bd2JpsiKstar'+trig].name.replace('Bd2JpsiKstar','Bs2JpsiPhi')
    print('Fitting %s category...' % name)
    fits[name] = optimize(lkhd_full_spline, method="hesse",
                    params=Parameters.load(dta_path+'input/params-full-baseline.json'),
                    kws={'data':  [cats['MC_Bs2JpsiPhi'+trig].time_d,
                                   cats['MC_Bd2JpsiKstar'+trig].time_d,
                                   cats['Bd2JpsiKstar'+trig].time_d],
                        'prob':   [cats['MC_Bs2JpsiPhi'+trig].lkhd_d,
                                   cats['MC_Bd2JpsiKstar'+trig].lkhd_d,
                                   cats['Bd2JpsiKstar'+trig].lkhd_d],
                        'weight': [cats['MC_Bs2JpsiPhi'+trig].weight_d,
                                   cats['MC_Bd2JpsiKstar'+trig].weight_d,
                                   cats['Bd2JpsiKstar'+trig].weight_d]},
                    verbose=True);
    fits[name].params.dump(out_dta+name)
    print('\n')
print('Fitting complete.')



################################################################################
