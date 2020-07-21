# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                    DECAY TIME ACCEPTANCE                                     #
#                                                                              #
################################################################################

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']


################################################################################


################################################################################
# %% Modules ###################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import pandas


from ipanema import initialize
initialize('cuda',1)
from ipanema import Sample, Parameters, Parameter
from ipanema import ristra, optimize

# get bsjpsikk and compile it with corresponding flags
import bsjpsikk
bsjpsikk.config['debug'] = 5
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
# to get Simon
bsjpsikk.config['use_perftag'] = 1
bsjpsikk.config['use_truetag'] = 0
# to get Peilian
#bsjpsikk.config['use_perftag'] = 0
#bsjpsikk.config['use_truetag'] = 1
bsjpsikk.get_kernels()

# input parameters
path = '/scratch17/marcos.romero/phis_samples/v0r2'
year = 2016
mode = 'MC_Bs2JpsiPhi'

################################################################################



################################################################################
#%% Load samples ###############################################################

# List of varaibles torestat allocate in device array
arr_data  = []
arr_data += ['cosK','cosL','hphi','time']              # angular variables
arr_data += ['X_M','sigmat']                           # mass and sigmat
arr_data += ['B_ID','B_ID', '0*B_ID', '0*B_ID']        # tagging

data = {}
for y in [2016]:
  data[f'{year}'] = {}
  data[f'{year}'] = Sample.from_root(f'/scratch18/EvtGenBsJpsiKK/bsmumukk_f0_phi_500k.root',treename='bsmumukk', cuts='MKK>=0.990 & MKK<=1.05 & time<15')
  data[f'{year}'].allocate(data=['theta_K', 'theta_mu', 'phi', 'time', '1000*MKK', '0*time', '-q', '-q', '0*time', '0*time'],lkhd='0*time')

#data = {}
#for y in [2016]:
  #data[f'{year}'] = {}
  #data[f'{year}'] = Sample.from_root(f'{path}/{year}/{mode}/test.root', cuts='time>=0.3 & time<15')
  #data[f'{year}'].allocate(data=arr_data,weight='sWeight',lkhd='0*time')

print(data['2016'].data[0])
# %% Prepare set of parameters -------------------------------------------------

# Some options
SWAVE = 1; DGZERO = 0; pars = Parameters()

# List of parameters
list_of_parameters = [#
Parameter(name='fSlon1',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon2',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon3',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon4',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon5',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon6',          value=0.,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
#
Parameter(name="fPlon",           value=0.5241,               min=0.5,    max=0.6,    free=True),
Parameter(name="fPper",           value=0.25,                 min=0.1,    max=0.3,    free=True),
#
Parameter(name="pSlon",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
Parameter(name="pPlon",           value=-0.03,                min=-0.5,   max=0.5,    free=True),
Parameter(name="pPpar",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
Parameter(name="pPper",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
#
Parameter(name='dSlon1',          value=+np.pi/2*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon2',          value=+np.pi/2*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon3',          value=+np.pi/2*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon4',          value=-np.pi/4*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon5',          value=-np.pi/4*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon6',          value=-np.pi/4*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
#
Parameter(name="dPlon",           value=0.00,                 min=-3.0,   max=3.0,    free=False),
Parameter(name="dPpar",           value=3.26,                 min= 0.0,   max=6.5,    free=True),
Parameter(name="dPper",           value=3.08,                 min= 0.8,   max=6.2,    free=True),
#
Parameter(name="lSlon",           value=1.,                   min=0.7,    max=1.3,    free=False),
Parameter(name="lPlon",           value=1.,                   min=0.7,    max=1.3,    free=True),
Parameter(name="lPpar",           value=1.,                   min=0.7,    max=1.3,    free=False),
Parameter(name="lPper",           value=1.,                   min=0.7,    max=1.3,    free=False),
#
#Parameter(name="Gd",              value= 0.65789,             min= 0.0,   max= 1.0,   free=False),
Parameter(name="Gs",              value= 0.6603,             min= 0.0,   max= 1.0,   free=True),
Parameter(name="DGs",             value= (1-DGZERO)*0.08,   min= 0.0,   max= 0.2,   free=(1-DGZERO)),
#Parameter(name="DGsd",            value= 0.03,                min=-0.1,   max= 0.1,   free=True),
Parameter(name="DM",              value=17.77,           min=16.0,   max=19.0,   free=True),
# CSP parameters
Parameter(name='CSP1',            value=0.867156706942*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP2',            value=0.928090569491*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP3',            value=0.905475343928*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP4',            value=0.948485509571*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP5',            value=0.973371863935*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP6',            value=0.983967674912*SWAVE,         min=0.0,    max=1.0,    free=False),
# time range
Parameter(name='tLL',           value=0.3,               min=0.0,    max=15.0,    free=False),
Parameter(name='tUL',           value=15.,                    min=0.0,    max=15.0,    free=False)
]

# update input parameters file
pars.add(*list_of_parameters); #pars.dump(taf_path+'/input/'+FITTING_SAMPLE)
print(pars)



#%% Build FCN



#%% Define fcn
def fcn_MC(parameters, data, weight = False):
  pars_dict = parameters.valuesdict()
  for y, dy in data.items():
    bsjpsikk.diff_cross_rate_full(dy.data, dy.lkhd, **pars_dict)
  prob = ristra.concatenate([dy.lkhd for dy in data.values()])
  print(prob-dy.lkhd)
  if weight:
    weight = ristra.concatenate([dy.weight for dy in data.values()])
    result = np.nan_to_num( (ristra.log(prob)*weight).get() )
    weight = weight.get()
  else:
    result = np.nan_to_num( (ristra.log(prob)       ).get() )
    result =  (ristra.log(prob)       ).get()
    #print(-2*result)
    weight = np.ones_like(result)
    print(result.sum(),weight.sum())
    print(-2*result.sum() + 2*weight.sum())
  return -2*result.sum() + 2*weight.sum()



def fcn_MC2(parameters, data, weight = False):
  pars_dict = parameters.valuesdict()
  bsjpsikk.diff_cross_rate_full(data.data, data.lkhd, **pars_dict)
  if weight:
    weight = data.weight
    result = (ristra.log(data.lkhd)*weight).get()
    weight = weight.get()
  else:
    result = (ristra.log(data.lkhd)).get()
    weight = np.ones_like(result)
    #print(result.sum(),weight.sum())
    #print(-2*result.sum() + 2*weight.sum())
  return -2*result + 2*weight


def fcn_MC3(parameters, data, weight = False):
  p = parameters.valuesdict()
  if not 'CSP' in p:
    p['CSP'] = []; p['fSlon'] = []; p['dSlon'] = [];
    for i in range(6):
      p['CSP'].append(p[f'CSP{i+1}']); p.pop(f'CSP{i+1}', None)
      p['fSlon'].append(p[f'fSlon{i+1}']); p.pop(f'fSlon{i+1}', None)
      p['dSlon'].append(p[f'dSlon{i+1}']); p.pop(f'dSlon{i+1}', None)
  # for k,v in p.items():
  #   print(k,v)
  bsjpsikk.diff_cross_rate_mc(data.data, data.lkhd, **p)
  if weight:
    weight = data.weight
    result = (ristra.log(data.lkhd)*weight).get()
    weight = weight.get()
  else:
    result = (ristra.log(data.lkhd)).get()
    weight = np.ones_like(result)
    #print(result.sum(),weight.sum())
    #print(-2*result.sum() + 2*weight.sum())
  return -2*result + 2*weight









# Test fcn
bsjpsikk.config['debug'] = 5
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.get_kernels()

fcn_MC3(pars, data=data['2016'], weight=False)
#fcn_MC2(pars, data=data['2016'], weight=False)


exit()

# Fit!
shit = optimize(fcn_MC3, method="minuit", params=pars, fcn_kwgs={"data": data['2016']}, verbose=False)
print(shit)








from iminuit import Minuit as minuit
from ipanema import optimizers

opt = optimizers.Optimizer(fcn_MC3, method="minuit", params=pars, fcn_kwgs={"data": data['2016']})
opt.prepare_fit()
opt._configure_minuit_(pars)


opt.optimize(method='minuit')
print(opt.result)












# try with naive minuit

# build a wrapper
def parseminuit(*fvars):
  for name, val in zip(opt.result.param_vary, fvars):
    pars[name].value = val
  return fcn_MC3(pars, data=data['2016'], weight=False).sum()

# give it a call!
crap = minuit(parseminuit, forced_parameters=opt.result.param_vary, **opt._configure_minuit_(pars), pedantic=False)
crap.migrad()
crap.hesse()



















































#
#
#
#
#
#
#
#
#
#
#
#
#
#
# shit = optimize(fcn_year_biased, method="minuit-hesse",
#                 params=cats['test_biased'].params,
#                 kws={ 'data': cats['test_biased'].data_d,
#                       'prob': cats['test_biased'].lkhd_d,
#                     'weight': cats['test_biased'].weight_d  },
#                 verbose=True);
#
#
# shit._minuit.migrad()
#
#
#
#
#
#
#
#
#
#
#
#
#
# #%% Fit all categories ---------------------------------------------------------
# fits = {}
# # fits['BdMC_biased'].chisqr
# # fits['BdMC_biased'].ndata
# # dir(fits['BdMC_biased'])
# # fits['BdMC_biased']._repr_html_()
#
#
# # Fit each sample
# for name, cat in zip(cats.keys(),cats.values()):
#   print('Fitting %s category...' % name)
#   if cat.params:
#     fits[name] = optimize(lkhd_single_spline, method="minuit-hesse",
#                           params=cat.params,
#                           kws={'data':cat.time_d,
#                                'prob': cat.lkhd_d,
#                                'weight': cat.weight_d});
#   print('\n')
#
# # Fit the ratio BsMC/BdMC
# if params['ratio']:
#   for trig in ['_unbiased', '_biased']:
#     fits['ratio'+trig] = optimize(lkhd_ratio_spline, method="minuit-hesse",
#                           params=params['ratio'],
#                           kws={'data':  [cats['BsMC'+trig].time_d,
#                                          cats['BdMC'+trig].time_d],
#                               'prob':   [cats['BsMC'+trig].lkhd_d,
#                                          cats['BdMC'+trig].lkhd_d],
#                               'weight': [cats['BsMC'+trig].weight_d,
#                                          cats['BdMC'+trig].weight_d]});
#
# # Full fit to get decay-time acceptance
# if params['full']:
#  for trig in ['_unbiased', '_biased']:
#    fits['full'+trig] = optimize(lkhd_full_spline, method="minuit-hesse",
#                           params=params['full'],
#                           kws={'data':  [cats['BsMC'+trig].time_d,
#                                          cats['BdMC'+trig].time_d,
#                                          cats['BdDT'+trig].time_d],
#                               'prob':   [cats['BsMC'+trig].lkhd_d,
#                                          cats['BdMC'+trig].lkhd_d,
#                                          cats['BdDT'+trig].lkhd_d],
#                               'weight': [cats['BsMC'+trig].weight_d,
#                                          cats['BdMC'+trig].weight_d,
#                                          cats['BdDT'+trig].weight_d]});
# print('Fitting complete.')
#
#
#
# #%% Plot all categories --------------------------------------------------------
# from ipanema import plotting
# for name, cat in zip(cats.keys(),cats.values()):
#   print('Plotting %s category...' % name)
#   filename = ppath+cat.name[:7]+'_'+name+'.pdf'
#   plot_single_spline(fits[name].params, cat.time_h, cat.weight_h, name = filename )
#   filename = ppath+cat.name[:7]+'_'+name+'_log.pdf'
#   plot_single_spline(fits[name].params, cat.time_h, cat.weight_h, name = filename, log= True )
# print('Plotting complete.')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def pull_hist(ref_counts, counts, counts_l, counts_h):
#   """
#   This function takes an array of ref_counts (reference histogram) and three
#   arrays of the objective histogram: counts, counts_l (counts' lower limit) and
#   counts_h (counts' higher limit). It returns the pull of counts wrt ref_counts.
#   """
#   residuals = counts - ref_counts;
#   pulls = np.where(residuals>0, residuals/counts_l, residuals/counts_h)
#   return pulls
#
#
# def hist(data, weights=None, bins=60, density = False, **kwargs):
#   """
#   This function is a wrap arround np.histogram so it behaves similarly to it.
#   Besides what np.histogram offers, this function computes the center-of-mass
#   bins ('cmbins') and the lower and upper limits for bins and counts. The result
#   is a ipo-object which has several self-explained attributes.
#   """
#
#   # Histogram data
#   counts, edges = np.histogram(data, bins = bins,
#                                weights = weights, density = False,
#                                **kwargs)
#   bincs = (edges[1:]+edges[:-1])*0.5;
#   norm = counts.sum()
#
#   # Compute the mass-center of each bin
#   cmbins = np.copy(bincs)
#   for k in range(0,len(edges)-1):
#     if counts[k] != 0:
#       cmbins[k] = np.median( data[(data>=edges[k]) & (data<=edges[k+1])] )
#
#   # Compute the error-bars
#   if weights is not None:
#     errl, errh = histogram.errors_poisson(counts)
#     errl = errl**2 + histogram.errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
#     errh = errh**2 + histogram.errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
#     errl = np.sqrt(errl); errh = np.sqrt(errh)
#   else:
#     errl, errh = histogram.errors_poisson(counts)
#
#   # Normalize if asked so
#   if density:
#     counts /= norm; errl /= norm;  errh /= norm;
#
#   # Construct the ipo-object
#   result = histogram.ipo(**{**{'counts':counts,
#                      'edges':edges, 'bins':bincs, 'cmbins': cmbins,
#                      'weights': weights, 'norm': norm,
#                      'density': density, 'nob': bins,
#                      'errl': errl, 'errh': errh,
#                     },
#                   **kwargs})
#   return result
#
#
# def compare_hist(data, weights = [None, None], density = False, **kwargs):
#   """
#   This function compares to histograms in data = [ref, obj] with(/out) weights
#   It returns two hisrogram ipo-objects, obj one with pulls, and both of them
#   normalized to one.
#   """
#   ref = hist(data[0], density = False, **kwargs, weights=weights[0])
#   obj = hist(data[1], density = False, **kwargs, weights=weights[1])
#   ref_norm = 1; obj_norm = 1;
#   if norm:
#     ref_norm = 1/ref.counts.sum(); obj_norm = 1/obj.counts.sum();
#   ref.counts = ref.counts*ref_norm; ref.errl *= ref_norm; ref.errh *= ref_norm
#   obj.counts = obj.counts*obj_norm; obj.errl *= obj_norm; obj.errh *= obj_norm
#   obj.add('pulls', pull_hist(ref.counts, obj.counts, obj.errl, obj.errh))
#   return ref, obj
#
#
#
#
# import sys
# sys.path.append(os.environ['PHIS_SCQ']+'tools')
# import importlib
# importlib.import_module('phis-scq-style')
# from scipy.interpolate import interp1d
#
#
#
# def pull_pdf(x_pdf, y_pdf, x_hist, y_hist, y_l, y_h):
#   """
#   This function compares one histogram with a pdf. The pdf is given with two
#   arrays x_pdf and y_pdf, these are interpolated (and extrapolated if needed),
#   contructing a cubic spline. The histogram takes x_hist (bins), y_hist(counts),
#   y_l (counts's lower limit) and y_h (counts' upper limit). The result is a
#   pull array between the histogram and the pdf.
#   (the pdf is expected to be correctly normalized)
#   """
#   s = interp1d(x_pdf, y_pdf, kind='cubic', fill_value='extrapolate')
#   residuals = y_hist - s(x_hist);
#   pulls = np.where(residuals>0, residuals/y_l, residuals/y_h)
#   return pulls
#
#
# def plot_single_spline(parameters, data, weight, log=False, name='test.pdf'):
#   ref = histogram.hist(data, weights=weight, bins = 100)
#   fig, axplot, axpull = plotting.axes_plotpull();
#   x = np.linspace(0.3,15,200)
#   y = lkhd_single_spline(parameters, x )
#   y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
#   axplot.plot(x,y)
#   axpull.fill_between(ref.bins,
#                       pull_pdf(x,y,ref.bins,ref.counts,ref.errl,ref.errh),
#                       0, facecolor="C0")
#   axplot.errorbar(ref.bins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
#   if log:
#     axplot.set_yscale('log')
#   axpull.set_xlabel(r'$t$ [ps]')
#   axplot.set_ylabel(r'Weighted candidates')
#   fig.savefig(name)
#   plt.close()
#
#
# plot_single_spline(fit['BdMC_biased'].params, time_h['BdMC_biased'], weight_h['BdMC_biased'])
# plot_single_spline(fit['BdMC_unbiased'].params, time_h['BdMC_unbiased'], weight_h['BdMC_unbiased'])
#
# for var in vars:
#   index = vars.index(var)
#   orig, ref = compare_hist([original_data[var].values,target_data[var].values], range=ranges[index], bins = 60, norm = True, cm = False, weights=[1/original_data['gb_weights'].values,target_data['sw'].values])
#   weig, ref = compare_hist([original_data[var].values,target_data[var].values], range=ranges[index], bins = 60, norm = True, cm = False, weights=[original_data['kinWeight'].values/original_data['gb_weights'].values,target_data['sw'].values])
#   axplot, axpull = plot_pull()
#   axplot.fill_between(ref['x'],ref['y'],0,facecolor='k',alpha=0.3,step='mid')
#   axplot.fill_between(orig['x'],orig['y'],0,facecolor='C3',alpha=0.3,step='mid')
#   axplot.errorbar(weig['x'],weig['y'], yerr = [weig['y_l'],weig['y_u']],  xerr = [weig['x_l'],weig['x_r']] , fmt='.')
#   axpull.set_xticks(axpull.get_xticks()[2:-1])
#   axplot.set_yticks(axplot.get_yticks()[1:-1])
#   axpull.fill_between(orig['x'],orig['pulls'],0, facecolor="C3")
#   axpull.fill_between(weig['x'],weig['pulls'],0, facecolor="C0")
#   axpull.set_xlabel(labels[index])
#   axplot.set_ylabel(r'Weighted candidates')
#   axplot.legend([r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ data',
#                  r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ MC',
#                  r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ MC + kinWeights'])
#   plt.savefig(var+'.pdf')
#   plt.close()
# histogram.hist(lkhd)
#






















""" FIT RESULTS FOR BIASED TRIGGER
+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BsMC       |   1.602 ± 0.034    |
| b2_BsMC       |    2.127 ± 0.03    |
| b3_BsMC       |   2.277 ± 0.037    |
| b4_BsMC       |   2.362 ± 0.035    |
| b5_BsMC       |   2.453 ± 0.037    |
| b6_BsMC       |   2.507 ± 0.043    |
| b7_BsMC       |   2.503 ± 0.042    |
| b8_BsMC       |   2.485 ± 0.038    |
+---------------+--------------------+

+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BdMC       |    1.49 ± 0.036    |
| b2_BdMC       |   1.884 ± 0.029    |
| b3_BdMC       |   1.974 ± 0.036    |
| b4_BdMC       |   2.025 ± 0.033    |
| b5_BdMC       |   2.066 ± 0.035    |
| b6_BdMC       |   2.155 ± 0.041    |
| b7_BdMC       |    2.05 ± 0.039    |
| b8_BdMC       |   2.021 ± 0.035    |
+---------------+--------------------+

+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BdData     |   1.393 ± 0.086    |
| b2_BdData     |   1.856 ± 0.071    |
| b3_BdData     |   1.839 ± 0.084    |
| b4_BdData     |   1.988 ± 0.081    |
| b5_BdData     |   1.946 ± 0.082    |
| b6_BdData     |    0.21 ± 0.01     |
| b7_BdData     |    1.846 ± 0.09    |
| b8_BdData     |   1.892 ± 0.082    |
+---------------+--------------------+
"""






"""
trig = '_biased'
out = optimize(lkhd_single_spline, method="emcee",
               params=params['BdMC'],
               burn=300, steps=2000, thin=20, nan_policy='omit',
               kws={'data':  time_d['BdMC'+trig],
                    'prob':  lkhd_d['BdMC'+trig],
                    'weight':weight_d['BdMC'+trig]}
              );

# single spline tester
trig = '_biased'
out = optimize(lkhd_single_spline, method="minuit-hesse",
               params=params['BdMC'],
               kws={'data':  time_d['BdMC'+trig],
                    'prob':  lkhd_d['BdMC'+trig],
                    'weight':weight_d['BdMC'+trig]}
              );

# double spline tester
trig = '_biased'
out = optimize(lkhd_ratio_spline, method="minuit-hesse",
              params=params['ratio'],
              kws={'data':   [cats['BsMC'+trig].time_d,
                              cats['BdMC'+trig].time_d],
                   'prob':   [cats['BsMC'+trig].lkhd_d,
                              cats['BdMC'+trig].lkhd_d],
                   'weight': [cats['BsMC'+trig].weight_d,
                              cats['BdMC'+trig].weight_d]}
              );



# triple spline terster
trig = '_biased'
out = optimize(lkhd_full_spline, method="minuit-hesse",
               params=params['full'],
               kws={'data':   [cats['BsMC'+trig].time_d,
                               cats['BdMC'+trig].time_d,
                               cats['BdDT'+trig].time_d],
                    'prob':   [cats['BsMC'+trig].lkhd_d,
                               cats['BdMC'+trig].lkhd_d,
                               cats['BdDT'+trig].lkhd_d],
                    'weight': [cats['BsMC'+trig].weight_d,
                               cats['BdMC'+trig].weight_d,
                               cats['BdDT'+trig].weight_d]}
              );

out.params.print()


cats['BdMC'+trig]

# Optimization finished in 3.0938 minutes.

"""

"""
+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BsMC       |   1,603 ± 0,039    |
| b2_BsMC       |   2,127 ± 0,035    |
| b3_BsMC       |   2,274 ± 0,042    |
| b4_BsMC       |   2,355 ± 0,040    |
| b5_BsMC       |   2,440 ± 0,042    |
| b6_BsMC       |   2,477 ± 0,048    |
| b7_BsMC       |   2,459 ± 0,046    |
| b8_BsMC       |   2,430 ± 0,043    |
| b1_BdMC       |   0,924 ± 0,033    |
| b2_BdMC       |   0,891 ± 0,019    |
| b3_BdMC       |   0,872 ± 0,023    |
| b4_BdMC       |   0,865 ± 0,020    |
| b5_BdMC       |   0,851 ± 0,020    |
| b6_BdMC       |   0,874 ± 0,024    |
| b7_BdMC       |   0,837 ± 0,022    |
| b8_BdMC       |   0,834 ± 0,020    |
| b1_BdData     |    1,51 ± 0,11     |
| b2_BdData     |    2,11 ± 0,10     |
| b3_BdData     |    2,13 ± 0,12     |
| b4_BdData     |    2,33 ± 0,12     |
| b5_BdData     |    2,32 ± 0,12     |
| b6_BdData     |    2,45 ± 0,14     |
| b7_BdData     |    2,23 ± 0,13     |
| b8_BdData     |    2,30 ± 0,12     |
+---------------+--------------------+
"""




# +---------------+--------------------+
# | Fit Parameter | Value ± ParabError |
# +---------------+--------------------+
# | b1_BsMC       |   1.555 ± 0.062    |
# | b2_BsMC       |   2.087 ± 0.061    |
# | b3_BsMC       |   2.244 ± 0.066    |
# | b4_BsMC       |   2.329 ± 0.067    |
# | b5_BsMC       |   2.385 ± 0.066    |
# | b6_BsMC       |   2.439 ± 0.086    |
# | b7_BsMC       |   2.528 ± 0.073    |
# | b8_BsMC       |   2.513 ± 0.067    |
# | r1_BdMC       |   0.951 ± 0.043    |
# | r2_BdMC       |   0.904 ± 0.029    |
# | r3_BdMC       |    0.882 ± 0.03    |
# | r4_BdMC       |   0.872 ± 0.028    |
# | r5_BdMC       |   0.869 ± 0.027    |
# | r6_BdMC       |   0.885 ± 0.034    |
# | r7_BdMC       |   0.813 ± 0.028    |
# | r8_BdMC       |   0.806 ± 0.025    |
# +---------------+--------------------+
