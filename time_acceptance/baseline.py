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
import uncertainties as unc
from uncertainties import unumpy as unp

from ipanema import ristra
from ipanema import Parameters, fit_report, optimize
from ipanema import histogram
from ipanema import Sample
from ipanema import getDataFile
from ipanema import plotting
from ipanema import wrap_unc, get_confidence_bands

def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--BsMC-sample',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-sample',
                      default = 'samples/MC_Bd2JpsiKstar_2016_test.json',
                      help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-sample',
                      default = 'samples/Bd2JpsiKstar_2016_test.json',
                      help='Bd2JpsiKstar data sample')
  # Output parameters
  parser.add_argument('--BsMC-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs_Bd_ratio/test_biased.json',
                      help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-params',
                      default = 'output/time_acceptance/parameters/2016/Bs2JpsiPhi/test_biased.json',
                      help='Bd2JpsiKstar data sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
                      default = 'baseline',
                      help='Configuration')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--flag',
                      default = 'test',
                      help='Year of data-taking')
  parser.add_argument('--trigger',
                      default = 'biased',
                      help='Trigger(s) to fit [comb/(biased)/unbiased]')
  # Report
  parser.add_argument('--pycode',
                      default = 'baseline',
                      help='Save a fit report with the results')

  return parser

################################################################################



################################################################################
#%% Configuration ##############################################################

# Parse arguments
args = vars(argument_parser().parse_args())
PATH = os.path.abspath(os.path.dirname(args['pycode']))
NAME = os.path.splitext(os.path.basename('time/baseline.py'))[0]
FLAG = args['flag']

# Select trigger to fit
if args['trigger'] == 'biased':
  trigger = 'biased'; cuts = "time>=0.3 & time<=15 & hlt1b==1"
elif args['trigger'] == 'unbiased':
  trigger = 'unbiased'; cuts = "time>=0.3 & time<=15 & hlt1b==0"
elif args['trigger'] == 'comb':
  trigger = 'comb'; cuts = "time>=0.3 & time<=15"
  print('not implemented!')
  exit()

print(f"\n{80*'='}\n{'= Settings':79}=\n{80*'='}\n")
print(f"{'path':>15}: {PATH:50}")
print(f"{'script':>15}: {NAME:50}")
print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
print(f"{'trigger':>15}: {args['trigger']:50}")
print(f"{'cuts':>15}: {cuts:50}\n")





# Initialize backend
from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],2)

# Get bsjpsikk model and configure it
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
    return -2*result + 2*weight.get()

def lkhd_ratio_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,2):
      samples.append(ristra.allocate(data))
      prob.append( ristra.allocate(np.zeros_like(data)) )
    bsjpsikk.ratio_spline_time_acceptance(data[0], data[1], prob[0], prob[1], **pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    bsjpsikk.ratio_spline_time_acceptance(data[0], data[1], prob[0], prob[1], **pars_dict)
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
    bsjpsikk.full_spline_time_acceptance(data[0], data[1], data[2], prob[0], prob[1], prob[2], **pars_dict)
    return [ p.get() for p in prob ]
  else:                               # Optimizer.optimize ready-to-use function
    bsjpsikk.full_spline_time_acceptance(data[0], data[1], data[2], prob[0], prob[1], prob[2], **pars_dict)
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
  axpull.fill_between(ref.cmbins,
                      histogram.pull_pdf(x,y,ref.cmbins,ref.counts,ref.errl,ref.errh),
                      0, facecolor="C0")
  axplot.errorbar(ref.cmbins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
  if log:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  fig.savefig(name)
  plt.close()

def plot_spline(params, time, weights, conf_level=1, name='test.pdf', bins=30, label=None):
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
  list_coeffs = [key for key in params if key[0]=='c']
  if not list_coeffs:
    list_coeffs = [key for key in params if key[0]=='b']
    if not list_coeffs:
      list_coeffs = [key for key in params if key[0]=='a']
      if not list_coeffs:
        print('shit')
      else:
        gamma = params['gamma_a'].value
        kind = 'single'
    else:
      gamma = params['gamma_b'].value
      if not [key for key in params if key[0]=='a']:
        kind = 'single'
      else:
        kind = 'ratio'
  else:
    gamma = params['gamma_c'].value
    if not [key for key in params if key[0]=='b']:
      kind = 'single'
    else:
      kind = 'full'


  # Prepare coeffs as ufloats
  coeffs = []
  for par in list_coeffs:
    if params[par].stdev:
      coeffs.append(unc.ufloat(params[par].value,params[par].stdev))
    else:
      coeffs.append(unc.ufloat(params[par].value,0))

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
  y = wrap_unc(bsjpsikk.acceptance_spline, x, *coeffs)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)

  # Manipulate data
  ref = histogram.hist(time, bins=bins, weights=weights)
  ref.counts *= int_pdf; ref.errl *= int_pdf; ref.errh *= int_pdf
  if kind == 'ratio':
    coeffs_a = [params[key].value for key in params if key[0]=='a']
    spline_a = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_a)
    ref.counts /= spline_a; ref.errl /= spline_a; ref.errh /= spline_a
  if kind == 'full':
    coeffs_a = [params[key].value for key in params if key[0]=='a']
    coeffs_b = [params[key].value for key in params if key[0]=='b']
    spline_a = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_a)
    spline_b = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_b)
    ref.counts /= spline_b; ref.errl /= spline_b; ref.errh /= spline_b
    #ref.counts *= spline_a; ref.errl *= spline_a; ref.errh *= spline_a
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
  y_upp, y_low = get_confidence_bands(x,y, sigma=conf_level)
  axplot.fill_between(x, y_upp/y_spl, y_low/y_spl, alpha=0.2, edgecolor="none",
                      label='$'+str(conf_level)+'\sigma$ confidence band')
  axplot.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'%s [a.u.]' % label)
  axplot.legend()
  fig.savefig(name)
  plt.close()

################################################################################



################################################################################
#%% Get data into categories ###################################################

print(f"\n{80*'='}\n{'= Loading categories':79}=\n{80*'='}\n")

# Select samples
samples = {}
samples['BsMC'] = os.path.join(args['BsMC_sample'])
samples['BdMC'] = os.path.join(args['BdMC_sample'])
samples['BdDT'] = os.path.join(args['BdDT_sample'])


cats = {}
for name, sample in zip(samples.keys(),samples.values()):
  print(f'Loading {sample} as {name} category')
  name = name[:4] # remove _sample
  if name == 'BsMC':
    label = (r'\mathrm{MC}',r'B_s^0')
  elif name == 'BdMC':
    label = (r'\mathrm{MC}',r'B^0')
  elif name == 'BdDT':
    label = (r'\mathrm{data}',r'B_s^0')
  cats[name] = Sample.from_file(sample, cuts=cuts)
  cats[name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+trigger
  cats[name].allocate(time='time',weight='sWeight*kinWeight',lkhd='0*time')
  param_path  = os.path.join(PATH,'init',NAME)
  param_path = os.path.join(param_path,f"{sample.split('/')[-2]}.json")
  cats[name].assoc_params(Parameters.load(param_path))
  cats[name].label = label
  cats[name].pars_path = os.path.dirname(args[f'{name}_params'])
  print( cats[name].pars_path )
  cats[name].figs_path = cats[name].pars_path.replace('parameters','figures')
  print( cats[name].figs_path )
  os.makedirs(cats[name].pars_path, exist_ok=True)
  os.makedirs(cats[name].figs_path, exist_ok=True)

################################################################################



################################################################################
#%% Fit all categories #########################################################
"""
fits = {}; FIT_EACH = 1; FIT_RATIO = 1; FIT_FULL = 1
# Fit each sample
if FIT_EACH:
  for name, cat in zip(cats.keys(),cats.values()):
    print('Fitting %s category...' % name)
    if cat.params:
      fits[cat.name] = optimize(lkhd_single_spline, method="hesse",
                            params=cat.params,
                            kwgs={'data': cat.time,
                                 'prob': cat.lkhd,
                                 'weight': cat.weight},
                            verbose=True);
      fits[cat.name].params.dump(TIMEACC_PATH+'/parameters/'+cat.name)
    fits[cat.name].label = r'$\varepsilon_{%s}^{%s}$' % cat.label
    print('\n')
  for name, cat in zip(cats.keys(),cats.values()):
    print('Plotting %s category...' % name)
    filename = TIMEACC_PATH+'/plots/'+cat.name+'_fit.pdf'
    plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                    name = filename )
    filename = TIMEACC_PATH+'/plots/'+cat.name+'_fit_log.pdf'
    plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                    name = filename, log= True )
    filename = TIMEACC_PATH+'/plots/'+cat.name+'_spline.pdf'
    plot_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
                       name = filename, label=fits[cat.name].label,
                       conf_level=1, bins=30 )






# Fit the ratio BsMC/BdMC ------------------------------------------------------
if FIT_RATIO:
  for trig in ['_biased']:
    name = cats['BdMC'+trig].name.replace('Bd2JpsiKstar','ratioBsBd')
    print('Fitting %s category...' % name)
    fits[name] = optimize(lkhd_ratio_spline, method="hesse",
                    params=cats['BsMC'+trig].params+cats['BdMC'+trig].params,
                    kwgs={'data':  [cats['BsMC'+trig].time,
                                   cats['BdMC'+trig].time],
                          'prob':  [cats['BsMC'+trig].lkhd,
                                   cats['BdMC'+trig].lkhd],
                         'weight': [cats['BsMC'+trig].weight,
                                   cats['BdMC'+trig].weight]},
                    verbose=True);
    fits[name].params.dump(TIMEACC_PATH+'/parameters/'+name)
    print('\n')

  for trig in ['_biased']:
    name = cats['BdMC'+trig].name.replace('Bd2JpsiKstar','ratioBsBd')
    plot_spline(
      {k:fits[name].params[k] for k in list(fits[name].params.keys())[0:12]},
      cats['BsMC'+trig].time.get(),
      cats['BsMC'+trig].weight.get(),
      name = TIMEACC_PATH+'/plots/'+cats['BsMC'+trig].name+'_spline.pdf',
      label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
      conf_level=1,
      bins=30
    )
    plot_spline(
      fits[name].params,
      cats['BdMC'+trig].time.get(),
      cats['BdMC'+trig].weight.get(),
      name = TIMEACC_PATH+'/plots/'+name+'_spline.pdf',
      label=r'$\varepsilon_{\mathrm{MC}}^{B^0/B_s^0}$',
      conf_level=1,
      bins=30
    )



"""


# Full fit to get decay-time acceptance ----------------------------------------
print(f"\n{80*'='}\n{'= Fitting three categories':79}=\n{80*'='}\n")

result = optimize(lkhd_full_spline, method="hesse",
            params=cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params,
            kwgs={'data': [cats['BsMC'].time,
                           cats['BdMC'].time,
                           cats['BdDT'].time],
                'prob':   [cats['BsMC'].lkhd,
                           cats['BdMC'].lkhd,
                           cats['BdDT'].lkhd],
                'weight': [cats['BsMC'].weight,
                           cats['BdMC'].weight,
                           cats['BdDT'].weight]},
            verbose=True);



# Dumping fit parameters -------------------------------------------------------
print(f"\n{80*'='}\n{'= Dumping parameters':79}=\n{80*'='}\n")

for name, cat in zip(cats.keys(),cats.values()):
  list_params = [par for par in cat.params if len(par) ==2]
  cat.params.add(*[result.params.get(par) for par in list_params])
  cat.params.dump(os.path.join(cat.pars_path,f'{FLAG}_{trigger}'))
  print( os.path.join(cat.pars_path,f'{FLAG}_{trigger}') )



# Plotting ---------------------------------------------------------------------
print(f"\n{80*'='}\n{'= Plotting':79}=\n{80*'='}\n")

for name, cat in zip(cats.keys(),cats.values()):
  plot_fcn_spline(
    cat.params,
    cat.time.get(),
    cat.weight.get(),
    name = os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit_log.pdf'),
    log=True
  )

for name, cat in zip(cats.keys(),cats.values()):
  plot_fcn_spline(
    cat.params,
    cat.time.get(),
    cat.weight.get(),
    name = os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit.pdf'),
    log=False
  )

"""
# BsMC
plot_spline(# BsMC
  cats['BsMC'].params,
  cats['BsMC'].time.get(),
  cats['BsMC'].weight.get(),
  name = os.path.join(cats['BsMC'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
  label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
  conf_level=1,
  bins=25
)

# ratio
plot_spline(# BsMC
  cats['BsMC'].params+cats['BdMC'].params,
  cats['BdMC'].time.get(),
  cats['BdMC'].weight.get(),
  name = os.path.join(cats['BdMC'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
  label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
  conf_level=1,
  bins=25
)

# BsDT
plot_spline(# BsMC
  cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params,
  cats['BdDT'].time.get(),
  cats['BdDT'].weight.get(),
  name = os.path.join(cats['BdDT'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
  label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
  conf_level=1,
  bins=25
)
"""
################################################################################
