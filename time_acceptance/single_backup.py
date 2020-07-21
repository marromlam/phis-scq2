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
from ipanema import plotting
from ipanema import wrap_unc, get_confidence_bands

def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--input-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
                      default = 'baseline',
                      help='Configuration')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'v0r0',
                      help='Version of the tuples to use')
  parser.add_argument('--trigger',
                      default = 'biased',
                      help='Trigger(s) to fit [comb/(biased)/unbiased]')
  # Report
  parser.add_argument('--pycode',
                      default = 'single',
                      help='Save a fit report with the results')

  return parser

################################################################################



################################################################################
#%% Configuration ##############################################################

# Parse arguments
args = vars(argument_parser().parse_args())
PATH = os.path.abspath(os.path.dirname(args['pycode']))
NAME = os.path.splitext(os.path.basename('time/baseline.py'))[0]
VERSION = args['version']
YEAR = args['year']
MODE = args['mode']
TRIGGER = args['trigger']

# Select trigger to fit
if args['trigger'] == 'biased':
  trigger = 'biased'; cuts = "time>=0.3 & time<=15 & hlt1b==1"
elif args['trigger'] == 'unbiased':
  trigger = 'unbiased'; cuts = "time>=0.3 & time<=15 & hlt1b==0"
elif args['trigger'] == 'comb':
  trigger = 'comb'; cuts = "time>=0.3 & time<=15"


shitty = f'{VERSION}_{trigger}_single'

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
# bsjpsikk.use_time_acc = 0
# bsjpsikk.use_time_offset = 0
# bsjpsikk.use_time_res = 0
# bsjpsikk.use_perftag = 0
# bsjpsikk.use_truetag = 1
# bsjpsikk.get_kernels()

#import badjanak as bsjpsikk

################################################################################



################################################################################
#%% Likelihood functions to minimize ###########################################

def lkhd_single_spline(parameters, data, weight = None, prob = None):
  pars_dict = list(parameters.valuesdict().values())
  if not prob: # for ploting, mainly
    data = ristra.allocate(data)
    prob = ristra.allocate(np.zeros_like(data.get()))
    bsjpsikk.splinexerf(data, prob, *pars_dict)
    return prob.get()
  else:
    bsjpsikk.splinexerf(data, prob, *pars_dict)
    if weight is not None:
      result = (ristra.log(prob)*weight).get()
    else:
      result = (ristra.log(prob)).get()
    return -2*result #+ 2*weight.get()



def lkhd_single_spline_2(parameters, data, weight = None, prob = None):
  pars_dict = list(parameters.values())
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
    # simon seems not to be using weight.get() at the end WARNIN!
    return -2*result #+ 2*weight.get()

def lkhd_ratio_spline(parameters, data, weight = None, prob = None):
  pars_dict = parameters.valuesdict()
  if not prob:                                             # for ploting, mainly
    samples = []; prob = []
    for sample in range(0,2):
      samples.append(ristra.allocate(data[sample]))
      prob.append( ristra.allocate(np.zeros_like(data[sample])) )
    bsjpsikk.ratio_spline_time_acceptance(samples[0], samples[1], prob[0], prob[1], **pars_dict)
    return prob[1].get()
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
      samples.append(ristra.allocate(data[sample]))
      prob.append( ristra.allocate(np.zeros_like(data[sample])) )
    bsjpsikk.full_spline_time_acceptance(samples[0], samples[1], samples[2], prob[0], prob[1], prob[2], **pars_dict)
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
  if name.split('/')[4].startswith('MC_Bs'): i = 0
  elif name.split('/')[4].startswith('MC_Bd'): i = 1
  else: i = 2
  print(i)
  ref = histogram.hist(data, weights=weight, bins = 100)
  fig, axplot, axpull = plotting.axes_plotpull();
  x = np.linspace(0.3,15,200)
  if len(parameters)>24:
    y = lkhd_full_spline(parameters, [x, x, x] )[i]
  elif len(parameters)>13:
    y = lkhd_ratio_spline(parameters, [x, x] )[i]
  else:
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
if MODE.startswith('MC_Bs') or MODE.startswith('TOY_Bs'):
  samples['BsMC'] = os.path.join(args['sample'])
elif MODE.startswith('MC_Bd') or MODE.startswith('TOY_Bd'):
  samples['BdMC'] = os.path.join(args['sample'])
elif MODE.startswith('Bd'):
  samples['BdDT'] = os.path.join(args['sample'])

cats = {}
for name, sample in zip(samples.keys(),samples.values()):
  print(f'Loading {sample} as {name} category')
  name = name[:4] # remove _sample
  if name == 'BsMC':
    label = (r'\mathrm{MC}',r'B_s^0')
    weight='(sw/gb_weights)*polWeight*pdfWeight*kinWeight'
  elif name == 'BdMC':
    label = (r'\mathrm{MC}',r'B^0')
    weight='sw*polWeight*pdfWeight*kinWeight'
  elif name == 'BdDT':
    label = (r'\mathrm{data}',r'B_s^0')
    weight='sw*kinWeight'
  cats[name] = Sample.from_root(sample, cuts=cuts)
  cats[name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+trigger
  #cats[name].assoc_params(args[f'input_params'])
  cats[name].assoc_params(args[f'input_params'].replace('TOY','MC').replace('2021','2018'))
  cats[name].allocate(time='time',lkhd='0*time')
  try:
    cats[name].allocate(weight=weight)
    cats[name].weight *= ristra.sum(cats[name].weight)/ristra.sum(cats[name].weight**2)
  except:
    print('There are no weights in this sample. Proceeding with weight=1')
    sigma_name = cats[name].params.find('sigma_.*')[0]
    print(f'Guessing your are fitting a TOY, so setting {sigma_name}=0')
    cats[name].allocate(weight='time/time')
    cats[name].params[sigma_name].value = 0
  knots = cats[name].params.find('k.*') + ['tLL','tUL']
  cats[name].knots = Parameters.build(cats[name].params, knots)
  [cats[name].params.pop(k, None) for k in knots]
  print(cats[name].params)
  print(cats[name].knots)
  cats[name].label = label
  cats[name].figs_path = os.path.dirname(args[f'output_params'])
  cats[name].figs_path = cats[name].figs_path.replace('params','figures')
  cats[name].pars_path = args[f'output_params']
  cats[name].tabs_path = cats[name].pars_path.replace('.json','.tex')
  cats[name].tabs_path = cats[name].tabs_path.replace('params','tables')
  #os.makedirs(os.path.dirname(args[f'{name}_output_params']), exist_ok=True)
  os.makedirs(cats[name].figs_path, exist_ok=True)
  os.makedirs(os.path.dirname(cats[name].tabs_path), exist_ok=True)

################################################################################



################################################################################
#%% Fit all categories #########################################################

print(f"\n{80*'='}\n{'= Fitting three categories':79}=\n{80*'='}\n")
"""
from iminuit import Minuit as minuit
from ipanema import ristra, Sample, Parameters, Parameter, Optimizer

cat = cats[list(cats.keys())[0]]

# Minuit wrapper
def wrapper_minuit(*fvars):
  p = {}
  for name, val in zip(list_of_pars, fvars):
    p[name] = val
  #print(cat.params)
  out = lkhd_single_spline_2(p, data = cat.time, prob = cat.lkhd, weight = cat.weight).sum()
  if np.isnan(out):
    return 1e12
  return out



def configure_minuit( pars, pars_list, **kwgs):
  def parameter_minuit_config(par):
    out = {par.name: par.init}
    lims = [None,None]
    if abs(par.min) != np.inf: lims[0] = par.min
    if abs(par.max) != np.inf: lims[1] = par.max
    if not par.free:
      out.update ({"fix_" + par.name: True})
    out.update ({"limit_" + par.name: tuple(lims)})
    return out

  config = {}
  for par in pars.keys():
    if par in pars_list:
      config.update(parameter_minuit_config(pars[par]))
  config.update(kwgs)
  return config



list_of_pars = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'mu_b', 'sigma_b', 'gamma_b']
print(list_of_pars)
dict_of_conf = {k:v for k,v in configure_minuit(cat.params, list_of_pars).items() }
for k,v in dict_of_conf.items():
  print(k,v)
"""

"""
x = np.arange(0.3,15.1,0.1)
y = ristra.allocate(np.zeros_like(x))
print(x[1]-x[0])
bsjpsikk.single_spline_time_acceptance(
      ristra.allocate(x), y,
      b0=1, b1=1.3, b2=1.5, b3=1.8, b4=2.1, b5=2.3, b6=2.2, b7=2.1, b8=2.0,
      mu=0.0, sigma=0.04, G=0.6,
      tLL = 0.3, tUL = 15,
    )

print(bsjpsikk.get_4cs([1, 1.3, 1.5, 1.8, 2.1, 2.3, 2.2, 2.1, 2.0]))

print(x)
print(y)


simon = np.array([1,1.2509988,1.3980877,1.4966164,1.5861674,1.6711439,1.7511891,1.8259436,1.8950355,1.9580883,2.0147256,2.0645886,2.1077268,2.1445974,2.1756756,2.2014364,2.2223551,2.2389056,2.2515254,2.2606041,2.2665279,2.2696831,2.2704562,2.2692335,2.2664012,2.2623459,2.2574537,2.252111,2.2466571,2.2411764,2.2356677,2.2301296,2.2245609,2.2189601,2.213326,2.2076572,2.2019525,2.1962104,2.1904296,2.1846089,2.1787468,2.1728422,2.1668935,2.1608995,2.1548589,2.1487704,2.1426326,2.1364441,2.1302037,2.1239101,2.1175618,2.1111576,2.1046962,2.0981761,2.0915962,2.084955,2.0782512,2.0714836,2.0646507,2.0577512,2.0507839,2.0437473,2.0366402,2.0294612,2.022209,2.0148823,2.0074798,2,1.9924812,1.9849624,1.9774436,1.9699248,1.962406,1.9548872,1.9473684,1.9398496,1.9323308,1.924812,1.9172932,1.9097744,1.9022556,1.8947368,1.887218,1.8796992,1.8721805,1.8646617,1.8571429,1.8496241,1.8421053,1.8345865,1.8270677,1.8195489,1.8120301,1.8045113,1.7969925,1.7894737,1.7819549,1.7744361,1.7669173,1.7593985,1.7518797,1.7443609,1.7368421,1.7293233,1.7218045,1.7142857,1.7067669,1.6992481,1.6917293,1.6842105,1.6766917,1.6691729,1.6616541,1.6541353,1.6466165,1.6390977,1.6315789,1.6240602,1.6165414,1.6090226,1.6015038,1.593985,1.5864662,1.5789474,1.5714286,1.5639098,1.556391,1.5488722,1.5413534,1.5338346,1.5263158,1.518797,1.5112782,1.5037594,1.4962406,1.4887218,1.481203,1.4736842,1.4661654,1.4586466,1.4511278,1.443609,1.4360902,1.4285714,1.4210526,1.4135338,1.406015,1.3984962])

print(np.amax(ristra.get(y)-simon))

##########
exit()
"""

"""
print('Fit is starting...')
crap = minuit(wrapper_minuit, forced_parameters=list_of_pars, **dict_of_conf) #, print_level=2, pedantic=True)
crap.strategy = 2
crap.migrad()
if not crap.migrad_ok():
  crap.migrad()
crap.hesse()
print('Fit is finished! Cross your fingers and pray Simon')
print(crap.values.values())



# Update pars
for name, val in zip(list_of_pars, crap.values.values()):
  if cat.params[name].free:
    cat.params[name].value = val
for name, val in zip(list_of_pars, crap.errors.values()):
  if cat.params[name].free:
    cat.params[name].stdev = val

print(cat.params)
"""




cat = cats[list(cats.keys())[0]]
result = optimize(fcn_call=lkhd_single_spline,
            method="minuit",
            params=cat.params,
            fcn_kwgs={'data': cat.time, 'prob': cat.lkhd, 'weight': cat.weight},
            verbose=True, strategy=0);

print(result)




# result = optimize(fcn_call=lkhd_single_spline,
#                   method="lbfgsb",
#                   params=cat.params,
#                   fcn_kwgs={'data': cat.time,
#                         'prob': cat.lkhd,
#                         'weight': cat.weight},
#                         verbose=True);
#
# print(result)


result = optimize(fcn_call=lkhd_single_spline,
                  method="cg",
                  params=cat.params,
                  fcn_kwgs={'data': cat.time,
                        'prob': cat.lkhd,
                        'weight': cat.weight},
                        verbose=True);

print(result)














exit()

cat.params = result.params

print(result)



# Plotting ---------------------------------------------------------------------
print(f"\n{80*'='}\n{'= Plotting':79}=\n{80*'='}\n")

for name, cat in zip(cats.keys(),cats.values()):
  plot_fcn_spline(
    result.params,
    cat.time.get(),
    cat.weight.get(),
    name = os.path.join(cat.figs_path,f'{VERSION}_single_{trigger}_fit_log.pdf'),
    log=True
  )
print(f"Plotted {os.path.join(cat.figs_path,f'{VERSION}_single_{trigger}_fit_log.pdf')}")

for name, cat in zip(cats.keys(),cats.values()):
  plot_fcn_spline(
    result.params,
    cat.time.get(),
    cat.weight.get(),
    name = os.path.join(cat.figs_path,f'{VERSION}_single_{trigger}_fit.pdf'),
    log=False
  )
print(f"Plotted {os.path.join(cat.figs_path,f'{VERSION}_single_{trigger}_fit.pdf')}")


# BsMC
plot_spline(# BsMC
  cat.params,
  cat.time.get(),
  cat.weight.get(),
  name = os.path.join(cat.figs_path,f'{VERSION}_single_{trigger}_spline.pdf'),
  label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
  conf_level=1,
  bins=25
)
print(f"Splines were plotted!")


# Dumping fit parameters -------------------------------------------------------
print(f"\n{80*'='}\n{'= Dumping parameters':79}=\n{80*'='}\n")

for name, cat in zip(cats.keys(),cats.values()):
  list_params = [par for par in cat.params if len(par) ==2]
  cat.params.add(*[result.params.get(par) for par in list_params])
  cat.params = cat.knots + cat.params
  cat.params.dump(cats[name].pars_path)
  # latex export
  with open(cat.tabs_path, "w") as text:
    text.write( cat.params.dump_latex( caption="""
    Coefficients of the \\textbf{%s} cubic spline for the $%s$
    \\texttt{\\textbf{%s}} $%s$ category in sigle fit.""" % (YEAR,cat.label[1],TRIGGER,cat.label[0]) ) )
  print( cat.pars_path )



################################################################################
