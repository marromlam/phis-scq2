# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import argparse
import os
import sys
import numpy as np
import hjson

# load ipanema
from ipanema import initialize
from ipanema import ristra, Parameters, optimize, Sample

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

import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and
from utils.helpers import  version_guesser, timeacc_guesser

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']


# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  parser.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--figure', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--mode', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--year', help='Year to fit')
  parser.add_argument('--version', help='Version of the tuples to use')
  parser.add_argument('--trigger', help='Trigger to fit')
  parser.add_argument('--timeacc', help='Different flag to ... ')
  parser.add_argument('--plot', help='Different flag to ... ')
  return parser

################################################################################




def plot_timeacc_fit(params, data, weight, mode, axes=None, log=False, label=None):
  # Look for axes
  if not axes:
    fig,axplot,axpull = plotting.axes_plotpull()
  else:
    fig,axplot,axpull = axes

  ref = histogram.hist(ristra.get(data), weights=ristra.get(weight), bins=100)



  x = np.linspace(params['tLL'].value, params['tUL'].value, 200)
  if mode == 'BsMC':  i = 0
  elif mode == 'BdMC': i = 1
  elif mode == 'BdRD': i = 2
  if len(params.fetch('gamma.*')) > 1: # then is the simultaneous fit or similar
    y = saxsbxscxerf(params, [x, x, x] )[i]
  else: # the single fit
    y = splinexerf(params, x )

  y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
  if label:
    axplot.plot(x,y, label=label)
  else:
    axplot.plot(x,y)
  axpull.fill_between(ref.cmbins,
                      histogram.pull_pdf(x,y,ref.cmbins,ref.counts,ref.errl,ref.errh),
                      0)
  axplot.errorbar(ref.cmbins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
  if log:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  return fig, axplot, axpull



def plot_timeacc_spline(params, time, weights, mode=None ,conf_level=1, name='test.pdf', bins=25, log=False, axes=False, modelabel=None, label=None):
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
  # Look for axes
  if not axes:
    fig, axplot = plotting.axes_plot()
  else:
    fig, axplot = axes

  # a = params.build(params,params.find('a.*')+['gamma_a'])
  # b = params.build(params,params.find('b.*')+['gamma_b'])
  # c = params.build(params,params.find('c.*')+['gamma_c'])
  #
  # print(a,b,c)
  # exit()
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
  print(kind)

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
  y = wrap_unc(badjanak.bspline, x, *coeffs)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)

  # Manipulate data
  ref = histogram.hist(ristra.get(time), bins=bins, weights=ristra.get(weights))
  ref.counts *= int_pdf; ref.errl *= int_pdf; ref.errh *= int_pdf
  if kind == 'ratio':
    coeffs_a = [params[key].value for key in params if key[0]=='a']
    spline_a = badjanak.bspline(ref.cmbins,*coeffs_a)
    ref.counts /= spline_a; ref.errl /= spline_a; ref.errh /= spline_a
  if kind == 'full':
    coeffs_a = [params[key].value for key in params if key[0]=='a']
    coeffs_b = [params[key].value for key in params if key[0]=='b']
    spline_a = badjanak.bspline(ref.cmbins,*coeffs_a)
    spline_b = badjanak.bspline(ref.cmbins,*coeffs_b)
    ref.counts /= spline_b; ref.errl /= spline_b; ref.errh /= spline_b
    #ref.counts *= spline_a; ref.errl *= spline_a; ref.errh *= spline_a
  counts_spline = interp1d(ref.bins, ref.counts, kind='cubic')
  int_5 = counts_spline(5)
  ref.counts /= int_5; ref.errl /= int_5; ref.errh /= int_5

  # Actual ploting
  axplot.set_ylim(0.4, 1.5)
  axplot.plot(x,y_nom/y_spl)
  axplot.errorbar(ref.cmbins,ref.counts,
                  yerr=[ref.errl,ref.errh],
                  xerr=[-ref.edges[:-1]+ref.cmbins,-ref.cmbins+ref.edges[1:]],
                  fmt='.', color='k')
  y_upp, y_low = get_confidence_bands(x, y, sigma=conf_level)
  axplot.fill_between(x, y_upp/y_spl, y_low/y_spl, alpha=0.2, edgecolor="none",
                      label='$'+str(conf_level)+'\sigma$ confidence band')
  if log:
    axplot.set_yscale('log')
  axplot.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'$\epsilon_{%s}$ [a.u.]' % modelabel)
  axplot.legend()
  return fig, axplot







################################################################################
#%% Run and get the job done ###################################################

def plotter(args,axes):

  # Parse arguments ------------------------------------------------------------
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC, MINER = timeacc_guesser(args['timeacc'])
  LOGSCALE = True if 'log' in args['plot'] else False
  PLOT = args['plot'][:-3] if LOGSCALE else args['plot']

  def trigger_scissors(trigger, CUT=""):
    if trigger == 'biased':
      CUT = cuts_and("hlt1b==1",CUT)
    elif trigger == 'unbiased':
      CUT = cuts_and("hlt1b==0",CUT)
    return CUT
  # Prepare the cuts

  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'mode':>15}: {MODE:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'plot':>15}: {PLOT:50}")
  print(f"{'logscale':>15}: {LOGSCALE:<50}")

  # List samples, params and tables
  samples = args['samples'].split(',')
  params = args['params'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  if TIMEACC == 'simul':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = 'kinWeight*'
  elif TIMEACC == 'nonkin':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = ''
  elif TIMEACC == '9knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = 'kinWeight*'
  elif TIMEACC == '12knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = 'kinWeight*'
  CUT = cuts_and(CUT,f'time>={knots[0]} & time<={knots[-1]}')


  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  cats = {}
  sw = f'sw_{VAR}' if VAR else 'sw'
  for i,m in enumerate(['MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar','Bd2JpsiKstar']):
    # Correctly apply weight and name for diffent samples
    if m=='MC_Bs2JpsiPhi':
      weight = f'{kinWeight}polWeight*pdfWeight*dg0Weight*sw/gb_weights'
      mode = 'BsMC';
    elif m=='MC_Bs2JpsiPhi_dG0':
      weight = f'{kinWeight}polWeight*pdfWeight*sw/gb_weights'
      mode = 'BsMC';
    elif m=='MC_Bd2JpsiKstar':
      weight = f'{kinWeight}polWeight*pdfWeight*sw'
      mode = 'BdMC';
    elif m=='Bd2JpsiKstar':
      weight = f'{kinWeight}{sw}'
      mode = 'BdRD';

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time',lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].assoc_params(params[i])

    # Attach labels and paths
    if MODE==m:
      MMODE=mode
      cats[mode].label = mode_tex(mode)
      cats[mode].figurepath = args['figure']


  if TIMEACC=='single':
    pars = []
  else:
    pars = Parameters()
    for cat in cats:
      pars = pars + cats[cat].params
  print(pars)

  if PLOT=='fit':
    axes = plot_timeacc_fit(pars,
                                          cats[MMODE].time, cats[MMODE].weight,
                                          MMODE, log=LOGSCALE, axes=axes,
                                          label=f"${args['version']} - {args['timeacc']}$")
  elif PLOT=='spline':
    axes = plot_timeacc_spline(pars,
                                          cats[MMODE].time, cats[MMODE].weight,
                                          mode=MMODE, log=LOGSCALE, axes=axes,
                                          conf_level=1,
                                          modelabel=mode_tex(MODE),
                                          label=f"${args['version']} - {args['timeacc']}$")
  return axes














if __name__ == '__main__':
  args = vars(argument_parser().parse_args())
  axes = plotting.axes_plotpull()

  initialize(os.environ['IPANEMA_BACKEND'], 1)
  from time_acceptance.fcn_functions import badjanak, saxsbxscxerf, splinexerf

  mix_timeacc = len(args['timeacc'].split('+')) > 1
  mix_version = len(args['version'].split('+')) > 1

  print(args['timeacc'],args['version'])
  if mix_timeacc and mix_version:
    print('shit')
  elif mix_timeacc:
    mixers = f"{args['timeacc']}".split('+')
  elif mix_version:
    mixers = f"{args['version']}".split('+')
  elif not mix_timeacc and not mix_version:
    print('no mix')
    mixers = False
  print(mixers)

  if mixers:
    params = []
    for i,m in enumerate(mixers):
      print(i,m, args['params'].split(',')[i::len(mixers)])
      params.append( args['params'].split(',')[i::len(mixers)] )
  else:
    params = args['params'].split(',')
  if mix_version:
    samples = []
    for i,m in enumerate(mixers):
      j = 3*i
      samples.append( args['samples'].split(',')[j:j+3] )


  # print("\n\n")
  # print(f"{args['params']}")
  # print("\n\n")
  # print(params)
  # print(samples)

  if 'spline' in args['plot']:
    axes = plotting.axes_plot()
  else:
    axes = plotting.axes_plotpull();

  if mix_timeacc:
    for i,m in enumerate(mixers):
      args = {
        "samples": f"{args['samples']}",
        "params":  f"{','.join(params[i])}",
        "figure":  args["figure"],
        "mode":    f"{args['mode']}",
        "year":    f"{args['year']}",
        "version": f"{args['version']}",
        "trigger": f"{args['trigger']}",
        "timeacc": f"{m}",
        "plot":    f"{args['plot']}"
      }
      axes = plotter(args, axes )
      axes[1].legend()
  elif mix_version:
    for i,m in enumerate(mixers):
      args = {
        "samples": f"{','.join(samples[i])}",
        "params":  f"{','.join(params[i])}",
        "figure":  args["figure"],
        "mode":    f"{args['mode']}",
        "year":    f"{args['year']}",
        "version": f"{m}",
        "trigger": f"{args['trigger']}",
        "timeacc": f"{args['timeacc']}",
        "plot":    f"{args['plot']}"
      }
      axes = plotter(args, axes )
      axes[1].legend()
  else:
    args = {
      "samples":  f"{args['samples']}",
      "params":   f"{','.join(params)}",
      "figure":   args["figure"],
      "mode":     f"{args['mode']}",
      "year":     f"{args['year']}",
      "version":  f"{args['version']}",
      "trigger":  f"{args['trigger']}",
      "timeacc":  f"{args['timeacc']}",
      "plot":     f"{args['plot']}"
    }
    axes = plotter(args, axes )

  watermark(axes[1],version=f"${version_guesser(args['version'])[0]}$",scale=1.01)
  axes[0].savefig(args['figure'])

  exit()


  plotter(args, axes=(axes))
  fig.savefig(args['figure'])





################################################################################




################################################################################