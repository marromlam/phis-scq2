# -*- coding: utf-8 -*-

import config
from utils.helpers import YEARS, version_guesser, timeacc_guesser, swnorm
from utils.strings import cammel_case_split, cuts_and
from utils.plot import mode_tex
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
import matplotlib.pyplot as plt
import complot
from ipanema.confidence import get_confidence_bands
from ipanema import uncertainty_wrapper
from ipanema import Sample
from ipanema import Parameters, fit_report, optimize
from ipanema import ristra
from uncertainties import unumpy as unp
import uncertainties as unc
from scipy.interpolate import interp1d
import importlib
import pandas
import platform
import uproot3 as uproot
import pandas as pd
from ipanema import ristra, Parameters, optimize, Sample, extrap1d
from ipanema import initialize
import hjson
import numpy as np
import sys
import os
import argparse
from trash_can.knot_generator import create_time_bins
__author__ = ['Marcos Romero']
__email__ = ['mromerol@cern.ch']

__all__ = ['plot_timeacc_fit', 'plot_timeacc_spline']

################################################################################
# %% Modules ###################################################################


# load ipanema

import os
import sys

# from ipanema import histogram
# from ipanema import plotting


# import some phis-scq utils

# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
# all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']


# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--figure', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--plot', help='Different flag to ... ')
  return p

################################################################################


def plot_timeacc_fit(params, data, weight, mode, axes=None, log=False,
                     label=None, nob=100, nop=200):
  # Look for axes
  if not axes:
    fig, axplot, axpull = complot.axes_plotpull()
  else:
    fig, axplot, axpull = axes

  knots = np.array(params.build(params, params.find('k.*'))).tolist()
  badjanak.config['knots'] = knots
  badjanak.get_kernels()

  ref = complot.hist(ristra.get(data), weights=ristra.get(weight), bins=nob,
                     range=(params['tLL'].value, params['tUL'].value))
  ref_x = ref.bins

  # Get x and y for pdf plot
  x = np.linspace(params['tLL'].value, params['tUL'].value, 200)

  # Get x and y for pdf plot
  x = np.linspace(params['tLL'].value, params['tUL'].value, nop)
  if mode == 'MC_Bs2JpsiPhi_dG0':
    i = 0
  elif mode == 'MC_Bd2JpsiKstar':
    i = 1
  elif mode == 'Bd2JpsiKstar':
    i = 2

  y = splinexerf(params, x)
  y_norm = splinexerf(params, ref_x)

  # normalize y to histogram counts     [[[I don't understand it completely...
  y *= np.trapz(ref.counts, ref_x) / np.trapz(y_norm, ref_x)
  # *abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))

  if label:
    axplot.plot(x, y, label=label)
  else:
    axplot.plot(x, y)
  pulls = complot.compute_pdfpulls(x, y, ref_x, ref.counts, *ref.yerr)
  axpull.fill_between(ref_x, pulls, 0)

  # plot histogram with errorbars
  axplot.errorbar(ref.bins, ref.counts, ref.yerr, fmt='.', color='k')

  # place labels and scales
  if log:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  return fig, axplot, axpull


def plot_timeacc_spline(params, time, weights, mode=None, conf_level=1, bins=45,
                        log=False, axes=False, modelabel=None, label=None,
                        timeacc=None):
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
  # Look for axes {{{

  if not axes:
    fig, axplot, axpull = complot.axes_plotpull()
    FMTCOLOR = 'k'
  else:
    fig, axplot, axpull = axes
    FMTCOLOR = next(axplot._get_lines.prop_cycler)['color']

  # }}}

  # Find all sets of parameters {{{

  a = params.build(params, params.find('a.*')) if params.find('a.*') else None
  b = params.build(params, params.find('b.*')) if params.find('b.*') else None
  c = params.build(params, params.find('c.*')) if params.find('c.*') else None
  knots = np.array(params.build(params, params.find('k.*'))).tolist()
  print("knots>", knots)
  badjanak.config['knots'] = knots
  badjanak.get_kernels()

  # }}}

  # Create some kinda lambda here {{{
  def splinef(time, *coeffs, BLOCK_SIZE=256):
    return badjanak.bspline(time, *coeffs, BLOCK_SIZE=BLOCK_SIZE)
  # }}}

  # exit()

  # try to guess what do you what to plot {{{

  kind = 'single'
  try:
    if mode in ('MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0'):
      mu = list(params.build(params, params.find('mu(.*)a(.*)')).keys())[0]
      mu = params[mu].value
      list_coeffs = list(a.keys())
      gamma = params['gamma_a'].value
    elif mode == 'MC_Bd2JpsiKstar':
      mu = list(params.build(params, params.find('mu(.*)b(.*)')).keys())[0]
      mu = params[mu].value
      list_coeffs = list(b.keys())
      gamma = params['gamma_b'].value
    elif mode == 'Bd2JpsiKstar':
      mu = list(params.build(params, params.find('mu(.*)c(.*)')).keys())[0]
      mu = params[mu].value
      list_coeffs = list(c.keys())
      gamma = params['gamma_c'].value
  except:
    gamma = params['gamma'].value

  # }}}

  print(mode, kind)

  # Prepare coeffs as ufloats
  coeffs = []
  for par in list_coeffs:
    if params[par].stdev:
      coeffs.append(unc.ufloat(params[par].value, params[par].stdev))
    else:
      coeffs.append(unc.ufloat(params[par].value, 0))
  print(coeffs)

  # Cook where should I place the bins {{{

  def distfunction(tLL, tUL, gamma, ti, nob):
    return np.log(-((np.exp(gamma * ti + gamma * tLL + gamma * tUL) * nob) /
                    (-np.exp(gamma * ti + gamma * tLL) + np.exp(gamma * ti + gamma * tUL) -
                     np.exp(gamma * tLL + gamma * tUL) * nob))) / gamma
  list_bins = [tLL]
  ipdf = []
  widths = []
  dummy = 0.4  # this is a general gamma to distribute the bins
  for k in range(0, bins):
    ti = list_bins[k]
    list_bins.append(distfunction(tLL, tUL, dummy, ti, bins))
    tf = list_bins[k + 1]
    ipdf.append(1.0 / ((-np.exp(-(tf * gamma)) + np.exp(-(ti * gamma))) / 1.0))
    widths.append(tf - ti)
  bins = np.array(list_bins)
  int_pdf = np.array(ipdf)

  # }}}

  # Manipulate data
  ref = complot.hist(ristra.get(time) - mu, bins=bins, weights=ristra.get(weights))
  ref.counts *= int_pdf
  ref.errl *= int_pdf
  ref.errh *= int_pdf

  # Manipulate the decay-time dependence of the efficiency
  x = ref.cmbins  # np.linspace(0.3,15,200)
  print(x)
  X = np.linspace(tLL, tUL, 2000)
  y = uncertainty_wrapper(splinef, x, *coeffs)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)
  y_norm = np.trapz(y_nom / y_spl, x)

  ylabel_str = r'$\varepsilon_{%s}$ [a.u.]' % modelabel
  if kind == 'ratio':
    coeffs_a = [params[key].value for key in params if key[0] == 'a']
    spline_a = splinef(ref.cmbins, *coeffs_a)
    ref.counts /= spline_a
    ref.errl /= spline_a
    ref.errh /= spline_a
    ylabel_str = r'$\varepsilon_{MC}^{B_d^0/B_s^0}$ [a.u.]'
  elif kind == 'fullBd':
    coeffs_a = [params[key].value for key in params if key[0] == 'a']
    coeffs_b = [params[key].value for key in params if key[0] == 'b']
    spline_a = splinef(ref.cmbins, *coeffs_a)
    spline_b = splinef(ref.cmbins, *coeffs_b)
    ref.counts /= spline_b
    ref.errl /= spline_b
    ref.errh /= spline_b
    ylabel_str = r'$\varepsilon_{RD}^{B_d^0}$ [a.u.]'
  elif kind == 'fullBs':
    coeffs_a = [params[key].value for key in params if key[0] == 'a']
    coeffs_b = [params[key].value for key in params if key[0] == 'b']
    spline_a = splinef(ref.cmbins, *coeffs_a)
    spline_b = splinef(ref.cmbins, *coeffs_b)
    # ref.counts /= spline_b; ref.errl /= spline_b; ref.errh /= spline_b
    ref.counts /= 1
    ref.errl /= 1
    ref.errh /= 1
    ylabel_str = r'$\varepsilon_{RD}^{B_s^0}$ [a.u.]'

    #ref.counts *= spline_a; ref.errl *= spline_a; ref.errh *= spline_a

  counts_spline = interp1d(ref.bins, ref.counts, kind='cubic')
  int_5 = counts_spline(5)
  ref.counts /= int_5
  ref.errl /= int_5
  ref.errh /= int_5

  ref_norm = np.trapz(ref.counts, ref.cmbins)
  #y_norm = np.trapz(y_nom/y_spl, x)

  # Splines for pdf ploting
  y_upp, y_low = get_confidence_bands(y, sigma=conf_level) / y_spl
  y_nom_s = interp1d(x[:-1], y_nom[:-1] / y_spl, kind='cubic')
  y_upp_s = interp1d(x[:-1], y_upp[:-1], kind='cubic')
  y_low_s = interp1d(x[:-1], y_low[:-1], kind='cubic')
  print(x[:-1])
  y_nom_s = extrap1d(y_nom_s)
  y_upp_s = extrap1d(y_upp_s)
  y_low_s = extrap1d(y_low_s)

  # Actual ploting
  axplot.set_ylim(0.4, 1.5)
  # axplot.set_ylim(0.96, 1.05)#0.96, 1.05
  # axplot.set_xlim(0.3, 3.05)#0.96, 1.05
  # axpull.set_ylim(-2, 2)  # 0.96, 1.05
  #axpull.set_yticks([-2, -1, 0, +1, +2])

  # Plot pdf
  axplot.plot(X, y_nom_s(X), color=FMTCOLOR if FMTCOLOR != 'k' else None)
  # Plot confidence bands
  axplot.errorbar(ref.cmbins, y_norm * ref.counts / ref_norm,
                  yerr=[ref.errl, ref.errh],
                  xerr=[ref.cmbins - ref.edges[:-1], ref.edges[1:] - ref.cmbins],
                  fmt='.', color=FMTCOLOR)

  axplot.fill_between(X, y_upp_s(X), y_low_s(X), alpha=0.2, edgecolor="none",
                      label=f'${conf_level}\sigma$ c.b. {label}')

  axpull.fill_between(ref.cmbins,
                      complot.compute_pdfpulls(
                          x, y_nom / y_spl, ref.cmbins, y_norm * ref.counts / ref_norm,
                          np.sqrt(ref.errl**2 + ref.errh**2), np.sqrt(ref.errl**2 + ref.errh**2)),
                      0)
  # If log, then log both axes
  if log:
    axplot.set_xscale('log')

  # Labeling
  axpull.set_xlabel("$t$ [ps]")
  axplot.set_ylabel(ylabel_str)
  axplot.legend()

  return fig, axplot, axpull


################################################################################
#%% Run and get the job done ###################################################

def plotter(args, axes):

  # Parse arguments ------------------------------------------------------------
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC = timeacc_guesser(args['timeacc'])
  LOGSCALE = True if 'log' in args['plot'] else False
  PLOT = args['plot'][:-3] if LOGSCALE else args['plot']
  LABELED = args['labeled']

  if LABELED:
    thelabel = f"${args['version']}$ {args['timeacc']}"
  else:
    thelabel = ""

  def trigger_scissors(trigger, CUT=""):
    if trigger == 'biased':
      CUT = cuts_and("hlt1b==1", CUT)
    elif trigger == 'unbiased':
      CUT = cuts_and("hlt1b==0", CUT)
    return CUT
  # Prepare the cuts

  CUT = ""  # bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'mode':>15}: {MODE:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'plot':>15}: {PLOT:50}")
  print(f"{'logscale':>15}: {LOGSCALE:<50}")

  # List samples, params and tables
  samples = args['samples'].split(',')
  params = args['params'].split(',')

  if len(config.years[YEAR]) > 1:
    for __y in (2015, 2016, 2017, 2018):
      if str(__y) in samples[0]:
        thelabel += f'  ${__y}$'

  # Prepare the cuts
  if TIMEACC['use_transverse_time']:
    time = 'timeT'
  else:
    time = 'time'
  if TIMEACC['use_truetime']:
    time = f'gen{time}'

  if TIMEACC['use_upTime']:
    tLL = config.general['upper_time_lower_limit']
  else:
    tLL = config.general['time_lower_limit']
  if TIMEACC['use_lowTime']:
    tUL = config.general['lower_time_upper_limit']
  else:
    tUL = config.general['time_upper_limit']
  print(TIMEACC['use_lowTime'], TIMEACC['use_upTime'])

  # Check timeacc flag to set knots and weights and place the final cut
  knots = create_time_bins(int(TIMEACC['nknots']), tLL, tUL).tolist()
  tLL, tUL = knots[0], knots[-1]

  # Cut is ready
  CUT = f'{time}>={tLL} & {time}<={tUL}'
  # place cut attending to trigger
  CUT = trigger_scissors(TRIGGER, CUT)

  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  cats = {}
  sw = 'sWeight'  # f'sw_{VAR}' if VAR else 'sw'
  for i, m in enumerate(['MC_Bd2JpsiKstar', 'Bd2JpsiKstar']):
    # Correctly apply weight and name for diffent samples
    if m == 'MC_Bs2JpsiPhi':
      if TIMEACC['corr']:
        weight = f'dg0Weight*{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*dg0Weight*{sw}/gb_weights'
      mode = 'MC_Bs2JpsiPhi'
      c = 'a'
    elif m == 'MC_Bs2JpsiPhi_dG0':
      if TIMEACC['corr']:
        weight = f'{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      mode = 'MC_Bs2JpsiPhi_dG0'
      c = 'a'
    elif m == 'MC_Bd2JpsiKstar':
      if TIMEACC['corr']:
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}'
      mode = 'MC_Bd2JpsiKstar'
      c = 'b'
    elif m == 'Bd2JpsiKstar':
      if TIMEACC['corr']:
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}{sw}'
      mode = 'Bd2JpsiKstar'
      c = 'c'

    # Load the sample
    cats[mode] = Sample.from_root(samples[0], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time=f'{time}', lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)
    cats[mode].assoc_params(params[0])

    # Attach labels and paths
    if MODE == m or MODE == 'Bs2JpsiPhi':
      MMODE = mode
      cats[mode].label = mode_tex(mode)
      cats[mode].figurepath = args['figure']

  if TIMEACC == 'single':
    pars = []
  else:
    pars = Parameters()
    for cat in cats:
      pars = pars + cats[cat].params
  # print(pars)

  if args['mode'] == 'Bs2JpsiPhi':
    MMMODE = 'Bs2JpsiPhi'
  else:
    MMMODE = MMODE
  if PLOT == 'fit':
    print(pars)
    axes = plot_timeacc_fit(pars,
                            cats[MMODE].time, cats[MMODE].weight,
                            mode=MMMODE, log=LOGSCALE, axes=axes,
                            label=thelabel)
  elif PLOT == 'spline':
    axes = plot_timeacc_spline(pars,
                               cats[MMODE].time, cats[MMODE].weight,
                               mode=MMMODE, log=LOGSCALE, axes=axes,
                               conf_level=1, timeacc=args['timeacc'],
                               modelabel=mode_tex(MODE),
                               label=thelabel)
  return axes


if __name__ == '__main__':
  args = vars(argument_parser().parse_args())
  axes = complot.axes_plotpull()
  print('hello')

  initialize(config.user['backend'], 1)
  from analysis.time_acceptance.fcn_functions import badjanak, saxsbxscxerf, splinexerf

  mix_timeacc = len(args['timeacc'].split('+')) > 1
  mix_version = len(args['version'].split('+')) > 1
  print(config.years[args['year']])
  mix_years = len(config.years[args['year']]) > 1

  print(args['timeacc'], args['version'], args['year'])

  if mix_timeacc and mix_version:
    print('shit')
  elif mix_timeacc:
    mixers = f"{args['timeacc']}".split('+')
  elif mix_version:
    mixers = f"{args['version']}".split('+')
  elif mix_years:
    mixers = config.years[args['year']]
  elif not mix_timeacc and not mix_version:
    print('no mix')
    mixers = False
  print(mixers)

  print('params from snakemake')
  print(args['params'].split(','))
  params = []
  for iy, yy in enumerate(config.years[args['year']]):
    __params = args['params'].split(',')[iy::4]
    if mixers and not mix_years:
      _params = []
      for i, m in enumerate(mixers):
        print(i, m, args['params'].split(',')[i::len(mixers)])
        _params.append(args['params'].split(',')[i::len(mixers)])
    else:
      _params = __params
    params.append(_params)

  samples = []
  for iy, yy in enumerate(config.years[args['year']]):
    __samples = args['samples'].split(',')[iy::4]
    if mix_version:
      _samples = []
      for i, m in enumerate(mixers):
        j = 3 * i
        _samples.append(args['samples'].split(',')[j:j + 3])
    else:
      _samples = __samples
    samples.append(_samples)

  print('PARAMS')
  print(params)
  print('TUPLES')
  print(samples)

  # print("\n\n")
  # print(f"{args['params']}")
  # print("\n\n")
  # print(params)
  # print(samples)

  if 'spline' in args['plot']:
    axes = complot.axes_plot()
    axes = None  # plotting.axes_plotpull()
  else:
    axes = complot.axes_plotpull()

  for iy, yy in enumerate(config.years[args['year']]):
    if mix_timeacc:
      for i, m in enumerate(mixers):
        args = {
            "samples": f"{args['samples']}",
            "params": f"{','.join(params[i])}",
            "figure": args["figure"],
            "mode": f"{args['mode']}",
            "year": f"{args['year']}",
            "version": f"{args['version']}",
            "trigger": f"{args['trigger']}",
            "timeacc": f"{m}",
            "plot": f"{args['plot']}",
            "labeled": True
        }
        axes = plotter(args, axes)
        axes[1].legend()
    elif mix_version:
      for i, m in enumerate(mixers):
        args = {
            "samples": f"{','.join(samples[i])}",
            "params": f"{','.join(params[i])}",
            "figure": args["figure"],
            "mode": f"{args['mode']}",
            "year": f"{args['year']}",
            "version": f"{m}",
            "trigger": f"{args['trigger']}",
            "timeacc": f"{args['timeacc']}",
            "plot": f"{args['plot']}",
            "labeled": True
        }
        axes = plotter(args, axes=axes)
        axes[1].legend()
    else:
      args = {
          "samples": f"{','.join(samples[iy])}",
          "params": f"{','.join(params[iy])}",
          "figure": args["figure"],
          "mode": f"{args['mode']}",
          "year": f"{args['year']}",
          "version": f"{args['version']}",
          "trigger": f"{args['trigger']}",
          "timeacc": f"{args['timeacc']}",
          "plot": f"{args['plot']}",
          "labeled": False
      }
      axes = plotter(args, axes)

  VWATERMARK = version_guesser(args['version'])[0]  # version to watermark plots
  if 'log' in args['plot'] and not 'spline' in args['plot']:
    watermark(axes[1], version=f"${VWATERMARK}$", scale=10.01)
  else:
    watermark(axes[1], version=f"${VWATERMARK}$", scale=1.01)
  axes[0].savefig(args['figure'])

  exit()

  plotter(args, axes=(axes))
  fig.savefig(args['figure'])


################################################################################


################################################################################
