from scipy.interpolate import UnivariateSpline
import argparse
import os
import sys
import numpy as np
import hjson
import logging
from ipanema import initialize
from ipanema import ristra, Parameters, optimize, Sample, extrap1d
import pandas as pd
import uproot3 as uproot
import platform
import pandas
import importlib
from scipy.interpolate import interp1d
import uncertainties as unc
from uncertainties import unumpy as unp
import complot
from ipanema import ristra
from ipanema import Parameters, fit_report, optimize
from ipanema import Sample
from ipanema import uncertainty_wrapper
from ipanema.confidence import get_confidence_bands
import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and
from utils.helpers import YEARS, version_guesser, timeacc_guesser, swnorm, trigger_scissors
from trash_can.knot_generator import create_time_bins
import config
__all__ = ['plot_timeacc_simul_fit', 'plot_timeacc_simul_spline']
__author__ = ['Marcos Romero']
__email__ = ['mromerol@cern.ch']


# Model {{{


LOG = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# load ipanema


# from ipanema import histogram
# from ipanema import plotting


# import some phis-scq utils

# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
# all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
# tLL = config.general['tLL']
# tUL = config.general['tUL']

# }}}


# Plot lifetime fit {{{

def plot_timeacc_simul_fit(params, data, weight, kind, axes=None, log=False,
                           label=None, nob=100):
  """
  This function plots the spline coming from a simultaneous fit

  Parameters
  ----------
  params: ipanema.Parameters
      Set of time acceptance parameters
  time: ipanema.ristra
      varaible to plot
  weights: ipanema.ristra
      Set of weights for the histogram
  """
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

  if kind == 'mc':
    i = 0
  elif kind == 'mc_control':
    i = 1
  elif kind == 'rd_control':
    i = 2

  # calculate pdf
  y = saxsbxscxerf(params, [x, x, x])[i]

  # normalize y to histogram counts
  y_norm = saxsbxscxerf(params, [ref_x, ref_x, ref_x])[i]
  y *= np.trapz(ref.counts, ref_x) / np.trapz(y_norm, ref_x)
  # *abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))

  # plot pdf
  axplot.plot(x, y, label=label)
  # plot pulls
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

# }}}


# Plot one spline {{{

def plot_timeacc_simul_spline(params, time, weights, mode=None, conf_level=1,
                              bins=25, log=False, axes=False, modelabel=None,
                              label=None, tLL=0.3, tUL=15):
  """
  This function plots the spline coming from a simultaneous fit

  Parameters
  ----------
  params: ipanema.Parameters
      Set of time acceptance parameters
  time: ipanema.ristra
      varaible to plot
  weights: ipanema.ristra
      Set of weights for the histogram

  Returns
  -------
  plt.figure
      Figure
  plt.axes
      Main plot axes
  plt.axes
      Pull plot axes

  Note
  ----
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
  kind = mode
  LOG.info(f"Plot kind is {kind}")

  # Look for axes {{{

  if not axes:
    fig, axplot, axpull = complot.axes_plotpull()
    FMTCOLOR = 'k'
  else:
    fig, axplot, axpull = axes
    FMTCOLOR = next(axplot._get_lines.prop_cycler)['color']

  # }}}

  # Find all sets of parameters {{{

  a, b, c = None, None, None
  if params.find('a.*'):
    a = params.build(params, params.find('a.*'))
  if params.find('b.*'):
    b = params.build(params, params.find('b.*'))
  if params.find('c.*'):
    c = params.build(params, params.find('c.*'))

  knots = np.array(params.build(params, params.find('k.*'))).tolist()
  badjanak.config['knots'] = knots
  badjanak.get_kernels()

  # }}}

  # Create some kinda lambda here {{{

  def splinef(time, mu, coeffs):
    return badjanak.bspline(time, coeffs)

  # }}}

  # try to guess what do you what to plot {{{

  mu = 0
  if mode == 'rd':
    LOG.error("We need to decide yet how to plot this")
    eff_label = r"\varepsilon_{RD}^{B_s^0}"
  if mode == 'rd_control':
    gamma = params['gamma_c'].value
    list_coeffs = list(c.keys())
    mu = list(params.build(params, params.find('mu(.*)c(.*)')).keys())[0]
    mu = params[mu].value
    eff_label = r"\varepsilon_{RD}^{B_d^0}"
  if mode == 'mc_control':
    gamma = params['gamma_b'].value
    mu = list(params.build(params, params.find('mu(.*)b(.*)')).keys())[0]
    mu = params[mu].value
    list_coeffs = list(b.keys())
    eff_label = r"\varepsilon_{MC}^{ratio}"
  if mode == 'mc':
    gamma = params['gamma_a'].value
    mu = list(params.build(params, params.find('mu(.*)a(.*)')).keys())[0]
    mu = params[mu].value
    list_coeffs = list(a.keys())
    eff_label = r"\varepsilon_{MC}^{B_s^0}"

  print(f"VELO misaligment: {mu}")
  # }}}

  # Prepare coeffs as ufloats
  coeffs = []
  for par in list_coeffs:
    if params[par].stdev:
      coeffs.append(unc.ufloat(params[par].value, params[par].stdev))
    else:
      coeffs.append(unc.ufloat(params[par].value, 0))
  LOG.debug(f"Coeffients for the main spline: {coeffs}")
  print(f"Coeffients for the main spline: {coeffs}")

  # cook where should I place the bins {{{

  def distfunction(tLL, tUL, gamma, ti, nob):
    return np.log(
        -((np.exp(gamma * ti + gamma * tLL + gamma * tUL) * nob) /
          (-np.exp(gamma * ti + gamma * tLL) + np.exp(gamma * ti + gamma * tUL) -
            np.exp(gamma * tLL + gamma * tUL) * nob)
          )) / gamma

  list_bins = [tLL]
  ipdf = []
  widths = []
  dummy_gamma = 0.4  # this is a general gamma to distribute the bins
  for k in range(0, bins):
    ti = list_bins[k]
    list_bins.append(distfunction(tLL, tUL, dummy_gamma, ti, bins))
    tf = list_bins[k + 1]
    ipdf.append(1.0 / ((-np.exp(-(tf * gamma)) + np.exp(-(ti * gamma))) / 1.0))
    widths.append(tf - ti)
  edges = np.array(list_bins)
  int_pdf = np.array(ipdf)

  # }}}

  # create the histogram for the dataset {{{
  # The binning obviously is computed in the previous step

  ref = complot.hist(ristra.get(time) - 0 * mu, bins=edges, center_of_mass=True,
                     weights=ristra.get(weights))
  ref_x = ref.bins
  ref_counts = ref.counts * int_pdf
  ref_yerr = [ref.yerr[0] * int_pdf, ref.yerr[1] * int_pdf]

  spline_compl = 1
  if kind == 'mc_control':
    coeffs_a = [params[key].value for key in params if key[0] == 'a']
    mu_a = 0
    spline_compl = splinef(ref_x, mu_a, coeffs_a)
  elif kind == 'rd_control':
    coeffs_b = [params[key].value for key in params if key[0] == 'b']
    mu_b = 0
    spline_compl = splinef(ref_x, mu_b, coeffs_b)

  # renormalize histogram
  ref_counts = ref_counts / spline_compl
  ref_yerr = [i / spline_compl for i in ref_yerr]

  # make it *unity* at 5 ps
  counts_spline = interp1d(
      ref_x, ref_counts, kind='cubic', fill_value='extrapolate')
  from csaps import csaps
  print(interp1d(ref_x, ref_counts, kind='cubic')(5))
  # counts_spline = UnivariateSpline(ref_x, ref_counts)
  # counts_spline.set_smoothing_factor(1000)
  # int_5 = counts_spline(5)
  # print(int_5)
  # counts_spline.set_smoothing_factor(0.)
  int_5 = counts_spline(5)
  print(int_5)
  # ref_counts = ref_counts / int_5
  # ref_yerr = [i / int_5 for i in ref_yerr]

  ref_norm = np.trapz(ref_counts, ref_x)

  # }}}

  # create efficiency spline {{{

  x = np.linspace(tLL, tUL, 200)
  y = uncertainty_wrapper(lambda p: splinef(ref_x, mu, p), coeffs)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(ref_x, y_nom, kind='cubic', fill_value='extrapolate')(5)
  y_norm = np.trapz(y_nom / y_spl, ref_x)

  # splines for pdf plotting
  # Here we ensure linear extrapolation has linear errors
  y_upp, y_low = get_confidence_bands(y, sigma=conf_level) / y_spl
  y_nom_s = interp1d(ref_x[:-1], y_nom[:-1] / y_spl, kind='cubic')
  y_upp_s = interp1d(ref_x[:-1], y_upp[:-1], kind='cubic')
  y_low_s = interp1d(ref_x[:-1], y_low[:-1], kind='cubic')
  y_nom_s = extrap1d(y_nom_s)
  y_upp_s = extrap1d(y_upp_s)
  y_low_s = extrap1d(y_low_s)

  # }}}

  # now we can set the histogram counts
  ref_y = y_norm * ref_counts / ref_norm
  ref_yerr = [y_norm * i / ref_norm for i in ref_yerr]

  # Actual ploting {{{

  axplot.set_ylim(0.4, 1.5)
  # axplot.set_xlim(0.3, 3)
  # axplot.set_ylim(0.96, 1.05)#0.96, 1.05
  # axplot.set_xlim(0.3, 3.05)#0.96, 1.05
  # axpull.set_ylim(-2, 2)  # 0.96, 1.05
  # axpull.set_yticks([-2, -1, 0, +1, +2])

  # Plot pdf
  axplot.plot(x, y_nom_s(x), color=FMTCOLOR if FMTCOLOR != 'k' else None)

  # Plot confidence bands
  axplot.errorbar(ref_x, ref_y, yerr=ref_yerr,
                  xerr=[-edges[0:-1] + ref_x, edges[1:] - ref_x],
                  fmt='.', color=FMTCOLOR)

  axplot.fill_between(x, y_upp_s(x), y_low_s(x), alpha=0.2, edgecolor="none",
                      label=f'{label}')

  _p = complot.compute_pdfpulls(ref_x, y_nom / y_spl, ref_x, ref_y, *ref_yerr)
  axpull.fill_between(ref_x, _p, 0, alpha=0.4)

  # if log, then log both axes
  if log:
    axplot.set_xscale('log')

  # Labeling
  axpull.set_xlabel("$t$ [ps]")
  axplot.set_ylabel(f"${eff_label}$")
  axplot.legend()

  # }}}

  return fig, axplot, axpull

# }}}


# shi shi shit {{{
# We need to create a plot abstraction so we can have different overlays.


def plot_wrapper(args, axes):

  # Parse arguments {{{

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(
      args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC = timeacc_guesser(args['timeacc'])
  LOGSCALE = True if 'log' in args['plot'] else False
  PLOT = args['plot'][:-3] if LOGSCALE else args['plot']
  LABELED = args['labeled']

  # }}}

  if LABELED:
    thelabel = f"${args['version']} - {args['timeacc']}$"
  else:
    thelabel = ""

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

  LOG.info(f"Time range: {TIMEACC['use_lowTime']}, {TIMEACC['use_upTime']}")

  # Check timeacc flag to set knots and weights and place the final cut
  knots = create_time_bins(int(TIMEACC['nknots']), tLL, tUL).tolist()
  tLL, tUL = knots[0], knots[-1]

  # Cut is ready
  CUT = f'{time}>={tLL} & {time}<={tUL}'
  # place cut attending to trigger
  CUT = trigger_scissors(TRIGGER, CUT)

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

  # Get data into categories ------------------------------
  LOG.info("Loading categories")

  cats = {}
  for i, s in enumerate(samples):
    # Correctly apply weight and name for diffent samples
    # MC_Bs2JpsiPhi {{{
    if ('MC_Bs2JpsiPhi' in s) and not ('MC_Bs2JpsiPhi_dG0' in s):
      m = 'MC_Bs2JpsiPhi'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*dg0Weight*sWeight'
      else:
        weight = f'dg0Weight*sWeight'
      mode = 'mc'
      c = 'a'
    # }}}
    # MC_Bs2JpsiPhi_dG0 {{{
    elif 'MC_Bs2JpsiPhi_dG0' in s:
      m = 'MC_Bs2JpsiPhi_dG0'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'mc'
      c = 'a'
    # }}}
    # MC_Bd2JpsiKstar {{{
    elif 'MC_Bd2JpsiKstar' in s:
      m = 'MC_Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'mc_control'
      c = 'b'
    # }}}
    # Bd2JpsiKstar {{{
    elif 'Bd2JpsiKstar' in s:
      m = 'Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kbsWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'rd_control'
      c = 'c'
    # }}}
    # MC_Bu2JpsiKplus {{{
    elif 'MC_Bu2JpsiKplus' in s:
      m = 'MC_Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'mc_control'
      c = 'b'
    # }}}
    # Bu2JpsiKplus {{{
    elif 'Bu2JpsiKplus' in s:
      m = 'Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kbsWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'rd_control'
      c = 'c'
    # }}}

    # Final parsing time acceptance and version configurations {{{
    if TIMEACC['use_oddWeight'] and "MC" in mode:
      weight = f"oddWeight*{weight}"
    if TIMEACC['use_veloWeight']:
      weight = f"veloWeight*{weight}"
    # if "bkgcat60" in args['version']:
    #     weight = weight.replace(f'sWeight', 'time/time')
    print(f"Weight is set to: {weight}")
    # }}}

    # Load the sample
    cats[mode] = Sample.from_root(s, cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time', lkhd='0*time', weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)
    LOG.info(f"Parameters for {mode}")
    LOG.info(params)
    cats[mode].assoc_params(params[i])

    # Attach labels and paths
    # if MODE == m or MODE == 'Bs2JpsiPhi':
    #     MMODE = mode
    #     cats[mode].label = mode_tex(mode)
    #     cats[mode].figurepath = args['figure']

  if TIMEACC == 'single':
    pars = []
  else:
    pars = Parameters()
    for cat in cats:
      pars = pars + cats[cat].params
  # print(pars)

  if MODE in ('MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0'):
    kind = 'mc'
  elif MODE in ('MC_Bd2JpsiKstar', 'MC_Bu2JpsiKplus'):
    kind = 'mc_control'
  elif MODE in ('Bd2JpsiKstar', 'Bu2JpsiKplus'):
    kind = 'rd_control'
  else:
    ValueError("I do not get this mode")

  if PLOT == 'fit':
    axes = plot_timeacc_simul_fit(pars,
                                  cats[mode].time, cats[mode].weight,
                                  kind=kind, log=LOGSCALE, axes=axes,
                                  label=thelabel)
  elif PLOT == 'spline':
    axes = plot_timeacc_simul_spline(pars,
                                     cats[kind].time, cats[kind].weight,
                                     mode=kind, log=LOGSCALE, axes=axes,
                                     conf_level=1,
                                     modelabel=mode_tex(MODE),
                                     label=thelabel, tLL=tLL, tUL=tUL)
  return axes

# }}}


# command line {{{
if __name__ == '__main__':

  DESCRIPTION = """
    dfdfd
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--figure', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--plot', help='Different flag to ... ')
  args = vars(p.parse_args())
  axes = complot.axes_plotpull()

  LOG.info("Hello")

  initialize(config.user['backend'], 1)
  from analysis.time_acceptance.fcn_functions import (badjanak,
                                                      saxsbxscxerf,
                                                      splinexerf)

  mix_timeacc = len(args['timeacc'].split('+')) > 1
  mix_version = len(args['version'].split('+')) > 1
  mix_years = len(config.years[args['year']]) > 1

  print(args['timeacc'], args['version'], args['year'])

  if mix_timeacc and mix_version:
    print('shit')
  elif mix_timeacc:
    mixers = f"{args['timeacc']}".split('+')
  elif mix_version:
    mixers = f"{args['version']}".split('+')
  elif not mix_timeacc and not mix_version:
    print('no mix')
    mixers = False
  LOG.info(f"Mixers are: {mixers}")

  params = []
  print(args['params'])
  years = config.years[args['year']]
  use_different_years = len(config.years[args['year']]) > 1
  use_different_years = len(config.years[args['year']]) > 1
  if use_different_years:
    LOG.info("Using different years")
    for ky, vy in enumerate(years):
      __params = args['params'].split(',')[ky::len(years)]
      if mixers:
        _params = []
        for i, m in enumerate(mixers):
          print(i, m, args['params'].split(',')[i::len(mixers)])
          _params.append(args['params'].split(',')[i::len(mixers)])
      else:
        _params = __params
      params.append(_params)
      print(params)
  else:
    __params = args['params'].split(',')
    if mixers:
      _params = []
      for i, m in enumerate(mixers):
        print(i, m, args['params'].split(',')[i::len(mixers)])
        _params.append(args['params'].split(',')[i::len(mixers)])
    else:
      _params = __params
    params.append(_params)
  print(params)

  samples = []
  if use_different_years:
    for year in config.years[args['year']]:
      if mix_version:
        _samples = []
        for i, m in enumerate(mixers):
          j = 3 * i
          _samples.append(args['samples'].split(',')[j:j + 3])
      else:
        _samples = args['samples'].split(',')
      samples.append(_samples)
  else:
    if mix_version:
      _samples = []
      for i, m in enumerate(mixers):
        j = 3 * i
        _samples.append(args['samples'].split(',')[j:j + 3])
    else:
      _samples = args['samples'].split(',')
    samples.append(_samples)

  print('PARAMS')
  print(params)
  print('TUPLES')
  print(samples)

  # chose kind of plot, and instantinate axes
  if 'spline' in args['plot']:
    axes = complot.axes_plot()
    axes = None  # plotting.axes_plotpull()
  else:
    axes = complot.axes_plotpull()

  for iy, yy in enumerate(config.years[args['year']]):
    samples0 = samples[iy]
    if mix_timeacc:
      for i, m in enumerate(mixers):
        args = {
            "samples": f"{','.join(samples[iy])}",
            # "samples": f"{args['samples']}",
            "params": f"{','.join(params[iy][i])}",
            "figure": args["figure"],
            "mode": f"{args['mode']}",
            "year": f"{args['year']}",
            "version": f"{args['version']}",
            "trigger": f"{args['trigger']}",
            "timeacc": f"{m}",
            "plot": f"{args['plot']}",
            "labeled": True
        }
        axes = plot_wrapper(args, axes)
        axes[1].legend()
    elif mix_version:
      for i, m in enumerate(mixers):
        args = {
            "samples": f"{','.join(samples[iy][i])}",
            "params": f"{','.join(params[iy][i])}",
            "figure": args["figure"],
            "mode": f"{args['mode']}",
            "year": f"{args['year']}",
            "version": f"{m}",
            "trigger": f"{args['trigger']}",
            "timeacc": f"{args['timeacc']}",
            "plot": f"{args['plot']}",
            "labeled": True
        }
        axes = plot_wrapper(args, axes=axes)
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
      axes = plot_wrapper(args, axes)

  VWATERMARK = version_guesser(args['version'])[0]  # watermark plots
  if 'log' in args['plot'] and 'spline' not in args['plot']:
    watermark(axes[1], version=f"${VWATERMARK}$", scale=10.01)
  else:
    watermark(axes[1], version=f"${VWATERMARK}$", scale=1.01)
  axes[0].savefig(args['figure'])

# }}}


# vim: fdm=marker
