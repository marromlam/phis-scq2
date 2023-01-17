__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


# Modules {{{

import argparse
import os
import hjson
import numpy as np
import complot

# load ipanema
from ipanema import initialize
from ipanema import ristra, Parameters, Sample, plot_conf2d, Optimizer

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and, printsec, printsubsec
from utils.helpers import version_guesser, timeacc_guesser
from utils.helpers import swnorm, trigger_scissors
from trash_can.knot_generator import create_time_bins

import config
# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']

# }}}


# Command Line Interface {{{

if __name__ == '__main__':
  DESCRIPTION = """
      This file contains 3 fcn functions to be minimized under ipanema3 framework
      those functions are, actually functions of badjanak kernels.
  """

  # Parse arguments {{{
  printsec("Time acceptance procedure")
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--resolutions', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--contour', help='Different flag to ... ')
  p.add_argument('--minimizer', default='minuit', help='Different flag to ... ')
  args = vars(p.parse_args())

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  print(args['version'])
  YEAR = args['year']
  MODE = 'Bs2JpsiPhi'
  TRIGGER = args['trigger']
  TIMEACC = timeacc_guesser(args['timeacc'])
  TIMEACC['use_upTime'] = TIMEACC['use_upTime'] | ('UT' in args['version'])
  TIMEACC['use_lowTime'] = TIMEACC['use_lowTime'] | ('LT' in args['version'])
  MINER = args['minimizer']

  # Get badjanak model and configure it
  initialize(config.user['backend'], 1)
  import analysis.time_acceptance.fcn_functions as fcns

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

  if 'T1' in args['version']:
    tLL, tUL = tLL, 0.9247
    fcns.badjanak.config['final_extrap'] = False
  elif 'T2' in args['version']:
    tLL, tUL = 0.9247, 1.9725
    fcns.badjanak.config['final_extrap'] = False
  elif 'T3' in args['version']:
    tLL, tUL = 1.9725, tUL
    # tLL, tUL = 2, tUL
  else:
    print("SAFE CUT")
    # "T1": "time < 0.9247"
    # "T2": "time > 0.9247 & time < 1.9725"
    # "T3": "time > 1.9725"

  # Check timeacc flag to set knots and weights and place the final cut
  knots = create_time_bins(int(TIMEACC['nknots']), tLL, tUL).tolist()
  if not fcns.badjanak.config['final_extrap']:
    knots[-2] = knots[-1]
  tLL, tUL = knots[0], knots[-1]

  # Cut is ready
  CUT = f'{time}>={tLL} & {time}<={tUL}'
  CUT = trigger_scissors(TRIGGER, CUT)         # place cut attending to trigger
  print("Applied cut:", CUT)

  # Print settings
  printsubsec("Settings")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'contour':>15}: {args['contour']:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  time_offset = args['resolutions'].split(',')
  time_offset = [Parameters.load(i) for i in time_offset]
  print(time_offset)
  oparams = args['params'].split(',')

  # }}}

  # Get data into categories {{{

  printsubsec(f"Loading categories")

  def samples_to_cats(samples, correct, oddity):
    cats = {}
    return cats

  cats = {}
  for i, m in enumerate(samples):
    # Correctly apply weight and name for diffent samples
    # MC_Bs2JpsiPhi {{{
    if ('MC_Bs2JpsiPhi' in m) and not ('MC_Bs2JpsiPhi_dG0' in m):
      m = 'MC_Bs2JpsiPhi'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*sWeight'
      else:
        weight = f'dg0Weight*sWeight'
      mode = 'signalMC'
      c = 'a'
    # }}}
    # MC_Bs2JpsiPhi_dG0 {{{
    elif 'MC_Bs2JpsiPhi_dG0' in m:
      m = 'MC_Bs2JpsiPhi_dG0'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'signalMC'
      c = 'a'
    # }}}
    # MC_Bd2JpsiKstar {{{
    elif 'MC_Bd2JpsiKstar' in m:
      m = 'MC_Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*pdfWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'controlMC'
      c = 'b'
    # }}}
    # Bd2JpsiKstar {{{
    elif 'Bd2JpsiKstar' in m:
      m = 'Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kbsWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'controlRD'
      c = 'c'
    # }}}
    # MC_Bu2JpsiKplus {{{
    elif 'MC_Bu2JpsiKplus' in m:
      m = 'MC_Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kbsWeight*polWeight*sWeight'
      else:
        weight = f'sWeight'
      mode = 'controlMC'
      c = 'b'
    # }}}
    # Bu2JpsiKplus {{{
    elif 'Bu2JpsiKplus' in m:
      m = 'Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kbsWeight*sWeight'
        # weight = f'sWeight'  # TODO: fix kbsWeight here, it should exist and be a reweight Bu -> Bs
      else:
        weight = f'sWeight'
      mode = 'controlRD'
      c = 'c'
    # }}}

    # Final parsing time acceptance and version configurations {{{
    if TIMEACC['use_oddWeight'] and "MC" in mode:
      weight = f"oddWeight*{weight}"
    if TIMEACC['use_veloWeight']:
      weight = f"veloWeight*{weight}"
    # if "bkgcat60" in args['version']:
    #   weight = weight.replace(f'sWeight', 'time/time')
    print(f"Weight is set to: {weight}")
    # }}}

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time=time, lkhd='0*time', weight=weight)
    print(np.min(cats[mode].time.get()), np.max(cats[mode].time.get()))
    cats[mode].weight = swnorm(cats[mode].weight)
    print(cats[mode].df[['time', 'sWeight']])
    # print(cats[mode].df['veloWeight'])

    # Add knots
    cats[mode].knots = Parameters()
    cats[mode].knots.add(*[
        {'name': f'k{j}', 'value': v, 'latex': f'k_{j}', 'free': False}
        for j, v in enumerate(knots[:-1])
    ])
    cats[mode].knots.add({'name': f'tLL', 'value': tLL,
                          'latex': 't_{ll}', 'free': False})
    cats[mode].knots.add({'name': f'tUL', 'value': tUL,
                          'latex': 't_{ul}', 'free': False})

    # Add coeffs parameters
    cats[mode].params = Parameters()
    cats[mode].params.add(*[
        {'name': f'{c}{j}{TRIGGER[0]}', 'value': 1.0,
         'latex': f'{c}_{j}^{TRIGGER[0]}',
         'free': False if j == 0 else True,  # 'min': 0.1, 'max': 10.
         } for j in range(len(knots[:-1]) + 2)
    ])
    cats[mode].params.add({'name': f'gamma_{c}',
                           'value': Gdvalue + resolutions[m]['DGsd'],
                           'latex': f'\Gamma_{c}', 'free': False})
    cats[mode].params.add({'name': f'mu_{c}',
                           'value': 1 * time_offset[i]['mu'].value,
                           'latex': f'\mu_{c}', 'free': False})
    _sigma = np.mean(cats[mode].df['sigmat'].values)
    print(f"sigmat = {resolutions[m]['sigma']} -> {_sigma}")
    cats[mode].params.add({'name': f'sigma_{c}',
                           'value': 0 * _sigma + 1 * resolutions[m]['sigma'],
                           'latex': f'\sigma_{c}', 'free': False})
    print(cats[mode].knots)
    print(cats[mode].params)

    # Attach labels and paths
    cats[mode].pars_path = oparams[i]

  # Configure kernel
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)

  # }}}

  # Time to fit {{{

  printsubsec(f"Simultaneous minimization procedure")
  fcn_call = fcns.saxsbxscxerf
  fcn_pars = cats['signalMC'].params + cats['controlMC'].params + cats['controlRD'].params
  fcn_kwgs = {
      'data': [cats['signalMC'].time, cats['controlMC'].time, cats['controlRD'].time],
      'prob': [cats['signalMC'].lkhd, cats['controlMC'].lkhd, cats['controlRD'].lkhd],
      'weight': [cats['signalMC'].weight, cats['controlMC'].weight, cats['controlRD'].weight],
      # TODO: flatend should be applied in badjanak, general
      # 'flatend': TIMEACC['use_flatend'],
      'tLL': tLL,
      'tUL': tUL
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)

  if MINER.lower() in ("minuit", "minos"):
    result = mini.optimize(method='minuit', verbose=True, strategy=2)
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    _res = mini.optimize(method='nelder', verbose=False)
    result = mini.optimize(method=MINER, params=_res.params, verbose=False)
  elif MINER.lower() in ('nelder'):
    result = mini.optimize(method='nelder', verbose=False)
  elif MINER.lower() in ('emcee'):
    _res = mini.optimize(method='minuit', verbose=False, tol=0.05)
    result = mini.optimize(method='emcee', verbose=False, params=_res.params,
                           steps=1000, nwalkers=100, behavior='chi2')
  print(result)

  # Do contours or scans if asked {{{

  if args['contour'] != "0":
    if len(args['contour'].split('vs')) > 1:
      fig, ax = plot_conf2d(
          mini, result, args['contour'].split('vs'), size=(50, 50))
      fig.savefig(cats[mode].pars_path.replace('tables', 'figures').replace(
          '.json', f"_scan{args['contour']}.pdf"))
    else:
      import matplotlib.pyplot as plt
      # x, y = result._minuit.profile(args['contour'], bins=100, bound=5, subtract_min=True)
      # fig, ax = plotting.axes_plot()
      # ax.plot(x,y,'-')
      # ax.set_xlabel(f"${result.params[ args['contour'] ].latex}$")
      # ax.set_ylabel(r"$L-L_{\mathrm{opt}}$")
      # fig.savefig(cats[mode].pars_path.replace('tables', 'figures').replace('.tex', f"_contour{args['contour']}.pdf"))
      result._minuit.draw_mnprofile(
          args['contour'], bins=20, bound=3, subtract_min=True, band=True, text=True)
      plt.savefig(cats[mode].pars_path.replace('tables', 'figures').replace('.json', f"_contour{args['contour']}.pdf"))

  # }}}

  # }}}

  # Writing results {{{

  printsec(f"Dumping parameters")

  for name, cat in zip(cats.keys(), cats.values()):
    list_params = cat.params.find('(a|b|c)(\d{1})(u|b)')
    print("Dumping:", list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])
    print(f"to: {cats[name].pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cats[name].pars_path)

  # }}}

# }}}


# vim: fdm=marker
# that's all folks!
