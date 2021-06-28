DESCRIPTION = """
    Lifetime shit shit shit
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import hjson

# load ipanema
from ipanema import initialize, plotting
from ipanema import ristra, Parameters, optimize, Sample, plot_conf2d, Optimizer

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and, printsec, printsubsec
from utils.helpers import version_guesser, timeacc_guesser
from utils.helpers import swnorm, trigger_scissors

import config
# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
tLL = config.general['tLL']
tUL = config.general['tUL']

if __name__ != '__main__':
  initialize(os.environ['IPANEMA_BACKEND'], 1)
  from time_acceptance.fcn_functions import splinexerf

################################################################################



################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  p = argparse.ArgumentParser(description='Compute single decay-time acceptance.')
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--contour', help='Different flag to ... ')
  args = vars(p.parse_args())
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC, NKNOTS, CORRECT, FLAT, LIFECUT, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'],1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  if EVT in ('evtOdd', 'evtEven'):
    time = 'gentime'
  else:
    time = 'time'
  CUT = f'{time}>={tLL} & {time}<={tUL}'
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}")
  print(f"{'contour':>15}: {args['contour']:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = all_knots[str(NKNOTS)]
  sWeight = "sw"



  # Get data into categories {{{ 

  printsubsec(f"Loading category")

  cats = {}
  for i, m in enumerate([MODE]):
    # Correctly apply weight and name for diffent samples
    if ('MC_Bs2JpsiPhi' in m) and not ('MC_Bs2JpsiPhi_dG0' in m):
      m = 'MC_Bs2JpsiPhi'
      if CORRECT:
        weight = f'kinWeight*polWeight*pdfWeight*{sWeight}/gb_weights'
      else:
        weight = f'dg0Weight*{sWeight}/gb_weights'
      # apply oddWeight if evtOdd in filename
      if EVT in ('evtEven', 'evtOdd'):
        weight = weight.replace('pdfWeight', 'dg0Weight*oddWeight')
        weight = weight.replace('dg0Weight', 'dg0Weight*oddWeight')
      mode = 'signalMC'; c = 'a'
    elif 'MC_Bs2JpsiPhi_dG0' in m:
      m = 'MC_Bs2JpsiPhi_dG0'
      if CORRECT:
        weight = f'kinWeight*polWeight*pdfWeight*{sWeight}/gb_weights'
      else:
        weight = f'{sWeight}/gb_weights'
      # apply oddWeight if evtOdd in filename
      if EVT in ('evtEven', 'evtOdd'):
        weight = weight.replace('pdfWeight', 'oddWeight')
      mode = 'signalMC'; c = 'a'
    elif 'MC_Bd2JpsiKstar' in m:
      m = 'MC_Bd2JpsiKstar'
      if CORRECT:
        weight = f'kinWeight*polWeight*pdfWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      # apply oddWeight if evtOdd in filename
      if EVT in ('evtEven', 'evtOdd'):
        weight = weight.replace('pdfWeight', 'oddWeight')
      mode = 'controlMC'; c = 'b'
    elif 'MC_Bu2JpsiKplus' in m:
      m = 'MC_Bu2JpsiKplus'
      if CORRECT:
        weight = f'kinWeight*polWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      # apply oddWeight if evtOdd in filename
      if EVT in ('evtEven', 'evtOdd'):
        weight = weight.replace('pdfWeight', 'oddWeight')
      mode = 'controlMC'; c = 'b'
    elif 'Bd2JpsiKstar' in m:
      m = 'Bd2JpsiKstar'
      if CORRECT:
        weight = f'kinWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      mode = 'controlRD'; c = 'c'
    print(weight)

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time=time, lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)
    print(cats[mode])

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
                     'free': True if j > 0 else False, 'min': 0.10, 'max': 50.0
                    } for j in range(len(knots[:-1])+2)
    ])
    cats[mode].params.add({'name': f'gamma_{c}',
                           'value': Gdvalue+resolutions[m]['DGsd'],
                           'latex': f'\Gamma_{c}', 'free': False})
    cats[mode].params.add({'name': f'mu_{c}',
                           'value': resolutions[m]['mu'],
                           'latex': f'\mu_{c}', 'free': False})
    cats[mode].params.add({'name': f'sigma_{c}',
                           'value': resolutions[m]['sigma'],
                           'latex': f'\sigma_{c}', 'free': False})
    print(cats[mode].knots)
    print(cats[mode].params)

    # Attach labels and paths
    cats[mode].label = mode_tex(mode)
    cats[mode].pars_path = oparams[i]



  # Configure kernel -----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)


  # Time to fit {{{

  printsubsec(f"Minimization procedure")
  fcn_call = fcns.splinexerf
  fcn_pars = cats[mode].params
  fcn_kwgs = {
      'data': cats[mode].time,
      'prob': cats[mode].lkhd,
      'weight': cats[mode].weight
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)
  if MINER.lower() in ("minuit", "minos"):
    result = mini.optimize(method=MINER, verbose=False, timeit=True, tol=0.05)
  elif MINER.lower() in ('bfgfs', 'lbfgsb','nelder'):
    result = mini.optimize(method=MINER, verbose=False)
  elif MINER.lower() in ('emcee'):
    result = mini.optimize(method='minuit', verbose=False)
    result = mini.optimize(method='emcee', verbose=False, steps=1000,
                           nwalkers=100, is_weighted=False)
  else:
    exit()
  print(result)

  # }}}


  # Do contours or scans if asked {{{

  if args['contour'] != "0":
    if len(args['contour'].split('vs'))>1:
      fig, ax = plot_conf2d(mini, result, args['contour'].split('vs'), size=(50,50))
      fig.savefig(cats[mode].tabs_path.replace('tables','figures').replace('.tex', f"_contour{args['contour']}.pdf"))
    else:
      import matplotlib.pyplot as plt
      # x, y = result._minuit.profile(args['contour'], bins=100, bound=5, subtract_min=True)
      # fig, ax = plotting.axes_plot()
      # ax.plot(x,y,'-')
      # ax.set_xlabel(f"${result.params[ args['contour'] ].latex}$")
      # ax.set_ylabel(r"$L-L_{\mathrm{opt}}$")
      result._minuit.draw_mnprofile(args['contour'], bins=30, bound=1, subtract_min=True, band=True, text=True)
      plt.savefig(cats[mode].tabs_path.replace('tables', 'figures').replace('.tex', f"_contour{args['contour']}.pdf"))

  # }}}


  # Writing results {{{ 

  printsec('Dumping parameters')
  cats[mode].params = cats[mode].knots + result.params
  print(f"Dumping json parameters to {cats[mode].pars_path}")
  cats[mode].params.dump(cats[mode].pars_path)

  # }}}

# }}}


# vim:foldmethod=marker
# that's all folks!
