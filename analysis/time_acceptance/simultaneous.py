DESCRIPTION = """
    This file contains 3 fcn functions to be minimized under ipanema3 framework
    those functions are, actually functions of badjanak kernels.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import hjson
import numpy as np

# load ipanema
from ipanema import initialize, plotting
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
all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
tLL = config.general['tLL']
tUL = config.general['tUL']



if __name__ != '__main__':
  import badjanak

################################################################################



################################################################################
# Run and get the job done #####################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  printsec("Time acceptance procedure")
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
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
  initialize(os.environ['IPANEMA_BACKEND'], 1)
  import time_acceptance.fcn_functions as fcns

  if TIMEACC['use_upTime']:
    tLL = 0.89
  if TIMEACC['use_lowTime']:
    tUL = 0.89
  print(TIMEACC['use_lowTime'], TIMEACC['use_upTime'])
  # Prepare the cuts
  CUT = trigger_scissors(TRIGGER)              # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')     # place decay-time cuts

  # Print settings
  printsubsec("Settings")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'contour':>15}: {args['contour']:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = all_knots[str(TIMEACC['nknots'])]
  if TIMEACC['use_lowTime'] or TIMEACC['use_upTime']:
    knots = create_time_bins(int(TIMEACC['nknots']), tLL, tUL)
    knots = knots.tolist()
  print(knots)
  sWeight = "sw"


  # Get data into categories ---------------------------------------------------
  printsubsec(f"Loading categories")

  def samples_to_cats(samples, correct, oddity):
    cats = {}
    return cats

  cats = {}
  for i,m in enumerate(samples):
    # Correctly apply weight and name for diffent samples
    # MC_Bs2JpsiPhi {{{
    if ('MC_Bs2JpsiPhi' in m) and not ('MC_Bs2JpsiPhi_dG0' in m):
      m = 'MC_Bs2JpsiPhi'
      if TIMEACC['corr']:
        weight = f'kinWeight*polWeight*pdfWeight*dg0Weight*{sWeight}/gb_weights'
      else:
        weight = f'dg0Weight*{sWeight}/gb_weights'
      # apply oddWeight if evtOdd in filename
      if TIMEACC['use_oddWeight']:
        weight = f"oddWeight*{weight}"
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      mode = 'signalMC'; c = 'a'
    # }}}
    # MC_Bs2JpsiPhi_dG0 {{{
    elif 'MC_Bs2JpsiPhi_dG0' in m:
      m = 'MC_Bs2JpsiPhi_dG0'
      if TIMEACC['corr']:
        weight = f'kinWeight*polWeight*pdfWeight*{sWeight}/gb_weights'
      else:
        weight = f'{sWeight}/gb_weights'
      # apply oddWeight if evtOdd in filename
      if TIMEACC['use_oddWeight']:
        weight = f"oddWeight*{weight}"
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      mode = 'signalMC'; c = 'a'
    # }}}
    # MC_Bd2JpsiKstar {{{
    elif 'MC_Bd2JpsiKstar' in m:
      m = 'MC_Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kinWeight*polWeight*pdfWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      # apply oddWeight if evtOdd in filename
      if TIMEACC['use_oddWeight']:
        weight = f"oddWeight*{weight}"
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      mode = 'controlMC'; c = 'b'
    # }}}
    # Bd2JpsiKstar {{{
    elif 'Bd2JpsiKstar' in m:
      m = 'Bd2JpsiKstar'
      if TIMEACC['corr']:
        weight = f'kinWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      mode = 'controlRD'; c = 'c'
    # }}}
    # MC_Bu2JpsiKplus {{{
    elif 'MC_Bu2JpsiKplus' in m:
      m = 'MC_Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kinWeight*polWeight*{sWeight}'
      else:
        weight = f'{sWeight}'
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      # apply oddWeight if evtOdd in filename
      if TIMEACC['use_oddWeight']:
        weight = f"oddWeight*{weight}"
      mode = 'controlMC'; c = 'b'
    # }}}
    # Bu2JpsiKplus {{{
    elif 'Bu2JpsiKplus' in m:
      m = 'Bu2JpsiKplus'
      if TIMEACC['corr']:
        weight = f'kinWeight*{sWeight}'
        # weight = f'{sWeight}'  # TODO: fix kinWeight here, it should exist and be a reweight Bu -> Bs
      else:
        weight = f'{sWeight}'
      if TIMEACC['use_veloWeight']:
        weight = f"veloWeight*{weight}"
      mode = 'controlRD'; c = 'c'
    # }}}
    print(weight)

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time', lkhd='0*time', weight=weight)
    print(np.min(cats[mode].time.get()), np.max(cats[mode].time.get()))
    cats[mode].weight = swnorm(cats[mode].weight)
    # print(cats[mode].df['veloWeight'])

    # Add knots
    cats[mode].knots = Parameters()
    cats[mode].knots.add(*[
                {'name':f'k{j}', 'value':v, 'latex':f'k_{j}', 'free':False}
                 for j,v in enumerate(knots[:-1])
               ])
    cats[mode].knots.add({'name':f'tLL', 'value':tLL,
                          'latex':'t_{ll}', 'free':False})
    cats[mode].knots.add({'name':f'tUL', 'value':tUL,
                          'latex':'t_{ul}', 'free':False})

    # Add coeffs parameters
    cats[mode].params = Parameters()
    cats[mode].params.add(*[
                    {'name':f'{c}{j}{TRIGGER[0]}', 'value':1.0,
                     'latex':f'{c}_{j}^{TRIGGER[0]}',
                     'free':False if j==0 else True, #'min':0.10, 'max':5.0
                    } for j in range(len(knots[:-1])+2)
    ])
    cats[mode].params.add({'name':f'gamma_{c}',
                           'value':Gdvalue+resolutions[m]['DGsd'],
                           'latex':f'\Gamma_{c}', 'free':False})
    cats[mode].params.add({'name':f'mu_{c}',
                           'value':resolutions[m]['mu'],
                           'latex':f'\mu_{c}', 'free':False})
    _sigma = np.mean(cats[mode].df['sigmat'].values)
    print(f"sigmat = {resolutions[m]['sigma']} -> {_sigma}")
    cats[mode].params.add({'name':f'sigma_{c}',
                           'value':0*_sigma + 1*resolutions[m]['sigma'],
                           'latex':f'\sigma_{c}', 'free':False})
    print(cats[mode].knots)
    print(cats[mode].params)

    # Attach labels and paths
    cats[mode].pars_path = oparams[i]



  # Configure kernel -----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)



  # Time to fit ----------------------------------------------------------------
  printsubsec(f"Simultaneous minimization procedure")
  fcn_call = fcns.saxsbxscxerf
  fcn_pars = cats['signalMC'].params+cats['controlMC'].params+cats['controlRD'].params
  fcn_kwgs={
    'data': [cats['signalMC'].time, cats['controlMC'].time, cats['controlRD'].time],
    'prob': [cats['signalMC'].lkhd, cats['controlMC'].lkhd, cats['controlRD'].lkhd],
    'weight': [cats['signalMC'].weight, cats['controlMC'].weight, cats['controlRD'].weight],
    'flatend': TIMEACC['use_flatend'],
    'tLL': tLL,
    'tUL': tUL
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)

  if MINER.lower() in ("minuit","minos"):
    result = mini.optimize(method='minuit', verbose=False, tol=0.1);
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    _res = mini.optimize(method='nelder', verbose=False);
    result = mini.optimize(method=MINER, params=_res.params, verbose=False);
  elif MINER.lower() in ('nelder'):
    result = mini.optimize(method='nelder', verbose=False)
  elif MINER.lower() in ('emcee'):
    _res = mini.optimize(method='minuit', verbose=False, tol=0.05)
    result = mini.optimize(method='emcee', verbose=False, params=_res.params,
                           steps=1000, nwalkers=100, behavior='chi2')
  print(result)

  # Do contours or scans if asked ----------------------------------------------
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



  # Writing results ------------------------------------------------------------
  printsec(f"Dumping parameters")

  for name, cat in zip(cats.keys(),cats.values()):
    list_params = cat.params.find('(a|b|c)(\d{1})(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cats[name].pars_path)


# vim:foldmethod=marker
# that's all folks!
