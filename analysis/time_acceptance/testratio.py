from utils.helpers import swnorm, trigger_scissors
from utils.helpers import version_guesser, timeacc_guesser
from utils.strings import cuts_and
from utils.plot import mode_tex
from ipanema import initialize
import hjson
import numpy as np
import os
import argparse
from ipanema import ristra, Parameters, optimize, Sample, plot_contours, Optimizer
DESCRIPTION = """
    This file contains 3 fcn functions to be minimized under ipanema3 framework
    those functions are, actually functions of badjanak kernels.
"""

__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


################################################################################
# %% Modules ###################################################################


# load ipanema

# import some phis-scq utils

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# Parse arguments for this script


def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  return p


if __name__ != '__main__':
  import badjanak


################################################################################


################################################################################
#%% Run and get the job done ###################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = 'Bs2JpsiPhi'
  TRIGGER = args['trigger']
  TIMEACC, CORR, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(config.user['backend'], 1 if YEAR in (2015, 2017) else -1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')
  otables = args['tables'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
  kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  if CORR == '9knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  elif CORR == '12knots':
    knots = [0.30, 0.43, 0.58, 0.74, 0.91, 1.11, 1.35,
             1.63, 1.96, 2.40, 3.01, 4.06, 9.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'

  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\nLoading categories\n{80*'='}\n")

  # Select samples
  cats = {}
  sw = f'sw_{VAR}' if VAR else 'sw'
  for i, m in enumerate(['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar']):
    # Correctly apply weight and name for diffent samples
    if m == 'MC_Bs2JpsiPhi':
      if CORR == 'Noncorr':
        weight = f'dg0Weight*{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*dg0Weight*{sw}/gb_weights'
      mode = 'BsMC'
      c = 'a'
    elif m == 'MC_Bs2JpsiPhi_dG0':
      if CORR == 'Noncorr':
        weight = f'{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      mode = 'BsMC'
      c = 'a'
    elif m == 'MC_Bd2JpsiKstar':
      if CORR == 'Noncorr':
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}'
      mode = 'BdMC'
      c = 'b'
    elif m == 'Bd2JpsiKstar':
      if CORR == 'Noncorr':
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}{sw}'
      mode = 'BdRD'
      c = 'c'
    print(weight)

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time', lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)
    print(cats[mode])
    print(cats[mode].weight)

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
         'free': True if j > 0 else False,  # 'min':0.10, 'max':5.0
         } for j in range(len(knots[:-1]) + 2)
    ])
    cats[mode].params.add({'name': f'gamma_{c}',
                           'value': Gdvalue + resolutions[m]['DGsd'],
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
    cats[mode].tabs_path = otables[i]

  # Configure kernel -----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels()

  # Time to fit ----------------------------------------------------------------
  print(f"\n{80*'='}\nSimultaneous minimization procedure\n{80*'='}\n")
  fcn_call = fcns.saxsbxerf
  fcn_pars = cats['BsMC'].params + cats['BdMC'].params
  fcn_kwgs = {
      'data': [cats['BsMC'].time, cats['BdMC'].time],
      'prob': [cats['BsMC'].lkhd, cats['BdMC'].lkhd],
      'weight': [cats['BsMC'].weight, cats['BdMC'].weight]
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)

  if MINER.lower() in ("minuit", "minos"):
    result = mini.optimize(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                           method='minuit', verbose=True, tol=0.05)
  elif MINER.lower() in ('bfgs', 'lbfgsb', 'nelder'):
    _res = optimize(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                    method='nelder', verbose=False)
    result = mini.optimize(fcn_call=fcn_call, params=_res.params, fcn_kwgs=fcn_kwgs,
                           method=MINER, verbose=False)
  elif MINER.lower() in ('emcee'):
    result = mini.optimize(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                           method='minuit', verbose=True, tol=0.05)
    result = mini.optimize(fcn_call=fcn_call, params=result.params, fcn_kwgs=fcn_kwgs,
                           method=MINER, verbose=False, steps=1000, nwalkers=100, behavior='chi2')
    import corner
    fig = corner.corner(result.flatchain)
    fig.savefig('myshit.pdf')
  print(result)
  for p in result.params:
    if 'c' in p:
      print(f"{result.params[p].uvalue:.4f}")

  __pars = []
  for p in result.params:
    if result.params[p].free:
      __pars.append(p)
  fig, ax = plot_contours(mini, result, __pars)
  #fig, ax = plot_contours(mini, result, [f"c{i}b" for i in range(2,4)])
  fig.savefig('dsafsdafdsafdsafdsfsdf.pdf')
  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\nDumping parameters\n{80*'='}\n")

  for name, cat in zip(cats.keys(), cats.values()):
    list_params = cat.params.find('(a|b|c)(\d{1})(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])

    print(f"Dumping tex table to {cats[name].tabs_path}")
    with open(cat.tabs_path, "w") as text:
      text.write(cat.params.dump_latex(caption=f"Time acceptance for the $\
      {mode_tex(f'{MODE}')}$ ${YEAR}$ {TRIGGER} category in simultaneous fit."))
    text.close()

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cats[name].pars_path)

################################################################################
# that's all folks!
