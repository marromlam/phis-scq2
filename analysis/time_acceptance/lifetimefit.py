# -*- coding: utf-8 -*-

from utils.helpers import version_guesser, timeacc_guesser, swnorm, trigger_scissors
from utils.strings import cammel_case_split, cuts_and
from utils.plot import mode_tex
from ipanema import ristra, Parameters, optimize, Sample
from ipanema import initialize
import hjson
import numpy as np
import sys
import os
import argparse
__all__ = []
__author__ = ['Marcos Romero']
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
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  parser.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--year', help='Year to fit')
  parser.add_argument('--version', help='Version of the tuples to use')
  parser.add_argument('--timeacc', help='Different flag to ... ')
  return parser


if __name__ != '__main__':
  import badjanak

################################################################################


"""
here i put current set of lifetime


args = {
  "sample": "/scratch17/marcos.romero/sidecar/2015/MC_Bd2JpsiKstar/v0r5.root,/scratch17/marcos.romero/sidecar/2015/Bd2JpsiKstar/v0r5.root",
  "output_params": "output/params/time_acceptance/2015/Bd2JpsiKstar/v0r5_lifetimefit_biased.json",
  "output_tables": "output/tables/time_acceptance/2015/Bd2JpsiKstar/v0r5_lifetimefit_biased.tex",
  "year": "2015",
  "version": "v0r5",
  "trigger": "biased",
  "timeacc": "simul",
}

"""


################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':

  # %% Parse arguments ----------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = 'Bd2JpsiKstar'
  TIMEACC, MINER = 'lifetime', 'minuit'

  # Get badjanak model and configure it
  initialize(config.user['backend'], 1 if YEAR in (2015, 2017) else 1)
  from time_acceptance.fcn_functions import saxsbxscxerf, fcn_test

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}\n")

  # List samples, params and tables
  samples = args['sample'].split(',')
  oparams = args['output_params'].split(',')
  otables = args['output_tables'].split(',')
  # Check timeacc flag to set knots and weights and place the final cut
  knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
  kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  if TIMEACC == 'nonkin':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = ''
  elif TIMEACC == '9knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  elif TIMEACC == '12knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'

  # %% Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  cats = {}
  sw = f'sw_{VAR}' if VAR else 'sw'
  for i, m in enumerate(['MC_Bd2JpsiKstar', 'Bd2JpsiKstar']):
    if m == 'MC_Bs2JpsiPhi':
      weight = f'{kinWeight}polWeight*pdfWeight*dg0Weight*{sw}/gb_weights'
      weight = f'{sw}/gb_weights'
      mode = 'BsMC'
      c = 'a'
    elif m == 'MC_Bs2JpsiPhi_dG0':
      weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      weight = f'{sw}/gb_weights'
      mode = 'BsMC'
      c = 'a'
    elif m == 'MC_Bd2JpsiKstar':
      weight = f'{kinWeight}polWeight*pdfWeight*{sw}'
      weight = f'{sw}'
      mode = 'BdMC'
      c = 'b'
    elif m == 'Bd2JpsiKstar':
      weight = f'{kinWeight}{sw}'
      weight = f'{sw}'
      mode = 'BdRD'
      c = 'c'

    cats[mode] = {}
    for t in ['biased', 'unbiased']:
      cats[mode][t] = {}
      for f, F in zip(['A', 'B'], ['(evtN % 2) == 0', '(evtN % 2) != 0']):
        cats[mode][t][f] = Sample.from_root(samples[i], share=SHARE, name=f"{mode}-{t}-{f}")
        cats[mode][t][f].chop(cuts_and(trigger_scissors(t), F, CUT))

        # allocate arrays
        cats[mode][t][f].allocate(time='time', lkhd='0*time')
        cats[mode][t][f].allocate(weight=weight)
        cats[mode][t][f].weight = swnorm(cats[mode][t][f].weight)
        print(cats[mode][t][f])

        # Add knots
        cats[mode][t][f].knots = Parameters()
        cats[mode][t][f].knots.add(*[
            {'name': f'k{j}', 'value': v, 'latex': f'k_{j}', 'free': False}
            for j, v in enumerate(knots[:-1])
        ])
        cats[mode][t][f].knots.add({'name': f'tLL', 'value': knots[0],
                                    'latex': 't_{ll}', 'free': False})
        cats[mode][t][f].knots.add({'name': f'tUL', 'value': knots[-1],
                                    'latex': 't_{ul}', 'free': False})

        # Add coeffs parameters
        cats[mode][t][f].params = Parameters()
        cats[mode][t][f].params.add(*[
            {'name': f'{c}{f}{j}{t[0]}', 'value': 1.0, 'latex': f'{c}_{f,j}^{t[0]}',
                     'free': True if j > 0 else False, 'min': 0.10, 'max': 5.0}
            for j in range(len(knots[:-1]) + 2)
        ])
        cats[mode][t][f].params.add({'name': f'gamma_{f}{c}',
                                     'value': Gdvalue + resolutions[m]['DGsd'],
                                     'latex': f'\Gamma_{f,c}', 'free': False})
        cats[mode][t][f].params.add({'name': f'mu_{f}{c}',
                                     'value': resolutions[m]['mu'],
                                     'latex': f'\mu_{f,c}', 'free': False})
        cats[mode][t][f].params.add({'name': f'sigma_{f}{c}',
                                     'value': resolutions[m]['sigma'],
                                     'latex': f'\sigma_{f,c}', 'free': False})
        # print(cats[mode][t][f].knots)
        # print(cats[mode][t][f].params)

        # Attach labels and paths
        cats[mode][t][f].label = mode_tex(mode)
        #cats[mode][t][f].pars_path = oparams[i]
        #cats[mode][t][f].tabs_path = otables[i]

  # %% Time to fit acceptance -----------------------------------------------------
  print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")

  result = {}
  for t in ['biased', 'unbiased']:
    print(cats['BdMC'][t])
    fcn_pars = cats['BdMC'][t]['A'].params + cats['BdMC'][t]['B'].params + cats['BdRD'][t]['B'].params
    fcn_kwgs = {
        'data': [cats['BdMC'][t]['A'].time, cats['BdMC'][t]['B'].time, cats['BdRD'][t]['B'].time],
        'prob': [cats['BdMC'][t]['A'].lkhd, cats['BdMC'][t]['B'].lkhd, cats['BdRD'][t]['B'].lkhd],
        'weight': [cats['BdMC'][t]['A'].weight, cats['BdMC'][t]['B'].weight, cats['BdRD'][t]['B'].weight]
    }

    if MINER.lower() in ("minuit", "minos"):
      result[t] = optimize(fcn_call=saxsbxscxerf,
                           params=fcn_pars,
                           fcn_kwgs=fcn_kwgs,
                           method=MINER,
                           verbose=True, timeit=True, strategy=1, tol=0.05)
    elif MINER.lower() in ('bfgs', 'lbfgsb'):
      result[t] = optimize(fcn_call=saxsbxscxerf,
                           params=fcn_pars,
                           fcn_kwgs=fcn_kwgs,
                           method=MINER,
                           verbose=True, timeit=True)

    print(result[t])
    print(result[t]._minuit.latex_matrix())

  # create ipanema.Parameters for lifetime fit (mu, sigma locked else free)
  #pars = cats['BdRD']['biased']['B'].params+cats['BdRD']['unbiased']['B'].params
  # pars.lock()
  pars = cats['BdRD']['biased']['A'].params + cats['BdRD']['unbiased']['A'].params
  # pars.lock()
  pars['gamma_Ac'].set(value=0.5, min=0.0, max=1, free=True)

  # clonning and locking paramters
  pars = Parameters.clone(pars)
  cats['BdRD']['unbiased']['A'].params = Parameters.clone(result['unbiased'].params)
  cats['BdRD']['biased']['A'].params = Parameters.clone(result['biased'].params)
  cats['BdRD']['unbiased']['A'].params.lock()
  cats['BdRD']['biased']['A'].params.lock()

  print(pars)
  print(result['unbiased'].params)
  print(result['biased'].params)

  for row in result['unbiased'].params.corr():
    for col in row:
      print(f"{col:.4f}  ", end='')
    print('\n', end='')
  # print(result['unbiased'].params.corr())
  # print(result['biased'].params.corr())
  MINOS = 'minos'

  # lifetime fit
  if MINER.lower() in ("minuit", "minos"):
    lifefit = optimize(fcn_call=fcn_test, params=pars,
                       fcn_kwgs={'cats': {
                           'biased': cats['BdRD']['biased']['A'],
                           'unbiased': cats['BdRD']['unbiased']['A']
                       },
                           'weight': True},
                       method=MINER,
                       verbose=True, strategy=1, tol=0.05)
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    0  # fix me!

  print(lifefit)
  print(lifefit.params.corr())

  print(f"tau = {1/lifefit.params['gamma_Ac'].uvalue:.2uL}")
  exit()
  # for k,v in result.params.items():
  #  print(f"{k:>10} : {v.value:+.8f} +/- {(v.stdev if v.stdev else 0):+.8f}")

  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Dumping parameters", f"\n{80*'='}\n")

  for name, cat in zip(cats.keys(), cats.values()):
    list_params = [par for par in cat.params if len(par) == 2]
    cat.params.add(*[result.params.get(par) for par in list_params])
    cat.params = cat.knots + cat.params

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params.dump(cats[name].pars_path)

    print(f"Dumping tex table to {cats[name].tabs_path}")
    with open(cat.tabs_path, "w") as text:
      text.write(cat.params.dump_latex(caption="""
      Time acceptance for the \\textbf{%s} $%s$ \\texttt{\\textbf{%s}} $%s$
      category in simultaneous fit.""" % (YEAR, cat.label[1], 0, cat.label[0])))
    print(cat.pars_path)

################################################################################
# that's all folks!
