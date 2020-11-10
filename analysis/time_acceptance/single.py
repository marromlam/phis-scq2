# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero Lamas']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import argparse
import os
import hjson
import numpy as np

# load ipanema
from ipanema import initialize
from ipanema import ristra, Parameters, optimize, Sample

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and
from utils.helpers import  version_guesser, timeacc_guesser
from utils.helpers import  swnorm, trigger_scissors

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description='Compute single decay-time acceptance.')
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
  initialize(os.environ['IPANEMA_BACKEND'],1)
  from time_acceptance.fcn_functions import splinexerf

################################################################################



################################################################################
#%% Run and get the job done ###################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC, CORR, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'],1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
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
  if TIMEACC == '9knots':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  elif TIMEACC == '12knots':
    knots = [0.30, 0.43, 0.58, 0.74, 0.91, 1.11,
             1.35, 1.63, 1.96, 2.40, 3.01, 4.06, 9.00, 15.0]
    kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'



  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\nLoading category\n{80*'='}\n")

  # Select samples
  cats = {}
  sw = f'sw_{VAR}' if VAR else 'sw'
  for i, m in enumerate([MODE]):
    # Correctly apply weight and name for diffent samples
    if m=='MC_Bs2JpsiPhi':
      if CORR=='Noncorr':
        weight = f'dg0Weight*{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*dg0Weight*{sw}/gb_weights'
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bs2JpsiPhi_dG0':
      if CORR=='Noncorr':
        weight = f'{sw}/gb_weights'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bd2JpsiKstar':
      if CORR=='Noncorr':
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}'
      mode = 'BdMC'; c = 'b'
    elif m=='Bd2JpsiKstar':
      if CORR=='Noncorr':
        weight = f'{sw}'
      else:
        weight = f'{kinWeight}{sw}'
      mode = 'BdRD'; c = 'c'
    print(weight)

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time', lkhd='0*time')
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
                    {'name':f'{c}{j}{TRIGGER[0]}', 'value':1.0,
                     'latex':f'{c}_{j}^{TRIGGER[0]}',
                     'free':True if j > 0 else False, 'min':0.10, 'max':3.0
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
    cats[mode].tabs_path = otables[i]



  # Configure kernel -----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)


  # Time to fit ----------------------------------------------------------------
  print(f"\n{80*'='}\nMinimization procedure\n{80*'='}\n")
  fcn_call = fcns.splinexerf
  fcn_pars = cats[mode].params
  fcn_kwgs = {
      'data': cats[mode].time,
      'prob': cats[mode].lkhd,
      'weight': cats[mode].weight
  }
  if MINER.lower() in ("minuit", "minos"):
    result = optimize(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                      method=MINER, verbose=True, timeit=True, tol=0.05)
  elif MINER.lower() in ('bfgfs', 'lbfgsb'):
    result = optimize(fcn_call=splinexerf, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                      method=MINER, verbose=False)
  print(result)





  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\nDumping parameters\n{80*'='}\n")

  cats[mode].params = cats[mode].knots + result.params

  print(f"Dumping json parameters to {cats[mode].pars_path}")
  cats[mode].params.dump(cats[mode].pars_path)

  print(f"Dumping tex table to {cats[mode].tabs_path}")
  with open(cats[mode].tabs_path, "w") as text:
    text.write( cats[mode].params.dump_latex(caption=f"Time acceptance for the \
    {VERSION} ${mode_tex(f'{MODE}')}$ ${YEAR}$ {TRIGGER} category in {TIMEACC} fit."))
  text.close()

################################################################################
# that's all folks!
