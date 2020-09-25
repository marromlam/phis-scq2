# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import os, sys
import hjson

from ipanema import initialize
from ipanema import ristra
from ipanema import Parameters, optimize
from ipanema import Sample

from utils.plot import mode_tex
from utils.strings import cuts_and
from utils.helpers import  version_guesser

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(
    description='Compute decay-time acceptance for one single dataset.')
  # Sample
  parser.add_argument('--sample',
    default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
    help='Full path to sample')
  # Output parameters
  parser.add_argument('--input-params',
    default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
    help='Path to input set of parameters')
  # Output parameters
  parser.add_argument('--output-params',
    default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
    help='Path to output set of parameters')
  parser.add_argument('--output-tables',
    default = 'output/time_acceptance/tables/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
    help='Path to output table of parameters in tex format')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
    default = 'MC_BdJpsiKstar',
    help='Mode to fit')
  parser.add_argument('--year',
    default = '2016',
    help='Year to fit')
  parser.add_argument('--version',
    default = 'v0r1',
    help='Version of the tuples to use')
  parser.add_argument('--trigger',
    default = 'biased',
    help='Trigger to fit, choose between comb, biased and unbiased')

  return parser

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

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'],1)
  from time_acceptance.fcn_functions import splinexerf, trigger_scissors

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {args['trigger']:50}")
  print(f"{'cut':>15}: {CUT:50}\n")



  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\n", "Loading category", f"\n{80*'='}\n")

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
    cats[name] = Sample.from_root(sample, cuts=CUT, share=SHARE)
    cats[name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+TRIGGER
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
    cats[name].pars_path = args[f'output_params']
    cats[name].tabs_path = args[f'output_tables']

  cat = cats[list(cats.keys())[0]]

  # Time to fit ----------------------------------------------------------------
  print(f"\n{80*'='}\n", "Minimization procedure", f"\n{80*'='}\n")
  selected_method = "minuit"

  result = optimize(fcn_call=splinexerf,
              method= selected_method,
              params=cat.params,
              fcn_kwgs={'data':cat.time, 'prob':cat.lkhd, 'weight':cat.weight},
              verbose=False, strategy=1);

  print(result)

  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Dumping parameters", f"\n{80*'='}\n")

  cat.params = cat.knots + result.params
  print(f"Dumping json parameters to {cats[name].pars_path}")
  cat.params.dump(cats[name].pars_path)

  print(f"Dumping tex table to {cats[name].tabs_path}")
  with open(cat.tabs_path, "w") as text:
    text.write( cat.params.dump_latex(caption=f"Time acceptance for the \
    ${mode_tex(f'{MODE}')}$ ${YEAR}$ {TRIGGER} category in single fit.") )
  text.close()

################################################################################
# that's all folks!
