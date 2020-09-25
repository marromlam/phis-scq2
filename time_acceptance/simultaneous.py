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
  parser.add_argument('--input-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--year', help='Year to fit')
  parser.add_argument('--version', help='Version of the tuples to use')
  parser.add_argument('--trigger', help='Trigger to fit')
  parser.add_argument('--timeacc', help='Different flag to ... ')
  return parser

if __name__ != '__main__':
  #import bsjpsikk # old
  import badjanak as bsjpsikk # charming new

################################################################################



################################################################################
#%% Likelihood functions to minimize ###########################################



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
  TIMEACC, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else -1)
  from time_acceptance.fcn_functions import saxsbxscxerf, trigger_scissors

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  iparams = args['input_params'].split(',')
  oparams = args['output_params'].split(',')
  otables = args['output_tables'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  if TIMEACC == 'simul':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
    kinWeight = 'kinWeight*'
  elif TIMEACC == 'nonkin':
    knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0],
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
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bs2JpsiPhi_dG0':
      weight = f'{kinWeight}polWeight*pdfWeight*sw/gb_weights'
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bd2JpsiKstar':
      weight = f'{kinWeight}polWeight*pdfWeight*sw'
      mode = 'BdMC'; c = 'b'
    elif m=='Bd2JpsiKstar':
      weight = f'{kinWeight}{sw}'
      mode = 'BdRD'; c = 'c'

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time',lkhd='0*time')
    cats[mode].allocate(weight=weight)

    # Add knots
    cats[mode].knots = Parameters()
    cats[mode].knots.add(*[
                 {'name':f'k{j}', 'value':v, 'latex':f'k_{j}', 'free':False}
                 for j,v in enumerate(knots[:-1])
               ])
    cats[mode].knots.add({'name':f'tLL', 'value':knots[0],
                          'latex':'t_{ll}', 'free':False})
    cats[mode].knots.add({'name':f'tUL', 'value':knots[-1],
                          'latex':'t_{ul}', 'free':False})

    # Add coeffs parameters
    cats[mode].params = Parameters()
    cats[mode].params.add(*[
                 {'name':f'{c}{j}', 'value':1.0, 'latex':f'{c}_{j}',
                 'free':True if j>0 else False, 'min':0.10, 'max':15.0}
                 for j in range(len(knots[:-1])+2)
               ])
    cats[mode].params.add({'name':f'gamma_{c}',
                           'value':Gdvalue+resolutions[m]['DGsd'],
                           'latex':f'\gamma_{c}', 'free':False})
    cats[mode].params.add({'name':f'mu_{c}',
                           'value':resolutions[m]['mu'],
                           'latex':f'\mu_{c}', 'free':False})
    cats[mode].params.add({'name':f'sigma_{c}',
                           'value':resolutions[m]['sigma'],
                           'latex':f'\sigma_{c}', 'free':False})
    print(cats[mode].knots)
    print(cats[mode].params)

    # Attach labels and paths
    cats[mode].label = mode_tex(mode)
    cats[mode].pars_path = oparams[i]
    cats[mode].tabs_path = otables[i]

  """
  exit()
  cats = {}
  for name, sample in zip(samples.keys(),samples.values()):
    print(f'Loading {sample} as {name} category')
    name = name[:4] # remove _sample
    if name == 'BsMC':
      label = (r'\mathrm{MC}',r'B_s^0')
      if SCRIPT == 'base':
        weight='(sw/gb_weights)*polWeight*pdfWeight*kinWeight'
      elif SCRIPT == 'nonkin':
        weight='(sw/gb_weights)'
      samplecut = f"({cuts}) {f'&({CUT})' if CUT else ' '}"
    elif name == 'BdMC':
      label = (r'\mathrm{MC}',r'B^0')
      if SCRIPT == 'base':
        weight='sw*polWeight*pdfWeight*kinWeight'
      elif SCRIPT == 'nonkin':
        weight='sw'
      samplecut = f"({cuts}) {f'&({CUT})' if CUT else ' '}"
    elif name == 'BdRD':
      label = (r'\mathrm{data}',r'B_s^0')
      if SCRIPT == 'base':
        weight='sw*kinWeight'
      elif SCRIPT == 'nonkin':
        weight='sw'
      samplecut = f"({cuts}) {f'&({CUT})' if CUT else ' '}"
    #print(samplecut)
    cats[name] = Sample.from_root(sample, cuts=samplecut)
    cats[name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+trigger
    #print(cats[name].df)
    cats[name].allocate(time='time',lkhd='0*time')
    cats[name].allocate(weight=weight)
    cats[name].weight *= ristra.sum(cats[name].weight)/ristra.sum(cats[name].weight**2)
    cats[name].assoc_params(args[f'{name}_input_params'])
    knots = cats[name].params.find('k.*') + ['tLL','tUL']
    cats[name].knots = Parameters.build(cats[name].params, knots)
    [cats[name].params.pop(k, None) for k in knots]
    for p in cats[name].params:
      if p.startswith('a') or p.startswith('b') or p.startswith('c'):
        cats[name].params[p].value = 1.0
        cats[name].params[p].init = 1.0
        cats[name].params[p].min = 0.1
        cats[name].params[p].max = 10.0
    print(cats[name].params)
    print(cats[name].knots)
    cats[name].label = label
    cats[name].pars_path = args[f'{name}_output_params']
    cats[name].tabs_path = args[f'{name}_output_tables']
  """
  # Time to fit ----------------------------------------------------------------
  print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
  fcn_pars = cats['BsMC'].params+cats['BdMC'].params+cats['BdRD'].params
  fcn_kwgs={
    'data': [cats['BsMC'].time, cats['BdMC'].time, cats['BdRD'].time],
    'prob': [cats['BsMC'].lkhd, cats['BdMC'].lkhd, cats['BdRD'].lkhd],
    'weight': [cats['BsMC'].weight, cats['BdMC'].weight, cats['BdRD'].weight]
  }
  if MINER.lower() in ("minuit","minos"):
    result = optimize(fcn_call=saxsbxscxerf, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                      method=MINER,
                      verbose=True, strategy=1, tol=0.1);
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    result = optimize(fcn_call=saxsbxscxerf, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                      method=MINER,
                      verbose=False);

  print(result)
  #for k,v in result.params.items():
  #  print(f"{k:>10} : {v.value:+.8f} +/- {(v.stdev if v.stdev else 0):+.8f}")

  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Dumping parameters", f"\n{80*'='}\n")

  for name, cat in zip(cats.keys(),cats.values()):
    list_params = [par for par in cat.params if len(par) ==2]
    cat.params.add(*[result.params.get(par) for par in list_params])
    cat.params = cat.knots + cat.params

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params.dump(cats[name].pars_path)

    print(f"Dumping tex table to {cats[name].tabs_path}")
    with open(cat.tabs_path, "w") as text:
      text.write( cat.params.dump_latex( caption="""
      Time acceptance for the \\textbf{%s} $%s$ \\texttt{\\textbf{%s}} $%s$
      category in simultaneous fit.""" % (YEAR,cat.label[1],TRIGGER,cat.label[0]) ) )
    print( cat.pars_path )

################################################################################
# that's all folks!
