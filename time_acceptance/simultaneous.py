# -*- coding: utf-8 -*-


__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']


################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import os, sys

from ipanema import initialize
from ipanema import ristra
from ipanema import Parameters, optimize
from ipanema import Sample

from utils.plot import mode_tex
from utils.strings import cammel_case_split

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--BsMC-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r0.root',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bd2JpsiKstar/v0r0.root',
    help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/Bd2JpsiKstar/v0r0.root',
    help='Bd2JpsiKstar data sample')
  # Output parameters
  parser.add_argument('--BsMC-input-params',
    default = 'time_acceptance/params/2016/MC_Bs2JpsiPhi_dG0/baseline.json',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-input-params',
    default = 'time_acceptance/params/2016/MC_Bd2JpsiKstar/baseline.json',
    help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-input-params',
    default = 'time_acceptance/params/2016/Bd2JpsiKstar/baseline.json',
    help='Bd2JpsiKstar data sample')
  # Output parameters
  parser.add_argument('--BsMC-output-params',
    default = 'output_new/params/time_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r0_baseline_biased.json',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-output-params',
    default = 'output_new/params/time_acceptance/2016/MC_Bd2JpsiKstar/v0r0_baseline_biased.json',
    help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-output-params',
    default = 'output_new/params/time_acceptance/2016/Bd2JpsiKstar/v0r0_baseline_biased.json',
    help='Bd2JpsiKstar data sample')
  # Output tables
  parser.add_argument('--BsMC-output-tables',
    default = 'output_new/params/time_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r0_baseline_biased.json',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--BdMC-output-tables',
    default = 'output_new/params/time_acceptance/2016/MC_Bd2JpsiKstar/v0r0_baseline_biased.json',
    help='Bd2JpsiKstar MC sample')
  parser.add_argument('--BdDT-output-tables',
    default = 'output_new/params/time_acceptance/2016/Bd2JpsiKstar/v0r0_baseline_biased.json',
    help='Bd2JpsiKstar data sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
    default = 'BdJpsiKstar',
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
  parser.add_argument('--script',
    default = 'baseline',
    help='Different flag to ... ')

  return parser

if __name__ != '__main__':
  #import bsjpsikk # old
  import badjanak as bsjpsikk # charming new

################################################################################



################################################################################
#%% Likelihood functions to minimize ###########################################

def saxsbxscxerf(params, data, weight=False, prob=None):
  pars = params.valuesdict()
  if not prob:
    samples = list( map(ristra.allocate,data) )
    prob = list( map(ristra.zeros_like,samples) )
    bsjpsikk.saxsbxscxerf(*samples, *prob, **pars)
    return [ p.get() for p in prob ]
  else:
    bsjpsikk.saxsbxscxerf(*data, *prob, **pars)
    if weight:
      result  = np.concatenate(( (ristra.log(prob[0])*weight[0]).get(),
                                 (ristra.log(prob[1])*weight[1]).get(),
                                 (ristra.log(prob[2])*weight[2]).get() ))
    else:
      result  = np.concatenate(( ristra.log(prob[0]).get(),
                                 ristra.log(prob[1]).get(),
                                 ristra.log(prob[2]).get() ))

    return -2*result#.sum()

# Bins of different varaibles
bin_vars = dict(
pt = ['(B_PT >= 0 & B_PT < 3.8e3)', '(B_PT >= 3.8e3 & B_PT < 6e3)', '(B_PT >= 6e3 & B_PT <= 9e3)', '(B_PT >= 9e3)'],
eta = ['(eta >= 0 & eta <= 3.3)', '(eta >= 3.3 & eta <= 3.9)', '(eta >= 3.9 & eta <= 6)'],
sigmat = ['(sigmat >= 0 & sigmat <= 0.031)', '(sigmat >= 0.031 & sigmat <= 0.042)', '(sigmat >= 0.042 & sigmat <= 0.15)']
)
################################################################################




################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':
  # Parse arguments
  args = vars(argument_parser().parse_args())
  VERSION = args['version']
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  FLAGS = cammel_case_split(args['script'])
  SCRIPT = FLAGS[0]; FLAGS.pop(0)
  if len(FLAGS) == 2:
    MINER = FLAGS[1]
    BINVAR, BIN = FLAGS[0][3:-1], FLAGS[0][-1]
    CUT = bin_vars[f'{BINVAR}'][int(BIN)-1]
  elif len(FLAGS) == 1:
    if FLAGS[0].startswith('Bin'):
      MINER = 'minuit';
      BINVAR, BIN = FLAGS[0][3:-1], FLAGS[0][-1]
      CUT = bin_vars[f'{BINVAR}'][int(BIN)-1]
    else:
      MINER = FLAGS[0]
      BINVAR, BIN = None, None
      CUT = None
  else:
    MINER = "minuit"
    BINVAR, BIN = None, None
    CUT = None
  print(f"SCRIPT = {SCRIPT}")

  # Initialize backend
  the_backend = os.environ['IPANEMA_BACKEND']
  try: # Odd years to gpu 1 and even ones to 2 (if there are 2 GPUs)
    initialize(the_backend, 1 if YEAR in (2015,2017) else 1)
  except: # Only one GPU :'(
    initialize(the_backend, 1)

  #import bsjpsikk # old
  import badjanak as bsjpsikk # charming new

  # Select trigger to fit                         WARNING, use pars to set cuts!
  if args['trigger'] == 'biased':
    trigger = 'biased'; cuts = "time>=0.3 & time<=15 & hlt1b==1"
  elif args['trigger'] == 'unbiased':
    trigger = 'unbiased'; cuts = "time>=0.3 & time<=15 & hlt1b==0"
  elif args['trigger'] == 'combined':
    trigger = 'combined'; cuts = "time>=0.3 & time<=15"

  # Print settings
  shitty = f'{VERSION}_{trigger}_single'
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {args['trigger']:50}")
  print(f"{'cuts':>15}: {cuts:50}")
  print(f"{'script':>15}: {SCRIPT.title():50}")
  print(f"{'cuts':>15}: {CUT if CUT else 'None':50}")
  print(f"{'minimizer':>15}: {MINER.title():50}\n")



  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  samples = {}
  samples['BsMC'] = os.path.join(args['BsMC_sample'])
  samples['BdMC'] = os.path.join(args['BdMC_sample'])
  samples['BdDT'] = os.path.join(args['BdDT_sample'])

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
      samplecut = f"({cuts})"
    elif name == 'BdMC':
      label = (r'\mathrm{MC}',r'B^0')
      if SCRIPT == 'base':
        weight='sw*polWeight*pdfWeight*kinWeight'
      elif SCRIPT == 'nonkin':
        weight='sw'
      samplecut = f"({cuts}) {f'&({CUT})' if CUT else ' '}"
    elif name == 'BdDT':
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

  # Time to fit ----------------------------------------------------------------
  print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
  fcn_pars = cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params
  fcn_kwgs={
    'data': [cats['BsMC'].time, cats['BdMC'].time, cats['BdDT'].time],
    'prob': [cats['BsMC'].lkhd, cats['BdMC'].lkhd, cats['BdDT'].lkhd],
    'weight': [cats['BsMC'].weight, cats['BdMC'].weight, cats['BdDT'].weight]
  }
  if MINER.lower() in ("minuit","minos"):
    result = optimize(fcn_call=saxsbxscxerf, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                      method=MINER,
                      verbose=False, strategy=1, tol=0.05);
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
