from utils.helpers import version_guesser, trigger_scissors, parse_angacc
from utils.strings import printsec
import config
from hep_ml import reweight
from warnings import simplefilter
from analysis import badjanak
from analysis.angular_acceptance.iterative_mc import acceptance_effect
from analysis.angular_acceptance.bdtconf_tester import bdtmesh
__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


# Modules {{{

import numpy as np
import os
import argparse
import pandas as pd

# load ipanema
from ipanema import (ristra, Sample, Parameters, initialize)
initialize(config.user['backend'], 1)

# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0

# reweighting config -- ignore future warnings
simplefilter(action='ignore', category=FutureWarning)


# import some phis-scq utils
# from analysis.angular_acceptance.iterative_mc import acceptance_effect
# 40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000

# }}}


# Run and get the job done {{{

if __name__ == '__main__':

  # Parse arguments {{{

  DESCRIPTION = """
    Computes angular acceptance with corrections in mHH, pB, pTB variables
    using an a reweight.
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data', help='Bs2JpsiPhi data sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC generator parameters')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC angular acceptance')
  p.add_argument('--output-weights-file', help='angWeights file')
  p.add_argument('--mode', help='Mode to compute angular acceptance with')
  p.add_argument('--year', help='Year to compute angular acceptance with')
  p.add_argument('--version', help='Version of the tuples')
  p.add_argument(
      '--angacc', help='corrected only or with acceptance effects')
  p.add_argument(
      '--trigger', help='Trigger to compute angular acceptance with')
  args = vars(p.parse_args())

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(
      args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  ANGACC = parse_angacc(args['angacc'])

  # Prepare the cuts
  if ANGACC['use_truetime']:
    time = 'gentime'
  else:
    time = 'time'

  # }}}

  # setting upper and lower time limits {{{

  if 'UT' in args['version']:
    tLL = config.general['upper_time_lower_limit']
  else:
    tLL = config.general['time_lower_limit']

  if 'LT' in args['version']:
    tUL = config.general['lower_time_upper_limit']
  else:
    tUL = config.general['time_upper_limit']

  if 'T1' in args['version']:
    tLL, tUL = tLL, 0.9247
    badjanak.config['final_extrap'] = False
  elif 'T2' in args['version']:
    tLL, tUL = 0.9247, 1.9725
    badjanak.config['final_extrap'] = False
  elif 'T3' in args['version']:
    tLL, tUL = 1.9725, tUL
    # tLL, tUL = 2, tUL
  else:
    print("SAFE CUT")

  CUT = f'{time}>={tLL} & {time}<={tUL}'

  # if version has bdt in name, then lets change it
  if 'bdt' in VERSION:
    bdtconfig = int(VERSION.split('bdt')[1])
    bdtconfig = bdtmesh(bdtconfig, config.general['bdt_tests'], False)
  else:
    bdtconfig = config.angacc['bdtconfig']
  reweighter = reweight.GBReweighter(**bdtconfig)

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {trigger_scissors(TRIGGER, CUT):50}")
  print(f"{'angacc':>15}: {'corrected':50}")
  print(f"{'bdtconfig':>15}: {list(bdtconfig.values())}\n")

  # }}}

  # Load samples {{{

  printsec("Loading categories")

  if VERSION == 'v0r0':
    args['input_params'] = args['input_params'].replace(
        'generator', 'generator_old')

  # Load Monte Carlo samples
  mc = Sample.from_root(args['sample_mc'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'])
  kinWeight = np.zeros_like(list(mc.df.index)).astype(np.float64)
  # Load corresponding data sample
  rd = Sample.from_root(args['sample_data'], share=SHARE, name='data')
  if VERSION != 'v0r0':
    mc.chop(trigger_scissors(TRIGGER, CUT))
    rd.chop(trigger_scissors(TRIGGER, CUT))

  # print(mc.df[['sw', 'sWeight']])
  # print(rd.df[['sw', 'sWeight']])

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]

  # if not using bkgcat==60, then don't use sWeight
  weight_rd = 'sWeight'
  weight_mc = 'polWeight*sWeight'
  if VERSION == 'v0r0':
    weight_mc = f"({weight_mc})*({trigger_scissors(TRIGGER)})"
    weight_rd = f"({weight_rd})*({trigger_scissors(TRIGGER)})"
  # weight_rd = 'sw'
  # weight_mc = 'polWeight*sw'

  # if mode is from Bs family, then use gb_weights
  # if "bkgcat60" in args['version']:
  #     weight_mc = 'polWeight'
  #     weight_rd = 'time/time'
  if ANGACC['use_oddWeight']:
    weight_rd = f'{weight_rd}*oddWeight'
  if ANGACC['use_pTWeight']:
    pTp = np.array(rd.df['pTHp'])
    pTm = np.array(rd.df['pTHm'])
    pT_acc = np.ones_like(rd.df['pTHp'])
    for k in range(len(pT_acc)):
      pT_acc[k] = acceptance_effect(pTp[k], 250**3)
      pT_acc[k] *= acceptance_effect(pTm[k], 250**3)
    rd.df['pTWeight'] = pT_acc
    weight_rd = f'{weight_rd}*pTWeight'

  print(f"Using weight = {weight_mc} for MC")
  print(f"Using weight = {weight_rd} for data")

  # Allocate some arrays with the needed branches
  if 'Bs2Jpsi' in MODE:
    mc.allocate(reco=reco + ['mHH', '0*mHH', 'genidB',
                'genidB', '0*mHH', '0*mHH'])
    mc.allocate(true=true + ['mHH', '0*mHH', 'genidB',
                'genidB', '0*mHH', '0*mHH'])
  elif 'Bd2JpsiKstar' in MODE:
    mc.allocate(reco=reco + ['mHH', '0*mHH', 'idB', 'idB', '0*mHH', '0*mHH'])
    mc.allocate(true=true + ['mHH', '0*mHH', 'idB', 'idB', '0*mHH', '0*mHH'])
  mc.allocate(pdf='0*time')
  mc.allocate(weight=weight_mc)

  print('Simulation sample')
  print(pd.concat((mc.df[['mHH', 'pB', 'pTB']],
                   mc.df.eval(weight_mc)), axis=1))
  print('Data sample')
  print(pd.concat((rd.df[['mHH', 'pB', 'pTB']],
                   rd.df.eval(weight_rd)), axis=1))

  # }}}

  # Compute standard kinematic weights {{{
  #     This means compute the kinematic weights using 'mHH','pB' and 'pTB'
  #     variables

  printsec('Compute angWeights correcting MC sample in kinematics')
  print(" * Computing kinematic GB-weighting in pTB, pB and mHH")

  reweighter.fit(original=mc.df[['mHH', 'pB', 'pTB']],
                 target=rd.df[['mHH', 'pB', 'pTB']],
                 original_weight=mc.df.eval(weight_mc),
                 target_weight=rd.df.eval(weight_rd))
  angWeight = reweighter.predict_weights(mc.df[['mHH', 'pB', 'pTB']])
  angWeight = np.where(mc.df.eval(weight_mc) != 0, angWeight, 0)
  kinWeight[list(mc.df.index)] = angWeight

  print(f"{'idx':>3} | {'sw':>11} | {'polWeight':>11} | {'angWeight':>11} ")
  for i in range(0, 20):
    if kinWeight[i] != 0:
      print(f"{str(i):>3} | {mc.df.eval('sWeight')[i]:+.8f} |",
            f"{mc.df['polWeight'][i]:+.8f} | {kinWeight[i]:+.8f} ")

  np.save(args['output_weights_file'], kinWeight)

  # }}}

  # Compute angWeights correcting with kinematic weights {{{
  #     This means compute the kinematic weights using 'mHH','pB' and 'pTB'
  #     variables
  badjanak.get_kernels()
  if 'Bs2Jpsi' in MODE:
    angacc = badjanak.get_angular_acceptance_weights(
        mc.true, mc.reco, mc.weight * ristra.allocate(angWeight),
        **mc.params.valuesdict(), tLL=tLL, tUL=tUL)
  elif 'Bd2JpsiKstar' in MODE:
    angacc = badjanak.get_angular_acceptance_weights_Bd(
        mc.true, mc.reco, mc.weight * ristra.allocate(angWeight),
        **mc.params.valuesdict())
  w, uw, cov, corr = angacc
  pars = Parameters()
  for i in range(0, len(w)):
    correl = {f'w{j}{TRIGGER[0]}': corr[i][j]
              for j in range(0, len(w)) if i > 0 and j > 0}
    pars.add({'name': f'w{i}{TRIGGER[0]}', 'value': w[i], 'stdev': uw[i],
              'correl': correl, 'free': False,
              'latex': f'w_{i}^{TRIGGER[0]}'})

  print("Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")

  # }}}

  # Writing results {{{
  #    Exporting computed results
  printsec("Dumping parameters")
  # Dump json file
  print(f"Dumping json parameters to {args['output_params']}")
  pars.dump(args['output_params'])

  # }}}

# }}}


# vim: fdm=marker
