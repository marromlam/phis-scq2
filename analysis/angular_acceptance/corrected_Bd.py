from hep_ml import reweight
from warnings import simplefilter
import badjanak
from utils.helpers import version_guesser, timeacc_guesser, trigger_scissors
from utils.strings import cammel_case_split, cuts_and
from utils.plot import mode_tex
from ipanema import ristra, Sample, Parameters
from ipanema import initialize
import hjson
import numpy as np
import sys
import os
import argparse
__all__ = []
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__ = ['mromerol@cern.ch']


################################################################################
# %% Modules ###################################################################


# load ipanema
initialize(config.user['backend'], 1)

# import some phis-scq utils
# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

# reweighting config
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)

# 40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000

# Parse arguments for this script


def argument_parser():
  p = argparse.ArgumentParser(description='Compute angular acceptance.')
  p.add_argument('--sample-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data', help='Bs2JpsiPhi data sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-file', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Configuration')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--trigger', help='Trigger(s) to fit [comb/(biased)/unbiased]')
  p.add_argument('--binvar', help='Different flag to ... ')
  return p


def printsec(string):
  print(f"\n{80*'='}\n{string}\n{80*'='}\n")

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
  #initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else -1)

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {trigger_scissors(TRIGGER, CUT):50}")
  print(f"{'angacc':>15}: {'corrected':50}")
  print(f"{'bdtconfig':>15}: {list(bdconfig.values())}\n")

  # %% Load samples ------------------------------------------------------------
  printsec("Loading categories")

  # Load Monte Carlo samples
  mc = Sample.from_root(args['sample_mc'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'])
  kinWeight = np.zeros_like(list(mc.df.index)).astype(np.float64)
  mc.chop(trigger_scissors(TRIGGER, CUT))
  print(mc.df[['X_M', 'B_P', 'B_PT']])
  # Load corresponding data sample
  rd = Sample.from_root(args['sample_data'], share=SHARE, name='data')
  rd.chop(trigger_scissors(TRIGGER, CUT))
  print(rd.df[['X_M', 'B_P', 'B_PT']])
  exit()

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  #true = [f'true{i}_GenLvl' for i in reco]
  weight_rd = f'(sw_{VAR})' if VAR else '(sw)'
  weight_mc = f'(polWeight*{weight_rd}/gb_weights)'
  print(weight_mc, weight_rd)
  # Allocate some arrays with the needed branches
  mc.allocate(reco=reco + ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
  mc.allocate(true=true + ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
  mc.allocate(pdf='0*time', ones='time/time', zeros='0*time')
  mc.allocate(weight=weight_mc)

  # %% Compute standard kinematic weights ---------------------------------------
  #     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
  #     variables
  printsec('Compute angWeights correcting MC sample in kinematics')
  print(f" * Computing kinematic GB-weighting in B_PT, B_P and X_M")

  reweighter.fit(original=mc.df[['X_M', 'B_P', 'B_PT']],
                 target=rd.df[['X_M', 'B_P', 'B_PT']],
                 original_weight=mc.df.eval(weight_mc),
                 target_weight=rd.df.eval(weight_rd))
  angWeight = reweighter.predict_weights(mc.df[['X_M', 'B_P', 'B_PT']])
  kinWeight[list(mc.df.index)] = angWeight

  print(f"{'idx':>3} | {'sw':>11} | {'polWeight':>11} | {'angWeight':>11} ")
  for i in range(0, 100):
    if kinWeight[i] != 0:
      print(f"{str(i):>3} | {mc.df['sWeight'][i]:+.8f} | {mc.df['polWeight'][i]:+.8f} | {kinWeight[i]:+.8f} ")

  np.save(args['output_weights_file'], kinWeight)

  # %% Compute angWeights correcting with kinematic weights ---------------------
  #     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
  #     variables
  print(" * Computing angular weights")

  angacc = badjanak.get_angular_cov(mc.true, mc.reco,
                                    mc.weight * ristra.allocate(angWeight),
                                    **mc.params.valuesdict())

  w, uw, cov, corr = angacc
  pars = Parameters()
  for i in range(0, len(w)):
    #print(f'w[{i}] = {w[i]:+.16f}')
    correl = {f'w{j}': cov[i][j] for j in range(0, len(w)) if i > 0 and j > 0}
    pars.add({'name': f'w{i}', 'value': w[i], 'stdev': uw[i], 'correl': correl,
              'free': False, 'latex': f'w_{i}'})
  print(f" * Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")

  print(f"{pars}")

  # Writing results ------------------------------------------------------------
  printsec("Dumping parameters")
  # Dump json file
  print(f"Dumping json parameters to {args['output_params']}")
  pars.dump(args['output_params'])
  # Export parameters in tex tables
  print(f"Dumping tex table to {args['output_tables']}")
  with open(args['output_tables'], "w") as tex_file:
    tex_file.write(
        pars.dump_latex(caption="""
      Kinematically corrected angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR, TRIGGER, MODE.replace('_', ' '))
        )
    )
  tex_file.close()
