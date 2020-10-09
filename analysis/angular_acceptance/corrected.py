# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import argparse
import os
import sys
import numpy as np
import hjson

# load ipanema
from ipanema import initialize
from ipanema import ristra, Sample, Parameters
initialize(os.environ['IPANEMA_BACKEND'],1)

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and
from utils.helpers import  version_guesser, timeacc_guesser

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000

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
  from time_acceptance.fcn_functions import trigger_scissors

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  #CUT = trigger_scissors(TRIGGER, CUT)         # place cut attending to trigger

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  #print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {list(bdconfig.values())}\n")



  # %% Load samples ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  # Load Monte Carlo samples
  mc = Sample.from_root(args['sample_mc'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'])
  # Load corresponding data sample
  rd = Sample.from_root(args['sample_data'], share=SHARE, name='data')

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  weight_rd = f'(sw_{VAR})' if VAR else '(sw)'
  weight_mc = f'(polWeight*{weight_rd}/gb_weights)'
  print(weight_mc,weight_rd)

  # Select trigger
  if TRIGGER == 'biased':
    trigger = 'biased';
    weight_mc += f'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)*({CUT})'
    weight_rd += f'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)*({CUT})'
  elif TRIGGER == 'unbiased':
    trigger = 'unbiased';
    weight_mc += f'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*({CUT})'
    weight_rd += f'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*({CUT})'
  elif TRIGGER == 'combined':
    trigger = 'combined';

  # Allocate some arrays with the needed branches
  mc.allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
  mc.allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
  mc.allocate(pdf='0*time', ones='time/time', zeros='0*time')
  mc.allocate(weight=weight_mc)



  #%% Compute standard kinematic weights ---------------------------------------
  #     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
  #     variables
  print(f"\n{80*'='}\nCompute angWeights correcting MC sample in kinematics\n{80*'='}\n")
  print(f" * Computing kinematic GB-weighting in B_PT, B_P and X_M")

  reweighter.fit(original        = mc.df[['X_M','B_P','B_PT']],
                 target          = rd.df[['X_M','B_P','B_PT']],
                 original_weight = mc.df.eval(weight_mc),
                 target_weight   = rd.df.eval(weight_rd));

  kinWeight = reweighter.predict_weights(mc.df[['X_M','B_P','B_PT']])
  kinWeight = np.where(mc.df.eval(weight_mc)!=0, kinWeight, 0)
  np.save(args['output_weights_file'], kinWeight)

  #%% Compute angWeights correcting with kinematic weights ---------------------
  #     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
  #     variables
  print(" * Computing angular weights")

  ang_acc = badjanak.get_angular_cov(mc.true, mc.reco, mc.weight*ristra.allocate(kinWeight), **mc.params.valuesdict() )
  w, uw, cov, corr = ang_acc
  pars = Parameters()
  for i in range(0,len(w)):
    #print(f'w[{i}] = {w[i]:+.16f}')
    correl = {f'w{j}':cov[i][j] for j in range(0,len(w)) if i>0 and j>0}
    pars.add({'name': f'w{i}', 'value': w[i], 'stdev': uw[i], 'correl': correl,
              'free': False, 'latex': f'w_{i}'})
  print(f"Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")

  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Dumping parameters", f"\n{80*'='}\n")
  # Dump json file
  print(f"Dumping json parameters to {args['output_params']}")
  pars.dump(args['output_params'])
  # Export parameters in tex tables
  print(f"Dumping tex table to {args['output_tables']}")
  with open(args['output_tables'], "w") as tex_file:
    tex_file.write(
      pars.dump_latex( caption="""
      Kinematically corrected angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR,TRIGGER,MODE.replace('_', ' ') )
      )
    )
  tex_file.close()