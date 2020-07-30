#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']




################################################################################
# %% Modules ###################################################################

import argparse
import os
import sys
import numpy as np
import uproot

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# reweighting config
from hep_ml import reweight
#reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 1})
reweighter = reweight.GBReweighter(n_estimators=40, learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={'subsample': 1})
#reweighter = reweight.GBReweighter(n_estimators=500, learning_rate=0.1, max_depth=2, min_samples_leaf=1000, gb_args={'subsample': 1})

#30:0.3:4:500
#20:0.3:3:1000

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute angular acceptance.')
  # Samples
  parser.add_argument('--sample-mc',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r1.root',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-data',
    default = '/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root',
    help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--input-params',
    default = 'angular_acceptance/params/2016/MC_Bs2JpsiPhi_dG0.json',
    help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
    default = 'output_new/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r1_Naive_Biased.json',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables',
    default = 'output_new/tables/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r1_Naive_Biased.tex',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-weights-file',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r1_angWeight.npy',
    help='Bs2JpsiPhi MC sample')
  # Configuration
  parser.add_argument('--mode',
    default = 'MC_Bs2JpsiPhi_dG0',
    help='Configuration')
  parser.add_argument('--year',
    default = '2016',
    help='Year of data-taking')
  parser.add_argument('--version',
    default = 'v0r1',
    help='Year of data-taking')
  parser.add_argument('--trigger',
    default = 'biased',
    help='Trigger(s) to fit [comb/(biased)/unbiased]')
  parser.add_argument('--binvar',
    default = None,
    help='Different flag to ... ')

  return parser

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
  if args['binvar']:
    BINVAR, BIN = args['binvar'][:-1], args['binvar'][-1]
    CUT = bin_vars[f'{BINVAR}'][int(BIN)-1]
  else:
    BINVAR, BIN = None, None
    CUT = None
  print(CUT)
  # %% Load samples ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")

  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {args['trigger']:50}")
  print(f"{'cuts':>15}: {CUT if CUT else 'None':50}\n")

  # Load Monte Carlo samples
  mc = Sample.from_root(args['sample_mc'], cuts=CUT)
  mc.assoc_params(args['input_params'])
  print(mc.df)
  print(Sample.from_root(args['sample_mc']).df)
  print(np.amax(Sample.from_root(args['sample_mc']).df['B_PT']))

  # Load corresponding data sample
  data = Sample.from_root(args['sample_data'], cuts=CUT)

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = ['true'+i+'_GenLvl' for i in reco]
  if BINVAR:
    weight_mc=f'(polWeight*sw/gb_weights)'
    weight_rd=f'(sw_{BINVAR})'
  else:
    weight_mc=f'(polWeight*sw/gb_weights)'
    weight_rd=f'(sw)'

  # Select trigger
  if TRIGGER == 'biased':
    trigger = 'biased';
    weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
    weight_rd += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
  elif TRIGGER == 'unbiased':
    trigger = 'unbiased';
    weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
    weight_rd += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
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
  print(f"\n{80*'='}\n",
        "Weight the MC samples to match data ones",
        f"\n{80*'='}\n")

  reweighter.fit(original        = mc.df[['X_M','B_P','B_PT']],
                 target          = data.df[['X_M','B_P','B_PT']],
                 original_weight = mc.df.eval(weight_mc),
                 target_weight   = data.df.eval(weight_rd));

  kinWeight = reweighter.predict_weights(mc.df[['X_M','B_P','B_PT']])
  kinWeight = np.where(mc.df.eval(weight_mc)!=0, kinWeight, 0)
  print(f"The kinematic-weighting in B_PT, B_P and X_M is done for {MODE}-{TRIGGER}")
  print(f"kinWeight: {kinWeight}")
  np.save(args['output_weights_file'], kinWeight)

  #%% Compute angWeights correcting with kinematic weights ---------------------
  #     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
  #     variables
  print(f"\n{80*'='}\n",
        "Compute angWeights correcting MC sample in kinematics",
        f"\n{80*'='}\n")

  print('Computing angular weights')
  ang_acc = badjanak.get_angular_cov(mc.true, mc.reco, mc.weight*ristra.allocate(kinWeight), **mc.params.valuesdict() )
  w, uw, cov, corr = ang_acc
  pars = Parameters()
  for i in range(0,len(w)):
    #print(f'w[{i}] = {w[i]:+.16f}')
    correl = {f'w{j}':cov[i][j] for j in range(0,len(w)) if i>0 and j>0}
    pars.add({'name': f'w{i}', 'value': w[i], 'stdev': uw[i], 'correl': correl,
              'free': False, 'latex': f'w_{i}'})


  # Writing results ------------------------------------------------------------
  print('Dumping parameters')
  pars.dump(args['output_params'])
  # Export parameters in tex tables
  print('Saving table of params in tex')
  with open(args['output_tables'], "w") as tex_file:
    tex_file.write(
      pars.dump_latex( caption="""
      Kinematically corrected angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR,TRIGGER,MODE.replace('_', ' ') )
      )
    )
  tex_file.close()
  print(f"Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")
