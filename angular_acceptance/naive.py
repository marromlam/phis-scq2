#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']




################################################################################
# %% Modules ###################################################################

import argparse
import os
import sys

# load ipanema
from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import Sample, Parameters

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.config['fast_integral'] = 0
badjanak.get_kernels(True)

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute angular acceptance.')
  # Samples
  parser.add_argument('--sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r1.root',
    help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--input-params',
    default = 'angular_acceptance/params/2016/MC_Bs2JpsiPhi_dG0.json',
    help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
    default = 'output_new/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r1_Naive_Biased.json',
    help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-tables',
    default = 'output_new/tables/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r1_Naive_Biased.tex',
    help='Bs2JpsiPhi MC sample')
  # Configuration file
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

  return parser

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

  # Load samples ---------------------------------------------------------------
  print(f"\n{80*'='}\n", "Loading category", f"\n{80*'='}\n")

  mc = Sample.from_root(args['sample'])
  mc.params = Parameters()
  import hjson
  this_pars = hjson.load(open( args['input_params'].replace('TOY','MC').replace('2021','2018') ))
  mc.params.add(*[ {"name":k, "value":v} for k,v in this_pars.items()])  # WARNING)
  # ----
  #mc.assoc_params(args['input_params'].replace('TOY','MC').replace('2021','2018'))

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = ['true'+i+'_GenLvl' for i in reco]
  weight='(polWeight*sw/gb_weights)'
  #weight='(sw/gb_weights)'

  # Select trigger
  if TRIGGER == 'biased':
    trigger = 'biased'; weight += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
  elif TRIGGER == 'unbiased':
    trigger = 'unbiased'; weight += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
  elif TRIGGER == 'combined':
    trigger = 'combined';

  # Allocate some arrays with the needed branches
  try:
    mc.allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
    mc.allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time'])
  except:
    print('Guessing you are working with TOY files. No X_M provided')
    mc.allocate(reco=reco+['0*time', '0*time', 'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    mc.allocate(true=true+['0*time', '0*time', 'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    weight = 'time/time'
  mc.allocate(pdf='0*time', ones='time/time', zeros='0*time')
  mc.allocate(weight=weight)


  # Compute angWeights without corrections -------------------------------------
  #     Let's start computing the angular weights in the most naive version, w/o
  #     any corrections
  print(f"\n{80*'='}\n",
        "Compute angWeights without correcting MC sample",
        f"\n{80*'='}\n")

  print('Computing angular weights')
  ang_acc = badjanak.get_angular_cov(mc.true, mc.reco, mc.weight, **mc.params.valuesdict() )
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
      Naive angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR,TRIGGER,MODE.replace('_', ' ') )
      )
    )
  tex_file.close()
  print(f"Naive angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")
