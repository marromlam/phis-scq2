#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE
-- """

################################################################################
#%% Modules ####################################################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters

# get bsjpsikk and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

################################################################################


pt_bins = [0, 3.8, 6, 9]
eta_bins = [0, 3.3, 3.9, 6]
sigmat_bins = [0, 0.031, 0.042, 0.150]


################################################################################
#%% ############################################################################


def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--input-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-tables',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
                      default = 'baseline',
                      help='Configuration')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'test',
                      help='Year of data-taking')
  parser.add_argument('--trigger',
                      default = 'biased',
                      help='Trigger(s) to fit [comb/(biased)/unbiased]')
  # Report
  parser.add_argument('--pycode',
                      default = 'single',
                      help='Save a fit report with the results')

  return parser

# Parse arguments
args = vars(argument_parser().parse_args())


YEAR = 2016
VERSION = 'v0r5'
MODE = 'MC_Bs2JpsiPhi'
TRIGGER = 'unbiased'
mc_params = f'analysis/params/generator/{2016}/MC_Bs2JpsiPhi_dG0.json'
mc_sample = f'/scratch17/marcos.romero/sidecar/{YEAR}/{MODE}/{VERSION}.root'
mc_weights = f'/scratch17/marcos.romero/sidecar/{YEAR}/{MODE}/{VERSION}_run2_simul_kkpWeight.root'
angular_acceptance = f'output/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_run2_simul_{TRIGGER}.json'
time_acceptance = f'output/params/time_acceptance/{YEAR}/Bd2JpsiKstar/{VERSION}_simul_{TRIGGER}.json'



#print(input_params_path,sample_path,output_tables_path)

YEAR = args['year']
VERSION = args['version']
MODE = args['mode']
TRIGGER = args['trigger']
input_params_path = args['input_params']
sample_path = args['sample']
output_tables_path = args['output_tables']
output_params_path = args['output_params']

#print(input_params_path,sample_path,output_tables_path)

################################################################################
################################################################################
################################################################################


pd

# %% Load samples --------------------------------------------------------------

# Load Monte Carlo sample
smc = Sample.from_root(mc_sample)
kin = Sample.from_root(mc_weights, treename='DecayTree')
smc.df = pd.concat([smc.df,kin.df],axis=1)
del kin
smc.assoc_params(mc_params.replace('TOY','MC').replace('2021','2018'))


# Select trigger
weight = '(polWeight*sw/gb_weights)'
if TRIGGER == 'biased':
  trigger = 'biased'; weight += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
elif TRIGGER == 'unbiased':
  trigger = 'unbiased'; weight += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
elif TRIGGER == 'comb':
  trigger = 'comb';

# Calculating the weight
n = len(smc.find('kkp.*')) # get last iteration number
smc.df['weight'] = smc.df.eval(f'angWeight{n}*kkpWeight{n}*polWeight*sw/gb_weights').values

# Load acceptances
timeacc = Parameters.load(time_acceptance)
angacc = Parameters.load(angular_acceptance)

time = np.array( Parameters.build(timeacc,timeacc.find('k.*')+['tUL'])  )
time
smc.subdfs = []
for i in range(0,len(time)-1):
  smc.subdfs.append( smc.df.query(f'time >= {time[i]} & time < {time[i+1]}') )



# Variables and branches to be used
reco  = ['cosK', 'cosL', 'hphi', 'time']
true  = ['gen'+i for i in reco]
reco += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']
true += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']

# Paths
#smc.tables_path = output_tables_path
#smc.params_path = output_params_path


################################################################################
################################################################################
################################################################################


#%% Compute angWeights without corrections ---------------------------------------
#     Let's start computing the angular weights in the most naive version, w/
#     any corrections
print(f"\n{80*'='}\n",
      "Compute angWeights without correcting MC sample",
      f"\n{80*'='}\n")


w = []
for i in range(0,len(time)-1):
  print(f'Computing angular weights for time >= {time[i]} & time < {time[i+1]}')
  vt = ristra.allocate(np.array(smc.subdfs[i].eval(true)).T)
  vr = ristra.allocate(np.array(smc.subdfs[i].eval(reco)).T)
  vw = ristra.allocate(smc.subdfs[i]['weight'].values)
  ang_acc = badjanak.get_angular_acceptance_weights(vt, vr, vw, **smc.params.valuesdict() )
  print(ang_acc[0])
  w.append( unp.uarray(ang_acc[0], ang_acc[1]) )

np.array(smc.subdfs[i].eval(true)).T.shape





for k,v in angacc.items():
  j = int(k[1:])
  y = [wi[j] for i, wi in enumerate(w)]
  if v.casket:
    v.casket.update({ 'baseline': {'x':time, 'y':y} })
  else:
    v.casket = {}
    v.casket.update({ 'baseline': {'x':time, 'y':y} })
  print(v.casket)

angacc['w5'].casket['baseline']['y']

w, uw, cov, corr = ang_acc
mc.w_uncorrected = Parameters()
for i in range(0,len(w)):
  print(f'w[{i}] = {w[i]:+.16f}')
  correl = {f'w{j}':cov[i][j] for j in range(0,len(w)) if i>0 and j>0}
  mc.w_uncorrected.add({'name': f'w{i}',
                        'value': w[i],
                        'stdev': uw[i],
                        'free': False,
                        'latex': f'w_{i}',
                        'correl': correl
                      })
# Dump the parameters
print('Dumping parameters')
mc.w_uncorrected.dump(mc.params_path)
