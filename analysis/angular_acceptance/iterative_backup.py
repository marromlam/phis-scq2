#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE


Benchmark:
Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz
nVidia GeForce GTX 1080 Ti
-- """


# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import uncertainties as unc
from uncertainties import unumpy as unp
from scipy.stats import chi2

# reweighting config
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# threading
import logging
import threading
import time
import multiprocessing

from iminuit import Minuit as minuit

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters, Parameter, Optimizer

# get bsjpsikk and compile it with corresponding flags
import bsjpsikk
bsjpsikk.config['debug'] = 0
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
bsjpsikk.config['use_perftag'] = 1
bsjpsikk.config['use_truetag'] = 0
bsjpsikk.get_kernels()

# reweighting config
from hep_ml import reweight

# reweighter = reweight.GBReweighter(n_estimators     = 40,
#                                    learning_rate    = 0.25,
#                                    max_depth        = 5,
#                                    min_samples_leaf = 500,
#                                    gb_args          = {'subsample': 1})

reweighter = reweight.GBReweighter(n_estimators     = 50,
                                   learning_rate    = 0.3,
                                   max_depth        = 10,
                                   min_samples_leaf = 1000,
                                   gb_args          = {'subsample': 1})



################################################################################
################################################################################
#%% ############################################################################



def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample-mc-std',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-mc-dg0',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-data',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--params-mc-std',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--params-mc-dg0',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angular-weights-mc-std',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angular-weights-mc-dg0',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-weights-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-weights-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-coeffs-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-coeffs-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-csp',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-weights-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-weights-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'test',
                      help='Year of data-taking')

  return parser


# Parse arguments
#try:
args = vars(argument_parser().parse_args())
YEARS = [int(y) for y in args['year'].split(',')] # years are int
VERSION = args['version']

samples_std   = args['sample_mc_std'].split(',')
samples_dg0   = args['sample_mc_dg0'].split(',')
samples_data  = args['sample_data'].split(',')

input_std_params = args['params_mc_std'].split(',')
input_dg0_params = args['params_mc_dg0'].split(',')
input_data_params = f'angular_acceptance/params/{2016}/Bs2JpsiPhi.json'

angWeight_std = args['angular_weights_mc_std'].split(',')
angWeight_dg0 = args['angular_weights_mc_dg0'].split(',')

output_tables = [f'output_new/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{VERSION}_iterative1_biased.tex',
                 f'output_new/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.tex',
                 f'output_new/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_biased.tex',
                 f'output_new/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.tex']
output_params = [f'output_new/params/angular_acceptance/{2015}/Bs2JpsiPhi/{VERSION}_iterative1_biased.json',
                 f'output_new/params/angular_acceptance/{2015}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.json',
                 f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_biased.json',
                 f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.json']

w_biased      = args['input_weights_biased'].split(',')
w_unbiased    = args['input_weights_unbiased'].split(',')

coeffs_biased      = args['input_coeffs_biased'].split(',')
coeffs_unbiased    = args['input_coeffs_unbiased'].split(',')

csp_factors    = args['input_csp'].split(',')
# except:
#   YEARS = [2016]
#   VERSION = 'v0r0'
#   input_std_params = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json']
#   input_dg0_params = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json']
#   input_data_params = f'angular_acceptance/params/{2016}/Bs2JpsiPhi.json'
#
#   samples_std   = [f'/scratch17/marcos.romero/phis_samples/{2016}/MC_Bs2JpsiPhi/{VERSION}.root']
#   samples_dg0   = [f'/scratch17/marcos.romero/phis_samples/{2016}/MC_Bs2JpsiPhi_dG0/{VERSION}.root']
#   samples_data  = [f'/scratch17/marcos.romero/phis_samples/{2016}/Bs2JpsiPhi/{VERSION}.root']
#
#   angWeight_std = [f'/scratch17/marcos.romero/phis_samples/{2016}/MC_Bs2JpsiPhi/{VERSION}_angWeight.root']
#   angWeight_dg0 = [f'/scratch17/marcos.romero/phis_samples/{2016}/MC_Bs2JpsiPhi_dG0/{VERSION}_angWeight.root']
#
#   output_tables = [f'output_new/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_biased.tex',
#                    f'output_new/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.tex']
#   output_params = [f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_biased.json',
#                    f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_iterative1_unbiased.json']
#
#   w_biased      = [f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_corrected_biased.json']
#   w_unbiased    = [f'output_new/params/angular_acceptance/{2016}/Bs2JpsiPhi/{VERSION}_corrected_unbiased.json']
#
#   coeffs_biased   = [f'v0r2/time_acceptance/parameters/{2016}/Bs2JpsiPhi/200221a_biased.json']
#   coeffs_unbiased = [f'v0r2/time_acceptance/parameters/{2016}/Bs2JpsiPhi/200221a_unbiased.json']







"""

input_std_params0 = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json',
                    f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json']
input_dg0_params0 = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json',
                    f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json']
input_data_params0 = f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json'

samples_std0   = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi/{FLAG}.root',
                 f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi/{FLAG}.root']
samples_dg00   = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi_dG0/{FLAG}.root',
                 f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi_dG0/{FLAG}.root']
samples_data0  = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/Bs2JpsiPhi/{FLAG}.root',
                 f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/Bs2JpsiPhi/{FLAG}.root']

angWeight_std0 = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root',
                 f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root']
angWeight_dg00 = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root',
                 f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root']


output_tables0 = [f'output/{VERSION}/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
                 f'output/{VERSION}/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex',
                 f'output/{VERSION}/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
                 f'output/{VERSION}/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex']
output_params0 = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
                 f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json',
                 f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
                 f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json']


w_biased0      = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_corrected_biased.json',
                 f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_corrected_biased.json']
w_unbiased0    = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json',
                 f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json']

coeffs_biased0   = [f'output/time_acceptance/{2015}/Bd2JpsiKstar/{FLAG}_baseline_biased.json',
                   f'output/time_acceptance/{2016}/Bd2JpsiKstar/{FLAG}_baseline_biased.json']
coeffs_unbiased0 = [f'output/time_acceptance/{2015}/Bd2JpsiKstar/{FLAG}_baseline_unbiased.json',
                   f'output/time_acceptance/{2016}/Bd2JpsiKstar/{FLAG}_baseline_unbiased.json']




print(input_std_params0)
print(input_std_params)
print(input_dg0_params0)
print(input_dg0_params)
print(input_data_params0)
print(input_data_params)

print(samples_std)
print(samples_std0)
print(samples_dg0)
print(samples_dg00)
print(samples_data)
print(samples_data0)

print(angWeight_std)
print(angWeight_std0)
print(angWeight_dg0)
print(angWeight_dg00)

print(output_params)
print(output_params0)


"""




#
#
# input_std_params = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json']
# input_dg0_params = [f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json',
#                     f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json']
# input_data_params = f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json'
#
# samples_std   = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2017}/MC_Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2018}/MC_Bs2JpsiPhi/{FLAG}.root']
# samples_dg0   = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi_dG0/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi_dG0/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2017}/MC_Bs2JpsiPhi_dG0/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2018}/MC_Bs2JpsiPhi_dG0/{FLAG}.root']
# samples_data  = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2017}/Bs2JpsiPhi/{FLAG}.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2018}/Bs2JpsiPhi/{FLAG}.root']
#
# angWeight_std = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2017}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2018}/MC_Bs2JpsiPhi/{FLAG}_angWeight.root']
# angWeight_dg0 = [f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2015}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2016}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2017}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root',
#                  f'/scratch17/marcos.romero/phis_samples/{VERSION}/{2018}/MC_Bs2JpsiPhi_dG0/{FLAG}_angWeight.root']
#
# output_tables = [f'output/{VERSION}/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_iterative1_biased.tex',
#                  f'output/{VERSION}/tables/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.tex']
# output_params = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json'
#                  f'output/{VERSION}/params/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json'
#                  f'output/{VERSION}/params/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_iterative1_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_iterative1_unbiased.json']
#
#
# w_biased      = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_corrected_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_corrected_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_corrected_biased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_corrected_biased.json']
# w_unbiased    = [f'output/{VERSION}/params/angular_acceptance/{2015}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2016}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2017}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json',
#                  f'output/{VERSION}/params/angular_acceptance/{2018}/Bs2JpsiPhi/{FLAG}_corrected_unbiased.json']
#
# coeffs_biased   = [f'v0r2/time_acceptance/parameters/{2015}/Bs2JpsiPhi/200221a_biased.json',
#                    f'v0r2/time_acceptance/parameters/{2016}/Bs2JpsiPhi/200221a_biased.json',
#                    f'v0r2/time_acceptance/parameters/{2017}/Bs2JpsiPhi/200221a_biased.json',
#                    f'v0r2/time_acceptance/parameters/{2018}/Bs2JpsiPhi/200221a_biased.json']
# coeffs_unbiased = [f'v0r2/time_acceptance/parameters/{2015}/Bs2JpsiPhi/200221a_unbiased.json',
#                    f'v0r2/time_acceptance/parameters/{2016}/Bs2JpsiPhi/200221a_unbiased.json',
#                    f'v0r2/time_acceptance/parameters/{2017}/Bs2JpsiPhi/200221a_unbiased.json',
#                    f'v0r2/time_acceptance/parameters/{2018}/Bs2JpsiPhi/200221a_unbiased.json']
#
#
#















# dddddddd
# coeffs_biased   = [f'output/v0r4/params/time_acceptance/{2015}/Bd2JpsiKstar/{FLAG}_baseline_biased.json',
#                    f'output/v0r4/params/time_acceptance/{2016}/Bd2JpsiKstar/{FLAG}_baseline_biased.json']
# coeffs_unbiased = [f'output/v0r4/params/time_acceptance/{2015}/Bd2JpsiKstar/{FLAG}_baseline_unbiased.json',
#                    f'output/v0r4/params/time_acceptance/{2016}/Bd2JpsiKstar/{FLAG}_baseline_unbiased.json']





# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\n",
      "Loading samples",
      f"\n{80*'='}\n")

# Lists of MC variables to load and build arrays
reco = ['cosK', 'cosL', 'hphi', 'time']
true = ['true'+i+'_GenLvl' for i in reco]
reco += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl']
true += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl']
weight_mc='(polWeight*sw/gb_weights)'

# Lists of data variables to load and build arrays
real  = ['cosK','cosL','hphi','time']                        # angular variables
real += ['X_M','sigmat']                                     # mass and sigmat
real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging
#real += ['0*B_ID','0*B_ID', '0*B_ID', '0*B_ID']  # tagging
weight_data='(sw)'

# Load Monte Carlo samples
mc = {}
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y} MC samples')
  mc[f'{y}'] = {}
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[samples_std,samples_dg0]):
    print(f' *  Loading {m}-{y} sample from\n    {v[i]}')
    mc[f'{y}'][f'{m}'] = Sample.from_root(v[i])
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[input_std_params,input_dg0_params]):
    print(f' *  Associating {m}-{y} parameters from\n    {v[i]}')
    mc[f'{y}'][f'{m}'].assoc_params(v[i])
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[angWeight_std,angWeight_dg0]):
    print(f' *  Attaching {m}-{y} kinWeight from\n    {v[i]}')
    mc[f'{y}'][f'{m}'].kinWeight = uproot.open(v[i])['DecayTree'].array('kinWeight')
    mc[f'{y}'][f'{m}'].path_to_weights = v[i]
    print(f"    {mc[f'{y}'][f'{m}'].kinWeight}")

for y, modes in mc.items():
  for m, v in modes.items():
    print(f' *  Allocating arrays in device for {m}-{y}')
    mc[f'{y}'][f'{m}'].allocate(reco=reco)
    mc[f'{y}'][f'{m}'].allocate(true=true)
    mc[f'{y}'][f'{m}'].allocate(pdf='0*time', ones='time/time', zeros='0*time')
    mc[f'{y}'][f'{m}'].allocate(weight=weight_mc)
    mc[f'{y}'][f'{m}'].allocate(biased='Jpsi_Hlt1DiMuonHighMassDecision_TOS==0')
    mc[f'{y}'][f'{m}'].allocate(unbiased='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
    mc[f'{y}'][f'{m}'].angular_weights = {'biased':0, 'unbiased':0}
    mc[f'{y}'][f'{m}'].kkpWeight = {}
    mc[f'{y}'][f'{m}'].pdfWeight = {}


# Load corresponding data sample
data = {}
mass = bsjpsikk.config['x_m']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[f'{y}'] = {}
  data[f'{y}']['combined'] = Sample.from_root(samples_data[i])
  csp = Parameters.load(csp_factors[i])  # <--- WARNING
  csp = csp.build(csp,csp.find('CSP.*'))
  csp_dev = ristra.allocate(np.array(csp.build(csp,csp.find('CSP.*'))))
  flavor= Parameters.load(f'output/v0r4/params/flavor_tagging/{y}/Bs2JpsiPhi/200506a.json')  # <--- WARNING
  resolution = Parameters.load(f'output_new/params/time_resolution/{y}/Bs2JpsiPhi/{VERSION}.json')
  for t, T in zip(['biased','unbiased'],[0,1]):
    print(f' *  Loading {y} sample in {t} category\n    {samples_data[i]}')
    this_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={T}) & (time>=0.3) & (time<=15)'
    data[f'{y}'][f'{t}'] = Sample.from_root(samples_data[i], cuts=this_cut)
    data[f'{y}'][f'{t}'].csp = csp
    data[f'{y}'][f'{t}'].csp_dev = csp_dev
    print(csp)
    print(resolution)
    data[f'{y}'][f'{t}'].flavor = flavor
    data[f'{y}'][f'{t}'].resolution = resolution
  for t, coeffs in zip(['biased','unbiased'],[coeffs_biased,coeffs_unbiased]):
    print(coeffs)
    print(f' *  Associating {y}-{t} time acceptance[{i}] from\n    {coeffs[i]}')
    c = Parameters.load(coeffs[i])
    data[f'{y}'][f'{t}'].timeacc = np.array(Parameters.build(c,c.fetch('c.*')))
    data[f'{y}'][f'{t}'].tLL = c['tLL'].value
    data[f'{y}'][f'{t}'].tUL = c['tUL'].value
    print(f"   {data[f'{y}'][f'{t}'].timeacc}")
  for t, weights in zip(['biased','unbiased'],[w_biased,w_unbiased]):
    print(f' *  Associating {y}-{t} angular weights from\n    {weights[i]}')
    w = Parameters.load(weights[i])
    data[f'{y}'][f'{t}'].angacc = np.array(Parameters.build(w,w.fetch('w.*')))
    print(f"   {data[f'{y}'][f'{t}'].angacc}")
  print(f' *  Allocating {y} arrays in device ')
  for d in [data[f'{y}']['biased'],data[f'{y}']['unbiased']]:
    sw = np.zeros_like(d.df['sw'])
    for l,h in zip(mass[:-1],mass[1:]):
      pos = d.df.eval(f'X_M>={l} & X_M<{h}')
      this_sw = d.df.eval(f'sw*(X_M>={l} & X_M<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
    d.df['sWeight'] = sw
    d.allocate(data=real,weight='sWeight',lkhd='0*time')
    d.angacc_dev = ristra.allocate(d.angacc)
    d.timeacc_dev = ristra.allocate(bsjpsikk.get_4cs(d.timeacc))


#%% define likelihood



pars = Parameters.load('angular_acceptance/Bs2JpsiPhi.json')
hey = hjson.load(open('angular_acceptance/params/2016/Bs2JpsiPhi.json'))
#hey = hjson.load(open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_dG0_'+str(0)+'.json'))
for k,v in hey.items():
  try:
    pars[k].set(value=v)
    pars[k].set(init=v)
  except:
    0
# pars['CSP1'].value = csp_dev.get()[0]
# pars['CSP2'].value = csp_dev.get()[1]
# pars['CSP3'].value = csp_dev.get()[2]
# pars['CSP4'].value = csp_dev.get()[3]
# pars['CSP5'].value = csp_dev.get()[4]
# pars['CSP6'].value = csp_dev.get()[5]
print(pars)

#pars = pars+Parameters.load(f'output/{VERSION}/params/flavor_tagging/{y}/Bs2JpsiPhi/200506a.json')+Parameters.load(f'output/{VERSION}/params/time_resolution/{y}/Bs2JpsiPhi/200506a.json')

bsjpsikk.config['debug']           = 0
bsjpsikk.config['debug_evt']       = 0
bsjpsikk.config['use_time_acc']    = 1
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res']    = 1
bsjpsikk.config['use_perftag']     = 0
bsjpsikk.config['use_truetag']     = 0
bsjpsikk.get_kernels()





def fcn_data(parameters, data, weight = False, lkhd0=False):
  pars_dict = parameters.valuesdict()
  likelihood = []
  for y, dy in data.items():
    for dt in [dy['biased'],dy['unbiased']]:
      bsjpsikk.diff_cross_rate_full(dt.data, dt.lkhd,
                                    w = dt.angacc,
                                    coeffs = dt.timeacc,
                                    **dt.csp.valuesdict(),
                                    #**dt.resolution.valuesdict(),
                                    **pars_dict)
      if weight:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd) * dt.weight).get() );
      else:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd)            ).get() );
  if lkhd0:
    return np.sum(np.concatenate(likelihood)) - (lkhd0-100)
  return np.sum(np.concatenate(likelihood))


#print( fcn_data(pars, data=data, weight=True) )


def fcn_data_opt(parameters, data, weight = False, lkhd0=False):
  pars = parameters.valuesdict()
  likelihood = []
  for dy in data.values():
    for dt in [dy['biased'],dy['unbiased']]:
      bsjpsikk.new_diff_rate(dt.data, dt.lkhd,
                             angacc = dt.angacc_dev, timeacc = dt.timeacc_dev,
                             CSP = dt.csp_dev,
                             tLL = dt.tLL, tUL = dt.tUL,
                             **pars)
      if weight:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd) * dt.weight).get() );
      else:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd)            ).get() );
  if lkhd0:
    return np.sum(np.concatenate(likelihood)) - (lkhd0-100)
  return np.sum(np.concatenate(likelihood))


# from timeit import default_timer as timer
# #
# t0 = timer()
# for i in range(10):
#   fcn_data(pars, data=data, weight=True)
# tf = timer()-t0
# print(tf)
# t0 = timer()
# for i in range(10):
#   fcn_data_opt(pars, data=data, weight=True)
# tf = timer()-t0
# print(tf)
#
#
#
#
#
#
#
# exit()









def fcn_data_dev(parameters, data, weight = False):
  pars_dict = parameters.valuesdict()
  likelihood = []; weights = []
  for dy in data.items():
      for dt in [dy['biased'],dy['unbiased']]:
        bsjpsikk.diff_cross_rate_full(dt.data, dt.lkhd,
                                      w = dt.angacc, coeffs = dt.timeacc,
                                      **pars_dict)
        ####np.save('caca.npy',dt.lkhd.get())
        if weight:
          likelihood.append( (-2*ristra.log(dt.lkhd)*dt.weight).get() );
          #weights.append( (2*dt.weight).get() );
        else:
          likelihood.append( (ristra.log(dt.lkhd)).get() );
          weights.append(np.ones_like(likelihood[-1]));
  #likelihood = np.column_stack(likelihood).ravel();
  #weights = np.column_stack(weights).squeeze();
  #likelihood = np.hstack(likelihood).squeeze();
  #weights = np.hstack(weights).squeeze();
  return likelihood[0].sum() + likelihood[1].sum() -(857867.611532611-100) #+ weights


bsjpsikk.config['debug']           = 0
bsjpsikk.config['debug_evt']       = 0
bsjpsikk.config['use_time_acc']    = 1
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res']    = 1
bsjpsikk.config['use_perftag']     = 0
bsjpsikk.config['use_truetag']     = 0
bsjpsikk.get_kernels()
#print( fcn_data(pars, data=data, weight=True) )

#print( fcn_data(pars, data=data, weight=True, lkhd0 = lkhd0) )

#exit()

##data[f'2016']['unbiased'].shape[0]+data[f'2016']['biased'].shape[0]
#
# r[0].sum()
# r[1].sum()
# r[2].sum()
# r[3].sum()
#
# r[0].sum()+r[1].sum()
#
#
#
# fcn_data(pars, data=data, weight=True).sum()
# 857867.611527497
#
# data[f'2016']['unbiased'].shape[0]+data[f'2016']['biased'].shape[0]
#856824.7150199823
#likelihood:
# 857867.611532611
# (857867.611532611-100)/183569
# Determined empirical constant of: 4.67327060414673

#
# print( data[f'2016']['unbiased'].df.query('time<0.3') )
# print( data[f'2016']['unbiased'].df.query('time>15') )
#
# print( data[f'2016']['unbiased'].df.query('X_M<990') )
# print( data[f'2016']['unbiased'].df.query('X_M>1050') )
#
#
#856824.7150199823
#
# print( data[f'2016']['biased'].df.query('time<0.3') )
# print( data[f'2016']['biased'].df.query('time>15') )
#
# print( data[f'2016']['biased'].df.query('X_M<990') )
# print( data[f'2016']['biased'].df.query('X_M>1050') )
#






#%% Prepate fitter






def minuit_fit(pars, data):
  # Set model to fit data
  bsjpsikk.config['debug']           = 0
  bsjpsikk.config['debug_evt']       = 0
  bsjpsikk.config['use_time_acc']    = 1
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res']    = 1
  bsjpsikk.config['use_perftag']     = 0
  bsjpsikk.config['use_truetag']     = 0
  bsjpsikk.get_kernels()

  lkhd0 = fcn_data(pars, data=data, weight=True)
  # Minuit wrapper
  def wrapper_minuit(*fvars):
    for name, val in zip(list_of_pars, fvars):
      pars[name].value = val
    result =  fcn_data(pars, data=data, weight=True, lkhd0=lkhd0)
    #exit()
    return result

  def configure_minuit( pars, pars_list, **kwgs):
    def parameter_minuit_config(par):
      out = {par.name: par.value}
      lims = [None,None]
      if abs(par.min) != np.inf: lims[0] = par.min
      if abs(par.max) != np.inf: lims[1] = par.max
      if not par.free:
        out.update ({"fix_" + par.name: True})
      out.update ({"limit_" + par.name: tuple(lims)})
      out.update ({"error_" + par.name: 1e-6})
      return out

    config = {}
    for par in pars.keys():
      if par in pars_list:
        config.update(parameter_minuit_config(pars[par]))
    config.update(kwgs)
    # for k,v in config.items():
    #   print(f'{k:>15} : {v}')
    return config

  # Get info for minuit
  #shit = Optimizer(fcn_data, params=pars); shit.prepare_fit()
  #list_of_pars = np.copy(shit.result.param_vary).tolist()
  list_of_pars = ['fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 'fPlon', 'fPper', 'pPlon', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6', 'dPpar', 'dPper', 'lPlon', 'DGs', 'DGsd', 'DM']
  dict_of_conf = configure_minuit(pars,list_of_pars)

  # Gor for it!
  print('Fit is starting...')
  crap = minuit(wrapper_minuit, forced_parameters=list_of_pars, **dict_of_conf, print_level=2, errordef=1, pedantic=False)
  crap.strategy = 0
  #crap.tol = 0.05
  crap.errordef = 1.0
  crap.migrad()
  crap.hesse()
  if not crap.migrad_ok():
    for i in range(0,10):
      crap.migrad()
      crap.hesse()
      if crap.migrad_ok():
        break
    if not crap.migrad_ok():
      print("Can't do better, sorry. Be aware of the precision loss")
      crap.hesse()
  print('Fit is finished! Cross your fingers and pray Simon')

  # Update pars
  #pars_fitted = Parameters.clone(pars)
  for name, val in zip(list_of_pars, crap.values.values()):
    pars[name].value = val
  for name, val in zip(list_of_pars, crap.errors.values()):
    pars[name].stdev = val
  #return pars_fitted


def kkp_weighting(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter.fit(original = original_v, target = target_v,
                 original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{t}.npy',kkpWeight)
  print(f"* GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")


def get_angular_acceptance(mc,t):
  # Select t
  if t == 'biased':
    trigger = mc.biased
  elif t == 'unbiased':
    trigger = mc.unbiased
  ang_acc = bsjpsikk.get_angular_cov(
              mc.true, mc.reco,
              trigger*mc.weight*ristra.allocate(mc.kkpWeight[i]*mc.kinWeight),
              **mc.params.valuesdict()
            )
  # Create parameters
  w, uw, cov, corr = ang_acc
  mc.angular_weights[t] = Parameters()
  for k in range(0,len(w)):
    correl = {f'w{j}':cov[k][j] for j in range(0,len(w)) if k>0 and j>0}
    mc.angular_weights[t].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                           'free': False, 'latex': f'w_{k}', 'correl': correl})
  print(f"{  np.array(mc.angular_weights[t])}")



# bsjpsikk.config['debug']           = 5
# bsjpsikk.config['debug_evt']       = 0
# bsjpsikk.config['use_time_acc']    = 1
# bsjpsikk.config['use_time_offset'] = 0
# bsjpsikk.config['use_time_res']    = 1
# bsjpsikk.config['use_perftag']     = 0
# bsjpsikk.config['use_truetag']     = 0
# bsjpsikk.get_kernels()



# data[f'2016']['unbiased'].angacc = np.array([
# 1.0,
# 1.026188051680966,
# 1.025919907820129,
# -0.0008080722059112641,
# 0.0007686171894090263,
# 0.0002055097707937297,
# 1.00647239083432,
# 0.000455221821734608,
# 0.0001466163291351719,
# -0.000731595571913009
# ])
# data[f'2016']['unbiased'].angacc = np.array([
# 1,
# 1.036616501815053,
# 1.036438881647181,
# -0.000767902108603735,
# 0.0002722187888972826,
# 0.0002375659824330591,
# 1.009448663033953,
# 2.073630681856376e-05,
# 4.092379705098602e-05,
# -0.003959703563916721
# ])
# data[f'2016']['biased'].angacc = np.array([
# 1.0,
# 1.020419588928056,
# 1.020502754804629,
# 0.002631350622172166,
# 0.003125427462874503,
# -0.0003293730619200012,
# 1.011599141342973,
# 0.0002557661696621679,
# 4.612016290721501e-06,
# -0.001331697639192716
# ])
# data[f'2016']['biased'].angacc = np.array([
# 1,
# 1.034440015714541,
# 1.034642153098812,
# 0.00272584738881403,
# 0.003038166631007048,
# -0.0002781312683095018,
# 1.020346829061547,
# 0.0001065078746602566,
# 6.226895891636155e-05,
# 0.001126252400056541
# ])






# data[f'2016']['unbiased'].timeacc = np.array([
# 1.0,
# 1.007332501212408,
# 1.029596127733144,
# 1.000472420511906,
# 0.9823659562668927,
# 0.9979454187495134,
# 1.004462717947248,
# 0.9834125194998904,
# 0.9808166980205768
# ])
# data[f'2016']['biased'].timeacc = np.array([
# 1.0,
# 1.485637143738824,
# 2.056236952115024,
# 2.117312551142014,
# 2.278232224247727,
# 2.289663906726421,
# 2.449815938926983,
# 2.23572899211712,
# 2.321796691949785
# ])





"""
bsjpsikk.config['x_m']
bsjpsikk.config['x_m'].index(990)
def check_bin(x):
  for m,M in zip(bsjpsikk.config['x_m'][:-1],bsjpsikk.config['x_m'][1:]):
    if (x>=m) & (x<M):
      return bsjpsikk.config['x_m'].index(M)

check_bin(1008)

bsjpsikk.get_kernels()
print(fcn_data(pars, data=data, weight=True))

+0.5760557769968792 // no bining
+0.5760557166305621 // sin 0.5
+0.5760682440100898 // con 0.5

# import pycuda.driver as cuda
# import pycuda.autoinit
# import numpy as np
# import pycuda.gpuarray as cu_array
# from pycuda.compiler import SourceModule
# Badjanak = SourceModule(
# ""
# #include <stdio.h>
# #include <math.h>
# #include <pycuda-complex.hpp>
#
# __global__
# void f(pycuda::complex<double> *a , pycuda::complex<double> *b)
# {
#   int evt = threadIdx.x + blockDim.x * blockIdx.x;
#   b[evt] = pycuda::complex<double>(2.0,3.0) + a[evt];
# };
# "")
# x = cu_array.to_gpu(np.complex128([+0.6893833775208801])).astype(np.complex128)
# y = cu_array.to_gpu(np.complex128([+0.])).astype(np.complex128)
#
# Badjanak.get_function('f')(x,y, block = (32,1,1))
# y.get()[0]
# x.get()[0]
# exit()



a = np.load('caca.npy')
b = np.loadtxt('/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/test_shit.txt',delimiter=', ')
b = b[:len(a)]
b = b[b[:,0].argsort()]
len(a)

for i in range(0,len(a)):
    #if a[i]!=0:
    diff = abs(b[i,1]-a[i])
    if diff > 1e-15:
      mass = data['2016']['unbiased'].df.iloc[i]['X_M']
      bin = data['2016']['unbiased'].df.iloc[i]['X_M']
      print(f'[{i:6.0f}, mass bin {check_bin(mass)}]  {b[i,1]:+.16f} - {a[i]:+.16f}  = {diff:.16f}')

"""




################################################################################
#%% Iterative procedure computing angWeights with corrections ##################

print(f"\n{80*'='}\n",
      "Iterative fitting procedure with pdf and kinematic-weighting",
      f"\n{80*'='}\n")



for i in range(1,7):
  # 1st step: fit data ---------------------------------------------------------
  print(f'Fitting Bs2JpsiPhi {"&".join(list(mc.keys()))} [iteration #{i}]')
  print('biased',data[f'{y}']['biased'].angacc)
  print('unbiased',data[f'{y}']['unbiased'].angacc)
  minuit_fit(pars, data)
  pars = Parameters.clone(pars)
  print(pars)


  pars_loaded = hjson.load(
                      open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_dG0_'+str(i-1)+'.json')
                )
  # for k, v in pars.items():
  #   print(f'{k:>10}: {v}')

  pars_comparison = Parameters.clone( pars )
  print(pars_comparison)

  for k, v in pars.items():
    try:
      pars_comparison[k].value = pars_loaded[k]
    except:
      0
    if v.free:
      try:
        print(f'{k:>10}: {v.value:+4.8f}   {pars_comparison[k].value:+4.8f}   {(v.value-pars_comparison[k].value):+2.2e}    {100*( (v.value-pars_comparison[k].value)/pars_comparison[k].value ):+4.4f}%')
      except:
        print(f'{k:>10}: {v.value:+4.8f}   ')

  print(f'FCN with SCQ parameters: {fcn_data(pars, data=data, weight=True):.16f}')
  print(f' FCN with HD parameters: {fcn_data(pars_comparison, data=data, weight=True):.16f}')
  #exit()
  # 2nd step: pdf weights ------------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0)
  print(f'\nPDF weighting MC samples to match Bs2JpsiPhi data [iteration #{i}]')
  bsjpsikk.config['debug']    = 0
  bsjpsikk.config['debug_evt'] = 0
  bsjpsikk.config['use_time_acc']    = 0
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res']    = 0
  bsjpsikk.config['use_perftag']     = 1
  bsjpsikk.config['use_truetag']     = 0
  bsjpsikk.get_kernels()


  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      print(f'* Calculating pdfWeight for {m}-{y} sample')
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **v.params.valuesdict(), mass_bins=1)
      original_pdf_h = v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **v.params.valuesdict(), mass_bins=1)
      original_pdf_h /= v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **pars.valuesdict())
      target_pdf_h = v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **pars.valuesdict())
      target_pdf_h /= v.pdf.get()
      v.pdfWeight[i] = np.nan_to_num(target_pdf_h/original_pdf_h)
      print(f"  pdfWeight[{i}]: {v.pdfWeight[i]}")
      # print(f'* Calculating pdfWeight for {m}-{y} sample')
      # bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **v.params.valuesdict(), mass_bins=1)
      # original_pdf_h = v.pdf.get()
      # bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **v.params.valuesdict(), mass_bins=1)
      # original_pdf_h /= v.pdf.get()
      # bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **pars_loaded)
      # target_pdf_h = v.pdf.get()
      # bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **pars_loaded)
      # target_pdf_h /= v.pdf.get()
      # print(f"  pdfWeight[{i}]: {np.nan_to_num(target_pdf_h/original_pdf_h)}")
  #exit()
  # 3rd step: kinematic weights ------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0).
  #   As a matter of fact, it's important to have data[y][combined] sample,
  #   the GBweighter gives different results when having those 0s or having
  #   nothing after cutting the sample.
  print(f'\nKinematic reweighting MC samples in K momenta [iteration #{i}]')
  threads = list()
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                      datefmt="%H:%M:%S")

  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      #v.kkpWeight = np.zeros_like(v.pdfWeight)
      for t, t_cut in zip(['biased','unbiased'],[0,1]):
        t_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={t_cut})*'
        #print(t_cut)
        original_v = v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']].values
        original_w = v.df.eval(t_cut+'polWeight*sw/gb_weights')*v.pdfWeight[i]*v.kinWeight
        target_v = data[f'{y}']['combined'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']].values
        target_w = data[f'{y}']['combined'].df.eval(t_cut+'sw').values
        # Run in single core (REALLY SLOW 10+ h)
        # kkp_weighting(original_v, original_w, target_v, target_w, v.kkpWeight[f'{t}'], y, m, t, 0)
        # Run in multithread mode (still has some problems with memory)
        # job = threading.Thread(target=kkp_weighting,
        #                        args=(original_v, original_w, target_v, target_w,
        #                        v.kkpWeight[f'{t}'], y, m, t))
        # Run multicore (about 12 minutes per iteration)
        job = multiprocessing.Process(target=kkp_weighting,
                               args=(original_v, original_w, target_v, target_w,
                               v.path, y, m, t, len(threads) ))
        threads.append(job); job.start()

  # Wait all processes to finish
  print(f'There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]


  # 4th step: angular weights --------------------------------------------------
  print(f'\nExtract angular weights [iteration #{i}]')
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      # write down code to save kkpWeight to root files
      # <CODE>
      b = np.load(os.path.dirname(v.path)+f'/kkpWeight_biased.npy')
      u = np.load(os.path.dirname(v.path)+f'/kkpWeight_unbiased.npy')
      v.kkpWeight[i] = b+u
      print(f' kkpWeight[{i}] = {v.kkpWeight[i][:20]}')
      for trigger in ['biased','unbiased']:
        print(f"Current angular weights for {m}-{y}-{trigger} sample are:")
        get_angular_acceptance(v,trigger)


  # 5th step: merge MC std and dg0 results -------------------------------------
  print(f'\nCombining MC_BsJpsiPhi and MC_BsJpsiPhi_dG0 [iteration #{i}]')
  for y, dy in mc.items(): # loop over years
    for trigger in ['biased','unbiased']:
      # Get angular weights for each MC
      std = dy['MC_BsJpsiPhi'].angular_weights[trigger]
      dg0 = dy['MC_BsJpsiPhi_dG0'].angular_weights[trigger]

      # Create w and cov arrays
      std_w = np.array([std[f'w{i}'].value for i in range(1,len(std))])
      dg0_w = np.array([dg0[f'w{i}'].value for i in range(1,len(dg0))])
      std_cov = std.correl_mat()[1:,1:];
      dg0_cov = dg0.correl_mat()[1:,1:];

      # Some matrixes
      std_covi = np.linalg.inv(std_cov)
      dg0_covi = np.linalg.inv(dg0_cov)
      cov_comb_inv = np.linalg.inv( std_cov + dg0_cov )
      cov_comb = np.linalg.inv( std_covi + dg0_covi )

      # Check p-value
      chi2_value = (std_w-dg0_w).dot(cov_comb_inv.dot(std_w-dg0_w));
      dof = len(std_w)
      prob = chi2.sf(chi2_value,dof)
      print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}')

      # Combine angular weights
      w = np.ones((dof+1))
      w[1:] = cov_comb.dot( std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T)  )

      # Combine uncertainties
      uw = np.zeros_like(w)
      uw[1:] = np.sqrt(np.diagonal(cov_comb))

      # Build correlation matrix
      corr = np.zeros((dof+1,dof+1))
      for k in range(1,cov_comb.shape[0]):
        for j in range(1,cov_comb.shape[1]):
          corr[k,j] = cov_comb[k][j]/np.sqrt(cov_comb[k][k]*cov_comb[j][j])

      # Create parameters
      merged_w = Parameters()
      for k in range(0,len(w)):
        correl = {f'w{j}':corr[k][j] for j in range(0,len(w)) if k>0 and j>0}
        merged_w.add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                      'free': False, 'latex': f'w_{k}', 'correl': correl})
      print(f"Current angular weights for Bs2JpsiPhi-{y}-{trigger} sample are:")
      print(merged_w)
      data[f'{y}'][trigger].angacc = np.array(merged_w)
      merged_w.dump(f'output_new/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_iteration{i}_{trigger}.json')


# Storing some weights in disk -------------------------------------------------
#    For future use of computed weights created in this loop, these should be
#    saved to the path where samples are stored.
#    GBweighting is slow enough once!
print('Storing weights in root file')
for y, dy in mc.items(): # loop over years
  for m, v in dy.items(): # loop over mc_std and mc_dg0
    pool = {}
    for iter, wvalues in v.pdfWeight.items():
      pool.update({f'pdfWeight{iter}': wvalues})
    for iter, wvalues in v.kkpWeight.items():
      pool.update({f'kkpWeight{iter}': wvalues})
    with uproot.recreate(v.path_to_weights,compression=None) as f:
      this_treename = '&'.join(map(str,YEARS))
      f['DecayTree'] = uproot.newtree({'kinWeight':np.float64})
      f['DecayTree'].extend({'kinWeight':v.kinWeight})
      f[this_treename] = uproot.newtree({var:np.float64 for var in pool})
      f[this_treename].extend(pool)
print(f' * Succesfully writen')




exit()


"""
mc['2015']['MC_BsJpsiPhi'].kkpWeight







    v.kkpWeight = np.zeros_like(v.pdfWeight)
    for t, t_cut in zip(['biased','unbiased'],[v.biased,v.unbiased]):
      print(f'* Reweighting {m}-{t} sample in p and pT of K+ and K- weight at {i} iteration.')
      reweighter.fit(original        = v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     target          = data[t].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     original_weight = t_cut.get()*v.weight.get()*v.pdfWeight*v.kinWeight,
                     target_weight   = data[t].df.eval('sw')
                    );
      kkpWeight = reweighter.predict_weights(v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']])
      kkpWeight = np.where(t_cut.get()*v.weight.get()*v.pdfWeight*v.kinWeight!=0, kkpWeight, 0)
      v.kkpWeight = np.where(kkpWeight!=0, kkpWeight, v.kkpWeight)
      print(f"  kkpWeight = {v.kkpWeight}")












  print('\nReweighting samples')
  for m, v in zip(['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0'],[mc_std,mc_dg0]):
    # do the pdf-weighting -----------------------------------------------------
    print(f'* Calculating pdfWeight in {i} iteration for {m} sample')
    bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **v.params.valuesdict())
    original_pdf_h = v.pdf.get()
    bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **v.params.valuesdict())
    original_pdf_h /= v.pdf.get()
    bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **pars.valuesdict())
    target_pdf_h = v.pdf.get()
    bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **pars.valuesdict())
    target_pdf_h /= v.pdf.get()
    v.pdfWeight = np.nan_to_num(target_pdf_h/original_pdf_h)
    print(f"  pdfWeight: {v.pdfWeight}")

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    v.kkpWeight = np.zeros_like(v.pdfWeight)
    for t, t_cut in zip(['biased','unbiased'],[v.biased,v.unbiased]):
      print(f'* Reweighting {m}-{t} sample in p and pT of K+ and K- weight at {i} iteration.')
      reweighter.fit(original        = v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     target          = data[t].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     original_weight = t_cut.get()*v.weight.get()*v.pdfWeight*v.kinWeight,
                     target_weight   = data[t].df.eval('sw')
                    );
      kkpWeight = reweighter.predict_weights(v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']])
      kkpWeight = np.where(t_cut.get()*v.weight.get()*v.pdfWeight*v.kinWeight!=0, kkpWeight, 0)
      v.kkpWeight = np.where(kkpWeight!=0, kkpWeight, v.kkpWeight)
      print(f"  kkpWeight = {v.kkpWeight}")

      # kinematic-weighting over P and PT of K+ and K- ---------------------------
      ang_acc = bsjpsikk.get_angular_cov(
                v.true,
                v.reco,
                t_cut*v.weight*ristra.allocate(v.kinWeight*v.kkpWeight),
                **v.params.valuesdict()
                )
      w, uw, cov, corr = ang_acc
      v.ang_weights[t] = Parameters()

      for k in range(0,len(w)):
        correl = {f'w{j}':cov[k][j] for j in range(0,len(w)) if k>0 and j>0}
        v.ang_weights[t].add({'name': f'w{k}',
                         'value': w[k],
                         'stdev': uw[k],
                         'free': False,
                         'latex': f'w_{k}',
                         'correl': correl


                          })
      print(f"Corrected angular weights for {m}-{YEAR}-{t} sample are:")
      print(f"{v.ang_weights[t]}")

  # merge sets
  print('\n\nCombining std and dg0 MC angular weights')
  for t, t_cut in zip(['biased','unbiased'],[v.biased,v.unbiased]):
    std = mc_std.ang_weights[t]
    dg0 = mc_dg0.ang_weights[t]
    print(f"Corrected angular weights for MC_BsJpsiPhi-{YEAR}-{t} sample are:")
    print(f"{std}")
    print(f"Corrected angular weights for MC_BsJpsiPhi_dG0-{YEAR}-{t} sample are:")
    print(f"{dg0}")

    std_cov = std.correl_mat()[1:,1:];
    dg0_cov = dg0.correl_mat()[1:,1:];
    std_covi = np.linalg.inv(std_cov)
    dg0_covi = np.linalg.inv(dg0_cov)

    std_w = np.array([std[f'w{i}'].value for i in range(1,len(std))])
    dg0_w = np.array([dg0[f'w{i}'].value for i in range(1,len(dg0))])

    cov_comb_inv = np.linalg.inv( std_cov + dg0_cov )
    cov_comb = np.linalg.inv( std_covi + dg0_covi )

    # Check p-value
    chi2_value = (std_w-dg0_w).dot(cov_comb_inv.dot(std_w-dg0_w));
    dof = len(std_w)
    prob = chi2.sf(chi2_value,dof)
    print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}')

    # Combine angular weights
    w = np.ones((dof+1))
    w[1:] = cov_comb.dot( std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T)  )

    # Combine uncertainties
    uw = np.zeros_like(w)
    uw[1:] = np.sqrt(np.diagonal(cov_comb))

    # Build correlation matrix
    corr = np.zeros((dof+1,dof+1))
    for k in range(1,cov_comb.shape[0]):
      for j in range(1,cov_comb.shape[1]):
        corr[k,j] = cov_comb[k][j]/np.sqrt(cov_comb[k][k]*cov_comb[j][j])

    # Create parameters
    merged_w = Parameters()
    for k in range(0,len(w)):
      correl = {f'w{j}':corr[k][j] for j in range(0,len(w)) if k>0 and j>0}
      merged_w.add({'name': f'w{k}',
                          'value': w[k],
                          'stdev': uw[k],
                          'free': False,
                          'latex': f'w_{k}',
                          'correl': correl
                        })
    print(f"Combined angular weights at iteration {i} for {YEAR}-{TRIGGER} sample are:")
    print(merged_w)

    data[t].ang_weights = np.array(merged_w)
"""
