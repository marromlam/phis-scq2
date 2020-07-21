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
from scipy.stats import chi2


# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters


################################################################################



################################################################################
#%% ############################################################################


def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  # Output parameters
  parser.add_argument('--weights-std',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--weights-dg0',
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
  parser.add_argument('--step',
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
VERSION = 'v0r0'
STEP = 'naive'
TRIGGER = 'unbiased'
weights_std_path = f'output_new/params/angular_acceptance/{YEAR}/MC_Bs2JpsiPhi/{VERSION}_{STEP}_{TRIGGER}.json'
weights_dg0_path = f'output_new/params/angular_acceptance/{YEAR}/MC_Bs2JpsiPhi_dG0/{VERSION}_{STEP}_{TRIGGER}.json'
output_tables_path = f'output_new/tables/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_{STEP}_{TRIGGER}.tex'
output_params_path = f'output_new/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_{STEP}_{TRIGGER}.json'

#print(weights_std_path,weights_dg0_path,output_tables_path)

YEAR = args['year']
VERSION = args['version']
STEP = args['step']
TRIGGER = args['trigger']
weights_std_path = args['weights_std']
weights_dg0_path = args['weights_dg0']
output_tables_path = args['output_tables']
output_params_path = args['output_params']

#print(weights_std_path,weights_dg0_path,output_tables_path)

################################################################################
################################################################################
################################################################################







# %% Load samples --------------------------------------------------------------

std = Parameters.load(weights_std_path)
dg0 = Parameters.load(weights_dg0_path)

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
for i in range(1,cov_comb.shape[0]):
  for j in range(1,cov_comb.shape[1]):
    corr[i,j] = cov_comb[i][j]/np.sqrt(cov_comb[i][i]*cov_comb[j][j])

# Create parameters
pars = Parameters()
for i in range(0,len(w)):
  print(f'w[{i}] = {w[i]:+.16f}')
  correl = {f'w{j}':corr[i][j] for j in range(0,len(w)) if i>0 and j>0}
  pars.add({'name': f'w{i}',
                        'value': w[i],
                        'stdev': uw[i],
                        'free': False,
                        'latex': f'w_{i}',
                        'correl': correl
                      })
print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8}")
for _i in range(len(pars.keys())):
  print(f"{np.array(std)[_i]:+1.5f} | {np.array(dg0)[_i]:+1.5f} | {pars[f'w{_i}'].uvalue:+1.2uP}")
# Dump the parameters
print('Dumping parameters')
pars.dump(output_params_path)
# Export parameters in tex tables
print('Saving table of params in tex')
with open(output_tables_path, "w") as tex_file:
  tex_file.write(
    pars.dump_latex( caption="""
    %s angular weights for \\textbf{%s} \\texttt{\\textbf{%s}}
    category using combined MC samples.""" % (STEP.title(),YEAR,TRIGGER)
    )
  )
tex_file.close()
print(f"Combined {STEP} angular weights for {YEAR}-{TRIGGER} sample are:")
print(pars)
