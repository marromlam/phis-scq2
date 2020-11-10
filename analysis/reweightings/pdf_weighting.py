#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']

__all__ = ['pdf_weighting']


################################################################################
#%% Modules ####################################################################

import argparse
import numpy as np
import pandas as pd
import uproot
import os, sys
import hjson
import importlib

from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import ristra, Sample, Parameters, Parameter, Optimizer

import badjanak

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser()
  p.add_argument('--input-file', help='File to add pdfWeight to')
  p.add_argument('--tree-name', help='Name of the original tree')
  p.add_argument('--output-file', help='File to store the ntuple with weights')
  p.add_argument('--target-params', help='Parameters of the target PDF')
  p.add_argument('--original-params', help='Gen parameters of input file')
  p.add_argument('--mode', help='Mode (MC_BsJpsiPhi or MC_BdJpsiKstar)')
  return p


################################################################################


################################################################################
# pdf_weighting ################################################################

def pdf_weighting(data, target_params, original_params, mode):
  # Modify flags, compile model and get kernels
  badjanak.config['debug_evt'] = 2930619
  badjanak.config['debug'] = 0
  badjanak.config['fast_integral'] = 0

  if mode == "MC_Bd2JpsiKstar":
    badjanak.config["x_m"] = [826, 861, 896, 931, 966]
    tad_vars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
                'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
                'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']
  elif mode.startswith("MC_Bs2JpsiPhi"):
    badjanak.config["x_m"] = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    tad_vars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
                'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
                'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']

  badjanak.get_kernels(True)
  cross_rate = badjanak.delta_gamma5_mc

  # Load file
  vars_h = np.ascontiguousarray(data[tad_vars].values)    # input array (matrix)
  vars_h[:,3] *= 1e3                                                # time in ps
  vars_h[:,5] *= 0                                                  # time in ps
  vars_h[:,8] *= 0                                                  # time in ps
  vars_h[:,9] *= 0                                                  # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                        # output array (pdf)
  # Allocate device_arrays
  vars_d = ristra.allocate(vars_h).astype(np.float64)
  pdf_d  = ristra.allocate(pdf_h).astype(np.float64)
  # Compute!
  original_params = Parameters.load(original_params)
  target_params = Parameters.load(target_params)
  cross_rate(vars_d,pdf_d,**original_params.valuesdict(),tLL=0.3,tUL=15.0);
  original_pdf_h = pdf_d.get()
  cross_rate(vars_d,pdf_d,**target_params.valuesdict(),tLL=0.3,tUL=15.0);
  target_pdf_h = pdf_d.get()
  np.seterr(divide='ignore', invalid='ignore')                 # remove warnings
  pdfWeight = np.nan_to_num(original_pdf_h/target_pdf_h)

  print(f"{'#':>3} | {'cosK':>11} | {'cosL':>11} | {'hphi':>11} | {'time':>11} | {'X_M':>14} | {'B_ID':>4} | {'original':>11} | {'target':>11} | {'pdfWeight':>11}")
  for i in range(0,20):
    print(f"{i:>3} | {vars_h[i,0]:>+.8f} | {vars_h[i,1]:>+.8f} | {vars_h[i,2]:>+.8f} | {vars_h[i,3]:>+.8f} | {vars_h[i,4]:>+4.8f} | {vars_h[i,6]:>+.0f} | {original_pdf_h[i]:>+.8f} | {target_pdf_h[i]:>+.8f} | {pdfWeight[i]:>+.8f}")

  return pdfWeight

################################################################################


################################################################################
# dg0_weighting ################################################################

def dg0_weighting(data, target_params, original_params, mode):
  # Modify flags, compile model and get kernels
  badjanak.config['debug_evt'] = 0
  badjanak.config['debug'] = 0
  badjanak.config['fast_integral'] = 0

  badjanak.config["x_m"] = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  tad_vars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
              'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
              'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']

  badjanak.get_kernels()
  cross_rate = badjanak.delta_gamma5_mc

  # Load file
  vars_h = np.ascontiguousarray(data[tad_vars].values)    # input array (matrix)
  vars_h[:,3] *= 1e3                                                # time in ps
  vars_h[:,5] *= 0                                                  # time in ps
  vars_h[:,8] *= 0                                                  # time in ps
  vars_h[:,9] *= 0                                                  # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                        # output array (pdf)

  # Allocate device_arrays
  vars_d = ristra.allocate(vars_h).astype(np.float64)
  pdf_d = ristra.allocate(pdf_h).astype(np.float64)

  # Compute!
  original_params = Parameters.load(original_params)
  target_params = Parameters.load(target_params)
  cross_rate(vars_d,pdf_d,**original_params.valuesdict(),tLL=0.3,tUL=15.0);
  original_pdf_h = pdf_d.get()
  cross_rate(vars_d,pdf_d,**target_params.valuesdict(),tLL=0.3,tUL=15.0);
  target_pdf_h = pdf_d.get()
  np.seterr(divide='ignore', invalid='ignore')                 # remove warnings
  dg0Weight = np.nan_to_num(original_pdf_h/target_pdf_h)

  print(f"{'#':>3} | {'cosK':>11} | {'cosL':>11} | {'hphi':>11} | {'time':>11} | {'X_M':>14} | {'B_ID':>4} | {'original':>11} | {'target':>11} | {'dg0Weight':>11}")
  for i in range(0,20):
    print(f"{i:>3} | {vars_h[i,0]:>+.8f} | {vars_h[i,1]:>+.8f} | {vars_h[i,2]:>+.8f} | {vars_h[i,3]:>+.8f} | {vars_h[i,4]:>+4.8f} | {vars_h[i,6]:>+.0f} | {original_pdf_h[i]:>+.8f} | {target_pdf_h[i]:>+.8f} | {dg0Weight[i]:>+.8f}")

  return dg0Weight

################################################################################



################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':
  args = vars(argument_parser().parse_args())
  print(f"\n{80*'='}\nPDF weighting\n{80*'='}\n")

  input_file = args['input_file']
  tree_name = args['tree_name']
  target_params = args['target_params']
  original_params = args['original_params']
  output_file = args['output_file']
  mode = args['mode']

  print(f'Loading {input_file}')
  df = uproot.open(input_file)[tree_name].pandas.df(flatten=None)
  df['pdfWeight'] = pdf_weighting(df, target_params, original_params, mode)
  print('pdfWeight was succesfully calculated')
  if 'MC_Bs2JpsiPhi' == mode:
    df['dg0Weight'] = dg0_weighting(df, target_params, original_params, mode)
    print('dg0Weight was succesfully calculated')

  # Save weights to file -------------------------------------------------------
  #    Save pdfWeight to the file
  if os.path.exists(output_file):
    print(f'Deleting previous {output_file}')
    os.remove(output_file)                               # delete file if exists
  print(f'Writing on {output_file}')
  with uproot.recreate(output_file) as f:
    f[tree_name] = uproot.newtree({var:'float64' for var in df})
    f[tree_name].extend(df.to_dict(orient='list'))
  print('pdfWeight was succesfully written\n')

################################################################################
