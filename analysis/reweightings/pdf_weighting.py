#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']
__all__ = ['pdf_weighting', 'dg0_weighting']


# Modules {{{ 

import argparse
import numpy as np
import uproot3 as uproot
from utils.strings import printsec
import os


from ipanema import initialize
from ipanema import ristra, Parameters
import config
initialize(config.user['backend'],1)

from analysis import badjanak
# kernel debugging handlers
badjanak.config['debug_evt'] = 0#2930619
badjanak.config['debug'] = 0
# since HD-fitter always use faddeva for the pdf integral, let also do it here
# this should impy identical pdf weights between us
badjanak.config['fast_integral'] = 0

# }}}


# PDF weighting {{{

def pdf_weighting(data, target_params, original_params, mode):
  # Modify flags, compile model and get kernels
  if mode in ("MC_Bd2JpsiKstar"):
    badjanak.config["mHH"] = [826, 861, 896, 931, 966]
    # WARNING : Here we should be using 511*X_ID/311
    data.eval("B_ID_GenLvl = 511*X_ID_GenLvl/313", inplace=True)
    avars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
             'truehelphi_GenLvl', 'B_TRUETAU_GenLvl', 'X_M', 'sigmat',
             'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']
             # 'X_ID_GenLvl', 'X_ID_GenLvl', 'X_ID_GenLvl', 'X_ID_GenLvl']
             # 'B_ID', 'B_ID', 'B_ID', 'B_ID']
    badjanak.get_kernels()
    cross_rate = badjanak.delta_gamma5_mc
  elif mode in ("MC_Bs2JpsiPhi", "MC_Bs2JpsiPhi_dG0", "MC_Bs2JpsiKK_Swave"):
    badjanak.config["mHH"] = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    avars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
             'truehelphi_GenLvl', 'B_TRUETAU_GenLvl', 'X_M', 'sigmat',
             'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']
    badjanak.get_kernels()
    cross_rate = badjanak.delta_gamma5_mc

  # Load file
  print(data[avars].values)
  vars_h = np.ascontiguousarray(data[avars].values)      # input array (matrix)
  vars_h[:,3] *= 1e3                                               # time in ps
  vars_h[:,5] *= 0                                                 # time in ps
  vars_h[:,8] *= 0                                                 # time in ps
  vars_h[:,9] *= 0                                                 # time in ps
  pdf_h = np.zeros(vars_h.shape[0])                        # output array (pdf)

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

  badjanak.config["mHH"] = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  avars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
           'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
           'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl', 'B_ID_GenLvl']

  badjanak.get_kernels()
  cross_rate = badjanak.delta_gamma5_mc

  # Load file
  vars_h = np.ascontiguousarray(data[avars].values)      # input array (matrix)
  vars_h[:,3] *= 1e3                                               # time in ps
  vars_h[:,5] *= 0                                                 # time in ps
  vars_h[:,8] *= 0                                                 # time in ps
  vars_h[:,9] *= 0                                                 # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                       # output array (pdf)

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

# }}}


# Run and get the job done {{{ 

if __name__ == '__main__':
  
  # parse comandline arguments
  p = argparse.ArgumentParser()
  p.add_argument('--input-file', help='File to add pdfWeight to')
  p.add_argument('--tree-name', help='Name of the original tree')
  p.add_argument('--output-file', help='File to store the ntuple with weights')
  p.add_argument('--target-params', help='Parameters of the target PDF')
  p.add_argument('--original-params', help='Gen parameters of input file')
  p.add_argument('--mode', help='Mode (MC_BsJpsiPhi or MC_BdJpsiKstar)')
  args = vars(p.parse_args())

  printsec(f"PDF weighting")

  # load arguments
  ifile = args['input_file']
  itree = args['tree_name']
  tparams = args['target_params']
  oparams = args['original_params']
  rfile = args['output_file']

  # check for dg0 string in output_file name, so derive what to do
  weight = 'dg0Weight' if 'dg0Weight' in rfile else 'pdfWeight'

  # add weight to dataframe
  print(f'Loading {ifile}')
  df = uproot.open(ifile)[itree].pandas.df()
  if weight == 'pdfWeight':
    pdfW = pdf_weighting(df, tparams, oparams, args['mode'])
    if args['mode'] == 'MC_BsJpsiPhi' or args['mode']=='MC_Bs2JpsiKK_Swave':
      pdfW /= np.array(df['dg0Weight'])
    df['pdfWeight'] = pdfW
    print('pdfWeight was succesfully calculated')
  elif weight == 'dg0Weight':
    df['dg0Weight'] = dg0_weighting(df, tparams, oparams, args['mode'])
    print('dg0Weight was succesfully calculated')

  # save weights to file
  with uproot.recreate(rfile) as rf:
      rf[itree] = uproot.newtree({var:'float64' for var in df})
      rf[itree].extend(df.to_dict(orient='list'))

# }}}


# vim: foldmethod=marker
