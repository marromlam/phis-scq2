#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Modules ----------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import hjson
import importlib

from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import ristra, Sample, Parameters, Parameter, Optimizer

import badjanak


################################################################################
################################################################################
def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file',
                      help='File to correct')
  parser.add_argument('--tree-name', default='DecayTree',
                      help='Name of the original tree')
  parser.add_argument('--output-file',
                      help='File to store the ntuple with weights')
  parser.add_argument('--target-params',
                      help='Parameters of the target PDF')
  parser.add_argument('--original-params', default='DecayTree',
                      help='Name of the original PDF')
  parser.add_argument('--mode',
                      help='Mode (MC_BsJpsiPhi or MC_BdJpsiKstar)')
  return parser
################################################################################
################################################################################



# pdf_weighting ----------------------------------------------------------------

def pdf_weighting(data, target_params, original_params, mode):
  # Modify flags, compile model and get kernels
  badjanak.config['debug_evt'] = 0
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
  pdf_d  = ristra.allocate(pdf_h).astype(np.float64)

  # Compute!
  print('Calc weights...')
  #original_params = hjson.load(open(original_params))
  #target_params = hjson.load(open(target_params))
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

  #bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights']
  # bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root')['PDFWeights']
  # #
  # print(bee.keys())
  # print(bee.array('PDFWeight')[0],bee.array('PDFWeight')[-1])
  # print(pdfWeight[0],pdfWeight[-1])
  # print(f"CHECK: {np.amax( pdfWeight - bee.array('PDFWeight') )}")
  #
  #

  return pdfWeight

# tree_name, output_file,
if __name__ == '__main__':
    parser = argument_parser()
    #pdf_weighting(**vars(args))
    args = vars(parser.parse_args())
    print(f"\n{80*'='}\n{'= PDF weighting':79}=\n{80*'='}\n")

    input_file = args['input_file']
    tree_name = args['tree_name']
    target_params = args['target_params']
    original_params = args['original_params']
    output_file = args['output_file']
    mode = args['mode']
    print(f"target: {args['target_params']}")
    print(f"original: {args['original_params']}")

    print('Loading file...')
    df = uproot.open(input_file)[tree_name].pandas.df(flatten=None)
    df['pdfWeight'] = pdf_weighting(df, target_params, original_params, mode)
    print('pdfWeight was succesfully calculated')

    # Save weights to file -------------------------------------------------------
    #    It would be nice that uproot.update worked, but it is not yet avaliable
    #    and the function is a placeholder only according to the author. So, at
    #    moment it is needed to load the whole df and write a new one :(.
    #    This produces some problems with large files, so we are using root_pandas
    #    to store files whilst uproot can't
    if os.path.exists(output_file):
      print('Deleting previous %s'  % output_file)
      os.remove(output_file)                               # delete file if exists
    #os.system('cp '+original_file+' '+output_file)
    print('Writing on %s' % output_file)
    #import root_pandas
    #root_pandas.to_root(df, output_file, key=tree_name)
    f = uproot.recreate(output_file)
    f[tree_name] = uproot.newtree({var:'float64' for var in df})
    f[tree_name].extend(df.to_dict(orient='list'))
    f.close()
    print('pdfWeight was succesfully written\n')
