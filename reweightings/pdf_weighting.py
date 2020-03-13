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

# openCL stuff - this can be changed to be handled by ipanema
import builtins
import reikna.cluda as cluda
api = cluda.ocl_api() # OpenCL API

builtins.THREAD = api.Thread.create()
builtins.CONTEXT = THREAD._context
builtins.BACKEND = 'opencl'
builtins.DEVICE = THREAD._device

import bsjpsikk

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

def pdf_weighting(input_file, tree_name, output_file,
                  target_params, original_params,
                  mode):

  print(f"\n{80*'='}\n{'= PDF weighting':79}=\n{80*'='}\n")
  # Modify flags, compile model and get kernels
  bsjpsikk.config['debug_evt'] = 1
  bsjpsikk.config['debug'] = 0
  bsjpsikk.config['use_time_acc'] = 0
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res'] = 0
  bsjpsikk.config['use_perftag'] = 1
  bsjpsikk.config['use_truetag'] = 0

  if mode == "MC_Bd2JpsiKstar":
    bsjpsikk.config["x_m"] = [826, 861, 896, 931, 966]
    tad_vars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
                'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
                'B_ID'] # Final names!
  elif mode.startswith("MC_Bs2JpsiPhi"):
    bsjpsikk.config["x_m"] = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    tad_vars = ['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl',
                'truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat',
                'B_ID_GenLvl'] # Final names!

  bsjpsikk.get_kernels()
  cross_rate = bsjpsikk.diff_cross_rate

  # Load file
  print('Loading file...')
  input_file = uproot.open(input_file)[tree_name]
  data = input_file.pandas.df(flatten=False)
  vars_h = np.ascontiguousarray(data[tad_vars].values)    # input array (matrix)
  vars_h[:,3] *= 1e3                                                # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                        # output array (pdf)
  # Allocate device_arrays
  vars_d = THREAD.to_device(vars_h).astype(np.float64)
  pdf_d  = THREAD.to_device(pdf_h).astype(np.float64)


  # Compute!
  print('Calc weights...')
  original_params = hjson.load(open(original_params))
  original_params
  target_params = hjson.load(open(target_params))
  cross_rate(vars_d,pdf_d,**original_params); original_pdf_h = pdf_d.get()
  cross_rate(vars_d,pdf_d,**target_params); target_pdf_h = pdf_d.get()
  np.seterr(divide='ignore', invalid='ignore')                 # remove warnings
  pdfWeight = np.nan_to_num(original_pdf_h/target_pdf_h)
  data['pdfWeight'] = pdfWeight
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
  import root_pandas
  root_pandas.to_root(data, output_file, key=tree_name)
  #f = uproot.recreate(output_file)
  #f[tree_name] = uproot.newtree({var:'float64' for var in data})
  #f[tree_name].extend(data.to_dict(orient='list'))
  #f.close()
  print('pdfWeight was succesfully written.')

  return 0


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    pdf_weighting(**vars(args))
