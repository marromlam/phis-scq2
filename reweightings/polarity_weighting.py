#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os



# %% polarity_weighting --------------------------------------------------------

def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--original-file',
                      help='File to correct')
  parser.add_argument('--original-treename', default='DecayTree',
                      help='Name of the original tree')
  parser.add_argument('--target-file',
                      help='File to reweight to')
  parser.add_argument('--target-treename', default='DecayTree',
                      help='Name of the target tree')
  parser.add_argument('--output-file',
                      help='File to store the ntuple with weights')
  return parser



def polarity_weighting(original_file, original_treename,
                       target_file, target_treename,
                       output_file):
  print(f"\n{80*'='}\n{'= Polarity weighting':79}=\n{80*'='}\n")
  # Get data
  try:
    _original_file = uproot.open(original_file)
    _original_tree = _original_file[original_treename]
    print('Loaded %s correctly' % original_file)
  except:
    print('polarity-weighting.py: Missing original file. Exiting...')
    raise
  try:
    _target_file = uproot.open(target_file)
    _target_tree = _target_file[target_treename]
    print('Loaded %s correctly' % target_file)
  except:
    print('polarity-weighting.py: Missing target file. Exiting...')
    raise

  if "polWeight".encode() in _original_tree.keys():
    print('polarity-weighting.py: This file already has a polWeight. Exiting...')
    exit()
  else:
    original_df       = _original_tree.pandas.df(flatten=False)
    original_polarity = original_df["Polarity"].values
    target_df         = _target_tree.pandas.df(branches="Polarity")
    target_polarity   = target_df["Polarity"].values
  print('DataFrames are ready')

  # Cook weights
  original_mean = np.mean(original_polarity)
  target_mean   = np.mean(target_polarity)
  target_down   = np.sum(np.where(original_polarity<0,1,0))
  target_up     = np.sum(np.where(original_polarity>0,1,0))

  weight_up   = 1
  weight_down = ( (1+target_mean) * (1-original_mean) ) /\
                ( (1-target_mean) * (1+original_mean) )

  weight_scale = (target_down + target_up) /\
                 (weight_down*target_down + weight_up*target_up)
  weight_up   *= weight_scale
  weight_down *= weight_scale

  polWeight = np.where(original_polarity<0, weight_up,weight_down)
  original_df['polWeight'] = polWeight
  print('polWeight was succesfully calculated')


  # Save weights to file
  #    It would be nice that uproot.update worked, but it is not yet avaliable
  #    and the function is a placeholder only according to the author. So, at
  #    moment it is needed to load the whole df and write a new one :(.
  #    This produces some problems with large files, so we are using root_pandas
  #    to store files whilst uproot can't
  if os.path.exists(output_file):
    print('Deleting previous %s'  % output_file)
    os.remove(output_file)                               # delete file if exists
  print('Writing on %s' % output_file)
  import root_pandas
  root_pandas.to_root(original_df, output_file, key=original_treename)
  #f = uproot.recreate(output_file)
  #f[original_treename] = uproot.newtree({var:'float64' for var in original_df})
  #f[original_treename].extend(original_df.to_dict(orient='list'))
  #f.close()
  print('polWeight was succesfully written.')

  return polWeight


if __name__ == '__main__':
  parser = argument_parser()
  args = parser.parse_args()
  polarity_weighting(**vars(args))
