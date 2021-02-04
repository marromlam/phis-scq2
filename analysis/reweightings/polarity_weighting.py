#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import uproot3 as uproot
from shutil import copyfile
import os

ROOT_PANDAS = True
if ROOT_PANDAS:
  import root_pandas
  import root_numpy


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
                       output_file, verbose=False):
  # Get data
  odf = uproot.open(original_file)[original_treename].pandas.df(flatten=None)
  original_polarity = odf["Polarity"].values
  #original_polarity = odf["magnet"].values
  tdf = uproot.open(target_file)[target_treename].pandas.df(branches="Polarity")
  target_polarity = tdf["Polarity"].values
  #tdf = uproot.open(targ#et_file)[target_treename].pandas.df(branches="magnet")
  #target_polarity = tdf["magnet"].values
  print('DataFrames are ready')
  print(odf)

  # Cook weights
  original_mean = np.mean(original_polarity)
  target_mean   = np.mean(target_polarity)
  original_down = np.sum(np.where(original_polarity<0,1,0))
  original_up   = np.sum(np.where(original_polarity>0,1,0))

  weight_up   = 1
  weight_down = ( (1+original_mean) * (1-target_mean) ) /\
                ( (1-original_mean) * (1+target_mean) )

  weight_scale = (original_down + original_up) /\
                 (weight_down*original_down + weight_up*original_up)
  weight_up   *= weight_scale
  weight_down *= weight_scale

  polWeight = np.where(original_polarity>0, weight_up, weight_down)

  # bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight']
  # bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root')['PolWeight']
  #
  # print(bee.array('PolWeight')[0],bee.array('PolWeight')[-1])
  # print(polWeight[0],polWeight[-1])
  # print(f"CHECK: {np.amax( polWeight - bee.array('PolWeight') )}")

  odf['polWeight'] = polWeight
  print('polWeight was succesfully calculated')
  print(odf)

  # Save weights to file
  #    It would be nice that uproot.update worked, but it is not yet avaliable
  #    and the function is a placeholder only according to the author. So, at
  #    moment it is needed to load the whole df and write a new one :(.
  #    This produces some problems with large files, so we are using root_pandas
  #    to store files whilst uproot can't
  if ROOT_PANDAS:
    copyfile(original_file, output_file)
    var = np.array(polWeight, dtype=[('polWeight', np.float64)])
    root_numpy.array2root(var, output_file, original_treename, mode='update')
  else:
    f = uproot.recreate(output_file)
    f[original_treename] = uproot.newtree({var:'float64' for var in odf})
    f[original_treename].extend(odf.to_dict(orient='list'))
    f.close()
  print('polWeight was succesfully written.')

  return polWeight


if __name__ == '__main__':
  parser = argument_parser()
  args = parser.parse_args()
  print(f"\n{80*'='}\nPolarity weighting\n{80*'='}\n")
  polarity_weighting(**vars(args), verbose=True)
  print(f"\n")
