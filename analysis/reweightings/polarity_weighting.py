DESCRIPTION = """
  Compute polarity weight between data and MC.
"""


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['polarity_weighting']


# Modules {{{

import argparse
import numpy as np
import uproot3 as uproot
from utils.strings import printsec

ROOT_PANDAS = False
if ROOT_PANDAS:
  from shutil import copyfile
  import root_numpy
  # import root_pandas

# }}}


# Polarity Weighting {{{

def polarity_weighting(original_polarity, target_polarity, verbose=False):
  # load polarity
  original_polarity = odf["Polarity"].values
  target_polarity = tdf["Polarity"].values

  # cook weights
  original_mean = np.mean(original_polarity)
  target_mean = np.mean(target_polarity)
  original_down = np.sum(np.where(original_polarity<0,1,0))
  original_up = np.sum(np.where(original_polarity>0,1,0))

  weight_up = 1
  weight_down = ( (1 + original_mean) * (1 - target_mean) ) /\
                ( (1 - original_mean) * (1 + target_mean) )

  weight_scale = (original_down + original_up) /\
                 (weight_down*original_down + weight_up*original_up)
  weight_up *= weight_scale
  weight_down *= weight_scale

  polWeight = np.where(original_polarity>0, weight_up, weight_down)
  odf['polWeight'] = polWeight

  # TO BE REMOVED
  # bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight']
  # bee = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root')['PolWeight']
  #
  # print(bee.array('PolWeight')[0],bee.array('PolWeight')[-1])
  # print(polWeight[0],polWeight[-1])
  # print(f"CHECK: {np.amax( polWeight - bee.array('PolWeight') )}")

  return odf

# }}}


# CMDLINE interface {{{

if __name__ == '__main__':

  # parse command line arguments
  p = argparse.ArgumentParser(DESCRIPTION)
  p.add_argument('--original-file',
                 help='File to correct')
  p.add_argument('--original-treename', default='DecayTree',
                 help='Name of the original tree')
  p.add_argument('--target-file',
                 help='File to reweight to')
  p.add_argument('--target-treename', default='DecayTree',
                 help='Name of the target tree')
  p.add_argument('--output-file',
                 help='File to store the ntuple with weights')
  args = vars(p.parse_args())

  # load tuples
  ofile = args['original_file']; tfile = args['target_file']
  otree = args['original_treename']; ttree = args['target_treename']
  odf = uproot.open(ofile)[otree].pandas.df(flatten=None)
  tdf = uproot.open(tfile)[ttree].pandas.df(branches="Polarity")

  # run 
  printsec("Polarity weighting")
  print(tdf)
  if 'magUp' in args['output_file'] or 'magDown' in args['output_file']:
    # since we are cutting the sample to have only one category of magnet
    # polarity, we should not compute the polWeight the regular way, otherwise
    # Infs will appear.
    odf.eval("polWeight=time/time", inplace=True)
  else:
    odf = polarity_weighting(odf, tdf, verbose=True)
  print(odf)

  # Save weights to file
  if ROOT_PANDAS:
    copyfile(ofile, tfile)
    polWeight = np.array(odf['polWeight'].values, dtype=[('polWeight', np.float64)])
    root_numpy.array2root(polWeight, ofile, otree, mode='update')
    # root_pandas.to_root(odf, args['output_file'], key=otree)
  else:
    with uproot.recreate(args['output_file']) as rf:
      rf[otree] = uproot.newtree({var:'float64' for var in odf})
      rf[otree].extend(odf.to_dict(orient='list'))
  print('polWeight was succesfully written.')

# }}}


# vim: foldmethod=marker
