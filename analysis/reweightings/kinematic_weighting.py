DESCRIPTION = """
  Kinematic reweighter
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['reweight']


# Modules {{{

import argparse
import numpy as np
import uproot3 as uproot
import ast
from shutil import copyfile
import math
import hjson
import yaml

ROOT_PANDAS = False
if ROOT_PANDAS:
  # import root_pandas
  import root_numpy

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml.reweight import GBReweighter

from utils.strings import printsec
from utils.helpers import trigger_scissors
from angular_acceptance.bdtconf_tester import bdtmesh
from config import timeacc

# }}}


# Reweightings functions {{{

def reweight(original, target, original_weight, target_weight,
             n_estimators, learning_rate, max_depth, min_samples_leaf, trunc):
  """
  This is the general reweighter for phis analysis.

  Parameters
  ----------
  original : pandas.DataFrame
  DataFrame for the original sample (the one which will be reweighted).
  Should only include the variables that will be reweighted.
  target : pandas.DataFrame
  DataFrame for the target sample.
  Should only include the variables that will be reweighted.
  original_weight : str
  String with the applied weight for the original sample.
  target_weight : str
  String with the applied weight for the target sample.
  n_estimators : int
  Number of estimators for the gb-reweighter.
  learning_rate : float
  Learning rate for the gb-reweighter.
  max_depth : int
  Maximum depth of the gb-reweighter tree.
  min_samples_leaf : int
  Minimum number of leaves for the gb-reweighter tree.

  Returns
  -------
  np.ndarray
  Array with the computed weights in order to original to be the same as target.
  """
  # setup the reweighter
  reweighter = GBReweighter(n_estimators     = int(n_estimators),
                            learning_rate    = float(learning_rate),
                            max_depth        = int(max_depth),
                            min_samples_leaf = int(min_samples_leaf),
                            gb_args          = {'subsample': 1})
  # perform the fit
  reweighter.fit(original=original, target=target,
                 original_weight=original_weight, target_weight=target_weight);

  # predict the weights
  kinWeight = reweighter.predict_weights(original)

  # use truncation if set, flush to zero
  if int(trunc):
    print('Apply a truncation at '+trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)

  # put to zero all reweight which are zero at start
  kinWeight = np.where(original_weight!=0, kinWeight, 0)

  return kinWeight

# }}}


# MOVE THIS TO INIT {{{

def kinematic_weighting(original_file, original_treename, original_vars,
                        original_weight, target_file, target_treename,
                        target_vars, target_weight, output_file, weight_set,
                        n_estimators, learning_rate, max_depth,
                        min_samples_leaf, gb_args, trunc=False):

  # find all needed branches
  # all_original_vars = original_vars + getStringVars(original_weight)
  # all_target_vars   = target_vars + getStringVars(target_weight)

  # fetch variables in original files
  print('Loading branches for original_sample')
  odf = uproot.open(original_file)[original_treename].pandas.df()
  print(odf)
  print('Loading branches for target_sample')
  tdf = uproot.open(target_file)[target_treename].pandas.df()
  print(tdf)

  print(f"Original weight = {original_weight}")
  print(odf.eval(original_weight))
  print(f"Target weight = {target_weight}")
  print(tdf.eval(target_weight))

  # sws = ['sw'] + [f'sw_{var}' for var in binned_vars.keys()]
  # kws = [f'{weight}'] + [f'{weight}_{var}' for var in binned_vars.keys()]

  # print('Original sWeights')
  # print(ovars_df[sws])
  # print('Target sWeights')
  # print(tvars_df[sws])
  print("Starting dataframe")
  print(odf, tdf)


  # Reweighting ---------------------------------------------------------------
  theWeight = np.zeros_like(list(odf.index)).astype(np.float64)
  for trig in ['biased', 'unbiased']:
    codf = odf.query(trigger_scissors(trig))
    ctdf = tdf.query(trigger_scissors(trig))
    cweight = reweight(codf.get(original_vars), ctdf.get(target_vars),
                       codf.eval(original_weight), ctdf.eval(target_weight),
                       n_estimators, learning_rate, max_depth,
                       min_samples_leaf, trunc)
    theWeight[list(codf.index)] = cweight
  odf[weight_set] = theWeight
  # new_vars.append(np.array(ovars_df[f'{weight}'],dtype=[(f'{weight}',np.float64)]))
  # for var,vvar in binned_vars.items():
  #   kinWeight = np.zeros_like(ovars_df[f'{weight}'].values)
  #   for cut in bin_vars[var]:
  #     original_weight_binned = original_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
  #     target_weight_binned = target_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
  #     print("ORIGINAL WEIGHT =", original_weight_binned)
  #     print("  TARGET WEIGHT =", target_weight_binned)
  #     kinWeight_ =  reweight(
  #                   ovars_df.get(original_vars),
  #                   tvars_df.get(target_vars),
  #                   ovars_df.eval(original_weight_binned),
  #                   tvars_df.eval(target_weight_binned),
  #                   n_estimators,learning_rate,max_depth,min_samples_leaf,
  #                   trunc)
  #     kinWeight = np.where(ovars_df.eval(cut.replace(var,vvar)), kinWeight_, kinWeight)
  #   new_vars.append( np.array(kinWeight,dtype=[(f'{weight}_{var}',np.float64)]) )
  #   ovars_df[f'{weight}_{var}'] = kinWeight 

  print('Final dataframe')
  print(odf)


  # Save weights to file ------------------------------------------------------
  print('Writing on %s' % output_file)
  if ROOT_PANDAS:
    copyfile(original_file, output_file)
    # for var in new_vars:
    #   root_numpy.array2root(var, output_file, original_treename, mode='update')
    theWeight = np.array(odf[weight_set].values, dtype=[(weight_set, np.float64)])
    root_numpy.array2root(theWeight, output_file, original_treename, mode='update')
    # root_pandas.to_root(odf, output_file, key=original_treename, mode='a')
  else:
    f = uproot.recreate(output_file)
    f[original_treename] = uproot.newtree({var:'float64' for var in odf})
    f[original_treename].extend(odf.to_dict(orient='list'))
    f.close()
  return odf[weight_set].values

# }}}


# Command Line Interface {{{

if __name__ == '__main__':
  # parse comandline arguments
  p = argparse.ArgumentParser()
  p.add_argument('--original-file', help='File to correct')
  p.add_argument('--original-treename', default='DecayTree', help='Name of the original tree')
  p.add_argument('--target-file', help='File to reweight to')
  p.add_argument('--target-treename', default='DecayTree', help='Name of the target tree')
  p.add_argument('--output-file', help='Branches string expression to calc the weight.')
  p.add_argument('--weight-set', default='kbsWeight', help='Branches string expression to calc the weight.')
  p.add_argument('--version', help='Branches string expression to calc the weight.')
  p.add_argument('--year', help='Branches string expression to calc the weight.')
  p.add_argument('--mode', help='Branches string expression to calc the weight.')
  args = vars(p.parse_args())

  with open('analysis/reweightings/config.yml') as file:
    reweight_config = yaml.load(file, Loader=yaml.FullLoader)
  reweight_config = reweight_config[args["weight_set"]][args["mode"]]

  with open('analysis/samples/branches.yaml') as file:
    sWeight = yaml.load(file, Loader=yaml.FullLoader)
  sWeight = sWeight[args['mode']]['sWeight']
  print(sWeight)

  args["original_vars"] = reweight_config["variables"]
  args["target_vars"] = reweight_config["variables"]
  args["original_weight"] = reweight_config["original"][0].format(sWeight=sWeight)
  args["target_weight"] = reweight_config["target"][0]

  # change bdt according to filename, if applies
  bdtconfig = timeacc['bdtconfig']
  if 'bdt' in args['version'].split('@')[0]:
    bdtconfig = int(version.split('bdt')[1])
    bdtconfig = bdtmesh(bdtconfig, settings.general['bdt_tests'], False)

  # delete some keys and update with bdtconfig
  del args['version']
  del args['year']
  del args['mode']
  args.update(**bdtconfig)
  args.update({"trunc": False})

  # run the kinematic weight
  printsec("Kinematic reweighting")
  for k,v in args.items():
    print(f"{k:>25} : {v}")
  kinematic_weighting(**args)

# }}}


#vim:foldmethod=marker
