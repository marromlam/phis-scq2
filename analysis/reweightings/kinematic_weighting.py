__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['reweight']


# Modules {{{

import config
from config import timeacc
from utils.helpers import trigger_scissors
from utils.strings import printsec
from hep_ml.reweight import GBReweighter
from warnings import simplefilter
import yaml
# import hjson
# import math
# from shutil import copyfile
# import ast
import uproot3 as uproot
import numpy as np
import argparse
from ipanema.tools.misc import get_vars_from_string

# TODO: create resimplement this
from analysis.angular_acceptance.bdtconf_tester import bdtmesh

simplefilter(action='ignore', category=FutureWarning)


# }}}


# Reweightings functions {{{

def reweight(original, target, original_weight, target_weight,
             n_estimators=20, learning_rate=0.1, max_depth=3,
             min_samples_leaf=1000, trunc=False):
  r"""
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
  Array with the computed weights in order to original to be the same as
  target.
  """
  # setup the reweighter
  reweighter = GBReweighter(n_estimators=int(n_estimators),
                            learning_rate=float(learning_rate),
                            max_depth=int(max_depth),
                            min_samples_leaf=int(min_samples_leaf),
                            gb_args={'subsample': 1})
  # perform the fit
  reweighter.fit(original=original, target=target,
                 original_weight=original_weight,
                 target_weight=target_weight)

  # predict the weights
  kinWeight = reweighter.predict_weights(original)

  # use truncation if set, flush to zero
  if int(trunc):
    print('Apply a truncation at ' + trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)

  # put to zero all reweight which are zero at start
  kinWeight = np.where(original_weight != 0, kinWeight, 0)

  return kinWeight

# }}}


# MOVE THIS TO INIT {{{

def kinematic_weighting(original_file, original_treename, original_vars,
                        original_weight, target_file, target_treename,
                        target_vars, target_weight, output_file, weight_set,
                        n_estimators, learning_rate, max_depth,
                        min_samples_leaf, gb_args, trunc=False):

  # fetch variables in original files
  print('Loading branches for original_sample')
  odf = uproot.open(original_file)[original_treename].pandas.df(flatten=None)
  try:
    odf['phiHH'] = odf.eval(
        "arctan((hminus_PY+hplus_PY)/(hminus_PX+hplus_PX))")
  except:
    odf['phiHH'] = odf.eval("time/time")
    print(f'You cannot calculate the phi of the phi for {original_file}')
  # print(odf)
  print('Loading branches for target_sample')
  tdf = uproot.open(target_file)[target_treename].pandas.df(flatten=None)
  try:
    tdf['phiHH'] = tdf.eval(
        "arctan((hminus_PY+hplus_PY)/(hminus_PX+hplus_PX))")
  except:
    tdf['phiHH'] = tdf.eval("time/time")
    print(f'You cannot calculate the phi of the phi for {target_file}')
  # print(tdf)

  # print(f"Original weight = {original_weight}")
  # print(odf.eval(original_weight))
  # print(f"Target weight = {target_weight}")
  # print(tdf.eval(target_weight))

  print("Starting dataframes")

  check_result = False
  if 'v0r0' in output_file:
    print("NOTE: This is v0r0 tuple, special config is loading")
    config.user['reweightings_per_trigger'] = False
    check_result = True
  if config.user['reweightings_per_trigger']:
    TRIGGER = ['biased', 'unbiased']
  else:
    TRIGGER = ['combined']

  # Reweighting
  theWeight = np.zeros_like(list(odf.index)).astype(np.float64)
  for trig in TRIGGER:
    if config.user['reweightings_per_trigger']:
      codf = odf.query(trigger_scissors(trig))  # .sample(frac=1)
      ctdf = tdf.query(trigger_scissors(trig))  # .sample(frac=1)
    else:
      codf = odf  # .sample(frac=1)  # .reset_index(drop=True)
      ctdf = tdf  # .sample(frac=1)  # .reset_index(drop=True)
    cw = reweight(codf.get(original_vars), ctdf.get(target_vars),
                  codf.eval(original_weight), ctdf.eval(target_weight),
                  n_estimators, learning_rate, max_depth,
                  min_samples_leaf, trunc)
    theWeight[list(codf.index)] = cw
  odf[weight_set] = theWeight

  # Check maximum difference wrt. to the kinWeight variable in the tuple
  if check_result and 'kinWeight' in odf.keys():
    print("Max. diff. wrt. Simon:",
          np.max(odf['kinWeight'] - odf[weight_set]))

  print('Final dataframe of weights')
  print(odf[get_vars_from_string(original_weight) + [weight_set]])

  # Save weights to file
  print(f'Writing to {output_file}')
  with uproot.recreate(output_file) as f:
    f[original_treename] = uproot.newtree({var: 'float64' for var in odf})
    f[original_treename].extend(odf.to_dict(orient='list'))
  return odf[weight_set].values

# }}}


# Command Line Interface {{{

if __name__ == '__main__':
  # parse comandline arguments
  p = argparse.ArgumentParser()
  p.add_argument('--original-file', help='File to correct')
  p.add_argument('--original-treename', default='DecayTree',
                 help='Name of the original tree')
  p.add_argument('--target-file', help='File to reweight to')
  p.add_argument('--target-treename', default='DecayTree',
                 help='Name of the target tree')
  p.add_argument('--output-file',
                 help='Branches string expression to calc the weight.')
  p.add_argument('--weight-set', default='kbsWeight',
                 help='Branches string expression to calc the weight.')
  p.add_argument('--version',
                 help='Branches string expression to calc the weight.')
  p.add_argument('--year',
                 help='Branches string expression to calc the weight.')
  p.add_argument('--mode',
                 help='Branches string expression to calc the weight.')
  args = vars(p.parse_args())

  with open('analysis/reweightings/config.yml') as file:
    reweight_config = yaml.load(file, Loader=yaml.FullLoader)
  reweight_config = reweight_config[args["weight_set"]][args["mode"]]

  original_mode = args['original_file'].split('/')[-2]
  target_mode = args['target_file'].split('/')[-2]
  if original_mode != args['mode']:
    raise ValueError(f"Original mode does not match {args['mode']}")

  with open('analysis/samples/branches.yaml') as file:
    sWeight = yaml.load(file, Loader=yaml.FullLoader)
  oSW = sWeight[original_mode]['sWeight']
  tSW = sWeight[target_mode]['sWeight']

  args["original_vars"] = reweight_config["variables"]
  args["target_vars"] = reweight_config["variables"]
  args["original_weight"] = reweight_config["original"][0].format(sWeight=oSW)
  args["target_weight"] = reweight_config["target"][0].format(sWeight=tSW)

  # change bdt according to filename, if applies
  bdtconfig = timeacc['bdtconfig']
  if 'bdt' in args['version'].split('@')[0]:
    bdtconfig = args['version'].split('@')[0].split('~')[0].split('bdt')[1]
    bdtconfig = int(bdtconfig)
    # bdtconfig = int(args['version'].split('bdt')[1])
    bdtconfig = bdtmesh(bdtconfig, config.general['bdt_tests'], False)

  # for v0r0 the GB config is the following one
  if 'v0r0' in args['version']:
    bdtconfig = {
        "n_estimators": 20,
        "learning_rate": 0.3,
        "max_depth": 3,
        "min_samples_leaf": 1000,
        "gb_args": {"subsample": 1}
    }

  # delete some keys and update with bdtconfig
  del args['version']
  del args['year']
  del args['mode']
  args.update(**bdtconfig)
  args.update({"trunc": False})

  # run the kinematic weight
  printsec("Kinematic reweighting")
  for k, v in args.items():
    print(f"{k:>25} : {v}")
  print(f"{'Original mode':>25} : {original_mode}")
  print(f"{'Target mode':>25} : {target_mode}")
  kinematic_weighting(**args)

# }}}


# vim: fdm=marker
