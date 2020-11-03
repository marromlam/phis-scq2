#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys

import ast
import math
import yaml
import hjson

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight

bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
binned_vars = {'eta':'B_ETA', 'pt':'B_PT', 'sigmat':'sigmat'}

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-file', help='File to correct')
    parser.add_argument('--original-treename', default='DecayTree', help='Name of the original tree')
    parser.add_argument('--original-vars', help='Names of the branches that will be used for the weighting')
    parser.add_argument('--original-weight', help='File to store the ntuple with weights')
    parser.add_argument('--target-file', help='File to reweight to')
    parser.add_argument('--target-treename', default='DecayTree', help='Name of the target tree')
    parser.add_argument('--target-vars', help='Names of the branches that will be used for the weighting')
    parser.add_argument('--target-weight', help='Branches string expression to calc the weight.')
    parser.add_argument('--output-file', help='Branches string expression to calc the weight.')
    parser.add_argument('--output-name', default='kinWeight', help='Branches string expression to calc the weight.')
    parser.add_argument('--n-estimators', default=20, help='Check hep_ml.reweight docs.')
    parser.add_argument('--learning-rate', default=0.3, help='Check hep_ml.reweight docs.')
    parser.add_argument('--max-depth', default=3,help='Check hep_ml.reweight docs.')
    parser.add_argument('--min-samples-leaf', default=1000, help='Check hep_ml.reweight docs.')
    parser.add_argument('--trunc', default=0, help='Cut value for kinWeight, all kinetic weights lower than trunc will be set to trunc.')

    return parser

# %% Variable extractor class and function -------------------------------------

class IdentifierExtractor(ast.NodeVisitor):
  def __init__(self):
    self.ids = set()
  def visit_Name(self, node):
    self.ids.add(node.id)

def getStringVars(FUN):
  extractor = IdentifierExtractor()
  extractor.visit(ast.parse(FUN))
  extractor.ids = extractor.ids - set(vars(math))
  return list(extractor.ids)



def computekinWeight(original, target, original_weight, target_weight,
                     n_estimators,learning_rate,max_depth,min_samples_leaf,
                     trunc):

  reweighter = reweight.GBReweighter(n_estimators     = int(n_estimators),
                                     learning_rate    = float(learning_rate),
                                     max_depth        = int(max_depth),
                                     min_samples_leaf = int(min_samples_leaf),
                                     gb_args          = {'subsample': 1})

  reweighter.fit(original = original, target = target,
                 original_weight = original_weight, target_weight = target_weight);

  kinWeight = reweighter.predict_weights(original)

  # Use truncation if set
  if int(trunc):
    print('Apply a truncation at '+trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)

  kinWeight = np.where(original_weight!=0, kinWeight, 0)
  return kinWeight



# %% kinematic_weighting -------------------------------------------------------

def kinematic_weighting(original_file, original_treename, original_vars, original_weight,
                        target_file, target_treename, target_vars, target_weight,
                        output_file, output_name,
                        n_estimators, learning_rate, max_depth,
                        min_samples_leaf, trunc):
  print(f"\n{80*'='}\n", "Kinematic weighting", f"\n{80*'='}\n")

  # %% Build pandas dataframes -------------------------------------------------
  original_vars = original_vars.split()
  target_vars = target_vars.split()

  # find all needed branches
  all_original_vars = original_vars + getStringVars(original_weight)
  all_target_vars   = target_vars + getStringVars(target_weight)

  # fetch variables in original files
  print('Loading branches for original_sample')
  file = uproot.open(original_file)[original_treename]
  ovars_df = file.pandas.df(flatten=None)

  print('Loading branches for target_sample')
  file = uproot.open(target_file)[target_treename]
  tvars_df = file.pandas.df(flatten=None)

  sws = ['sw']+[f'sw_{var}' for var in binned_vars.keys()]
  kws = ['kinWeight']+[f'kinWeight_{var}' for var in binned_vars.keys()]
  try:
    print('Original sWeights')
    print(ovars_df[sws])
    print('Target sWeights')
    print(tvars_df[sws])
  except:
    0
  # %% Reweighting -------------------------------------------------------------
  ovars_df['kinWeight'] = computekinWeight(
                   ovars_df.get(original_vars),
                   tvars_df.get(target_vars),
                   ovars_df.eval(original_weight),
                   tvars_df.eval(target_weight),
                   n_estimators,learning_rate,max_depth,min_samples_leaf,
                   trunc)
  try:
    for var,vvar in binned_vars.items():
      kinWeight = np.zeros_like(ovars_df['kinWeight'].values)
      for cut in bin_vars[var]:
        original_weight_binned = original_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
        target_weight_binned = target_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
        print("ORIGINAL WEIGHT =", original_weight_binned)
        print("  TARGET WEIGHT =", target_weight_binned)
        kinWeight_ = computekinWeight(
                ovars_df.get(original_vars),
                tvars_df.get(target_vars),
                ovars_df.eval(original_weight_binned),
                tvars_df.eval(target_weight_binned),
                n_estimators,learning_rate,max_depth,min_samples_leaf,
                trunc)
        kinWeight = np.where(ovars_df.eval(cut.replace(var,vvar)), kinWeight_, kinWeight)
      ovars_df[f'kinWeight_{var}'] = kinWeight
  except:
    print('There arent such branches')

  print('Original kinWeights')
  print(ovars_df[kws])

  # %% Save weights to file ----------------------------------------------------
  if os.path.exists(output_file):
    print('Deleting previous %s'  % output_file)
    os.remove(output_file)                               # delete file if exists
  print('Writing on %s' % output_file)
  f = uproot.recreate(output_file)
  f[original_treename] = uproot.newtree({var:'float64' for var in ovars_df})
  f[original_treename].extend(ovars_df.to_dict(orient='list'))
  f.close()

  return kinWeight

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    kinematic_weighting(**vars(args))
