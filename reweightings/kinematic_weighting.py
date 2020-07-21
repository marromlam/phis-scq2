#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
from hep_ml import reweight
import ast
import math
import yaml


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



# %% kinematic_weighting -------------------------------------------------------

def kinematic_weighting(original_file, original_treename, original_vars, original_weight,
                        target_file, target_treename, target_vars, target_weight,
                        output_file,
                        n_estimators, learning_rate, max_depth,
                        min_samples_leaf, trunc):
  print(f"\n{80*'='}\n{'= Kinematic weighting':79}=\n{80*'='}\n")
  # %% Build pandas dataframes -------------------------------------------------
  original_vars = original_vars.split()
  target_vars = target_vars.split()

  # find all needed branches
  all_original_vars = original_vars + getStringVars(original_weight)
  all_target_vars   = target_vars + getStringVars(target_weight)

  # fetch variables in original files
  print('Loading branches for original_sample')
  file = uproot.open(original_file)[original_treename]
  original_vars_df = file.pandas.df(flatten=None)
  original_weight = original_vars_df.eval(original_weight)

  print('Loading branches for target_sample')
  file = uproot.open(target_file)[target_treename]
  target_vars_df = file.pandas.df(branches=all_target_vars)
  target_weight = target_vars_df.eval(target_weight)

  # %% Reweighting -------------------------------------------------------------

  reweighter = reweight.GBReweighter(n_estimators     = int(n_estimators),
                                     learning_rate    = float(learning_rate),
                                     max_depth        = int(max_depth),
                                     min_samples_leaf = int(min_samples_leaf),
                                     gb_args          = {'subsample': 1})

  print('Reweighting...')
  print(f"original_weight = {original_weight}")
  print(f"target_weight = {target_weight}")
  reweighter.fit(original         = original_vars_df.get(original_vars),
                 target           = target_vars_df.get(target_vars),
                 original_weight  = original_weight,
                 target_weight    = target_weight);

  kinWeight = reweighter.predict_weights(original_vars_df.get(original_vars))
  print(f"kinWeight = {kinWeight}")

  # Use truncation if set
  if int(trunc):
    print('Apply a truncation at '+trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)
  kinWeight = np.where(original_weight!=0, kinWeight, 0)
  original_vars_df['kinWeight'] = kinWeight

  # %% Save weights to file ----------------------------------------------------
  if os.path.exists(output_file):
    print('Deleting previous %s'  % output_file)
    os.remove(output_file)                               # delete file if exists
  #os.system('cp '+original_file+' '+output_file)
  print('Writing on %s' % output_file)
  #import root_pandas
  #root_pandas.to_root(original_vars_df, output_file, key=original_treename)
  f = uproot.recreate(output_file)
  f[original_treename] = uproot.newtree({var:'float64' for var in original_vars_df})
  f[original_treename].extend(original_vars_df.to_dict(orient='list'))
  f.close()

  return kinWeight

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    kinematic_weighting(**vars(args))
