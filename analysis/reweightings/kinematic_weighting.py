

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['computekinWeight']

# %% Modules -------------------------------------------------------------------
import argparse
import numpy as np
import uproot3 as uproot
import ast
from shutil import copyfile
import math
import hjson

ROOT_PANDAS = True
if ROOT_PANDAS:
  import root_pandas
  import root_numpy

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight

bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
binned_vars = {'etaB':'B_ETA', 'pTB':'B_PT', 'sigmat':'sigmat'}

def argument_parser():

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
  print(original_weight)
  print(target_weight)
  reweighter.fit(original=original, target=target,
                 original_weight=original_weight, target_weight=target_weight);

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
  print(f"\n{80*'='}\nKinematic weighting\n{80*'='}\n")

  # %% Build pandas dataframes -------------------------------------------------
  original_vars = original_vars.split()
  target_vars = target_vars.split()
  # new_vars = []

  # find all needed branches
  # all_original_vars = original_vars + getStringVars(original_weight)
  # all_target_vars   = target_vars + getStringVars(target_weight)

  # fetch variables in original files
  print('Loading branches for original_sample')
  odf = uproot.open(original_file)[original_treename].pandas.df(flatten=None)

  print('Loading branches for target_sample')
  tdf= uproot.open(target_file)[target_treename].pandas.df(flatten=None)

  # create branch according to file name
  if 'kbuWeight' in output_file:
    weight = 'kbuWeight'
  elif 'oddWeight' in output_file:
    weight = 'oddWeight'
  else:
    weight = 'kinWeight'

  # sws = ['sw'] + [f'sw_{var}' for var in binned_vars.keys()]
  # kws = [f'{weight}'] + [f'{weight}_{var}' for var in binned_vars.keys()]
  
  # print('Original sWeights')
  # print(ovars_df[sws])
  # print('Target sWeights')
  # print(tvars_df[sws])
  
  # %% Reweighting -------------------------------------------------------------
  odf[f'{weight}'] = computekinWeight(
                       odf.get(original_vars), tdf.get(target_vars),
                       odf.eval(original_weight), tdf.eval(target_weight),
                       n_estimators, learning_rate, max_depth,
                       min_samples_leaf, trunc)


  # new_vars.append(np.array(ovars_df[f'{weight}'],dtype=[(f'{weight}',np.float64)]))
  # for var,vvar in binned_vars.items():
  #   kinWeight = np.zeros_like(ovars_df[f'{weight}'].values)
  #   for cut in bin_vars[var]:
  #     original_weight_binned = original_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
  #     target_weight_binned = target_weight.replace("sw",f"sw_{var}"+f"*({cut})".replace(var,vvar))
  #     print("ORIGINAL WEIGHT =", original_weight_binned)
  #     print("  TARGET WEIGHT =", target_weight_binned)
  #     kinWeight_ =  computekinWeight(
  #                   ovars_df.get(original_vars),
  #                   tvars_df.get(target_vars),
  #                   ovars_df.eval(original_weight_binned),
  #                   tvars_df.eval(target_weight_binned),
  #                   n_estimators,learning_rate,max_depth,min_samples_leaf,
  #                   trunc)
  #     kinWeight = np.where(ovars_df.eval(cut.replace(var,vvar)), kinWeight_, kinWeight)
  #   new_vars.append( np.array(kinWeight,dtype=[(f'{weight}_{var}',np.float64)]) )
  #   ovars_df[f'{weight}_{var}'] = kinWeight 
  print('Original kinWeights')
  print(ovars_df[kws])

  # Save weights to file ------------------------------------------------------
  print('Writing on %s' % output_file)
  if ROOT_PANDAS:
    root_pandas.to_root(ovars_df, output_file, key=original_treename, mode='a')
    # copyfile(original_file, output_file)
    # for var in new_vars:
    #   root_numpy.array2root(var, output_file, original_treename, mode='update')
  else:
    f = uproot.recreate(output_file)
    f[original_treename] = uproot.newtree({var:'float64' for var in ovars_df})
    f[original_treename].extend(ovars_df.to_dict(orient='list'))
    f.close()
  return kinWeight

if __name__ == '__main__':
  # Parse comandline arguments
  p = argparse.ArgumentParser()
  p.add_argument('--original-file', help='File to correct')
  p.add_argument('--original-treename', default='DecayTree', help='Name of the original tree')
  p.add_argument('--original-vars', help='Names of the branches that will be used for the weighting')
  p.add_argument('--original-weight', help='File to store the ntuple with weights')
  p.add_argument('--target-file', help='File to reweight to')
  p.add_argument('--target-treename', default='DecayTree', help='Name of the target tree')
  p.add_argument('--target-vars', help='Names of the branches that will be used for the weighting')
  p.add_argument('--target-weight', help='Branches string expression to calc the weight.')
  p.add_argument('--output-file', help='Branches string expression to calc the weight.')
  p.add_argument('--output-name', default='kinWeight', help='Branches string expression to calc the weight.')
  p.add_argument('--n-estimators', default=20, help='Check hep_ml.reweight docs.')
  p.add_argument('--learning-rate', default=0.3, help='Check hep_ml.reweight docs.')
  p.add_argument('--max-depth', default=3,help='Check hep_ml.reweight docs.')
  p.add_argument('--min-samples-leaf', default=1000, help='Check hep_ml.reweight docs.')
  p.add_argument('--trunc', default=0, help='Cut value for kinWeight, all kinetic weights lower than trunc will be set to trunc.')
  p = argument_parser()
  args = vars(p.parse_args())
  # run the kinematic weight
  kinematic_weighting(**args)
