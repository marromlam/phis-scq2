#!/usr/bin/env python
# -*- coding: utf-8 -*-



# %% Modules -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
from hep_ml import reweight
import ast
import math
import json


# openCL stuff
cl_path = os.path.join(os.environ['PHIS_SCQ'],'opencl')
import pyopencl as cl                      # Import the OpenCL GPU computing API
import pyopencl.array as cl_array                        # Import PyOpenCL Array
context = cl.create_some_context()                      # Initialize the Context
queue   = cl.CommandQueue(context)

sys.path.append(cl_path)
from Badjanak import *

# %% Define some shit ----------------------------------------------------------

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

SWICHER = 0
BDT_CONF = 4

# %% Build pandas dataframes ---------------------------------------------------

if SWICHER == 0:
  # Bs MC --> Bs data ----------------------------------------- [MOVE ELSEWHERE]
  original_paths    = [
  "/home3/marcos.romero/phis-scq/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root",
  "/home3/marcos.romero/phis-scq/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_polarityweight.root",
  "/home3/marcos.romero/phis-scq/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root"
  ]
  target_paths    = [
  "/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root"
  ]
  original_vars     = ["B_PT","X_M"]
  target_vars       = ["B_PT","X_M"]
  original_weight   = '(sw/gb_weights)*polWeight*PDFWeight'
  target_weight     = 'sw'
elif SWICHER == 1:
  # Bd data --> Bs data --------------------------------------- [MOVE ELSEWHERE]
  original_paths = [
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat.root"
  ]
  target_paths = [
  "/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root"
  ]
  original_vars     = ["B_PT","B_P"]
  target_vars       = ["B_PT","B_P"]
  original_weight   = 'nsig_sw'
  target_weight     = 'sw'
elif SWICHER == 2:
  # Bd MC --> Bd data ----------------------------------------- [MOVE ELSEWHERE]
  original_paths = [
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat.root",
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root",
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root"
  ]
  target_paths = [
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat.root",
  "/home3/marcos.romero/phis-scq/BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat_kinematicweight.root"
  ]
  original_vars     = ["B_PT","X_M"]
  target_vars       = ["B_PT","X_M"]
  original_weight   = 'nsig_sw*PolWeight*PDFWeight'
  target_weight     = 'nsig_sw*kinWeight'







uproot.open('/home3/marcos.romero/phis-scq/BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat.root2')



def polarity_weighting(original_file, original_treename,
                       target_file, target_treename,
                       output_file):
  # Get data -------------------------------------------------------------------
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

  if "PolWeight".encode() in _original_tree.keys():
    print('polarity-weighting.py: This file already has a polarity weight. Exiting...')
    exit()
  else:
    original_df       = _original_tree.pandas.df()
    original_polarity = original_df["Polarity"].values
    target_df         = _target_tree.pandas.df(branches="Polarity")
    target_polarity   = target_df["Polarity"].values
  print('DataFrames are ready')

  # Cook weights ---------------------------------------------------------------
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


  # Save weights to file -------------------------------------------------------
  #    It would be nice that uproot.update worked, but it is not yet avaliable
  #    and the function is a placeholder only according to the author. So, at
  #    moment it is needed to load the whole df and write a new one :(.
  #    This can produce some problems with large files, let's hope we don't need
  #    large ones.
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  #os.system('cp '+original_file+' '+output_file)
  print('Writing on %s' % output_file)
  with uproot.recreate(output_file,compression=None) as f:
    f[original_treename] = uproot.newtree({var:'float64' for var in original_df})
    f[original_treename].extend(original_df.to_dict(orient='list'))
  f.close()
  print('polWeight was succesfully written.')

  return polWeight





def pdf_weighting(original_file, tree_name, output_file,
                  target_params, original_params, mode = 'MC_BsJpsiPhi'):

  # Flags
  config = {
    "USE_TIME_ACC":    "0",# NO  time acceptance
    "USE_TIME_OFFSET": "0",# NO  time offset
    "USE_TIME_RES":    "0",# USE time resolution
    "USE_PERFTAG":     "1",# USE perfect tagging
    "USE_TRUETAG":     "0",# NO  true tagging
  }
  if mode == "MC_BdJpsiKstar":
    config["NMASSBINS"] = "4"
    config["X_M"] = "{826, 861, 896, 931, 966}"
  elif mode == "MC_BsJpsiPhi":
    config["NMASSBINS"] = "6"
    config["X_M"] = "{990, 1008, 1016, 1020, 1024, 1032, 1050}"

  # Compile model and get kernels
  BsJpsiKK     = Badjanak(cl_path,'cl',context,queue,**config)
  getCrossRate = BsJpsiKK.getCrossRate

  # Load file
  input_file = uproot.open(original_file)[tree_name]
  print('Loaded %s correctly' % original_file)
  data = input_file.pandas.df()
  print('DataFrames are ready')

  # Prepare host arrays
  tad_vars = ['cosThetaKRef_GenLvl','cosThetaMuRef_GenLvl','phiHelRef_GenLvl',
              'time_GenLvl', 'X_M','sigmat','B_ID_GenLvl'] # simon vars
  tad_vars = ['truehelcosthetaK','truehelcosthetaL','truehelphi',
              'B_TRUETAU', 'X_M','sigmat','B_ID_GenLvl']
  vars_h = np.ascontiguousarray(data[tad_vars].values)    # input array (matrix)
  vars_h[:,3] *= 1e3                                                # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                        # output array (pdf)

  # Allocate device_arrays
  vars_d = cl_array.to_device(queue,vars_h).astype(np.float64)
  pdf_d  = cl_array.to_device(queue,pdf_h).astype(np.float64)

  # Compute!
  print('Calc weights...')
  original_pdf_h = getCrossRate(vars_d,pdf_d,original_params,7)     # 7 mKK bins
  target_pdf_h   = getCrossRate(vars_d,pdf_d,target_params,1)        # 1 mKK bin
  np.seterr(divide='ignore', invalid='ignore')                 # remove warnings
  pdfWeight = np.nan_to_num(original_pdf_h/target_pdf_h)
  data['pdfWeight'] = pdfWeight
  print('pdfWeight was succesfully calculated')

  # Save weights to file
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  print('Writing on %s' % output_file)
  with uproot.recreate(output_file,compression=None) as f:
    f["DecayTree"] = uproot.newtree({var:'float64' for var in data})
    f["DecayTree"].extend(data.to_dict(orient='list'))
  f.close()
  print('pdfWeight was succesfully written.')

  return pdfWeight

original_weight
getStringVars(original_weight)

def kinematic_weighting(original_file, original_treename, original_vars, original_weight,
                         target_file, target_treename, target_vars, target_weight,
                         output_file,
                         n_estimators, learning_rate, max_depth,
                         min_samples_leaf, trunc):
  # %% Build pandas dataframes -------------------------------------------------
  # find all needed branches
  all_original_vars = original_vars + getStringVars(original_weight)
  all_target_vars   = target_vars + getStringVars(target_weight)
  # fetch variables in original files
  print('Loaded %s correctly' % original_file)
  _file = uproot.open(original_file)
  _tree = _file[original_treename]
  original_df = _tree.pandas.df()
  # fetch variables in  target files
  print('Loaded %s correctly' % original_file)
  _file = uproot.open(target_file)
  _tree = _file[target_treename]
  target_df = _tree.pandas.df(branches = all_target_vars)
  print('DataFrames are ready')
  # check if all needed branches are placed in df's
  for var in all_original_vars:
    if var not in original_df.keys():
      raise RuntimeError('Variable %s is missing.' % var); exit()
  for var in all_target_vars:
    if var not in target_df.keys():
      raise RuntimeError('Variable %s is missing.' % var); exit()
  # finishing: cooking dataframe
  original_weight = original_df.eval(original_weight)
  target_weight   = target_df.eval(target_weight)
  # %% Reweighting -------------------------------------------------------------
  reweighter = reweight.GBReweighter(n_estimators     = n_estimators,
                                     learning_rate    = learning_rate,
                                     max_depth        = max_depth,
                                     min_samples_leaf = min_samples_leaf,
                                     gb_args          = {'subsample': 1})
  print('Reweighting...')
  reweighter.fit(original         = original_df.get(original_vars),
                 target           = target_df.get(target_vars),
                 original_weight  = original_weight,
                 target_weight    = target_weight);

  kinWeight = reweighter.predict_weights(original_df.get(original_vars))
  print('kinWeight was succesfully calculated')
  # Use truncation if set
  if trunc:
    print('Apply a truncation at '+trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)
  kinWeight = np.where(original_weight!=0,kinWeight,0)
  original_df['kinWeight'] = kinWeight
  # %% Plot histograms ---------------------------------------------------------
  # plt.hist(kinWeight,100, range=(0,1))
  # TODO: write plot functions here!
  # %% Save weights to file ----------------------------------------------------
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  print('Writing on %s' % output_file)
  with uproot.recreate(output_file,compression=None) as f:
    f[original_treename] = uproot.newtree({var:'float64' for var in original_df})
    f[original_treename].extend(original_df.to_dict(orient='list'))
  f.close()
  print('kinWeight was succesfully written.')
  return kinWeight










# polarity-weighting
path = '/home3/marcos.romero/phis-scq/'
original_file     = path+"MC_Bs2JpsiPhi_2016_selected_bdt_v0r1.root"
target_file       = path+'BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = path+"MC_Bs2JpsiPhi_2016_selected_bdt_polWeight_v0r1.root"

polarity_weighting(original_file, original_treename,target_file, target_treename,output_file)



# pdf-weighting
path = '/home3/marcos.romero/phis-scq/'
input_file     = path+"MC_Bs2JpsiPhi_2016_selected_bdt_polWeight_v0r1.root"
tree_name      = 'DecayTree'
output_file    = path+"MC_Bs2JpsiPhi_2016_selected_bdt_pdfWeight_v0r1.root"
original_params = json.load(open('/home3/marcos.romero/phis-scq/input/tad-2016-both-simon1.json'))
target_params   = json.load(open('/home3/marcos.romero/phis-scq/input/tad-2016-both-simon2.json'))

pdf_weighting(input_file, tree_name, output_file, target_params, original_params)



original_file     = "/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_pdfWeight_v0r1.root"
target_file       = '/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = '/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_kinWeight_v0r1.root'
original_vars     = ["B_PT","X_M"]
target_vars       = ["B_PT","X_M"]
original_weight   = 'polWeight*pdfWeight/gb_weights'
target_weight     = 'sw'
n_estimators      = 20
learning_rate     = 0.3
max_depth         = 3
min_samples_leaf  = 1000
trunc             = 0

kinematic_weighting(original_file, original_treename, original_vars, original_weight, target_file, target_treename, target_vars, target_weight, output_file, n_estimators, learning_rate, max_depth, min_samples_leaf, trunc)












def polarity_weighting(original_file, original_treename,
                       target_file, target_treename,
                       output_file):
  # Get data -------------------------------------------------------------------
  if isinstance(original_file,str):
    _file = uproot.open(original_file)
    _tree = _file[original_treename]
  else:
    print('polarity-weighting.py: Missing original file. Exiting...')
    exit()

  if "PolWeight".encode() in _tree.keys():
    print('polarity-weighting.py: This file already has a polarity weight. Exiting...')
    exit()
  else:
    original_df       = _tree.pandas.df()
    original_polarity = original_df["Polarity"].values

  if isinstance(target_file,str):
    _file = uproot.open(target_file)
    _tree = _file[target_treename]
  else:
    print('polarity-weighting.py: Missing target file. Exiting...')
    exit()

  if "PolWeight".encode() in _tree.keys():
    print('polarity-weighting.py: This file already has a polarity weight. Exiting...')
    exit()
  else:
    target_df       = _tree.pandas.df(branches="Polarity")
    target_polarity = target_df["Polarity"].values

  # Cook weights ---------------------------------------------------------------
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


  # Save weights to file -------------------------------------------------------
  #    It would be nice that uproot.update worked, but it is not yet avaliable
  #    and the function is a placeholder only according to the author. So, at
  #    moment it is needed to load the whole df and write a new one :(.
  #    This can produce some problems with large files, let's hope we don't need
  #    large ones.
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  #os.system('cp '+original_file+' '+output_file)
  with uproot.recreate(output_file,compression=None) as f:
    f[original_treename] = uproot.newtree({var:'float64' for var in original_df})
    f[original_treename].extend(original_df.to_dict(orient='list'))
  f.close()
  print(output_file+' was succesfully written.')

  return polWeight


polarity_weighting(original_file, original_treename, target_file, target_treename, output_file)










aja = uproot.open(target_file)['DecayTree']

aja['sw']



original_file     = "/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_polWeight_v0r1.root"
target_file       = '/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = '/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_kinWeight_v0r1.root'
original_vars     = ["B_PT","X_M"]
target_vars       = ["B_PT","X_M"]
original_weight   = '(1/gb_weights)*polWeight*pdfWeight'
target_weight     = 'sw'

n_estimators =20
learning_rate = 0.3
max_depth = 3
min_samples_leaf= 1000
trunc = 0



all_target_vars[2]


def kinematic_weighting(original_file, original_treename, original_vars, original_weight,
                         target_file, target_treename, target_vars, target_weight,
                         output_file,
                         n_estimators, learning_rate, max_depth,
                         min_samples_leaf, trunc):
  # %% Build pandas dataframes -------------------------------------------------
  # find all needed branches
  all_original_vars = original_vars + getStringVars(original_weight)
  all_target_vars   = target_vars + getStringVars(target_weight)
  # fetch variables in original files
  print('Loading branches for original_sample')
  _file = uproot.open(original_file)
  _tree = _file[original_treename]
  original_df = _tree.pandas.df()
  # fetch variables in  target files
  print('Loading branches for target_sample')
  _file = uproot.open(target_file)
  _tree = _file[target_treename]
  target_df = _tree.pandas.df(branches = all_target_vars)
  # check if all needed branches are placed in df's
  for var in all_original_vars:
    if var not in original_df.keys():
      raise RuntimeError('Variable %s is missing.' % var); exit()
  for var in all_target_vars:
    if var not in target_df.keys():
      raise RuntimeError('Variable %s is missing.' % var); exit()
  # finishing: cooking dataframe
  original_weight = original_df.eval(original_weight)
  target_weight   = target_df.eval(target_weight)
  # %% Reweighting -------------------------------------------------------------
  reweighter = reweight.GBReweighter(n_estimators     = n_estimators,
                                     learning_rate    = learning_rate,
                                     max_depth        = max_depth,
                                     min_samples_leaf = min_samples_leaf,
                                     gb_args          = {'subsample': 1})
  print('Reweighting...')
  reweighter.fit(original         = original_df.get(original_vars),
                 target           = target_df.get(target_vars),
                 original_weight  = original_weight,
                 target_weight    = target_weight);

  kinWeight = reweighter.predict_weights(original_df.get(original_vars))
  # Use truncation if set
  if trunc:
    print('Apply a truncation at '+trunc)
    kinWeight[kinWeight > float(trunc)] = float(trunc)
  kinWeight = np.where(original_weight!=0,kinWeight,0)
  original_df['kinWeight'] = kinWeight
  # %% Plot histograms ---------------------------------------------------------
  # plt.hist(kinWeight,100, range=(0,1))
  # TODO: write plot functions here!
  # %% Save weights to file ----------------------------------------------------
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  with uproot.recreate(output_file,compression=None) as f:
    f[original_treename] = uproot.newtree({var:'float64' for var in original_df})
    f[original_treename].extend(original_df.to_dict(orient='list'))
  f.close()
  print(output_file+' was succesfully written.')
  return kinWeight



import sys
sys.path.append(os.environ['PHIS_SCQ']+'tools')
import importlib
importlib.import_module('phis-scq-style')


from scipy.stats import chi2
from scipy.optimize import fsolve
import math
import uproot





def poisson_interval(data, a=0.318):
  """
  Uses chisquared info to get the poisson interval.
  """
  from scipy.stats import chi2
  low, high = (chi2.ppf(a/2, 2*data) / 2, chi2.ppf(1-a/2, 2*data + 2) / 2)
  return np.array(data-low), np.array(high-data)



def plot_pull():
  fig, (axplot,axpull) = plt.subplots(2, 1,
                                      sharex=True,
                                      gridspec_kw = {'height_ratios':[10, 3],
                                                     'hspace': 0.0}
                                      )
  axpull.xaxis.set_major_locator(plt.MaxNLocator(8))
  axplot.yaxis.set_major_locator(plt.MaxNLocator(8))
  axpull.set_ylim(-7, 7)
  axpull.set_yticks([-5, 0, +5])
  # axpull.set_xticks(axpull.get_xticks()[1:-1])
  # axplot.set_yticks(axplot.get_yticks()[1:-1])
  axplot.tick_params(which='major', length=8, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axplot.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='major', length=8, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='minor', length=6, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  return axplot, axpull






def hist_ext(data, bins, weights = None, norm = False, cm = False, **kwargs):
  counts, edges = np.histogram(data, bins=bins,
                               weights = weights, density = norm,
                               **kwargs)
  bincs = (edges[1:]+edges[:-1])*0.5
  cmbins = None
  if cm: # computes the mass-center of each bin
    np.copy(bincs)
    for k in range(0,len(edges)-1):
      if counts[k] != 0:
        cmbins[k] = np.median( data[(data>=edges[k]) & (data<=edges[k+1])] )
  if weights is not None:
    errl, errh = poisson_interval(counts)
    errl = errl**2 + sum_w2(data, weights = weights, bins = bins, **kwargs)**2
    errh = errh**2 + sum_w2(data, weights = weights, bins = bins, **kwargs)**2
    errl = np.sqrt(errl); errh = np.sqrt(errh)
  else:
    errl, errh = poisson_interval(counts)

  result = ipo(**{**{'counts':counts,
                     'edges':edges, 'bins':bincs, 'cmbins': cmbins,
                     'weights': weights,
                     'norm': norm, 'cm':cm, 'nob': bins,
                     'errl': errl, 'errh': errh,
                    },
                  **kwargs})
  return result

import uproot
vars   = ['B_PT', 'B_P', 'X_M']
ranges = [(0,40000),(0,500000),(990,1050)]
labels = [r"$p_T(B_s^0)$ [MeV/$c$]",r"$p(B_s^0)$ [MeV/$c$]",r"$m(K^+K^-)$ [MeV/$c^2$]"]

labels[0]
uproot.open(target_file).keys()
original_data = uproot.open(output_file)['DecayTree'].pandas.df(branches = vars +["gb_weights","kinWeight","Jpsi_Hlt1DiMuonHighMassDecision_TOS"] )
target_data   = uproot.open(target_file)['DecayTree'].pandas.df(branches = vars +["sw","Jpsi_Hlt1DiMuonHighMassDecision_TOS"] )

original_data = uproot.open(output_file)['DecayTree'].pandas.df(branches = vars +["gb_weights","kinWeight","Jpsi_Hlt1DiMuonHighMassDecision_TOS"] )
target_data   = uproot.open(target_file)['DecayTree'].pandas.df(branches = vars +["sw","Jpsi_Hlt1DiMuonHighMassDecision_TOS"] )

trigger = 1
original_data = original_data[ original_data['Jpsi_Hlt1DiMuonHighMassDecision_TOS'] == trigger]
target_data = target_data[ target_data['Jpsi_Hlt1DiMuonHighMassDecision_TOS'] == trigger]


original_file     = "/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_pdfWeight_v0r1.root"
target_file       = '/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = '/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_kinWeight_v0r1.root'
original_vars     = ["B_PT","X_M"]
target_vars       = ["B_PT","X_M"]
original_weight   = 'polWeight*pdfWeight/gb_weights'
target_weight     = 'sw'
n_estimators      = 20
learning_rate     = 0.3
max_depth         = 3
min_samples_leaf  = 1000
trunc             = 0

import numpy as np
hist_ext(data, bins, weights = None, norm = False, cm = False, **kwargs)
shit = hist_ext(target_data['B_PT'], range=ranges[0], bins = 60, norm = True, cm = False, weights=target_data['sw'])




class ipo(object):
  """docstring for ipanema-plot-object (ipo)."""

  def __init__(self, **kwargs):
    super(ipo, self).__init__()
    #self.arg = arg
    for arg in kwargs:
      self.add(arg,kwargs[arg])

  def add(self, name, value):
    self.__setattr__(name, value)




    orgl_y, orgl_x, orgl_xl, orgl_xr, orgl_yl, orgl_yu = hist_ext(data[0], bins=bins, norm = False, cm=cm, **kwargs, weights=weights[0])

ejem= ipo()
ejem.add('caca',2)
ejem.caca



def hist_pull(data, weights = [None, None], bins = 60, norm = False, cm = False, **kwargs):
  orgl_y, orgl_x, orgl_xl, orgl_xr, orgl_yl, orgl_yu = hist_ext(data[0], bins=bins, norm = False, cm=cm, **kwargs, weights=weights[0])
  trgt_y, trgt_x, trgt_xl, trgt_xr, trgt_yl, trgt_yu = hist_ext(data[1], bins=bins, norm = False, cm=cm, **kwargs, weights=weights[1])
  orgl_norm = 1; trgt_norm = 1;
  if norm:
    orgl_norm = 1/orgl_y.sum(); trgt_norm = 1/trgt_y.sum();
  orgl_y = orgl_y*orgl_norm; orgl_yl *= orgl_norm; orgl_yu *= orgl_norm
  trgt_y = trgt_y*trgt_norm; trgt_yl *= trgt_norm; trgt_yu *= trgt_norm
  pulls = pull_errors(trgt_y, orgl_y, orgl_yl, orgl_yu)

  orgl_out = {}
  orgl_out['x'], orgl_out['y']     = orgl_x, orgl_y
  orgl_out['x_l'], orgl_out['x_r'] = orgl_xl, orgl_xr
  orgl_out['y_l'], orgl_out['y_u'] = orgl_yl, orgl_yu
  orgl_out['pulls'] = pulls
  trgt_out = {}
  trgt_out['x'], trgt_out['y']     = trgt_x, trgt_y
  trgt_out['x_l'], trgt_out['x_r'] = trgt_xl, trgt_xr
  trgt_out['y_l'], trgt_out['y_u'] = trgt_yl, trgt_yu
  return orgl_out, trgt_out









def sum_w2(x, weights=None, range= None, bins = 60 ):
  if weights is not None:
    values = np.histogram(x, bins, range, weights=weights*weights)[0]
  else:
    values = np.histogram(x, bins, range)[0]
  return np.sqrt(values)





def pull_errors(target, orgl_y, orgl_yl = None, orgl_yu = None):
  # This function ...
  residuals = orgl_y - target;
  pulls = np.where(residuals>0, residuals/orgl_yl, residuals/orgl_yu)
  return pulls








# Simon tuples --------
_file1 = uproot.open("/home3/marcos.romero/phis-scq/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root")["DecayTree"].arrays(["B_PT","B_P","X_M","sw","gb_weights"])
_file2 = uproot.open('/scratch03/marcos.romero/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
_file3 = uproot.open("/home3/marcos.romero/phis-scq/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root")["DecayTree"].arrays(["B_PT","B_P","X_M","sw"])

original_data      = _file1
original_sweight   = (_file1[b'sw']/_file1[b'gb_weights'])
original_kinweight = _file2
target_data        = _file3
target_sweight     = _file3[b'sw']

vars   = ['B_PT', 'B_P', 'X_M']
ranges = [(0,40000),(0,500000),(990,1050)]

MAX_EVENTS = 100000


for var in vars:
  index = vars.index(var); var = var.encode()
  orig, ref = hist_pull([original_data[var][:MAX_EVENTS],target_data[var][:MAX_EVENTS]], range=ranges[index], bins = 60, norm = True, cm = False, weights=[original_sweight[:MAX_EVENTS],target_sweight[:MAX_EVENTS]])
  weig, ref = hist_pull([original_data[var][:MAX_EVENTS],target_data[var][:MAX_EVENTS]], range=ranges[index], bins = 60, norm = True, cm = False, weights=[(original_kinweight*original_sweight)[:MAX_EVENTS],target_sweight[:MAX_EVENTS]])
  axplot, axpull = plot_pull()
  axplot.fill_between(ref['x'],ref['y'],0,facecolor='k',alpha=0.3,step='mid')
  axplot.fill_between(orig['x'],orig['y'],0,facecolor='C3',alpha=0.3,step='mid')
  axplot.errorbar(weig['x'],weig['y'], yerr = [weig['y_l'],weig['y_u']],  xerr = [weig['x_l'],weig['x_r']] , fmt='.')
  axpull.set_xticks(axpull.get_xticks()[2:-1])
  axplot.set_yticks(axplot.get_yticks()[1:-1])
  axpull.fill_between(weig['x'],weig['pulls'],0)
  axpull.set_xlabel(r'$'+var.decode()+'$')
  axplot.set_ylabel(r'Weighted candidates')
  axplot.legend([r'$B_{s}^{0} \rightarrow J/\psi \phi$ data',
                 r'$B_{s}^{0} \rightarrow J/\psi \phi$ MC',
                 r'$B_{s}^{0} \rightarrow J/\psi \phi$ MC + kinWeights'])
  plt.savefig(var.decode()+'.pdf')
  plt.close()


r'$B_s^0 \rightarrow J\psi \phi $ data'





dc

plt.plot(a['pulls'])




#axplot.margins(x=0.01, y=0.02); axpull.margins(x=0.01, y=0.02)
axpull.set_ylim(-7, 7)
axpull.set_ylim(-5, 5)







print(dir(plt.MaxNLocator(6).tick_values))
axpull.xaxis.set_major_locator(plt.MaxNLocator(8)[1:-1])










from matplotlib.ticker import FuncFormatter, MaxNLocator

plt.draw()


MaxNLocator(6)


def plotFit(bins,data,errl,errh,lins,pdf, figname = 'figure.pdf',
                                legends = None,
                                labels  = ["x","y"],
                                title   = "",
                                style   = "-"):
  fitdict = {}; fitdict["pdf"] = {}; fitdict["legend"] = {}
  AR = (1+np.sqrt(5))/2
  fig, (axplot, axpull) = plt.subplots(2, 1, figsize=(10, (10+2)/AR),
                                       sharex=True,
                                       gridspec_kw = {'height_ratios':[10, 2]})

  width = bins[1]-bins[0]; fitdict["width"] = width
  for k in range(0,len(pdf)):
    if legends:
      fitdict["legend"][str(k)] = legends[k]
      axplot.plot(lins, pdf[k]*np.ones_like(lins),
                  color = Color(k),
                  label=legends[k],
                  linestyle = style[k])
      axplot.legend()
    else:
      axplot.plot(lins, pdf[k]*np.ones_like(lins), color = Color(k))
      #plt.show()
    fitdict["pdf"][str(k)] = (pdf[k]*np.ones_like(lins)).tolist()
  #lins = np.array(lins.tolist())
  #pdf[0] = np.array(pdf[0].tolist())
  #print 'lins:', lins
  #print 'pdf[0]:',pdf[0]
  #print lins.shape, pdf[0].shape
  splinepdf = interp1d(lins, pdf[0]*np.ones_like(lins), kind='cubic',fill_value='extrapolate')
  #fill_value='extrapolate'
  quadf = sc.integrate.quad
  integrpdf = [quadf(splinepdf,bin-width/2,bin+width/2)[0] for bin in bins]
  #bins_rebin, data_rebin, pdf_rebin = DataRebining(bins, data, [splinepdf(bins)], 10, mask = False)
  #bins_rebin, data_rebin, pdf_rebin = bins, data, [np.float64(integrpdf)/width]
  bins_rebin, data_rebin, pdf_rebin = bins, data, [splinepdf(bins)]
  fitdict["bins"] = bins_rebin.tolist(); fitdict["lins"] = lins.tolist()
  fitdict["data"] = bins_rebin.tolist();
  fitdict["pdfbins"] = pdf_rebin[0].tolist()

  pulls = pullHist(data_rebin, errl, errh, pdf_rebin[0] ) # + 0*splinepdf(bins)
  axplot.errorbar(bins_rebin, data_rebin,
                  yerr=[errl, errh], xerr = 0.0*width,
                  fmt='.', color='k', linewidth=1, capthick=1, capsize = 3)
  axpull.fill_between(bins_rebin, 0, pulls, facecolor=Color("blue"),
                                            alpha=0.7,
                                            edgecolor=Color("blue"))
  fitdict["errl"] = errl.tolist(); fitdict["errh"] = errh.tolist();
  fitdict["pulls"] = pulls.tolist()

  axplot.minorticks_on(); axpull.minorticks_on()
  axplot.margins(x=0.01, y=0.02); axpull.margins(x=0.01, y=0.02)
  axplot.tick_params(which='major', length=8, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axplot.tick_params(which='minor', length=6, width=1, direction='in',
                    bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='major', length=8, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  axpull.tick_params(which='minor', length=6, width=1, direction='in',
                     bottom=True, top=True, left=True, right=True)
  axpull.set_ylim(-5, 5)
  axplot.locator_params(axis='x', nbins=17)
  axplot.locator_params(axis='y', nbins=17/AR)
  axpull.locator_params(axis='x', nbins=17)
  axpull.locator_params(axis='y', nbins=1)
  axpull.set_xlabel(labels[0])
  print "width = ",width
  axplot.set_ylabel(labels[1]+"/(%.4f)" % width)
  if title != "":
    axplot.set_title(r+title, color='black')
  fig.subplots_adjust(top=0.8)
  fig.tight_layout()
  fitdict["labels"] = labels
  fitdict["title"] = title
  fitdict["name"] = figname
  #print fitdict
  try:
    with open(os.path.join(rpath + figname) + ".json", 'w') as outfile:
      json.dump(fitdict, outfile)
  except:
    print("Error saving json results file.")
  try:
    plt.savefig(os.path.join(rpath + figname) + ".pdf")
    tikzplotlib.save(os.path.join(rpath + figname) + ".tex")
    print("Figure saved as '" + figname + "' at RESULTS")
  except:
    print("Error saving Figure.")
  #plt.show()
  plt.close()

################################################################################












plt.plot(xo,yo,'C2',xw,yw,'C1',xt,yt,'k')
plt.fill_between(xo,yo,0*yo)
plt.legend(['MC','MC weighted','data'])
plt.step(xo,yo,xw,yw,xt,yt)
plt.xlabel(r'\verb|'+vars[0]+'|')
plt.ylabel(r'Events')
plt.legend('')
data['B_P']



path = os.environ['PHIS_SCQ']

import path.tools









# %% Check if we are as pro as simon... ----------------------------------------


kinWeight = kinematic_weighting(original_paths, original_vars, original_weight,
                        target_paths, target_vars, target_weight)

if   SWICHER == 1:
  simon = uproot.open('/home3/marcos.romero/phis-scq/BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat_BdDataToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
elif SWICHER == 0:
  simon = uproot.open('/scratch03/marcos.romero/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
elif SWICHER == 2:
  simon = uproot.open('/scratch03/marcos.romero/SideCar/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_BdMCToBdData_BaselineDef_15102018.root')['weights'].array('kinWeight')

plt.plot(simon-kinWeight)
plt.show()






# : value for key, value in variable}
