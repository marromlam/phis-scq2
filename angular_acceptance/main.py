#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE




-- """


# %% Modules -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import platform
import json
from hep_ml import reweight

# Kernel paths
import os
path = os.environ['PHIS_SCQ']
#cl_path = os.path.join(os.environ['PHIS_SCQ'],'opencl')
cu_path = os.path.join(os.environ['PHIS_SCQ'],'cuda')

# openCL stuff
# import pyopencl as cl
# import pyopencl.array as cl_array
# context = cl.create_some_context()
# queue   = cl.CommandQueue(context)
# sys.path.append(cl_path)
# from Badjanak import *

# CUDA stuff
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as cu_array
sys.path.append(cu_path)
from Badjanak import *

# testing...
MAX_EVENTS = 5

sys.path.append("/home3/marcos.romero/ipanema3/")
from ipanema import Parameters#, fit_report, minimize
import uncertainties as unc
from uncertainties import unumpy as unp


# Load parameters
params = Parameters.load(os.environ['PHIS_SCQ']+'/params/ang-peilian.json');



# %% Get data (now a test file) ------------------------------------------------

dpath  = '/scratch03/marcos.romero/BsJpsiKK/Root/'
dpath += 'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw'

# test files
mc_path   = '/scratch03/marcos.romero/phisRun2/testers/MC_JpsiPhi_sample2016_kinWeight.root'
data_path = '/scratch03/marcos.romero/phisRun2/testers/JpsiPhi_sample2016.root'

mc_file   = uproot.open(mc_path)["DecayTree"]
data_file = uproot.open(data_path)["DecayTree"]

vars_true  = ['truehelcosthetaK','truehelcosthetaL','truehelphi','B_TRUETAU']
vars_reco  = ['helcosthetaK','helcosthetaL','helphi','time']
vars_kin   = ['hplus_P', 'hplus_PT', 'hminus_P', 'hminus_PT']
#vars_kin  += ['muplus_P', 'muplus_PT', 'muminus_P', 'muminus_PT']

mc_branches   = vars_true + vars_reco+vars_kin + ['sw','gb_weights','B_BKGCAT','kinWeight']
mc_df   = mc_file.pandas.df(branches=mc_branches)
mc_df   = mc_df[(mc_df['B_BKGCAT']==0) | (mc_df['B_BKGCAT']==50)]    # BKG cuts
data_branches = vars_reco + vars_kin + ['sw']
data_df = data_file.pandas.df(branches=data_branches)

mc_true_h = np.ascontiguousarray(mc_df[vars_true].values); mc_true_h[:,3] *=1e3
mc_reco_h = np.ascontiguousarray(mc_df[vars_reco].values)
mc_kinWeight_h = np.ascontiguousarray(mc_df['kinWeight'].values)
mc_sw_h   = np.ascontiguousarray(mc_df.eval('sw/gb_weights').values)
#mc_sw_h  *= np.sum(mc_sw_h)/np.sum(mc_sw_h**2)

data_reco_h = np.ascontiguousarray(data_df[vars_reco].values)
data_sw_h   = np.ascontiguousarray(data_df['sw'].values)
#data_sw_h  *= np.sum(data_sw_h)/np.sum(data_sw_h**2)


#data_true_h = data_true_h[:MAX_EVENTS,:MAX_EVENTS]
#data_reco_h = data_reco_h[:MAX_EVENTS,:]
#sweights_h  = sweights_h[:MAX_EVENTS]

pdf_true_h  = np.zeros(mc_true_h.shape[0])
fk_true_h   = np.zeros([mc_true_h.shape[0],10])
pdf_reco_h  = np.zeros(mc_reco_h.shape[0])
fk_reco_h   = np.zeros([mc_reco_h.shape[0],10])

# Variables in gpuarray
mc_true_d   = cu_array.to_gpu(mc_true_h).astype(np.float64)
mc_reco_d   = cu_array.to_gpu(mc_reco_h).astype(np.float64)
mc_sw_d     = cu_array.to_gpu(mc_sw_h).astype(np.float64)
data_reco_d = cu_array.to_gpu(data_reco_h).astype(np.float64)
data_sw_d   = cu_array.to_gpu(data_sw_h).astype(np.float64)

pdf_true_d  = cu_array.to_gpu(pdf_true_h).astype(np.float64)
fk_true_d   = cu_array.to_gpu(fk_true_h).astype(np.float64)
pdf_reco_d  = cu_array.to_gpu(pdf_reco_h).astype(np.float64)
fk_reco_d   = cu_array.to_gpu(fk_true_h).astype(np.float64)



# %% Get kernels ---------------------------------------------------------------

# Flags
config = json.load(open(path+'/angular-acceptance/config.json'))
#config.update({'DEBUG':'0'})
#config.update({'DEBUG_EVT':'1'})

# Compile model and get kernels
BsJpsiKK = Badjanak(cu_path,**config);
getAngularWeights = BsJpsiKK.getAngularWeights;
getAngularCov = BsJpsiKK.getAngularCov;



# %% Run -----------------------------------------------------------------------

# Get the angular weights
w = getAngularWeights(mc_reco_d, params.valuesdict())


""" ----------------------------------------------------------------------------
EXPECTED RESULTS 2016 MC DG0

TRUE:
[ 1.00000000e+00  1.02729089e+00  1.02720681e+00  3.09926795e-05
 -2.04624605e-04 -4.84271268e-05  1.01057052e+00  6.71033433e-04
 -6.01351452e-04 -1.97206856e-03]

RECO:
[ 1.00000000e+00  1.02753308e+00  1.02746135e+00  4.36813344e-05
 -6.23355718e-05 -3.59750950e-05  1.01080218e+00  7.24432185e-04
 -6.05650041e-04 -1.79225349e-03]
---------------------------------------------------------------------------- """



# Calculate the covariance matrix of the normalization weights (for all 10)
w, uw = getAngularCov(data_reco_d, params.valuesdict())
weights = unp.uarray(w,uw)
weights
""" ----------------------------------------------------------------------------
EXPECTED RESULTS 2016 MC DG0
  w1/w0:  1.0275338 +- 0.0008291
  w2/w0:  1.0274621 +- 0.0008281
  w3/w0:  0.0000437 +- 0.0005396
  w4/w0: -0.0000623 +- 0.0003696
  w5/w0: -0.0000360 +- 0.0003480
  w6/w0:  1.0108026 +- 0.0005279
  w7/w0:  0.0007244 +- 0.0004852
  w8/w0: -0.0006057 +- 0.0004709
  w9/w0: -0.0017923 +- 0.0010610
---------------------------------------------------------------------------- """



# reweighting config
n_estimators      = 20
learning_rate     = 0.3
max_depth         = 3
min_samples_leaf  = 1000

reweighter = reweight.GBReweighter(n_estimators     = n_estimators,
                                   learning_rate    = learning_rate,
                                   max_depth        = max_depth,
                                   min_samples_leaf = min_samples_leaf,
                                   gb_args          = {'subsample': 1})


# Angular weights after kinematic reweighting
w_initial = getAngularWeights(mc_reco_d, params.valuesdict())
w_initial/w_initial[0]

# Reweight MC to data in p an pT of K+ and K-
mc_weight = mc_sw_h * mc_kinWeight_h
data_weight = data_sw_h
reweighter.fit(original = mc_df[vars_reco], target = data_df[vars_reco],
               original_weight = mc_weight, target_weight = data_weight);
kin_weight = reweighter.predict_weights(mc_df[vars_reco])
kin_weight
