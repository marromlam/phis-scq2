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
import uncertainties as unc
from uncertainties import unumpy as unp

import ipanema

ipanema.initialize('cuda',1)

# Kernel paths


# Get bsjpsikk and compile it with corresponding flags
import bsjpsikk
bsjpsikk.config['debug'] = 0
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
bsjpsikk.config['use_perftag'] = 0
bsjpsikk.config['use_truetag'] = 1
bsjpsikk.get_kernels()

# testing...
MAX_EVENTS = None




# Load parameters
params = ipanema.Parameters.load('angular_acceptance/input/Bs2JpsiPhi_2016.json').valuesdict();

def alpha(x, y=1):
  z = x/y
  return z*( (z.sum())/((z**2).sum()) )

# %% Get data ------------------------------------------------------------------
cut = '(B_BKGCAT==0 | B_BKGCAT==50) & hlt1b == 1'
#cut = '(B_BKGCAT==0 | B_BKGCAT==50)'

mc_path   = '/scratch17/marcos.romero/phis_samples/v0r2/2015/MC_Bs2JpsiPhi/test_selected_bdt_sw.root'
data_path = '/scratch17/marcos.romero/phis_samples/v0r2/2015/Bs2JpsiPhi/test.root'

# Load MC samples
mc_std = ipanema.Sample.from_root(
  '/scratch03/marcos.romero/phisRun2/cooked_test_files/2016/MC_Bs2JpsiPhi/test_kinWeight.root',
  entrystop=MAX_EVENTS, cuts=cut)
mc_dg0 = ipanema.Sample.from_root(
  '/scratch03/marcos.romero/phisRun2/cooked_test_files/2016/MC_Bs2JpsiPhi_dG0/test_kinWeight.root',
  entrystop=MAX_EVENTS, cuts=cut)

# Concat those samples
mc = ipanema.Sample.from_pandas(pd.concat([mc_std.df, mc_dg0.df]))
# # Load data sample
# data_sample = ipanema.Sample.from_root(
#   '/scratch03/marcos.romero/phisRun2/cooked_test_files/2015/Bs2JpsiPhi/test_kinWeight.root',
#   entrystop=MAX_EVENTS)

mc = mc_dg0
mc.df.eval('sWeight = polWeight*sw/gb_weights', inplace=True)
mc.df.eval('weight = @alpha(sWeight)', inplace=True)
mc.df.eval('sWeight = @alpha(sw,gb_weights)', inplace=True)
print(mc.df.shape)
# Allocate some arrays
reco = ['helcosthetaK', 'helcosthetaL', 'helphi']
true = ['true'+i for i in reco]
genlvl = ['true'+i+'_GenLvl' for i in reco]
mc.allocate(reco=reco+['time', 'X_M', 'sigmat', 'B_ID'])
#mc.allocate(kinWeight='kinWeight', sWeight='sWeight', weight='sWeight*kinWeight')
mc.allocate(weight='weight')
mc.allocate(polWeight='polWeight', sWeight='sWeight')
mc.allocate(true=true+['1000*B_TRUETAU', 'X_M', 'sigmat', 'B_ID'])
mc.allocate(genlvl=genlvl+['1000*B_TRUETAU_GenLvl','X_M', 'sigmat', 'B_ID_GenLvl'])
mc.allocate(pdf='0*time')
mc.allocate(ones='time/time')
mc.df.keys()

w = bsjpsikk.get_angular_weights(mc.true, mc.true, mc.weight, params)
print("w/w0 = "+10*"%+.4f  " % tuple(np.nan_to_num(w/w[0]).tolist()))
w = bsjpsikk.get_angular_weights(mc.true, mc.true, mc.sWeight*mc.polWeight, params)
print("w/w0 = "+10*"%+.4f  " % tuple(np.nan_to_num(w/w[0]).tolist()))

# %% Run -----------------------------------------------------------------------

# Get the angular weights
w = bsjpsikk.get_angular_weights(mc.true, mc.true, mc.ones, params)
print("w/w0 = "+10*"%+.4f  " % tuple(np.nan_to_num(w/w[0]).tolist()))
w = bsjpsikk.get_angular_weights(mc.reco, mc.reco, mc.ones, params)
print("w/w0 = "+10*"%+.4f  " % tuple(np.nan_to_num(w/w[0]).tolist()))




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
w, uw = bsjpsikk.get_angular_cov(mc.reco, mc.ones, params.valuesdict())
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

  w1/w0:  1.0269630 +- 0.0008287
  w2/w0:  1.0268884 +- 0.0008278
  w3/w0:  0.0000433 +- 0.0005392
  w4/w0: -0.0002540 +- 0.0003694
  w5/w0: -0.0000474 +- 0.0003480
  w6/w0:  1.0103428 +- 0.0005275
  w7/w0:  0.0006896 +- 0.0004849
  w8/w0: -0.0005949 +- 0.0004707
  w9/w0: -0.0017298 +- 0.0010612
---------------------------------------------------------------------------- """



#%% reweighting config
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
