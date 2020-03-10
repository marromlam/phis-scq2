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
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
bsjpsikk.config['use_perftag'] = 0
bsjpsikk.config['use_truetag'] = 1
bsjpsikk.get_kernels()

# x = ipanema.ristra.linspace(0.3,15,100)
# y = bsjpsikk.acceptance_spline(x.get())
# plt.plot(x.get(),y)


# testing...
MAX_EVENTS = None



# Load parameters
params = ipanema.Parameters.load('angular_acceptance/input/Bs2JpsiPhi_2016.json');



# %% Get data ------------------------------------------------------------------

mc_path   = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test.root'
data_path = '/scratch17/marcos.romero/phis_samples/v0r2/2016/Bs2JpsiPhi/test.root'

mc_sample = ipanema.Sample.from_root(mc_path,entrystop=MAX_EVENTS)
data_sample = ipanema.Sample.from_root(data_path,entrystop=MAX_EVENTS)

# Allocate some arrays
reco = ['helcosthetaK', 'helcosthetaL', 'helphi', 'time']
true = ['true'+i for i in reco]
genlvl = ['true'+i+'_GenLvl' for i in reco]
mc_sample.allocate(reco=reco, weight='kinWeight')
mc_sample.allocate(true=true)
mc_sample.allocate(genlvl=genlvl)
mc_sample.allocate(pdf='0*time')
mc_sample.allocate(ones='time/time')



# %% Run -----------------------------------------------------------------------

# Get the angular weights
#params.valuesdict()

ipanema.ristra.ones(10)
#mc_sample.reco
w = bsjpsikk.get_angular_weights(mc_sample.true,
                                 mc_sample.ones,
                                 params.valuesdict() )

w/w[0]

mc_sample.true

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
