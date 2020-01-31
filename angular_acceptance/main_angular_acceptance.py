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

# testing...
MAX_EVENTS = 5


import uncertainties as unc
from uncertainties import unumpy as unp
from ipanema import Parameters, fit_report, optimize
from ipanema import histogram
from ipanema import Category
from ipanema import getDataFile

# Load parameters
#params = Parameters.load(os.environ['PHIS_SCQ']+'/params/ang-peilian.json');
#params


path  = os.environ['PHIS_SCQ']
samples_path = os.environ['PHIS_SCQ'] + 'samples/'
dta_path = path + 'decay-time-acceptance/'
out_dta = path + 'output/decay-time-acceptance/'
ppath = out_dta + 'plots/'


#%% Load samples ---------------------------------------------------------------
samples = {}
samples['MC_Bs2JpsiPhi_DG0_2016__baseline'] = samples_path+'MC_Bs2JpsiPhi_DG0_2016__baseline.json'
samples['MC_Bs2JpsiPhi_2016__baseline'] = samples_path+'MC_Bs2JpsiPhi_2016__baseline.json'
samples['Bs2JpsiPhi_2016__baseline'] = samples_path+'Bs2JpsiPhi_2016__baseline.json'
Category.open(samples['MC_Bs2JpsiPhi_DG0_2016__baseline'])

triggers = {'biased':1, 'unbiased':0}
mc = {}; data = {}
for name, sample in zip(samples.keys(),samples.values()):
  name_ = '_'.join( name.split('__')[0].split('_')[:-1] )
  for trig, bool_ in zip(triggers.keys(),triggers.values()):
    s_ = name+'__'+trig
    if name[:2] == 'MC':
      mc[s_] = Category.open(sample, cuts='hlt1b=={}'.format(bool_))
      mc[s_].to_device(lkhd='0*time')
      mc[s_].to_device(sWeight='sWeight')
      mc[s_].to_device(data=['cosK','cosL','hphi','time','X_M','sigma_t','q'])
      param_path = os.environ['PHIS_SCQ']+'time-angular-fit/input/'+mc[s_].name
      mc[s_].assoc_params(Parameters.load(param_path+'.json'))
      mc[s_].name += '_'+trig
      print('Loaded sample %s correctly as MC category' % name)
    else:
      data[s_] = Category.open(sample, cuts='hlt1b=={}'.format(bool_))
      data[s_].to_device(lkhd='0*time')
      data[s_].to_device(sWeight='sWeight')
      data[s_].to_device(data=['cosK','cosL','hphi','time','X_M','sigma_t','q'])
      param_path = os.environ['PHIS_SCQ']+'time-angular-fit/input/'+data[s_].name
      data[s_].assoc_params(Parameters.load(param_path+'.json'))
      data[s_].name += '_'+trig
      print('Loaded sample %s correctly as DATA category' % name)


data['Bs2JpsiPhi_2016__baseline__biased'].params.latex_dumps()


# %% Get kernels ---------------------------------------------------------------

# Flags
config = json.load(open(path+'/angular_acceptance/config/baseline.json'))
#config.update({'DEBUG':'0'})
#config.update({'DEBUG_EVT':'1'})

# Compile model and get kernels
kernel_config = config['kernel_config']
BsJpsiKK = Badjanak(cu_path,**kernel_config);
getAngularWeights = BsJpsiKK.getAngularWeights;
getAngularCov = BsJpsiKK.getAngularCov;
getCrossRate = BsJpsiKK.getCrossRate;



# %% Run -----------------------------------------------------------------------
"""
STEPS
#1  Compute angWeights without correcting MC sample.
#2  Weight the MC samples to match data in B kinematics and mKK
#3  Compute angWeights correcting MC sample with step 2 correc and determine
    a new set of angWeights
#4  Weight the corrected MC to match data in the iterative variables
    (namely p and pT of K+ and K-)
#5  Compute angWeights correcting MC sample with step 2+4 correc and determine
    a new set of angWeights
Go to #4 until convergence is reached.
"""

# Build 2 dicts
w0  = {key:np.zeros(10) for key in triggers.keys()}
w   = {key:np.zeros(10) for key in triggers.keys()}
kW1 = {key:0 for key in mc.keys() if key[:2] == 'MC'}
kW2 = {key:0 for key in mc.keys() if key[:2] == 'MC'}
pW  = {key:0 for key in mc.keys() if key[:2] == 'MC'}



print('STEP 1: Compute angWeights without correcting MC sample');
for trig in w0.keys():
  w0[trig] = np.zeros(10)
  for key in mc:
    if key[-len(trig)-2:] == "__"+trig:
      w0[trig] += getAngularWeights(mc[key].data_d,
                                    mc[key].sWeight_d/mc[key].sWeight_d,
                                    mc[key].params.valuesdict())
      print(key+" parameters");print(mc[key].params.print())
  print("\tCurrent angular weights for %s trigger category are:" % trig)
  print("\t"+10*"%+.5lf   " % tuple(w0[trig]/w0[trig][0]))




print('STEP 2: Weight the MC samples to match data ones')
# This means compute the kinematic weights, which hopefully is done :)
kin_vars = ['B_PT', 'X_M']
for key in kW1:
  reweighter.fit(original        = cats[key].df[kin_vars],
                 target          = cats[key[3:]].df[kin_vars],
                 original_weight = cats[key].sWeight_h,
                 target_weight   = cats[key[3:]].sWeight_h);
  kW1[key] = reweighter.predict_weights(cats[key].df[kin_vars])
print("\tThe kinematic-weighting in B_PT and X_M was done")



# STEP 3: Compute angWeights correcting MC sample with step 2 corrections
print('STEP 3: Compute angWeights correcting MC sample with step 2 corrections')
# This implies using kinWeights as the angular-acceptance weights
for key in w0:
  w0[key] = getAngularWeights(
              cats[key].data_d,
              cu_array.to_gpu(kW1[key]*cats[key].sWeight_h).astype(np.float64),
              params.valuesdict()
            )
  print("\tCurrent angular weights for %s are:" % cats[key].name)
  print("\t"+10*"%+.5lf   " % tuple(w0[key]/w0[key][0]))


# STEPS 4 & 5: Weight MC to match data in the iterative variables namely
#              p and pT of K+ and K-
for key in kW2:
  kW2[key] = np.ones_like(kW1[key])
kin_vars = ['hplus_P', 'hplus_PT', 'hminus_P', 'hminus_PT']

has_converged = False
while not has_converged:
  for key in kW2:
    print('STEP 4: Weight MC to match data in the iterative variables')
    # do the pdf-weighting
    print('\tStarting %s pdf-weighting' % cats[key].name)
    getCrossRate(cats[key].data_d,cats[key].lkhd_d,cats[key[3:]].params.valuesdict(),
                 mass_bins=1, coeffs=False)
    pdf_obs = cats[key].lkhd_d.get()
    getCrossRate(cats[key].data_d,cats[key].lkhd_d,cats[key].params.valuesdict(),
                 mass_bins=1, coeffs=False)
    pdf_gen = cats[key].lkhd_d.get()
    np.seterr(divide='ignore', invalid='ignore')
    pW[key] = np.nan_to_num(pdf_obs/pdf_gen)
    print("\tpdf-weights:",pW[key])

    # kinematic-weighting over P and PT of K+ and K-
    print('\tStarting %s kinematic-weighting' % cats[key].name)
    reweighter.fit(
      original = cats[key].df[kin_vars],
      target = cats[key[3:]].df[kin_vars],
      original_weight = kW1[key]*kW2[key]*pW[key]*cats[key].sWeight_h,
      target_weight = cats[key[3:]].sWeight_h
    );
    kW2[key] = reweighter.predict_weights(cats[key].df[kin_vars])
    print("\tkinematic-weights:",kW2[key])

    print('STEP 5: Compute angWeights correcting MC sample with steps 2+4')

    w[key] = getAngularWeights(
                cats[key].data_d,
                cu_array.to_gpu(
                  kW1[key]*kW2[key]*cats[key].sWeight_h*pW[key]
                ).astype(np.float64),
                params.valuesdict()
             )
    print("\tCurrent angular weights for %s are:" % cats[key].name)
    print("\t"+10*"%+.5lf   " % tuple(w0[key]/w0[key][0]))
  has_converged = True



#6 Compute uncertainties
for key in w:
  print("\tThe %s's angular weights are:" % cats[key].name)
  ang_weig, uang_weig = getAngularCov(
                          cats[key].data_d,
                          cu_array.to_gpu(
                            kW1[key]*kW2[key]*cats[key].sWeight_h*pW[key]
                          ).astype(np.float64),
                          params.valuesdict()
                        )
  angular_weights = unp.uarray(ang_weig,uang_weig)
  for item in angular_weights:
    print("\t",item)
