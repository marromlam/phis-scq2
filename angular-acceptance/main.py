#!/usr/bin/env python
# -*- coding: utf-8 -*-



# %% Modules -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import platform
import json
import numpy as np  # Import Numpy number tools

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



# %% Load parameters -----------------------------------------------------------
params = Parameters()
params.load(os.environ['PHIS_SCQ']+'/params/ang-peilian.json');



# %% Get kernels ---------------------------------------------------------------

# Flags
config = json.load(open(path+'/angular-acceptance/config.json'))
#config.update({'DEBUG':'0'})
#config.update({'DEBUG_EVT':'1'})

# Compile model and get kernels
BsJpsiKK = Badjanak(cu_path,**config);
getAngularWeights = BsJpsiKK.getAngularWeights;



# %% Get data (now a test file) ------------------------------------------------

dpath  = 'BsJpsiKK/Root/'
dpath += 'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw'

if platform.system() == "Darwin":
  rfile = uproot.open('/Volumes/Santiago/Scratch03/'+dpath+'.root')["DecayTree"]
elif platform.system() == "Linux":
  rfile = uproot.open('/scratch03/marcos.romero/'+dpath+'.root')["DecayTree"]

vars_true  = ['truehelcosthetaK','truehelcosthetaL','truehelphi','B_TRUETAU']
vars_reco  = ['helcosthetaK','helcosthetaL','helphi','time']
vars_other = ['sw','B_BKGCAT']

data = rfile.pandas.df(branches=vars_true+vars_reco+vars_other)
data = data[(data['B_BKGCAT']==0) | (data['B_BKGCAT']==50)]

data_true_h = np.ascontiguousarray(data[vars_true].values)
data_reco_h = np.ascontiguousarray(data[vars_reco].values)
data_true_h[:,3] *=1e3 # in ps
#sweights_h   = np.array(data['sw'].values)
#sweights_h  *= np.sum(sweights_h)/np.sum(sweights_h**2)

#data_true_h = data_true_h[:MAX_EVENTS,:MAX_EVENTS]
#data_reco_h = data_reco_h[:MAX_EVENTS,:]
#sweights_h  = sweights_h[:MAX_EVENTS]

pdf_true_h  = np.zeros(data_true_h.shape[0])
fk_true_h   = np.zeros([data_true_h.shape[0],10])
pdf_reco_h  = np.zeros(data_reco_h.shape[0])
fk_reco_h   = np.zeros([data_reco_h.shape[0],10])

# Variables in gpuarray
data_true_d = cu_array.to_gpu(data_true_h).astype(np.float64)
data_reco_d = cu_array.to_gpu(data_reco_h).astype(np.float64)
#sweights_d  = cu_array.to_gpu(sweights_h).astype(np.float64)
pdf_true_d  = cu_array.to_gpu(pdf_true_h).astype(np.float64)
fk_true_d   = cu_array.to_gpu(fk_true_h).astype(np.float64)
pdf_reco_d  = cu_array.to_gpu(pdf_reco_h).astype(np.float64)
fk_reco_d   = cu_array.to_gpu(fk_true_h).astype(np.float64)



# %% Run -----------------------------------------------------------------------

weights = getAngularWeights(data_true_d, params.valuesdict())
weights




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
