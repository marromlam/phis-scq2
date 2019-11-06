#!/usr/bin/env python
# -*- coding: utf-8 -*-



# %% Modules -------------------------------------------------------------------
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy.physics.quantum.spin import Rotation
from sympy.abc import i, k, l, m
import math
import uproot
import os
import platform


import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import json

BLOCK_SIZE = 5#56
MAX_EVENTS = 5

def getGrid(Nevts, BLOCK_SIZE):
  Nbunch = Nevts *1. / BLOCK_SIZE
  if Nbunch > int(Nbunch):
    Nbunch = int(Nbunch) +1
  else :
    Nbunch = int(Nbunch)
  return Nbunch


# %% Load parameters -----------------------------------------------------------
tpath = '/home3/marcos.romero/phis-scq/angular-acceptance/'
subs_dict = json.load(open(tpath+'input/time-angular-distribution.json','r') )



# %% Get data (now a test file) ------------------------------------------------

dpath = 'BsJpsiKK/Root/'+\
'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root'

if platform.system() == "Darwin":
    rfile = uproot.open('/Volumes/Santiago/Scratch03/'+dpath)["DecayTree"]
elif platform.system() == "Linux":
    rfile = uproot.open('/scratch03/marcos.romero/'+dpath)["DecayTree"]



data = rfile.pandas.df(branches=['B_TRUETAU','truehelcosthetaK',
                                'truehelcosthetaL', 'truehelphi','helcosthetaK',
                                'helcosthetaL','helphi','time','sw'])

data = data[(data['time']>0.3) & (data['time']<15) &
            (1e3*data['B_TRUETAU']>0.3) & (1e3*data['B_TRUETAU']<15)]     # cuts

vars_true_h = np.stack((
                        data['truehelcosthetaK'].values,
                        data['truehelcosthetaL'].values,
                        data['truehelphi'].values,
                        1e3*data['B_TRUETAU'].values
                      ), axis=-1)

vars_reco_h = np.stack((
                        data['helcosthetaK'].values,
                        data['helcosthetaL'].values,
                        data['helphi'].values,
                        data['time'].values
                      ), axis=-1)

vars_true_h = vars_true_h[:MAX_EVENTS,:]
vars_reco_h = vars_reco_h[:MAX_EVENTS,:]

sweights_h   = np.array(data['sw'].values)[:MAX_EVENTS]
sweights_h  *= np.sum(sweights_h)/np.sum(sweights_h**2)


# Variables in gpuarray
vars_true_d = gpuarray.to_gpu(vars_true_h).astype(np.float64)
vars_reco_d = gpuarray.to_gpu(vars_reco_h).astype(np.float64)
sweights_d  = gpuarray.to_gpu(sweights_h).astype(np.float64)


# Result containers
pdf_true_h  = np.zeros(vars_true_h.shape[0])
fk_true_h   = np.zeros([vars_true_h.shape[0],10])
pdf_reco_h  = np.zeros(vars_reco_h.shape[0])
fk_reco_h   = np.zeros([vars_reco_h.shape[0],10])
weights_h   = np.zeros([vars_reco_h.shape[0],10])

pdf_true_d  = gpuarray.to_gpu(pdf_true_h).astype(np.float64)
fk_true_d   = gpuarray.to_gpu(fk_true_h).astype(np.float64)
pdf_reco_d  = gpuarray.to_gpu(pdf_reco_h).astype(np.float64)
fk_reco_d   = gpuarray.to_gpu(fk_true_h).astype(np.float64)
weights_d   = gpuarray.to_gpu(weights_h).astype(np.float64)



# %% Run -----------------------------------------------------------------------

# Get cuda kernels
kernel_path = '/home3/marcos.romero/phis-scq/cuda/AngularAcceptance2.cu'
AngularAcceptance = SourceModule(open(kernel_path,"r").read());
cuDiffRate  = AngularAcceptance.get_function("pyDiffRate")
cuFcoeffs   = AngularAcceptance.get_function("pyFcoeffs")
cuAngularWeights   = AngularAcceptance.get_function("pyAngularWeights")

# Define kernel wrappers
def getCrossRate(vars,pdf,pars):
  """
  Look at kernel definition to see help
  """
  cuDiffRate( vars, pdf,
              np.float64(pars["G"]), np.float64(pars["DG"]),
              np.float64(pars["DM"]), np.float64(pars["CSP"]),
              np.float64(pars["APlon"]), np.float64(pars["ASlon"]),
              np.float64(pars["APpar"]), np.float64(pars["APper"]),
              np.float64(pars["phisPlon"]), np.float64(pars["phisSlon"]),
              np.float64(pars["phisPpar"]), np.float64(pars["phisPper"]),
              np.float64(pars["deltaSlon"]), np.float64(pars["deltaPlon"]),
              np.float64(pars["deltaPpar"]), np.float64(pars["deltaPper"]),
              np.float64(pars["lPlon"]), np.float64(pars["lSlon"]),
              np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
              np.int32(MAX_EVENTS),
              block=(BLOCK_SIZE,1,1),
              grid=(getGrid(pdf.shape[0], BLOCK_SIZE),1,1))
  return pdf.get()



def getAngularCoeffs(vars, fk):
  """
  Look at kernel definition to see help
  """
  cuFcoeffs(vars, fk, np.int32(MAX_EVENTS),
            block=(BLOCK_SIZE,10,1),
            grid=(getGrid(fk.shape[0], BLOCK_SIZE),1,1))
  return fk.get()



def getAngularWeights(vars_true,vars_reco,weights,pars):
  """
  getAngularWeights(vars_true,vars_reco,weights,pars):

    In:
            vars_true:  eventsx4 matrix that stores [cosK, cosK, hphi, time]
                        variables in gpuarray -- true variables
            vars_reco:  eventsx4 matrix that stores [cosK, cosK, hphi, time]
                        variables in gpuarray -- reconstructed variables
              weights:  1x10 gpuarray vector where angular weights will be
                        stored
                 pars:  python dict of all diff cross rate parameters, see below
                        their (key) names

    Out:
                    0:  returns weights in host version, in a 1x10 np.array

  Look at kernel definition to see more help
  """
  cuAngularWeights( vars_true,vars_reco, weights,
              np.float64(pars["G"]), np.float64(pars["DG"]),
              np.float64(pars["DM"]), np.float64(pars["CSP"]),
              np.float64(pars["APlon"]), np.float64(pars["ASlon"]),
              np.float64(pars["APpar"]), np.float64(pars["APper"]),
              np.float64(pars["phisPlon"]), np.float64(pars["phisSlon"]),
              np.float64(pars["phisPpar"]), np.float64(pars["phisPper"]),
              np.float64(pars["deltaSlon"]), np.float64(pars["deltaPlon"]),
              np.float64(pars["deltaPpar"]), np.float64(pars["deltaPper"]),
              np.float64(pars["lPlon"]), np.float64(pars["lSlon"]),
              np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
              np.int32(MAX_EVENTS),
              block=(BLOCK_SIZE,1,1),
              grid=(getGrid(vars_reco.shape[0], BLOCK_SIZE),1,1))
  return weights.get()

#print(help(getAngularWeights))

# Compute and store in host
fk_true_h  = getAngularCoeffs(vars_true_d, fk_true_d)
pdf_true_h = getCrossRate(vars_true_d, pdf_true_d, subs_dict)
fk_reco_h  = getAngularCoeffs(vars_reco_d, fk_reco_d)
pdf_reco_h = getCrossRate(vars_reco_d, pdf_reco_d, subs_dict)

# Compute angular weights
weights = fk_true_h/pdf_reco_h[:,None] #sweights_h[:,None]*
weights = (weights.T/weights.T[0]).T                            # w[i,:]/w_[i,0]

for k in range(0,len(weights)):
  meh = 'event %d:\t' % k
  w0 = weights[k,0]
  for subitem in weights[k]:
    meh += "%+3.4f \t" % (subitem)
  print(meh)

# print(weights)
weights_h = getAngularWeights(vars_true_d,vars_reco_d,weights_d,subs_dict)


"""
# %% expected results
+0.1023	+0.0040	+0.0017	+0.0009	+0.0050	-0.0254	+0.0357	-0.0030	+0.0150	-0.1210
+0.0802	+0.0228	+0.0266	+0.0490	+0.0052	-0.0048	+0.0595	+0.0045	-0.0042	+0.1382
+0.0032	+0.0763	+0.0704	+0.0160	-0.0165	+0.0236	+0.0124	-0.0325	+0.0465	+0.0126
+0.0006	+0.0479	+0.0422	-0.0882	+0.0010	+0.0010	+0.0591	+0.0098	+0.0105	+0.0115
+0.0000	+0.0687	+0.0874	-0.0130	-0.0022	-0.0007	+0.0152	+0.0608	+0.0191	-0.0011
"""
