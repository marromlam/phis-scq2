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

BLOCK_SIZE = 64#56


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
                                'helcosthetaL','helphi','time','sw','B_BKGCAT'])

# data = data[(data['time']>0.3) & (data['time']<15) &
#             (1e3*data['B_TRUETAU']>0.3) & (1e3*data['B_TRUETAU']<15)]     # cuts

#data = data[(data['time']>0.3) & (data['time']<15)]
#data = data[(1e3*data['B_TRUETAU']>=0.3) & (1e3*data['B_TRUETAU']<=15)]     # cuts
data = data[(data['B_BKGCAT']==0) | (data['B_BKGCAT']==50)]

print('Number of events = %d\n' % len(data['time']))

MAX_EVENTS = 5
MAX_EVENTS = len(data['time'])

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
weights_h   = np.zeros(10)

pdf_true_d  = gpuarray.to_gpu(pdf_true_h).astype(np.float64)
fk_true_d   = gpuarray.to_gpu(fk_true_h).astype(np.float64)
pdf_reco_d  = gpuarray.to_gpu(pdf_reco_h).astype(np.float64)
fk_reco_d   = gpuarray.to_gpu(fk_true_h).astype(np.float64)
weights_d   = gpuarray.to_gpu(weights_h).astype(np.float64)



# %% Run -----------------------------------------------------------------------

# Get cuda kernels
kernel_path = '/home3/marcos.romero/phis-scq/cuda/AngularAcceptance.cu'
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
  return weights.get()/weights.get()[0]

#print(help(getAngularWeights))

# Compute and store in host
fk_true_h  = getAngularCoeffs(vars_true_d, fk_true_d)
pdf_true_h = getCrossRate(vars_true_d, pdf_true_d, subs_dict)
fk_reco_h  = getAngularCoeffs(vars_reco_d, fk_reco_d)
pdf_reco_h = getCrossRate(vars_reco_d, pdf_reco_d, subs_dict)

# Compute angular weights
weights = fk_reco_h/pdf_reco_h[:,None] #sweights_h[:,None]*

# print("pdf0/pdf         = " +    "%+.4f  " % pdf_reco_h[0])
# print("w     (1 event)  = " + 10*"%+.4f  " % tuple(weights[0].tolist()))
# print("w/w0  (1 event)  = " + 10*"%+.4f  " % tuple((weights[0]/weights[0][0]).tolist()))

weights = np.sum(weights,axis=0)



# print("pdf0/pdf         = " +  5*"%+.4f  " % tuple(pdf_reco_h.tolist()))
# print("w     (5 events) = " + 10*"%+.4f  " % tuple(weights.tolist()))
# print("w/w0  (5 events) = " + 10*"%+.4f  " % tuple((weights/weights[0]).tolist()))


#weights = (weights.T/weights.T[0]).T                            # w[i,:]/w_[i,0]
"""
import numpy as np
weights = np.array([[ 1.27041849e+00,  4.81670511e-02,  2.13231541e-02,
         1.16184577e-02,  6.48522850e-02, -3.13108729e-01,
         4.42973067e-01, -3.82948648e-02,  1.84888728e-01,
        -1.50034819e+00],
       [ 4.03716928e+00,  1.17950485e+00,  1.36891348e+00,
         2.52203883e+00,  2.89228532e-01, -2.68321576e-01,
         3.03181726e+00,  2.50642183e-01, -2.32524451e-01,
         6.99713071e+00],
       [ 1.71650700e-01,  3.70242235e+00,  3.43331503e+00,
         7.91060646e-01, -8.56237113e-01,  1.19570502e+00,
         6.14270935e-01, -1.61976214e+00,  2.26194086e+00,
         6.49430630e-01],
       [ 5.91776993e-03,  2.77040467e+00,  2.45818009e+00,
        -5.11095663e+00,  2.45092926e-02,  2.60522378e-02,
         3.41562895e+00,  5.88826141e-01,  6.25894793e-01,
         2.84344202e-01],
       [ 2.22863286e-03,  4.21957803e+00,  5.38197488e+00,
        -7.83173223e-01, -1.82839540e-01, -5.58480216e-02,
         9.35153383e-01,  3.74534888e+00,  1.14401035e+00,
        -9.13041853e-02]])

for k in range(0,len(weights)):
  meh = 'event %d:\t' % k
  w0 = weights[k,0]
  for subitem in weights[k]:
    meh += "%+3.4f \t" % (subitem/w0)
  print(meh)
"""

"""
      #     f0/pdf    f1/pdf    f2/pdf    f3/pdf    f4/pdf    f5/pdf    f6/pdf    f7/pdf    f8/pdf    f9/pdf
event 0:   +1.2704   +0.0482   +0.0213   +0.0116   +0.0649   -0.3131   +0.4430   -0.0383   +0.1849   -1.5003
event 1:   +4.0372   +1.1795   +1.3689   +2.5220   +0.2892   -0.2683   +3.0318   +0.2506   -0.2325   +6.9971
event 2:   +0.1717   +3.7024   +3.4333   +0.7911   -0.8562   +1.1957   +0.6143   -1.6198   +2.2619   +0.6494
event 3:   +0.0059   +2.7704   +2.4582   -5.1110   +0.0245   +0.0261   +3.4156   +0.5888   +0.6259   +0.2843
event 4:   +0.0022   +4.2196   +5.3820   -0.7832   -0.1828   -0.0558   +0.9352   +3.7453   +1.1440   -0.0913

for 1 event
      w:    1.2704    0.0482    0.0213    0.0116    0.0649    0.3131    0.4430    0.0383    0.1849    1.5003
   w/w0:    1.0000    0.0379    0.0167    0.0091    0.0510   -0.2464    0.3486   -0.0301    0.1455   -1.1809
for 5 events
      w:    5.4873    11.920    12.664   -2.5694   -0.6604    0.5844     8.4398   2.9267    3.9842    6.3392
   w/w0:    1.0000    2.1722    2.3077   -0.4682   -0.1203    0.1065     1.5380   0.5333    0.7260    1.1552
"""



weights_h = getAngularWeights(vars_true_d,vars_reco_d,weights_d,subs_dict)
print(weights_h)

"""
# %% expected results
+0.1023	+0.0040	+0.0017	+0.0009	+0.0050	-0.0254	+0.0357	-0.0030	+0.0150	-0.1210
+0.0802	+0.0228	+0.0266	+0.0490	+0.0052	-0.0048	+0.0595	+0.0045	-0.0042	+0.1382
+0.0032	+0.0763	+0.0704	+0.0160	-0.0165	+0.0236	+0.0124	-0.0325	+0.0465	+0.0126
+0.0006	+0.0479	+0.0422	-0.0882	+0.0010	+0.0010	+0.0591	+0.0098	+0.0105	+0.0115
+0.0000	+0.0687	+0.0874	-0.0130	-0.0022	-0.0007	+0.0152	+0.0608	+0.0191	-0.0011
"""
