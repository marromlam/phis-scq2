#!/usr/bin/env python
# -*- coding: utf-8 -*-



# Imports ----------------------------------------------------------------------
import sys
sys.path.append("../")
#from ipanema import Parameters, fit_report, minimize
import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import matplotlib.pyplot as plt
import corner

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy.physics.quantum.spin import Rotation
from sympy.abc import i, k, l, m
import math
#import uproot
import os
import platform


# import pycuda.driver as cuda
# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# from pycuda.compiler import SourceModule


# %% Prepare context: were OpenCL function should run --------------------------
context = cl.Context([cl.get_platforms()[0].get_devices()[1]])
queue = cl.CommandQueue(context)  # Instantiate a Queue



subs_dict = {
  "APlon":      np.float64(0.7217582697828963),
  "ASlon":      np.float64(0.0),
  "APpar":      np.float64(0.47983226235842036),
  "APper":      np.float64(0.4988246184782784),
  "CSP":        np.float64(1),
  "DG":         np.float64(0.0),
  "DM":         np.float64(17.8),
  "G":          np.float64(0.66137),
  "deltaPlon":  np.float64(0.0),
  "deltaSlon":  np.float64(3.07),
  "deltaPpar":  np.float64(3.3),
  "deltaPper":  np.float64(3.07),
  "lPlon":      np.float64(1),
  "lSlon":      np.float64(1),
  "lPpar":      np.float64(1),
  "lPper":      np.float64(1),
  "phisPlon":   np.float64(0.07),
  "phisSlon":   np.float64(0.07),
  "phisPpar":   np.float64(0.07),
  "phisPper":   np.float64(0.07)
}



test_data_h = np.array([
  [-0.97683865, -0.6333215 ,  1.375432  ,  0.4200498 ],
  [ 0.67043906, -0.05669873, -2.3950727 ,  2.5682745 ],
  [ 0.2930908 , -0.88998055,  0.96165264,  2.1690416 ],
  [ 0.05597173,  0.0991468 , -0.8175671 ,  1.423794  ],
  [-0.02103739, -0.8631929 ,  2.837755  ,  2.3820534 ]]).astype(np.float64)


test_out_h = np.zeros_like(test_data_h[:,0]).astype(np.float64)
test_fk_h = np.zeros_like(10*[test_data_h[:,0]]).astype(np.float64).T

test_data_d = pycl_array.to_device(queue,test_data_h)
test_out_d  = pycl_array.to_device(queue,test_out_h)
test_fk_d   = pycl_array.to_device(queue,test_fk_h)




# %% Prepare CUDA model --------------------------------------------------------
module = cl.Program(context, open("/Users/marcos/phis-scq/opencl/AngularAcceptance.ocl","r").read() ).build()  # Create the OpenCL program


def model(data, lkhd, amp, period, shift, decay):
    cudaModel.shitModel(queue, data.shape, None, data.data, lkhd.data,
                  np.float32(amp), np.float32(period), np.float32(shift),
                  np.float32(decay), np.int16(len(data)) )
    return lkhd.get()












#%% run

shit  = SourceModule(open('/home3/marcos.romero/JpsiKKAna/cuda/AngularAcceptance2.cu',"r").read());
foo   = shit.get_function("pyDiffRate")
bar   = shit.get_function("pyFcoeffs")

foo( test_data_d,  test_out_d,  subs_dict["G"],  subs_dict["DG"],  subs_dict["DM"],  subs_dict["CSP"],
  subs_dict["APlon"],  subs_dict["ASlon"],  subs_dict["APpar"],  subs_dict["APper"],
  subs_dict["phisPlon"],  subs_dict["phisSlon"],  subs_dict["phisPpar"],  subs_dict["phisPper"],
  subs_dict["deltaSlon"],  subs_dict["deltaPlon"],  subs_dict["deltaPpar"],  subs_dict["deltaPper"],
  subs_dict["lPlon"],  subs_dict["lSlon"],  subs_dict["lPpar"],  subs_dict["lPper"], np.int32(len(test_out_h)),
  block=(5,1,1), grid=(1,1,1))

bar( test_data_d,  test_fk_d, np.int32(len(test_data_d)), block=(5,10,1), grid=(1,1,1))
test_fk_h  = test_fk_d.get()
test_out_h = test_out_d.get()

for item in (test_fk_h):
  meh = ''
  for subitem in item:
    meh += "%+3.4f" % subitem + "\t"
  print(meh)


"""
# %% expected results
+0.1023	+0.0040	+0.0017	+0.0009	+0.0050	-0.0254	+0.0357	-0.0030	+0.0150	-0.1210
+0.0802	+0.0228	+0.0266	+0.0490	+0.0052	-0.0048	+0.0595	+0.0045	-0.0042	+0.1382
+0.0032	+0.0763	+0.0704	+0.0160	-0.0165	+0.0236	+0.0124	-0.0325	+0.0465	+0.0126
+0.0006	+0.0479	+0.0422	-0.0882	+0.0010	+0.0010	+0.0591	+0.0098	+0.0105	+0.0115
+0.0000	+0.0687	+0.0874	-0.0130	-0.0022	-0.0007	+0.0152	+0.0608	+0.0191	-0.0011
"""
