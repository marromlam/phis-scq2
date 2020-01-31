# -*- coding: utf-8 -*-

import os
import builtins

import pyopencl as cl
import pyopencl.array as cl_array



import numpy as np



# This file path
PATH = os.path.dirname(os.path.abspath(__file__))
DEVICE = builtins.DEVICE
CONTEXT = builtins.CONTEXT
QUEUE = builtins.THREAD



# Default compile flags
#     The compile_flags is a dict where each key substitutes a same-named
#     string in the kernel file by its value: #define KEY {KEY} <-- value

debug =           0 # no prints
debug_evt =       5 # number of events to debug
use_time_acc =    0 # no  time acceptance
use_time_offset = 0 # no  time offset
use_time_res =    0 # use time resolution
use_perftag =     1 # use perfect tagging
use_truetag =     0 # no  true tagging
nknots =          7
ntimebins =       8
sigma_t =         0.15
knots =           [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00]
nmassbins =       6
x_m =             [990, 1008, 1016, 1020, 1024, 1032, 1050]
tristan =         [1,1,1,0,0,0,1,0,0,0]

#   "DEBUG":           "0",# no prints
#   "DEBUG_EVT":       "1",# number of events to debug
#   "USE_TIME_ACC":    "0",# NO  time acceptance
#   "USE_TIME_OFFSET": "0",# NO  time offset
#   "USE_TIME_RES":    "0",# USE time resolution
#   "USE_PERFTAG":     "1",# USE perfect tagging
#   "USE_TRUETAG":     "0",# NO  true tagging
#   "NKNOTS":          "7",
#   "NTIMEBINS":       "8",
#   "SIGMA_T":         "0.15",
#   "KNOTS":           "{0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00}",
#   "NMASSBINS":       "6",
#   "X_M":             "{990, 1008, 1016, 1020, 1024, 1032, 1050}",
#   "TRISTAN":         "{1,1,1,0,0,0,1,0,0,0}"



def compile_flags():
  dict_flags = {
    "DEBUG":           str(debug),
    "DEBUG_EVT":       str(debug_evt),
    "USE_TIME_ACC":    str(use_time_acc),
    "USE_TIME_OFFSET": str(use_time_offset),
    "USE_TIME_RES":    str(use_time_res),
    "USE_PERFTAG":     str(use_perftag),
    "USE_TRUETAG":     str(use_truetag),
    "NKNOTS":          str(nknots),
    "NTIMEBINS":       str(ntimebins),
    "SIGMA_T":         str(sigma_t),
    "KNOTS":           str(knots).replace('[','{').replace(']','}'),
    "NMASSBINS":       str(nmassbins),
    "X_M":             str(x_m).replace('[','{').replace(']','}'),
    "TRISTAN":         str(tristan).replace('[','{').replace(']','}')
  }
  return dict_flags



# Compiler
#     Compile kernel against given BACKEND
def compile():
  kernel_path = os.path.join(PATH,'Badjanak.cl')
  Badjanak = cl.Program(CONTEXT,
                        open(kernel_path,"r").read().format(**compile_flags())
                       ).build(options=["-I "+PATH])
  return Badjanak



# Get kernels
def get_kernels():
  Badjanak = compile()
  items = Badjanak.kernel_names.split(';')
  #
  #
  for item in items:
    setattr(Badjanak, 'KERNEL_'+item[2:], Badjanak.__getattr__(item))
  __MODEL = Badjanak



# Update property
def update_property(self,property,value):
  setattr(self, property, value)
  __MODEL = get_kernels()



__MODEL = get_kernels()



# Functions
#     Here pythonic versions of KERNEL functions are defined. There are wrappers
#     that are simpler to interact with

def cross_rate(vars,pdf,pars,mass_bins=7,coeffs=None,BLOCK_SIZE=32):
  """
  Look at kernel definition to see help
  """
  if not coeffs:                           # this is only for testing purposes
    coeffs = np.array([[-1.513859, 11.996029, -13.678216,  5.410743],
                       [-1.340649, 11.100117, -12.133541,  4.522999],
                       [ 2.371407, -1.137433,   1.314317, -0.402956],
                       [ 0.895665,  2.141992,  -1.114887,  0.196846],
                       [ 2.579169, -0.434798,   0.199802, -0.026739],
                       [ 1.649324,  0.491956,  -0.108090,  0.007356],
                       [ 1.898947,  0.060150,   0.000000,  0.000000]])

  pars_keys = sorted(pars.keys())
  CSP   = np.float64([pars[key] for key in pars_keys if key[:3] == 'CSP'])
  ASlon = np.float64([pars[key] for key in pars_keys if key[:5] == 'ASlon'])
  FP    = abs(1-ASlon)
  APlon = FP*pars['APlon']
  APper = FP*pars['APper']
  APpar = FP*(1-pars['APlon']-pars['APper'])
  dSlon = np.float64([pars[key] for key in pars_keys if key[:5] == 'dSlon'])

  __MODEL.KERNELDiffRate(self.queue,
            (int(np.ceil(pdf.shape[0]/BLOCK_SIZE)),),
            (BLOCK_SIZE,mass_bins,1),
            vars.data, pdf.data,
            np.float64(pars["Gd"]+pars["DGsd"]),
            np.float64(pars["DGs"]),
            np.float64(pars["DM"]),
            cl_array.to_device(self.queue, CSP).astype(np.float64).data,
            cl_array.to_device(self.queue, np.sqrt(ASlon)).astype(np.float64).data,
            cl_array.to_device(self.queue, np.sqrt(APlon)).astype(np.float64).data,
            cl_array.to_device(self.queue, np.sqrt(APpar)).astype(np.float64).data,
            cl_array.to_device(self.queue, np.sqrt(APper)).astype(np.float64).data,
            np.float64(pars["pPlon"]), np.float64(pars["pSlon"]),
            np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
            cl_array.to_device(self.queue, dSlon).astype(np.float64).data,
            np.float64(pars["dPlon"]),
            np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
            np.float64(pars["lPlon"]), np.float64(pars["lSlon"]),
            np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
            np.float64(0.3), np.float64(15),
            cl_array.to_device(self.queue, coeffs).astype(np.float64).data,
            np.int32(pdf.shape[0]), g_times_l = True )

  return pdf.get()



def angular_weights(self,
                      vars_true,vars_reco,
                      weights,
                      pars,
                      coeffs=None,
                      BLOCK_SIZE=32):
  """
  getAngularWeights(vars_true,vars_reco,weights,pars):

    In:
            vars_true:  eventsx4 matrix that stores [cosK, cosL, hphi, time]
                        variables in gpuarray -- true variables
            vars_reco:  eventsx4 matrix that stores [cosK, cosL, hphi, time]
                        variables in gpuarray -- reconstructed variables
              weights:  1x10 gpuarray vector where angular weights will be
                        stored
                 pars:  python dict of all diff cross rate parameters, see
                        below their (key) names

    Out:
                    0:  returns weights in host version, in a 1x10 np.array

  Look at kernel definition to see more help
  """
  if not coeffs:                           # this is only for testing purposes
    coeffs = np.array([[-1.513859, 11.996029, -13.678216,  5.410743],
                       [-1.340649, 11.100117, -12.133541,  4.522999],
                       [ 2.371407, -1.137433,   1.314317, -0.402956],
                       [ 0.895665,  2.141992,  -1.114887,  0.196846],
                       [ 2.579169, -0.434798,   0.199802, -0.026739],
                       [ 1.649324,  0.491956,  -0.108090,  0.007356],
                       [ 1.898947,  0.060150,   0.000000,  0.000000]])
  ASlon = pars['ASlon']
  FP    = abs(1-ASlon)
  APlon = FP*pars['APlon']
  APper = FP*pars['APper']
  APpar = FP*(1-pars['APlon']-pars['APper'])
  __MODEL.KERNELAngularWeights(self.queue,
                       (int(np.ceil(vars_true.shape[0]/BLOCK_SIZE)),),
                       (BLOCK_SIZE,1,1),
  vars_true.data,vars_reco.data, weights.data,
              np.float64(pars["Gd"]+pars["DGsd"]), np.float64(pars["DGs"]),
              np.float64(pars["DM"]), np.float64(pars["CSP"]),
              np.float64(np.sqrt(APlon)), np.float64(np.sqrt(ASlon)),
              np.float64(np.sqrt(APpar)), np.float64(np.sqrt(APper)),
              np.float64(pars["pPlon"]), np.float64(pars["pSlon"]),
              np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
              np.float64(pars["dSlon"]), np.float64(pars["dPlon"]),
              np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
              np.float64(pars["lPlon"]), np.float64(pars["lSlon"]),
              np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
              np.float64(0.3), np.float64(15),
              cl_array.to_device(self.queue, coeffs).astype(np.float64).data,
              np.int32(vars_true.shape[0]), g_times_l = True )
  print(weights)
  return weights.get()/weights.get()[0]
