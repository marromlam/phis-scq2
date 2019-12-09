# -*- coding: utf-8 -*-

import os
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule
import numpy as np


class Badjanak():
  """
  docstring for Badjanak.
  """

  def __init__(self,  path, **kwargs):
    #super(Badjanak, self).__init__()

    # Some settings
    self.path = path
    #self.interface = interface
    #self.context = context
    #self.queue = queue

    # Default complile flags
    #     The compile_flags is a dict where each key substitutes a same-named
    #     string in the kernel file by its value: #define KEY {KEY} <-- value
    self.compile_flags = {
      "DEBUG":           "1",# no prints
      "DEBUG_EVT":       "5",# number of events to debug
      "USE_TIME_ACC":    "0",# NO  time acceptance
      "USE_TIME_OFFSET": "0",# NO  time offset
      "USE_TIME_RES":    "0",# USE time resolution
      "USE_PERFTAG":     "1",# USE perfect tagging
      "USE_TRUETAG":     "0",# NO  true tagging
      "NKNOTS":          "7",
      "NTIMEBINS":       "8",
      "SIGMA_T":         "0.15",
      "KNOTS":           "{0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00}",
      "NMASSBINS":       "6",
      "X_M":             "{990, 1008, 1016, 1020, 1024, 1032, 1050}",
      "TRISTAN":         "{1,1,1,0,0,0,1,0,0,0}"
    }

    # Update compile_flags with the ones provided
    for configurable in kwargs.keys():
      if configurable in self.compile_flags.keys():
        self.compile_flags[configurable] = kwargs[configurable]
      else:
        print(configurable+' is not in compile_flags!.')

    # Get kernels
    self.getKernels()

  def getGrid(self, n, BLOCK_SIZE):
    Nbunch = n *1. / BLOCK_SIZE
    if Nbunch > int(Nbunch):
        Nbunch = int(Nbunch) +1
    else :
        Nbunch = int(Nbunch)
    return Nbunch

  def updateProperty(self,property,value):
    setattr(self, property, value)
    self.getKernels()

  def compileCU(self):
    kernel_path = os.path.join(self.path,'Badjanak.cu')
    #print(self.compile_flags)
    Badjanak = SourceModule(open(kernel_path,"r").read()
                            .format(**self.compile_flags),
                            no_extern_c=False, arch=None, code=None,
                            include_dirs=[self.path])
    return Badjanak

  def getKernels(self):
    #try:
    self.__Badjanak = self.compileCU()
    items = ['pyFcoeffs', 'pyAngularWeights', 'pyDiffRate']
    for item in items:
        setattr(self, 'k'+item[2:], self.__Badjanak.get_function(item))
    #except:
    #  print('Error!')



  def getCrossRate(self,vars,pdf,pars,mass_bins=7,coeffs=None,BLOCK_SIZE=32):
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

    self.kDiffRate(queue,
              (int(np.ceil(pdf.shape[0]/BLOCK_SIZE)),),
              (BLOCK_SIZE,mass_bins,1),
              vars.data, pdf.data,
              np.float64(pars["Gd"]+pars["DGsd"]),
              np.float64(pars["DGs"]),
              np.float64(pars["DM"]),
              cu_array.to_gpu(queue, CSP).astype(np.float64).data,
              cu_array.to_gpu(queue, np.sqrt(ASlon)).astype(np.float64).data,
              cu_array.to_gpu(queue, np.sqrt(APlon)).astype(np.float64).data,
              cu_array.to_gpu(queue, np.sqrt(APpar)).astype(np.float64).data,
              cu_array.to_gpu(queue, np.sqrt(APper)).astype(np.float64).data,
              np.float64(pars["pSlon"]), np.float64(pars["pPlon"]),
              np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
              cu_array.to_gpu(queue, dSlon).astype(np.float64).data,
              np.float64(pars["dPlon"]),
              np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
              np.float64(pars["lSlon"]), np.float64(pars["lPlon"]),
              np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
              np.float64(0.3), np.float64(15),
              cu_array.to_gpu(queue, coeffs).astype(np.float64).data,
              np.int32(pdf.shape[0]), g_times_l = True )

    return pdf.get()


  def getAngularWeights(self, vars_true, pars, coeffs=None, BLOCK_SIZE=32):
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
    ASlon   = pars['ASlon']
    FP      = abs(1-ASlon)
    APlon   = FP*pars['APlon']
    APper   = FP*pars['APper']
    APpar   = FP*(1-pars['APlon']-pars['APper'])
    weights = cu_array.to_gpu(np.zeros(10)).astype(np.float64)

    self.kAngularWeights(
          vars_true, weights,
          np.float64(pars["Gd"]+pars["DGsd"]), np.float64(pars["DGs"]),
          np.float64(pars["DM"]), np.float64(pars["CSP"]),
          np.float64(np.sqrt(ASlon)), np.float64(np.sqrt(APlon)), np.float64(np.sqrt(APpar)), np.float64(np.sqrt(APper)),
          np.float64(pars["pSlon"]), np.float64(pars["pPlon"]), np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
          np.float64(pars["dSlon"]), np.float64(pars["dPlon"]), np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
          np.float64(pars["lSlon"]), np.float64(pars["lPlon"]), np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
          np.float64(0.3), np.float64(15),
          cu_array.to_gpu(coeffs).astype(np.float64),
          np.int32(vars_true.shape[0]),
          block = (BLOCK_SIZE,1,1),
          grid = (int(np.ceil(vars_true.shape[0]/BLOCK_SIZE)),1,1)
        )
    result = weights.get()
    return result/result[0]



#cu_path = os.path.join(os.environ['PHIS_SCQ'],'cuda')
#BsJpsiKK = Badjanak(cu_path)
#dir(BsJpsiKK)
