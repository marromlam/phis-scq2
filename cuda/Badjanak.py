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
    self.knots = np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00])
    self.nknots = 7
    #self.interface = interface
    #self.context = context
    #self.queue = queue

    # Default complile flags
    #     The compile_flags is a dict where each key substitutes a same-named
    #     string in the kernel file by its value: #define KEY {KEY} <-- value
    self.compile_flags = {
      "DEBUG":           "0",# no prints
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
    items = ['pyDiffRate',
             'pyFcoeffs', 'pyAngularWeights', 'pyAngularCov',
             'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc']
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
    return result#/result[0]



  def getAngularCov(self, vars_true, pars, coeffs=None, BLOCK_SIZE=32):
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
    _weights = self.getAngularWeights(vars_true, pars)
    weights = cu_array.to_gpu(_weights).astype(np.float64)
    cov_mat = cu_array.to_gpu(np.zeros([10,10])).astype(np.float64)

    self.kAngularCov(
          vars_true, weights, cov_mat,
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
    w = weights.get(); cov = cov_mat.get()

    cov = cov + cov.T - np.eye(cov.shape[0])*cov # fill the lower-triangle
    final_cov = np.zeros_like(cov)
    for i in range(0,cov.shape[0]):
      for j in range(0,cov.shape[1]):
        final_cov[i,j] = 1.0/(w[0]*w[0])*(
                                  w[i]*w[j]/(w[0]*w[0])*cov[0][0]+cov[i][j]-
                                  w[i]/w[0]*cov[0][j]-w[j]/w[0]*cov[0][i]);

    return w/w[0], np.sqrt(np.diagonal(final_cov)) #result#/result[0]



  def getSingleTimeAcc(self, time, lkhd, params, BLOCK_SIZE=32):
    """
      In:
                   time:  1D gpuarray with time to be fitted
                   lkhd:  1D gpuarray where likelihood is being stored
                 params:  dict of params {'name': value}
             BLOCK_SIZE:  device block/workgroup size

      Out:
                   void

    Look at pySingleTimeAcc kernel definition to see more help.
    """
    b = [params['b_0'],params['b_1'],params['b_2'],params['b_3'],params['b_4'],
                       params['b_5'],params['b_6'],params['b_7'],params['b_8']]
    mu    = np.float64(params['mu']);
    sigma = np.float64(params['sigma']);
    gamma = np.float64(params['gamma']);
    self.kSingleTimeAcc(time, lkhd,
                        cu_array.to_gpu(self.get4coeffs(b)).astype(np.float64),
                        mu, sigma, gamma,
                        np.float64(0.3),np.float64(15),
                        np.int32(lkhd.shape[0]),
                        block = (BLOCK_SIZE,1,1),
                        grid = (int(np.ceil(lkhd.shape[0]/BLOCK_SIZE)),1,1)
                       )

  def getRatioTimeAcc(self, time, lkhd, params, BLOCK_SIZE=32):
    """
    In:
            time: 2D list of 1D gpuarray with time to be fitted, the expected
                  format is [time1,time2] (tipically 1:BsMC 2:BdMC)
            lkhd: 2D list of 1D gpuarray where likelihood is being stored
         params1: dict of params {'name': value}
      BLOCK_SIZE: device block/workgroup size

    Out:
            void

    Look at kernel definition to see more help
    """
    b = [params['b_0'],params['b_1'],params['b_2'],params['b_3'],params['b_4'],
                       params['b_5'],params['b_6'],params['b_7'],params['b_8']]
    r = [params['r_0'],params['r_1'],params['r_2'],params['r_3'],params['r_4'],
                       params['r_5'],params['r_6'],params['r_7'],params['r_8']]
    mu1    = np.float64(params['mu_BsMC']);
    mu2    = np.float64(params['mu_BdMC'])
    sigma1 = np.float64(params['sigma_BsMC']);
    sigma2 = np.float64(params['sigma_BdMC'])
    gamma1 = np.float64(params['gamma_BsMC']);
    gamma2 = np.float64(params['gamma_BdMC'])
    size1  = np.int32(lkhd[0].shape[0]);
    size2  = np.int32(lkhd[1].shape[0])
    size_max = max(size1,size2)
    self.kRatioTimeAcc(time[0], time[1], lkhd[0], lkhd[1],
                       cu_array.to_gpu(self.get4coeffs(b)).astype(np.float64),
                       cu_array.to_gpu(self.get4coeffs(r)).astype(np.float64),
                       mu1, sigma1, gamma1, mu2, sigma2, gamma2,
                       np.float64(0.3),np.float64(15),
                       size1, size2,
                       block = (BLOCK_SIZE,1,1),
                       grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
                      )



  def getFullTimeAcc(self, time, lkhd, params, BLOCK_SIZE=32):
    """
    In:
            time: 3D list of 1D gpuarray with time to be fitted, the expected
                  format is [time1,time2] (tipically 1:BsMC 2:BdMC)
            lkhd: 3D list of 1D gpuarray where likelihood is being stored
         params1: dict of params {'name': value}
      BLOCK_SIZE: device block/workgroup size

    Out:
            void

    Look at kernel definition to see more help
    """
    b = [params['b_0'],params['b_1'],params['b_2'],params['b_3'],params['b_4'],
                       params['b_5'],params['b_6'],params['b_7'],params['b_8']]
    r = [params['r_0'],params['r_1'],params['r_2'],params['r_3'],params['r_4'],
                       params['r_5'],params['r_6'],params['r_7'],params['r_8']]
    c = [params['c_0'],params['c_1'],params['c_2'],params['c_3'],params['c_4'],
                       params['c_5'],params['c_6'],params['c_7'],params['c_8']]
    mu1    = np.float64(params['mu_BsMC'])
    mu2    = np.float64(params['mu_BdMC'])
    mu3    = np.float64(params['mu_BdDT'])
    sigma1 = np.float64(params['sigma_BsMC'])
    sigma2 = np.float64(params['sigma_BdMC'])
    sigma3 = np.float64(params['sigma_BdDT'])
    gamma1 = np.float64(params['gamma_BsMC'])
    gamma2 = np.float64(params['gamma_BdMC'])
    gamma3 = np.float64(params['gamma_BdDT'])
    size1  = np.int32(lkhd[0].shape[0])
    size2  = np.int32(lkhd[1].shape[0])
    size3  = np.int32(lkhd[2].shape[0])
    size_max = max(size1,size2,size3)
    self.kFullTimeAcc(time[0], time[1], time[2], lkhd[0], lkhd[1], lkhd[2],
                       cu_array.to_gpu(self.get4coeffs(b)).astype(np.float64),
                       cu_array.to_gpu(self.get4coeffs(r)).astype(np.float64),
                       cu_array.to_gpu(self.get4coeffs(c)).astype(np.float64),
                       mu1, sigma1, gamma1, mu2, sigma2, gamma2, mu3, sigma3, gamma3,
                       np.float64(0.3),np.float64(15),
                       size1, size2, size3,
                       block = (BLOCK_SIZE,1,1),
                       grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
                      )










  def getKnot(self, i, knots, n):
    if (i<=0):        i = 0;
    elif (i>=n):      i = n
    return knots[i]

  def get4coeffs(self, listcoeffs):
    n = self.nknots
    result = []                                           # list of bin coeffs C
    def u(j): return self.getKnot(j,self.knots,self.nknots-1)
    for i in range(0,self.nknots-1):
          a, b, c, d = listcoeffs[i:i+4]                    # bspline coeffs b_i
          C = []                                   # each bin 4 coeffs c_{bin,i}
          C.append(-((b*u(-2 + i)*pow(u(1 + i),2))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i)))) +
           (a*pow(u(1 + i),3))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
           (c*pow(u(-1 + i),2)*u(1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           (b*u(-1 + i)*u(1 + i)*u(2 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           (c*u(-1 + i)*u(i)*u(2 + i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (b*u(i)*pow(u(2 + i),2))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (d*pow(u(i),3))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
           (c*pow(u(i),2)*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
          C.append((2*b*u(-2 + i)*u(1 + i))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
           (3*a*pow(u(1 + i),2))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
           (b*pow(u(1 + i),2))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
           (c*pow(u(-1 + i),2))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           (b*u(-1 + i)*u(1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           (2*c*u(-1 + i)*u(1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           (b*u(-1 + i)*u(2 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           (b*u(1 + i)*u(2 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           (c*u(-1 + i)*u(i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (c*u(-1 + i)*u(2 + i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
           (2*b*u(i)*u(2 + i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (c*u(i)*u(2 + i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
           (b*pow(u(2 + i),2))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (c*pow(u(i),2))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
           (3*d*pow(u(i),2))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) -
           (2*c*u(i)*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
          C.append(-((b*u(-2 + i))/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*
                (-u(i) + u(1 + i)))) + (3*a*u(1 + i))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
           (2*b*u(1 + i))/
            ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
           (b*u(-1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           (2*c*u(-1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           (b*u(1 + i))/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*
              (-u(-1 + i) + u(2 + i))) + (c*u(1 + i))/
            ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           (b*u(2 + i))/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*
              (-u(-1 + i) + u(2 + i))) + (c*u(-1 + i))/
            ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (b*u(i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
           (c*u(i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           (2*b*u(2 + i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
           (c*u(2 + i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
           (2*c*u(i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) -
           (3*d*u(i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
           (c*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
          C.append(-(a/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i)))) +
           b/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
           b/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
           c/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
           b/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           c/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
           c/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
           d/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
          result.append(C)
    m = C[1] + 2*C[2]*u(n) + 3*C[3]*u(n)**2
    C = [C[0] + C[1]*u(n) + C[2]*u(n)**2 + C[3]*u(n)**3 - m*u(n),m,0,0]
    result.append(C)
    return np.array(result)

# cu_path = os.path.join(os.environ['PHIS_SCQ'],'cuda')
# BsJpsiKK = Badjanak(cu_path)
# dir(BsJpsiKK)
