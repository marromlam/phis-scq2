# -*- coding: utf-8 -*-

import os
import builtins

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule

import numpy as np



# This file path
PATH = os.path.dirname(os.path.abspath(__file__))
BACKEND = builtins.BACKEND
DEVICE = builtins.DEVICE
CONTEXT = builtins.CONTEXT
QUEUE = builtins.THREAD



# Default compile flags
#     The compile_flags is a dict where each key substitutes a same-named
#     string in the kernel file by its value: #define KEY {KEY} <-- value
global config
config = dict(
debug =           0, # no prints
debug_evt =       1, # number of events to debug
use_time_acc =    0, # no  time acceptance
use_time_offset = 0, # no  time offset
use_time_res =    0, # use time resolution
use_perftag =     1, # use perfect tagging
use_truetag =     0, # no  true tagging
nknots =          7,
ntimebins =       8,
sigma_t =         0.15,
knots =           [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00],
nmassbins =       6,
x_m =             [990, 1008, 1016, 1020, 1024, 1032, 1050],
tristan =         [1,1,1,0,0,0,1,0,0,0],
ang_acc =         [1,1,1,0,0,0,1,0,0,0],
nterms =          10,
csp =             [1,1,1,1,1,1,1,1,1,1]
)



def flagger():
  #global config
  dict_flags = {}
  for key, value in zip(config.keys(),config.values()):
    dict_flags[key.upper()] = str(value).replace('[','{').replace(']','}')
  return dict_flags



# Compiler
#     Compile kernel against given BACKEND
def compile():
  kernel_path = os.path.join(PATH,'Badjanak.cu')
  Badjanak = SourceModule(open(kernel_path,"r").read().format(**flagger()),
                          no_extern_c=False, arch=None, code=None,
                          include_dirs=[PATH])
  return Badjanak



# Get kernels
def get_kernels():
  global __KERNELS
  Badjanak = compile()
  items = ['pyDiffRate',
           'pyFcoeffs', 'pyAngularWeights', 'pyAngularCov',
           'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc', 'pySpline']
  for item in items:
    setattr(Badjanak, item[2:], Badjanak.get_function(item))
  __KERNELS = Badjanak



# Update property
def update_property(self,property,value):
  global __KERNELS
  setattr(property, value)
  __KERNELS = get_kernels()



get_kernels()



# Functions
#     Here pythonic versions of KERNEL functions are defined. There are wrappers
#     that are simpler to interact with


# def updateProperty(self,property,value):
#   setattr(property, value)
#   getKernels()

# def compileCU(self):
#   kernel_path = os.path.join(path,'Badjanak.cu')
#   # print(config)
#   #print(open(kernel_path,"r").read().format(**compile_flags))
#   Badjanak = SourceModule(open(kernel_path,"r").read()
#                           .format(**compile_flags),
#                           no_extern_c=False, arch=None, code=None,
#                           include_dirs=[path])
#   return Badjanak

# def getKernels(self):
#   #try:
#   __Badjanak = compileCU()
#   items = ['pyDiffRate',
#            'pyFcoeffs', 'pyAngularWeights', 'pyAngularCov',
#            'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc', 'pySpline']
#   for item in items:
#     setattr('k'+item[2:], __Badjanak.get_function(item))
#   #except:
#   #  print('Error!')



def diff_cross_rate(
      vars,pdf,
      Gs = 0.66137, DGs = 0.08, DM = 17.7,
      fSlon = [1],
      fPlon = 0.7217, fPper = 0.4988,
      dSlon = [3.07],
      dPlon = 0, dPpar = 3.30, dPper = 3.07,
      pSlon = 0,
      pPlon = -0.03, pPpar = 0, pPper = 0,
      lSlon = 1,
      lPlon = 1, lPpar = 1, lPper = 1,
      c = [1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9],
      BLOCK_SIZE=32):
  """
  Look at kernel definition to see help
  """
  ASlon = np.atleast_1d(fSlon)
  FP = abs(1-ASlon)
  APlon = FP*fPlon; APper = FP*fPper; APpar = FP*abs(1-APlon-APper) # Amplitudes
  dSlon = np.atleast_1d(np.array(dSlon)) + dPper                 # Strong phases

  __KERNELS.DiffRate(
    vars, pdf,
    np.float64(Gs), np.float64(DGs), np.float64(DM),
    #cu_array.to_gpu(CSP).astype(np.float64),
    cu_array.to_gpu(np.sqrt(ASlon)).astype(np.float64),
    cu_array.to_gpu(np.sqrt(APlon)).astype(np.float64),
    cu_array.to_gpu(np.sqrt(APpar)).astype(np.float64),
    cu_array.to_gpu(np.sqrt(APper)).astype(np.float64),
    np.float64(pSlon+pPlon),
    np.float64(pPlon), np.float64(pPpar+pPlon), np.float64(pPper+pPlon),
    cu_array.to_gpu(dSlon).astype(np.float64),
    np.float64(dPlon), np.float64(dPpar), np.float64(dPper),
    np.float64(lSlon*lPlon),
    np.float64(lPlon), np.float64(lPpar*lPlon), np.float64(lPper*lPlon),
    np.float64(0.3), np.float64(15),
    cu_array.to_gpu(get_4cs(c)).astype(np.float64),
    np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,config['nmassbins']+1,1),
    grid = (int(np.ceil(vars.shape[0]/BLOCK_SIZE)),1,1))



def getAngularWeights(data, weight, pars, coeffs=None, BLOCK_SIZE=32):
  """
  getAngularWeights(data,vars_reco,weights,pars):

    In:
            data:  eventsx4 matrix that stores [cosK, cosL, hphi, time]
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
  ang_acc = cu_array.to_gpu(np.zeros(10)).astype(np.float64)

  kAngularWeights(
        data, weight, ang_acc,
        np.float64(["Gd"]+pars["DGsd"]), np.float64(pars["DGs"]),
        np.float64(pars["DM"]), np.float64(pars["CSP"]),
        np.float64(np.sqrt(ASlon)), np.float64(np.sqrt(APlon)), np.float64(np.sqrt(APpar)), np.float64(np.sqrt(APper)),
        np.float64(pars["pSlon"]), np.float64(pars["pPlon"]), np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
        np.float64(pars["dSlon"]), np.float64(pars["dPlon"]), np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
        np.float64(pars["lSlon"]), np.float64(pars["lPlon"]), np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
        np.float64(0.3), np.float64(15),
        cu_array.to_gpu(coeffs).astype(np.float64),
        np.int32(data.shape[0]),
        block = (BLOCK_SIZE,1,1),
        grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1)
      )
  result = ang_acc.get()
  return result#/result[0]



def getAngularCov(data, weights, pars, coeffs=None, BLOCK_SIZE=32):
  """
  getAngularWeights(data,vars_reco,weights,pars):

    In:
            data:  eventsx4 matrix that stores [cosK, cosL, hphi, time]
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
  ang_acc_ = getAngularWeights(data, weights, pars)
  ang_acc = cu_array.to_gpu(ang_acc_).astype(np.float64)
  cov_mat = cu_array.to_gpu(np.zeros([10,10])).astype(np.float64)

  kAngularCov(
        data, weights, ang_acc, cov_mat,
        np.float64(pars["Gd"]+pars["DGsd"]), np.float64(pars["DGs"]),
        np.float64(pars["DM"]), np.float64(pars["CSP"]),
        np.float64(np.sqrt(ASlon)), np.float64(np.sqrt(APlon)), np.float64(np.sqrt(APpar)), np.float64(np.sqrt(APper)),
        np.float64(pars["pSlon"]), np.float64(pars["pPlon"]), np.float64(pars["pPpar"]), np.float64(pars["pPper"]),
        np.float64(pars["dSlon"]), np.float64(pars["dPlon"]), np.float64(pars["dPpar"]), np.float64(pars["dPper"]),
        np.float64(pars["lSlon"]), np.float64(pars["lPlon"]), np.float64(pars["lPpar"]), np.float64(pars["lPper"]),
        np.float64(0.3), np.float64(15),
        cu_array.to_gpu(coeffs).astype(np.float64),
        np.int32(data.shape[0]),
        block = (BLOCK_SIZE,1,1),
        grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1)
      )
  w = ang_acc.get(); cov = cov_mat.get()

  cov = cov + cov.T - np.eye(cov.shape[0])*cov # fill the lower-triangle
  final_cov = np.zeros_like(cov)
  for i in range(0,cov.shape[0]):
    for j in range(0,cov.shape[1]):
      final_cov[i,j] = 1.0/(w[0]*w[0])*(
                                w[i]*w[j]/(w[0]*w[0])*cov[0][0]+cov[i][j]-
                                w[i]/w[0]*cov[0][j]-w[j]/w[0]*cov[0][i]);

  return w/w[0], np.sqrt(np.diagonal(final_cov)) #result#/result[0]



def single_spline_time_acceptance(
      time, lkhd,
      b0=1, b1=1.3, b2=1.5, b3=1.8, b4=2.1, b5=2.3, b6=2.2, b7=2.1, b8=2.0,
      mu=0.0, sigma=0.04, gamma=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=256
    ):
  """
    In:
                 time:  1D gpuarray with time to be fitted
                 lkhd:  1D gpuarray where likelihood is being stored
              *params:  list of parameters
           BLOCK_SIZE:  device block/workgroup size

    Out:
                 void

  Look at pySingleTimeAcc kernel definition to see more help.
  """
  b = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
  __KERNELS.SingleTimeAcc(
              time, lkhd, # input, output
              cu_array.to_gpu(get_4cs(b)).astype(np.float64),
              np.float64(mu), np.float64(sigma), np.float64(gamma),
              np.float64(tLL),np.float64(tUL),
              np.int32(lkhd.shape[0]),
              block = (BLOCK_SIZE,1,1),
              grid = (int(np.ceil(lkhd.shape[0]/BLOCK_SIZE)),1,1)
            )



def ratio_spline_time_acceptance(
      time_a, time_b, lkhd_a, lkhd_b,
      a0=1, a1=1.3, a2=1.5, a3=1.8, a4=2.1, a5=2.3, a6=2.2, a7=2.1, a8=2.0,
      mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
      b0=1, b1=1.2, b2=1.4, b3=1.7, b4=2.2, b5=2.2, b6=2.1, b7=2.0, b8=1.9,
      mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=256):
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
  a = [a0, a1, a2, a3, a4, a5, a6, a7, a8]
  b = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_max = max(size_a,size_b)
  __KERNELS.RatioTimeAcc(
              time_a, time_b, lkhd_a, lkhd_b,
              cu_array.to_gpu(get_4cs(a)).astype(np.float64),
              cu_array.to_gpu(get_4cs(b)).astype(np.float64),
              np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
              np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
              np.float64(tLL),np.float64(tUL),
              size_a, size_b,
              block = (BLOCK_SIZE,1,1),
              grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
            )



def full_spline_time_acceptance(
      time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
      a0=1, a1=1.3, a2=1.5, a3=1.8, a4=2.1, a5=2.3, a6=2.2, a7=2.1, a8=2.0,
      mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
      b0=1, b1=1.2, b2=1.4, b3=1.7, b4=2.2, b5=2.2, b6=2.1, b7=2.0, b8=1.9,
      mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
      c0=1, c1=1.2, c2=1.4, c3=1.7, c4=2.2, c5=2.2, c6=2.1, c7=2.0, c8=1.9,
      mu_c=0.0, sigma_c=0.04, gamma_c=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=32):
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
  a = [a0, a1, a2, a3, a4, a5, a6, a7, a8]
  b = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
  c = [c0, c1, c2, c3, c4, c5, c6, c7, c8]
  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_c  = np.int32(lkhd_c.shape[0])
  size_max = max(size_a,size_b,size_c)
  __KERNELS.FullTimeAcc(
              time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
              cu_array.to_gpu(get_4cs(a)).astype(np.float64),
              cu_array.to_gpu(get_4cs(b)).astype(np.float64),
              cu_array.to_gpu(get_4cs(c)).astype(np.float64),
              np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
              np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
              np.float64(mu_c), np.float64(sigma_c), np.float64(gamma_c),
              np.float64(tLL),np.float64(tUL),
              size_a, size_b, size_c,
              block = (BLOCK_SIZE,1,1),
              grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
            )



def acceptance_spline(
      time, spline,
      b0=1, b1=1.3, b2=1.5, b3=1.8, b4=2.1, b5=2.3, b6=2.2, b7=2.1, b8=2.0,
      BLOCK_SIZE=32):
  coeffs = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
  coeffs_d = cu_array.to_gpu(get_4cs(coeffs)).astype(np.float64)
  n_evt    = len(time)
  __KERNELS.Spline(time, spline, coeffs_d, np.int32(n_evt),
                   block = (BLOCK_SIZE,1,1),
                   grid = (int(np.ceil(n_evt/BLOCK_SIZE)),1,1)
                  )



def get_knot(i, knots, n):
  if (i<=0):        i = 0;
  elif (i>=n):      i = n
  return knots[i]



def get_4cs(listcoeffs):
  n = config['nknots']
  result = []                                           # list of bin coeffs C
  def u(j): return get_knot(j,config['knots'],config['nknots']-1)
  for i in range(0,config['nknots']-1):
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
