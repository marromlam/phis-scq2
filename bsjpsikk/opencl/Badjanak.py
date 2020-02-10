# -*- coding: utf-8 -*-

import os
import builtins

import pyopencl as cl
import pyopencl.array as cl_array



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
ntimebins =       8,
sigma_t =         0.15,
knots =           [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00],
x_m =             [990,1050],#[990, 1008, 1016, 1020, 1024, 1032, 1050],
tristan =         [1,1,1,0,0,0,1,0,0,0],
ang_acc =         [1,1,1,0,0,0,1,0,0,0],
csp =             [1]
)



def flagger():
  #global config
  dict_flags = {}
  for key, value in zip(config.keys(),config.values()):
    if key == 'x_m':
      dict_flags['nmassbins'.upper()] = len(value)-1
      dict_flags['nmassknots'.upper()] = len(value)
    if key == 'knots':
      dict_flags['nknots'.upper()] = len(value)
    if key == 'tristan':
      dict_flags['nterms'.upper()] = len(value)
    dict_flags[key.upper()] = str(value).replace('[','{').replace(']','}')
  return dict_flags



# Compiler
#     Compile kernel against given BACKEND
def compile():
  kernel_path = os.path.join(PATH,'Badjanak.cl')
  print(open(kernel_path,"r").read().format(**flagger()))
  Badjanak = cl.Program(CONTEXT,
                        open(kernel_path,"r").read().format(**flagger())
                       ).build(options=["-I "+PATH])
  return Badjanak



# Get kernels
def get_kernels():
  global __KERNELS
  Badjanak = compile()
  items = Badjanak.kernel_names.split(';')
  #
  #
  for item in items:
    setattr(Badjanak, item[2:], Badjanak.__getattr__(item))
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

def diff_cross_rate(
      vars, pdf,
      Gs = 0.66137, DGs = 0.08, DM = 17.7,
      fSlon = [0],
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
  dSlon = np.atleast_1d(np.array(dSlon)) + dPper
  # print(f'ASlon = {ASlon} with {ASlon.dtype}')
  # print(f'APlon = {APlon} with {APlon.dtype}')
  # print(f'APper = {APper} with {APper.dtype}')
  # print(f'APpar = {APpar} with {APpar.dtype}')

  __KERNELS.DiffRate(
    THREAD._queue,
    (int(np.ceil(pdf.shape[0]/BLOCK_SIZE)),),
    (BLOCK_SIZE,len(config['csp']),1),
    vars.data, pdf.data,
    np.float64(Gs), np.float64(DGs), np.float64(DM),
    #cl_array.to_device(QUEUE, CSP).astype(np.float64).data,
    QUEUE.to_device(np.sqrt(ASlon)).astype(np.float64).data,
    QUEUE.to_device(np.sqrt(APlon)).astype(np.float64).data,
    QUEUE.to_device(np.sqrt(APpar)).astype(np.float64).data,
    QUEUE.to_device(np.sqrt(APper)).astype(np.float64).data,
    np.float64(pSlon+pPlon),
    np.float64(pPlon), np.float64(pPpar+pPlon), np.float64(pPper+pPlon),
    QUEUE.to_device(dSlon).astype(np.float64).data,
    np.float64(dPlon), np.float64(dPpar), np.float64(dPper),
    np.float64(lSlon*lPlon),
    np.float64(lPlon), np.float64(lPpar*lPlon), np.float64(lPper*lPlon),
    np.float64(0.3), np.float64(15),
    QUEUE.to_device(get_4cs(c)).astype(np.float64).data,
    np.int32(pdf.shape[0]), g_times_l = True
    )

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
  __KERNELS.KERNELAngularWeights(QUEUE,
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
              cl_array.to_device(QUEUE, coeffs).astype(np.float64).data,
              np.int32(vars_true.shape[0]), g_times_l = True )
  print(weights)
  return weights.get()/weights.get()[0]


def get_knot(i, knots, n):
  if (i<=0):        i = 0;
  elif (i>=n):      i = n
  return knots[i]



def get_4cs(listcoeffs):
  n = len(config['knots'])
  result = []                                           # list of bin coeffs C
  def u(j): return get_knot(j,config['knots'],len(config['knots'])-1)
  for i in range(0,len(config['knots'])-1):
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
