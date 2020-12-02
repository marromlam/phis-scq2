# -*- coding: utf-8 -*-

import os
import builtins

from reikna.cluda import functions, dtypes

import numpy as np
import ipanema
from ipanema import ristra

import warnings

#import pycuda.driver as cuda
#import pycuda.autoinit
#import pycuda.gpuarray as cu_array
from pycuda.compiler import SourceModule

# This file path
try:
  PATH = os.path.dirname(os.path.abspath(__file__))
except:
  PATH = '/home3/marcos.romero/phis-scq/bsjpsikk/cuda'
  ipanema.initialize('cuda',1)
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
    if key == 'x_m':
      dict_flags['nmassbins'.upper()] = len(value)-1
      dict_flags['nmassknots'.upper()] = len(value)
    if key == 'knots':
      dict_flags['nknots'.upper()] = len(value)
    if key == 'tristan':
      dict_flags['nterms'.upper()] = len(value)
    dict_flags[key.upper()] = str(value).replace('[','{').replace(']','}')
  for key, value in config.items():
    print(f'{key.upper():<20}: {value}')
  return dict_flags




# Compiler
#     Compile kernel against given BACKEND
def compile():
  kernel_path = os.path.join(PATH,'Badjanak.cu')
  #print( open(kernel_path,"r").read().format(**flagger()) )
  Badjanak = SourceModule(open(kernel_path,"r").read().format(**flagger()),
                          no_extern_c=False, arch=None, code=None,
                          include_dirs=[PATH])
  #Badjanak = THREAD.compile(open(kernel_path,"r").read().format(**flagger()),
  #                          compiler_options={f'-I {PATH}'})
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
    #setattr(Badjanak, item[2:], Badjanak.__getattr__(item))
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
      vars, pdf,
      Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7, CSP   = 1,
      fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
      dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
      pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
      lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
      fSlon1=0, fSlon2=0, fSlon3=0,fSlon4=0,fSlon5=0, fSlon6=0, # binned mass
      dSlon1=0, dSlon2=0, dSlon3=0,dSlon4=0,dSlon5=0, dSlon6=0, # binned mass
      CSP1=0, CSP2=0, CSP3=0,CSP4=0,CSP5=0, CSP6=0,             # binned mass
      Gs = None,
      nknots = 7,
      knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00],
      coeffs = [1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9],
      use_fk=1,
      BLOCK_SIZE=32):
  """
  Look at kernel definition to see help
  """
  # print(f"{fSlon}\n{fPlon}\n{fPper}\n{dSlon}")
  if CSP1: # then use binned mass
    CSP = [c for c in (CSP1, CSP2, CSP3, CSP4, CSP5, CSP6) if c != 0]
    fSlon = [c for c in (fSlon1, fSlon2, fSlon3, fSlon4, fSlon5, fSlon6) if c != 0]
    dSlon = [c for c in (dSlon1, dSlon2, dSlon3, dSlon4, dSlon5, dSlon6) if c != 0]
    #CSP   = [CSP1, CSP2, CSP3, CSP4, CSP5, CSP6]
    #fSlon = [fSlon1, fSlon2, fSlon3, fSlon4, fSlon5, fSlon6]
    #dSlon = [dSlon1, dSlon2, dSlon3, dSlon4, dSlon5, dSlon6]
  ASlon = np.atleast_1d(fSlon)
  FP = abs(1-ASlon)
  APlon = FP*fPlon; APper = FP*fPper; APpar = FP*abs(1-fPlon-fPper) # Amplitudes
  dSlon = np.atleast_1d(dSlon) + dPper                           # Strong phases
  CSP   = np.atleast_1d(CSP)                                       # CSP factors
  if Gs==None: Gs = Gd+DGsd+DGd
  # print(f"{ASlon}\n{APlon}\n{APper}\n{APpar}\n{dSlon}")
  # print(len(CSP),CSP)
  __KERNELS.DiffRate(
    vars, pdf,
    np.float64(Gs), np.float64(DGs), np.float64(DM),
    ristra.allocate(CSP).astype(np.float64),
    ristra.allocate(np.sqrt(ASlon)).astype(np.float64),
    ristra.allocate(np.sqrt(APlon)).astype(np.float64),
    ristra.allocate(np.sqrt(APpar)).astype(np.float64),
    ristra.allocate(np.sqrt(APper)).astype(np.float64),
    np.float64(pSlon+pPlon),
    np.float64(pPlon), np.float64(pPpar+pPlon), np.float64(pPper+pPlon),
    ristra.allocate(dSlon).astype(np.float64),
    np.float64(dPlon), np.float64(dPpar), np.float64(dPper),
    np.float64(lSlon*lPlon),
    np.float64(lPlon), np.float64(lPpar*lPlon), np.float64(lPper*lPlon),
    np.float64(0.3), np.float64(15),

    ristra.allocate(get_4cs(coeffs)).astype(np.float64),
    np.int32(use_fk), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,len(CSP),1),
    grid = (int(np.ceil(vars.shape[0]/BLOCK_SIZE)),1,1))


# def cross_rate_parser(parameters):
#   pars = dict(
#   Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7,
#   CSP = 1,                     # by default we asume only one mass bin and CSP=1
#   fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
#   dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
#   pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
#   lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
#   fSlon1=None,      # if more than 1 mass bin, then fSloni, we take care of them
#   dSlon1=None,      # if more than 1 mass bin, then dSloni, we take care of them
#   CSP1=None,          # if more than 1 mass bin, then CSPi, we take care of them
#   Gs = None,   # by default we assume Gs = Gd+DGsd, but Gs can be directly input
#   timeacc = np.array([1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9]),
#   angacc = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
#   # Time resolution parameters
#   sigma_offset = 0.01297,
#   sigma_slope = 0.8446,
#   sigma_curvature = 0,
#   mu = 0,
#   # Flavor tagging parameters
#   eta_os = 0.3602,
#   eta_ss = 0.4167,
#   p0_os = 0.389,
#   p0_ss = 0.4325,
#   p1_os = 0.8486,
#   p2_os = 0.,
#   p1_ss = 0.9241,
#   p2_ss = 0.,
#   dp0_os = 0.009,
#   dp0_ss = 0,
#   dp1_os = 0.0143,
#   dp2_os = 0.,
#   dp1_ss = 0,
#   dp2_ss = 0,
#   tLL = 0.3,
#   tUL = 15.0,
#   )
#   #pars.update(parameters)
#   len_csp = len([v for k,v in parameters.items() if k[:3]=='CSP'])
#   if pars['CSP1']: # then use binned mass
#     pars['CSP']   = [ pars['CSP'+str(i)]   for i in range(1,len_csp) ]
#     pars['fSlon'] = [ pars['fSlon'+str(i)] for i in range(1,len_csp) ]
#     pars['dSlon'] = [ pars['dSlon'+str(i)] for i in range(1,len_csp) ]
#     #CSP   = [CSP1, CSP2, CSP3, CSP4, CSP5, CSP6]
#     #fSlon = [fSlon1, fSlon2, fSlon3, fSlon4, fSlon5, fSlon6]
#     #dSlon = [dSlon1, dSlon2, dSlon3, dSlon4, dSlon5, dSlon6]
#   pars['CSP']   = np.atleast_1d(pars['CSP'])
#   #pars['fSlon'] = np.atleast_1d(pars['fSlon'])
#   pars['dSlon'] = np.atleast_1d(pars['dSlon'])
#   pars['ASlon'] = np.atleast_1d(pars['fSlon'])
#   FP = abs(1-pars['ASlon'])
#   pars['APlon'] = FP*pars['fPlon'];
#   pars['APper'] = FP*pars['fPper'];
#   pars['APpar'] = FP*abs(1-pars['fPlon']-pars['fPper'])        # Amplitudes
#   pars['dSlon'] = pars['dSlon'] + pars['dPper']               # Strong phases
#   if pars['Gs'] == None:
#     #print('no Gs')
#     pars['Gs'] = pars['Gd']+pars['DGsd']#+pars['DGd']
#   pars['nknots'] = len(pars['knots'])
#   # for k, v in pars.items():
#   #  print(f'{k:>10}:  {v}')
#   return pars




def cross_rate_parser(parameters):
    pars = dict(
    Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7, CSP   = 1,
    fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
    dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
    pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
    lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
    fSlon1=0, #fSlon2=0, fSlon3=0,fSlon4=0,fSlon5=0, fSlon6=0, # binned mass
    dSlon1=0, #dSlon2=0, dSlon3=0,dSlon4=0,dSlon5=0, dSlon6=0, # binned mass
    CSP1=None, #CSP2=0, CSP3=0,CSP4=0,CSP5=0, CSP6=0,             # binned mass
    Gs = None,
    knots = np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00]),
    coeffs = np.array([1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9]),
    w = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    # Time resolution parameters
    sigma_offset = 0.01297,
    sigma_slope = 0.8446,
    sigma_curvature = 0,
    mu = 0,
    # Flavor tagging parameters
    eta_os = 0.3602,
    eta_ss = 0.4167,
    p0_os = 0.389,
    p0_ss = 0.4325,
    p1_os = 0.8486,
    p2_os = 0.,
    p1_ss = 0.9241,
    p2_ss = 0.,
    dp0_os = 0.009,
    dp0_ss = 0,
    dp1_os = 0.0143,
    dp2_os = 0.,
    dp1_ss = 0,
    dp2_ss = 0,
    tLL = 0.3,
    tUL = 15.0,
    )
    pars.update(parameters)
    len_csp = len([pars[p] for p in pars if p[:3]=='CSP'])
    pars['mass_bins'] = len_csp-1
    if pars['CSP1']: # then use binned mass
      pars['CSP']   = [ pars['CSP'+str(i)]   for i in range(1,len_csp)]
      pars['fSlon'] = [ pars['fSlon'+str(i)] for i in range(1,len_csp)]
      pars['dSlon'] = [ pars['dSlon'+str(i)] for i in range(1,len_csp)]
      #CSP   = [CSP1, CSP2, CSP3, CSP4, CSP5, CSP6]
      #fSlon = [fSlon1, fSlon2, fSlon3, fSlon4, fSlon5, fSlon6]
      #dSlon = [dSlon1, dSlon2, dSlon3, dSlon4, dSlon5, dSlon6]
    pars['CSP']   = np.atleast_1d(pars['CSP'])
    #pars['fSlon'] = np.atleast_1d(pars['fSlon'])
    pars['dSlon'] = np.atleast_1d(pars['dSlon'])
    pars['ASlon'] = np.atleast_1d(pars['fSlon'])
    FP = abs(1-pars['ASlon'])
    pars['APlon'] = FP*pars['fPlon'];
    pars['APper'] = FP*pars['fPper'];
    pars['APpar'] = FP*abs(1-pars['fPlon']-pars['fPper'])        # Amplitudes
    pars['dSlon'] = pars['dSlon'] + pars['dPper']               # Strong phases
    if pars['Gs'] == None:
      #print('no Gs')
      pars['Gs'] = pars['Gd']+pars['DGsd']#+pars['DGd']
    pars['nknots'] = len(pars['knots'])
    # for k, v in pars.items():
    #  print(f'{k:>10}:  {v}')
    return pars










def cross_rate_parser2(parameters):
    pars = dict(
    Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7, CSP   = 1,
    fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
    dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
    pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
    lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
    Gs = None,
    knots = np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00]),
    coeffs = np.array([1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9]),
    w = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    # Time resolution parameters
    sigma_offset = 0.01297,
    sigma_slope = 0.8446,
    sigma_curvature = 0,
    mu = 0,
    # Flavor tagging parameters
    eta_os = 0.3602,
    eta_ss = 0.4167,
    p0_os = 0.389,
    p0_ss = 0.4325,
    p1_os = 0.8486,
    p2_os = 0.,
    p1_ss = 0.9241,
    p2_ss = 0.,
    dp0_os = 0.009,
    dp0_ss = 0,
    dp1_os = 0.0143,
    dp2_os = 0.,
    dp1_ss = 0,
    dp2_ss = 0,
    tLL = 0.3,
    tUL = 15.0,
    )
    pars.update(parameters)
    #pars['fSlon'] = np.atleast_1d(pars['fSlon'])
    pars['fSlon'] = [pars['fSlon1'],pars['fSlon2'],pars['fSlon3'],pars['fSlon4'],pars['fSlon5'],pars['fSlon6']]
    pars['dSlon'] = [pars['dSlon1'],pars['dSlon2'],pars['dSlon3'],pars['dSlon4'],pars['dSlon5'],pars['dSlon6']]
    pars['dSlon'] = np.atleast_1d(pars['dSlon'])
    pars['ASlon'] = np.atleast_1d(pars['fSlon'])
    FP = abs(1-pars['ASlon'])
    pars['APlon'] = FP*pars['fPlon'];
    pars['APper'] = FP*pars['fPper'];
    pars['APpar'] = FP*abs(1-pars['fPlon']-pars['fPper'])        # Amplitudes
    pars['dSlon'] = pars['dSlon'] + pars['dPper']               # Strong phases
    if pars['Gs'] == None:
      #print('no Gs')
      pars['Gs'] = pars['Gd']+pars['DGsd']+pars['DGd']
    pars['nknots'] = len(pars['knots'])
    # for k, v in pars.items():
    #  print(f'{k:>10}:  {v}')
    return pars






def diff_cross_rate_full( data, pdf, use_fk=1, BLOCK_SIZE=256, mass_bins=None, verbose=False, **parameters):
  """
  Look at kernel definition to see help
  """
  p = cross_rate_parser(parameters)
  if verbose:
    print(f"\n\n{'delta_gamma5 '+67*'-'}\n")
    print('You are running under the following configuration:')
    #print(f"Number of Events: {nevt}")
    print(f"Block size in device: {BLOCK_SIZE}")
    print(f"Time range: [{p['tLL']}, {p['tUL']}] ps")

    print(f"Gs : {p['Gs']}, DGs : {p['DGs']}, DM : {p['DM']}")
    print(f"CSP: {p['CSP']}")
    print(f"ASlon: {p['ASlon']}")
    print(f"APlon: {p['APlon']}")
    print(f"APpar: {p['APpar']}")
    print(f"APper: {p['APper']}")

    print(f"pSlon: {p['pSlon']}, pPlon: {p['pPlon']}, pPpar: {p['pPpar']}, pPper: {p['pPper']}")
    print(f"dSlon: {p['pSlon']}, dPlon: {p['pPlon']}, dPpar: {p['pPpar']}, dPper: {p['pPper']}")
    print(f"lSlon: {p['pSlon']}, lPlon: {p['pPlon']}, lPpar: {p['pPpar']}, lPper: {p['pPper']}")

    print(f"\nTime resolution parameters")
    print(f"sigma_offset: {p['sigma_offset']} sigma_slope: {p['sigma_slope']} sigma_curvature: {p['sigma_curvature']} mu:{p['mu']}")

    print(f"\nTagging parameters")
    print(f"eta_os: {p['eta_os']} eta_ss: {p['eta_ss']}")
    print(f"p0_os: {p['p0_os']} p1_os: {p['p1_os']} p2_os: {p['p2_os']}")
    print(f"dp0_os: {p['dp0_os']} dp1_os: {p['dp1_os']} dp2_os: {p['dp2_os']}")
    print(f"p0_ss: {p['p0_ss']} p1_ss: {p['p1_ss']} p2_ss: {p['p2_ss']}")
    print(f"dp0_ss: {p['dp0_ss']} dp1_ss: {p['dp1_ss']} dp2_ss: {p['dp2_ss']}")

    print(f"\nTime acceptance")
    print(f"{p['coeffs']}")
    print(f"\nAngular acceptance")
    print(f"{p['w']}")

    #print(f"{'APlon':>15} : {APlon}")
    print(f"\n{80*'-'}\n\n")
  if mass_bins:
    print(p['CSP'])
    p['CSP'] = np.array([1])
  else:
    mass_bins = len(p['CSP'])
  __KERNELS.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64(p['DGsd']+0.65789),
    np.float64(p['DGs']),
    np.float64(p['DM']),
    ristra.allocate(p['CSP']).astype(np.float64),
    ristra.allocate(np.sqrt(p['ASlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APpar'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APper'])).astype(np.float64),
    np.float64(p['pSlon']+p['pPlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']+p['pPlon']),
    np.float64(p['pPper']+p['pPlon']),
    ristra.allocate(p['dSlon']).astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']*p['lPlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']*p['lPlon']),
    np.float64(p['lPper']*p['lPlon']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']), np.float64(p['sigma_slope']), np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    ristra.allocate(get_4cs(p['coeffs'])).astype(np.float64),
    # Angular acceptance
    ristra.allocate(p['w']).astype(np.float64),
    #np.int32(use_fk), np.int32(pdf.shape[0]),
    np.int32(use_fk), np.int32(mass_bins), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,1,1),
    #block = (BLOCK_SIZE,len(p['CSP']),1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))



def delta_gamma5(input, output,
                  # Time-dependent angular distribution
                  G, DG, DM,
                  CSP,
                  ASlon, APlon, APpar, APper,
                  pSlon, pPlon, pPpar, pPper,
                  dSlon, dPlon, dPpar, dPper,
                  lSlon, lPlon, lPpar, lPper,
                  # Time limits
                  tLL, tUL,
                  # Time resolution
                  sigma_offset, sigma_slope, sigma_curvature,
                  mu,
                  # Flavor tagging
                  eta_bar_os, eta_bar_ss,
                  p0_os,  p1_os, p2_os,
                  p0_ss,  p1_ss, p2_ss,
                  dp0_os, dp1_os, dp2_os,
                  dp0_ss, dp1_ss, dp2_ss,
                  # Time acceptance
                  timeacc,
                  # Angular acceptance
                  angacc,
                  # Flags
                  use_fk=1, use_time_acc = 0, use_time_offset = 0,
                  use_time_res = 0, use_perftag = 1, use_truetag = 0,
                  BLOCK_SIZE=256, **crap):

  """
  Look at kernel definition to see help
  The aim of this function is to be the fastest wrapper
  """
  __KERNELS.DiffRate(
    # Input and output arrays
    input, output,
    # Differential cross-rate parameters
    np.float64(DGsd), np.float64(DGs), np.float64(DM),
    CSP.astype(np.float64),
    ASlon.astype(np.float64), APlon.astype(np.float64), APpar.astype(np.float64), APper.astype(np.float64),
    np.float64(pSlon),                 np.float64(pPlon),                 np.float64(pPpar),                 np.float64(pPper),
    dSlon.astype(np.float64),          np.float64(dPlon),                 np.float64(dPpar),                 np.float64(dPper),
    np.float64(lSlon),                 np.float64(lPlon),                 np.float64(lPpar),                 np.float64(lPper),
    # Time range
    np.float64(tLL), np.float64(tUL),
    # Time resolution
    np.float64(sigma_offset), np.float64(sigma_slope), np.float64(sigma_curvature),
    np.float64(mu),
    # Flavor tagging
    np.float64(eta_os), np.float64(eta_ss),
    np.float64(p0_os), np.float64(p1_os), np.float64(p2_os),
    np.float64(p0_ss), np.float64(p1_ss), np.float64(p2_ss),
    np.float64(dp0_os), np.float64(dp1_os), np.float64(dp2_os),
    np.float64(dp0_ss), np.float64(dp1_ss), np.float64(dp2_ss),
    # Decay-time acceptance
    timeacc.astype(np.float64),
    # Angular acceptance
    angacc.astype(np.float64),
    np.int32(use_fk), np.int32(len(CSP)), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,1,1),
    #np.int32(use_fk), np.int32(pdf.shape[0]),   # should check which is faster!
    #block = (BLOCK_SIZE,len(CSP']),1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))





def new_diff_rate(data, pdf, use_fk=1, BLOCK_SIZE=256, verbose=False, **p):

  mass_bins = np.int32(len(p['CSP']))
  zero = np.float64(0)

  # Get all bin parameters and put them in ristras
  if mass_bins > 1:
    fSlon = [p['fSlon1'],p['fSlon2'],p['fSlon3'],p['fSlon4'],p['fSlon5'],p['fSlon6']]
    fSlon = ristra.allocate(np.float64(fSlon)).astype(np.float64)
    dSlon = [p['dSlon1'],p['dSlon2'],p['dSlon3'],p['dSlon4'],p['dSlon5'],p['dSlon6']]
    dSlon = ristra.allocate(np.float64(dSlon)).astype(np.float64)
  else:
    fSlon = [p['fSlon']]
    fSlon = ristra.allocate(np.float64(fSlon)).astype(np.float64)
    dSlon = [p['dSlon']]
    dSlon = ristra.allocate(np.float64(dSlon)).astype(np.float64)


  # parameters
  DGs = np.float64(p["DGs"])
  DM = np.float64(p["DM"])
  # Parse Gs value
  if 'Gs' in p:
    print('there is Gs')
    Gs = np.float64(p['Gs'])
  else:
    Gs = np.float64(p['Gd']+p['DGsd'])#+p['DGd'])


  # Compute fractions of S and P wave objects
  FP = abs(1-fSlon)
  ASlon = ristra.sqrt( fSlon ).astype(np.float64)
  APlon = ristra.sqrt( FP*p['fPlon'] ).astype(np.float64);
  APper = ristra.sqrt( FP*p['fPper'] ).astype(np.float64);
  APpar = ristra.sqrt( FP*abs(1-p['fPlon']-p['fPper']) ).astype(np.float64)

  # Strong phases
  dPlon = np.float64(p['dPlon'])
  dPper = np.float64(p['dPper']) - dPlon
  dPpar = np.float64(p['dPpar']) - dPlon
  dSlon = ( dSlon + dPper ).astype(np.float64)

  # Weak phases
  pPlon = np.float64(p['pPlon'])
  pSlon = np.float64(p['pSlon']) + pPlon
  pPpar = np.float64(p['pPpar']) + pPlon
  pPper = np.float64(p['pPper']) + pPlon

  # Lambdas
  lPlon = np.float64(p['lPlon'])
  lSlon = np.float64(p['lSlon']) * lPlon
  lPpar = np.float64(p['lPpar']) * lPlon
  lPper = np.float64(p['lPper']) * lPlon

  # Time range
  tLL = np.float64(p['tLL'])
  tUL = np.float64(p['tUL'])

  # Time resolution
  sigma_offset = np.float64(p['sigma_offset']) if 'sigma_offset' in p else zero
  sigma_slope = np.float64(p['sigma_slope'])  if 'sigma_slope' in p else zero
  sigma_curvature = np.float64(p['sigma_curvature'])  if 'sigma_curvature' in p else zero
  mu = np.float64(p['mu'])

  # Tagging
  sigma_offset = np.float64(p['sigma_offset']) if 'sigma_offset' in p else zero
  sigma_slope = np.float64(p['sigma_slope'])  if 'sigma_slope' in p else zero
  sigma_curvature = np.float64(p['sigma_curvature'])  if 'sigma_curvature' in p else zero
  mu = np.float64(p['mu'])

  # # Tagging
  eta_os = np.float64(p['eta_os']) if 'eta_os' in p else zero
  eta_ss = np.float64(p['eta_ss']) if 'eta_ss' in p else zero
  p0_os = np.float64(p['p0_os']) if 'p0_os' in p else zero
  p1_os = np.float64(p['p1_os']) if 'p1_os' in p else zero
  p2_os = np.float64(p['p2_os']) if 'p2_os' in p else zero
  p0_ss = np.float64(p['p0_ss']) if 'p0_ss' in p else zero
  p1_ss = np.float64(p['p1_ss']) if 'p1_ss' in p else zero
  p2_ss = np.float64(p['p2_ss']) if 'p2_ss' in p else zero
  dp0_os = np.float64(p['dp0_os']) if 'dp0_os' in p else zero
  dp1_os = np.float64(p['dp1_os']) if 'dp1_os' in p else zero
  dp2_os = np.float64(p['dp2_os']) if 'dp2_os' in p else zero
  dp0_ss = np.float64(p['dp0_ss']) if 'dp0_ss' in p else zero
  dp1_ss = np.float64(p['dp1_ss']) if 'dp1_ss' in p else zero
  dp2_ss = np.float64(p['dp2_ss']) if 'dp2_ss' in p else zero

  # Other flags
  nevt = np.int32(pdf.shape[0])


  # Verbose mode
  # if verbose:
  #   print('You are running under the following configuration:')
  #   print(f"Number of Events: {nevt}")
  #   print(f"Block size in device: {BLOCK_SIZE}")
  #   print(f"Time range: [{tLL}, {tUL}] ps")
  #
  #   print(f"Gs : {Gs}, DGs : {DGs}, DM : {DM}")
  #   print(f"CSP: {CSP}")
  #   print(f"ASlon: {ASlon}")
  #   print(f"APlon: {APlon}")
  #   print(f"APpar: {APpar}")
  #   print(f"APper: {APper}")
  #
  #   print(f"pSlon: {pSlon}, pPlon: {pPlon}, pPpar: {pPpar}, pPper: {pPper}")
  #   print(f"dSlon: {pSlon}, dPlon: {pPlon}, dPpar: {pPpar}, dPper: {pPper}")
  #   print(f"lSlon: {pSlon}, lPlon: {pPlon}, lPpar: {pPpar}, lPper: {pPper}")
  #
  #   print(f"Time resolution parameters")
  #   print(f"sigma_offset: {sigma_offset} sigma_slope: {sigma_slope} sigma_curvature: {sigma_curvature} mu:{mu}")
  #   print(f"Tagging parameters")
  #   print(f"eta_os: {eta_os} eta_ss: {eta_ss}")
  #   print(f"p0_os: {p0_os} p1_os: {p1_os} p2_os: {p2_os}")
  #   print(f"dp0_os: {dp0_os} dp1_os: {dp1_os} dp2_os: {dp2_os}")
  #   print(f"p0_ss: {p0_ss} p1_ss: {p1_ss} p2_ss: {p2_ss}")
  #   print(f"dp0_ss: {dp0_ss} dp1_ss: {dp1_ss} dp2_ss: {dp2_ss}")
  #
  #   print(f"{'APlon':>15} : {APlon}")
  #   for k, v in p.items():
  #     print(f'{k:>15} : {v}')

  __KERNELS.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    Gs, DGs, DM, p['CSP'],
    ASlon, APlon, APpar, APper,
    pSlon, pPlon, pPpar, pPper,
    dSlon, dPlon, dPpar, dPper,
    lSlon, lPlon, lPpar, lPper,
    tLL, tUL,
    # Time resolution
    sigma_offset, sigma_slope, sigma_curvature, mu,
    # Flavor tagging
    eta_os, eta_ss,
    p0_os, p1_os, p2_os, p0_ss, p1_ss, p2_ss,
    dp0_os, dp1_os, dp2_os, dp0_ss, dp1_ss, dp2_ss,
    # Decay-time acceptance
    p['timeacc'],
    # Angular acceptance
    p['angacc'], np.int32(use_fk),
    # Some flags
    mass_bins, nevt,
    block = (BLOCK_SIZE,1,1),
    #block = (BLOCK_SIZE, mass_bins, 1),
    grid = (int(np.ceil(nevt/BLOCK_SIZE)), 1, 1)
  )



def diff_cross_rate_full2( data, pdf, use_fk=1, BLOCK_SIZE=2, **parameters):
  """
  Look at kernel definition to see help
  """
  #print('\n')
  p = cross_rate_parser2(parameters)
  print(len(p['CSP']))
  for k, v in p.items():
   print(f'{k:>10}:  {v}')
  print('\n\n')
  __KERNELS.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64(p['DGsd']+0.65789),
    np.float64(p['DGs']),
    np.float64(p['DM']),
    p['CSP'],
    ristra.allocate(np.sqrt(p['ASlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APpar'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APper'])).astype(np.float64),
    np.float64(p['pSlon']+p['pPlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']+p['pPlon']),
    np.float64(p['pPper']+p['pPlon']),
    ristra.allocate(p['dSlon']).astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']*p['lPlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']*p['lPlon']),
    np.float64(p['lPper']*p['lPlon']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']), np.float64(p['sigma_slope']), np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    np.int32(p['nknots']),
    ristra.allocate(p['knots']).astype(np.float64),
    p['coeffs'],
    # Angular acceptance
    p['w'],
    np.int32(use_fk), np.int32(len(p['CSP'])), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,1,1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))






def diff_cross_rate_full_ready(
    data, pdf,
    Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7,
    fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
    dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
    pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
    lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
    CSP = ristra.allocate( np.atleast_1d([1]) ).astype(np.float64),
    Gs = None,
    knots = ristra.allocate(np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00]) ).astype(np.float64), # optimize this!
    timeacc = ristra.allocate(np.array([1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9]) ).astype(np.float64),
    angacc = ristra.allocate(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])).astype(np.float64),
    # Time resolution parameters
    sigma_offset = 0.01297,
    sigma_slope = 0.8446,
    sigma_curvature = 0,
    mu = 0,
    # Flavor tagging parameters
    eta_os = 0.3602,
    eta_ss = 0.4167,
    p0_os = 0.389,
    p0_ss = 0.4325,
    p1_os = 0.8486,
    p2_os = 0.,
    p1_ss = 0.9241,
    p2_ss = 0.,
    dp0_os = 0.009,
    dp0_ss = 0,
    dp1_os = 0.0143,
    dp2_os = 0.,
    dp1_ss = 0,
    dp2_ss = 0,
    tLL = 0.3,
    tUL = 15.0,
    use_fk=1, BLOCK_SIZE=32, **shit_args):
  """
  Look at kernel definition to see help
  """

  dSlon = np.atleast_1d(dSlon)
  ASlon = np.atleast_1d(fSlon)
  FP = abs(1-ASlon)

  if not Gs:
    Gs = Gd + DGsd + DGd

  __KERNELS.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64( Gs ),
    np.float64( DGs ),
    np.float64( DM ),
    CSP.astype(np.float64),
    ristra.allocate(  np.sqrt(ASlon)  ).astype(np.float64),
    ristra.allocate(  np.sqrt(FP* fPlon )  ).astype(np.float64),
    ristra.allocate(  np.sqrt(FP* fPper )  ).astype(np.float64),
    ristra.allocate(  np.sqrt(FP*abs(1- fPlon - fPper ))  ).astype(np.float64),
    np.float64( pSlon + pPlon ),
    np.float64( pPlon ),
    np.float64( pPpar + pPlon ),
    np.float64( pPper + pPlon ),
    ristra.allocate( dSlon + dPper ).astype(np.float64),
    np.float64( dPlon ),
    np.float64( dPpar ),
    np.float64( dPper ),
    np.float64( lSlon * lPlon ),
    np.float64( lPlon ),
    np.float64( lPpar * lPlon ),
    np.float64( lPper * lPlon ),
    np.float64( tLL ), np.float64( tUL ),
    # Time resolution
    np.float64( sigma_offset ), np.float64( sigma_slope ), np.float64( sigma_curvature ),
    np.float64( mu ),
    # Flavor tagging
    np.float64( eta_os ), np.float64( eta_ss ),
    np.float64( p0_os ), np.float64( p1_os ), np.float64( p2_os ),
    np.float64( p0_ss ), np.float64( p1_ss ), np.float64( p2_ss ),
    np.float64( dp0_os ), np.float64( dp1_os ), np.float64( dp2_os ),
    np.float64( dp0_ss ), np.float64( dp1_ss ), np.float64( dp2_ss ),
    # Decay-time acceptance
    np.int32( len(knots) ),
    knots.astype(np.float64),
    timeacc.astype(np.float64),
    # Angular acceptance
    angacc.astype(np.float64),
    np.int32(use_fk), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,len(CSP),1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))



def diff_cross_rate_mc( data, pdf,
      Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, Gs = None,
      DM = 17.7, CSP = 1,
      fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
      dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
      pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
      lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
      tLL = 0.3, tUL = 15.0,
      knots = np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00]),
      coeffs = [1, 1.2, 1.4, 1.7, 2.2, 2.2, 2.1, 2.0, 1.9],
      w = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
      # Time resolution parameters
      sigma_offset = 0.,
      sigma_slope = 0.,
      sigma_curvature = 0,
      mu = 0,
      # Flavor tagging parameters
      eta_os = 0., eta_ss = 0.,
      p0_os = 0., p0_ss = 0., p1_os = 0., p2_os = 0., p1_ss = 0., p2_ss = 0.,
      dp0_os = 0., dp0_ss = 0, dp1_os = 0., dp2_os = 0., dp1_ss = 0, dp2_ss = 0,
      use_fk=1,
      BLOCK_SIZE=32):

  """
  Look at kernel definition to see help
  """
  if Gs==None: Gs = Gd+DGsd+DGd
  ASlon = np.atleast_1d(fSlon)
  CSP = np.atleast_1d(CSP)
  FP = abs(1-ASlon)
  APlon = FP*fPlon; APper = FP*fPper; APpar = FP*abs(1-fPlon-fPper) # Amplitudes
  dSlon = np.atleast_1d(dSlon) + dPper                           # Strong phases

  __KERNELS.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64(Gs),
    np.float64(DGs),
    np.float64(DM),
    ristra.allocate(np.array(CSP)).astype(np.float64),
    ristra.allocate(np.sqrt(ASlon)).astype(np.float64),
    ristra.allocate(np.sqrt(APlon)).astype(np.float64),
    ristra.allocate(np.sqrt(APpar)).astype(np.float64),
    ristra.allocate(np.sqrt(APper)).astype(np.float64),
    np.float64(pSlon+pPlon),
    np.float64(pPlon),
    np.float64(pPpar+pPlon),
    np.float64(pPper+pPlon),
    ristra.allocate(np.array(dSlon)).astype(np.float64),
    np.float64(dPlon),
    np.float64(dPpar),
    np.float64(dPper),
    np.float64(lSlon*lPlon),
    np.float64(lPlon),
    np.float64(lPpar*lPlon),
    np.float64(lPper*lPlon),
    np.float64(tLL), np.float64(tUL),
    # Time resolution
    np.float64(sigma_offset), np.float64(sigma_slope), np.float64(sigma_curvature),
    np.float64(mu),
    # Flavor tagging
    np.float64(eta_os), np.float64(eta_ss),
    np.float64(p0_os), np.float64(p1_os), np.float64(p2_os),
    np.float64(p0_ss), np.float64(p1_ss), np.float64(p2_ss),
    np.float64(dp0_os), np.float64(dp1_os), np.float64(dp2_os),
    np.float64(dp0_ss), np.float64(dp1_ss), np.float64(dp2_ss),
    # Decay-time acceptance
    np.int32(len(knots)),
    ristra.allocate(knots).astype(np.float64),
    ristra.allocate(get_4cs(coeffs)).astype(np.float64),
    # Angular acceptance
    ristra.allocate(w).astype(np.float64),
    # Flags and cuda management
    np.int32(use_fk), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,len(CSP),1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))


"""
data_ = ipanema.ristra.allocate(np.array([[-2.19668930e-01,  4.49032911e-01,  2.20482633e+00,  6.41448580e-01, 1.02986146e+03,  4.12645720e-02,  5.31000000e+02]]))
pdf_ = ipanema.ristra.allocate(np.array([0]))
pars_ = dict([('lSlon', 1),
             ('lPlon', 1),
             ('lPpar', 1),
             ('lPper', 1),
             ('pSlon', 0),
             ('pPlon', -0.03),
             ('pPpar', 0),
             ('pPper', 0),
             ('DGd', 0),
             ('DGsd', 0.0034801),
             ('DGs', 0.08543),
             ('Gd', 0.65789),
             ('DM', 17.8),
             ('fPlon', 0.52428),
             ('fPper', 0.25005),
             ('dPpar', 3.26),
             ('dPper', 3.08),
             ('fSlon', 0),
             ('CSP', 1),
             ('dSlon', 0)])



data_.T.shape[0]
diff_cross_rate_full(data_, pdf_, use_fk=1, BLOCK_SIZE=32, **pars_)
pdf_





config['debug'] = 4
config['debug_evt'] = 1
config['use_time_acc'] = 0
config['use_time_offset'] = 0
config['use_time_res'] = 0
# to get Simon
config['use_perftag'] = 1
config['use_truetag'] = 0
# to get Peilian
#config['use_perftag'] = 0
#config['use_truetag'] = 1
get_kernels()

"""

"""

G                  : +0.661370
DG                 : +0.085430
DM                 : +17.800000
CSP                : +1.000000
ASlon              : +0.000000
APlon              : +0.724072
APpar              : +0.475047
APper              : +0.500050
pSlon              : -0.030000
pPlon              : -0.030000
pPpar              : -0.030000
pPper              : -0.030000
dSlon              : +3.080000
dPlon              : +0.000000
dPper              : +3.080000
dPpar              : +3.260000
lSlon              : +1.000000
lPlon              : +1.000000
lPper              : +1.000000
lPpar              : +1.000000
tLL                : +0.300000
tUL                : +15.000000
COEFFS             : -0.141887	+5.891100	-8.353963	+4.681980
                     +0.671944	+1.681632	-1.096259	+0.510886
                     +1.818299	-2.097563	+3.056702	-1.010345
                     -2.016313	+6.423797	-3.255416	+0.548202
                     +2.037987	+0.218237	-0.089314	+0.009750
                     +2.309906	-0.052779	+0.000725	-0.000221
                     +2.426316	-0.075188	+0.000000	+0.000000
INTEGRAL           : ta=1.24613252	tb=0.09632043	tc=0.03824251	td=0.02563429
INPUT              : cosK=-0.219669	cosL=+0.449033	hphi=+2.204826	time=+0.641449	q=+1.000000
RESULT             : pdf=0.01304335	ipdf=4.79047689	pdf/ipdf=0.002722766634107
                   : pdfB=+0.003261	pdBbar=+0.027139	ipdfB=+1.197619	ipdfBbar=+1.198388
                   : dta=+1.000000
TIME ACC           : ta=0.65451577	tb=0.01792890	tc=0.26810538	td=-0.59681568
ANGULAR PART   (0) : +0.524280	+1.000000	-0.999550	+0.000000	-0.029996	+0.006898
ANGULAR PART   (1) : +0.225670	+1.000000	-0.999550	+0.000000	-0.029996	+0.061333
ANGULAR PART   (2) : +0.250050	+1.000000	+0.999550	+0.000000	+0.029996	+0.041052
ANGULAR PART   (3) : +0.237547	+0.000000	+0.029511	-0.179030	-0.983401	-0.064931
ANGULAR PART   (4) : +0.343968	-0.992998	+0.992551	+0.000000	+0.029785	+0.012898
ANGULAR PART   (5) : +0.362072	-0.000000	-0.029939	+0.061554	+0.997655	+0.017540
ANGULAR PART   (6) : +0.000000	+1.000000	+0.999550	+0.000000	+0.029996	+0.047649
ANGULAR PART   (7) : +0.000000	+0.000000	+0.005370	+0.983844	-0.178949	-0.033898
ANGULAR PART   (8) : +0.000000	-0.000000	-0.000000	-0.000000	-0.000000	-0.046101
ANGULAR PART   (9) : +0.000000	+0.000000	-0.001846	-0.998104	+0.061526	-0.036259

"""




#get_angular_weights(data_, data_, ipanema.ristra.ones(data_.shape[0]))


"""
__global__
void pyFcoeffs(double *data, double *fk,  int Nevt)
{{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int k = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= Nevt) {{ return; }}
  fk[i*10+k]= 9./(16.*M_PI)*getF(data[i*10+0],data[i*10+1],data[i*10+2],k+1);
}}
"""
def get_angular_fk(data, BLOCK_SIZE=32):
  fk = ristra.allocate( np.zeros((data.shape[0],10)) ).astype(np.float64)
  __KERNELS.Fcoeffs( data, fk,
    np.int32(data.shape[0]),
    block = (BLOCK_SIZE,1,1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),10,1)
  )
  return fk.get()






def get_angular_weights(true, reco, weight, BLOCK_SIZE=32, **parameters):
  """
  Look at kernel definition to see help
  """
  p = cross_rate_parser(parameters)

  # print(BLOCK_SIZE)
  # print('true shape', true.shape)
  # print('reco shape', reco.shape)
  # print('grid shape', (int(np.ceil(true.shape[0]/BLOCK_SIZE)),1,1) )
  ang_acc = ipanema.ristra.zeros(10)

  __KERNELS.AngularWeights(
    true, reco, weight, ang_acc,
    # Differential cross-rate parameters
    np.float64(p['Gs']),
    np.float64(p['DGs']),
    np.float64(p['DM']),
    ristra.allocate(p['CSP']).astype(np.float64),
    ristra.allocate(np.sqrt(p['ASlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APlon'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APpar'])).astype(np.float64),
    ristra.allocate(np.sqrt(p['APper'])).astype(np.float64),
    np.float64(p['pSlon']+p['pPlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']+p['pPlon']),
    np.float64(p['pPper']+p['pPlon']),
    ristra.allocate(p['dSlon']).astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']*p['lPlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']*p['lPlon']),
    np.float64(p['lPper']*p['lPlon']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']), np.float64(p['sigma_slope']), np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    np.int32(p['nknots']),
    ristra.allocate(p['knots']).astype(np.float64),
    ristra.allocate(get_4cs(p['coeffs'])).astype(np.float64),
    # Angular acceptance
    ristra.allocate(p['w']).astype(np.float64),
    np.int32(true.shape[0]),
    block = (BLOCK_SIZE,1,1),
    grid = (int(np.ceil(true.shape[0]/BLOCK_SIZE)),1,1)
  )
  result = ang_acc.get()
  # print(weight.get().sum())
  # for i in range(0,len(result)):
  #     print(f'result[{i}] = {result[i]:+.16f}   {result[i]/result[0]:+.16f}')
  return result#/result[0]






def get_angular_cov(true, reco, weight, BLOCK_SIZE=32, **parameters):
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
  p = cross_rate_parser(parameters)
  ang_acc = ipanema.ristra.zeros(10)
  ang_acc_ = get_angular_weights(true, reco, weight, BLOCK_SIZE, **parameters)
  ang_acc = ristra.allocate(ang_acc_).astype(np.float64)
  cov_mat = ristra.allocate(np.zeros([10,10])).astype(np.float64)
  scale = np.sum(ipanema.ristra.get(weight))

  __KERNELS.AngularCov(
        true, reco, weight, ang_acc, cov_mat, np.float64(scale),
        # Differential cross-rate parameters
        np.float64(p['Gs']),
        np.float64(p['DGs']),
        np.float64(p['DM']),
        ristra.allocate(p['CSP']).astype(np.float64),
        ristra.allocate(np.sqrt(p['ASlon'])).astype(np.float64),
        ristra.allocate(np.sqrt(p['APlon'])).astype(np.float64),
        ristra.allocate(np.sqrt(p['APpar'])).astype(np.float64),
        ristra.allocate(np.sqrt(p['APper'])).astype(np.float64),
        np.float64(p['pSlon']+p['pPlon']),
        np.float64(p['pPlon']),
        np.float64(p['pPpar']+p['pPlon']),
        np.float64(p['pPper']+p['pPlon']),
        ristra.allocate(p['dSlon']).astype(np.float64),
        np.float64(p['dPlon']),
        np.float64(p['dPpar']),
        np.float64(p['dPper']),
        np.float64(p['lSlon']*p['lPlon']),
        np.float64(p['lPlon']),
        np.float64(p['lPpar']*p['lPlon']),
        np.float64(p['lPper']*p['lPlon']),
        np.float64(p['tLL']), np.float64(p['tUL']),
        # Time resolution
        np.float64(p['sigma_offset']), np.float64(p['sigma_slope']), np.float64(p['sigma_curvature']),
        np.float64(p['mu']),
        # Flavor tagging
        np.float64(p['eta_os']), np.float64(p['eta_ss']),
        np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
        np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
        np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
        np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
        # Decay-time acceptance
        np.int32(p['nknots']),
        ristra.allocate(p['knots']).astype(np.float64),
        ristra.allocate(get_4cs(p['coeffs'])).astype(np.float64),
        # Angular acceptance
        ristra.allocate(p['w']).astype(np.float64),
        np.int32(true.shape[0]),
        block = (BLOCK_SIZE,1,1),
        grid = (int(np.ceil(true.shape[0]/BLOCK_SIZE)),1,1)
      )
  w = ang_acc.get(); cov = cov_mat.get()

  warnings.filterwarnings('ignore')
  cov = cov + cov.T - np.eye(cov.shape[0])*cov # fill the lower-triangle
  final_cov = np.zeros_like(cov)
  corr = np.zeros_like(cov)
  # Cov relative to f0
  for i in range(0,cov.shape[0]):
    for j in range(0,cov.shape[1]):
      final_cov[i,j] = 1.0/(w[0]*w[0])*(
                                w[i]*w[j]/(w[0]*w[0])*cov[0][0]+cov[i][j]-
                                w[i]/w[0]*cov[0][j]-w[j]/w[0]*cov[0][i]);
  final_cov[np.isnan(final_cov)] = 0
  final_cov = np.where(np.abs(final_cov)<1e-12,0,final_cov)
  # Correlation matrix
  for i in range(0,cov.shape[0]):
    for j in range(0,cov.shape[1]):
      corr[i,j] = final_cov[i][j]/np.sqrt(final_cov[i][i]*final_cov[j][j])
  return w/w[0], np.sqrt(np.diagonal(final_cov)), final_cov, corr #result#/result[0]













def splinexerf(
      time, lkhd,
      b0=1, b1=1.3, b2=1.5, b3=1.8, b4=2.1, b5=2.3, b6=2.2, b7=2.1, b8=2.0,
      mu=0.0, sigma=0.04, G=0.6,
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
  # print(b)
  # print(mu,sigma,G)
  # print('\n')
  __KERNELS.SingleTimeAcc(
              time, lkhd, # input, output
              ristra.allocate(get_4cs(b)).astype(np.float64),
              np.float64(mu), np.float64(sigma), np.float64(G),
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
              ristra.allocate(get_4cs(a)).astype(np.float64),
              ristra.allocate(get_4cs(b)).astype(np.float64),
              np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
              np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
              np.float64(tLL),np.float64(tUL),
              size_a, size_b,
              block = (BLOCK_SIZE,1,1),
              grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
            )



def saxsbxscxerf(
      time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
      a0=1, a1=1.3, a2=1.5, a3=1.8, a4=2.1, a5=2.3, a6=2.2, a7=2.1, a8=2.0,
      mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
      b0=1, b1=1.2, b2=1.4, b3=1.7, b4=2.2, b5=2.2, b6=2.1, b7=2.0, b8=1.9,
      mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
      c0=1, c1=1.2, c2=1.4, c3=1.7, c4=2.2, c5=2.2, c6=2.1, c7=2.0, c8=1.9,
      mu_c=0.0, sigma_c=0.04, gamma_c=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=256):
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
  # print(a)
  # print(f"a0={a0}  a1={a1}  a2={a2}  a3={a3}  a4={a4}  a5={a5}  a6={a6}  a7={a7}  a8={a8}")
  # print(f"b0={b0}  b1={b1}  b2={b2}  b3={b3}  b4={b4}  b5={b5}  b6={b6}  b7={b7}  b8={b8}")
  # print(f"c0={c0}  c1={c1}  c2={c2}  c3={c3}  c4={c4}  c5={c5}  c6={c6}  c7={c7}  c8={c8}")
  # print(f"mu_a={mu_a}, sigma_a={sigma_a}, gamma_a={gamma_a}")
  # print(f"mu_b={mu_b}, sigma_b={sigma_b}, gamma_b={gamma_a}")
  # print(f"mu_c={mu_c}, sigma_c={sigma_c}, gamma_c={gamma_a}")

  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_c  = np.int32(lkhd_c.shape[0])
  size_max = max(size_a,size_b,size_c)
  __KERNELS.FullTimeAcc(
              time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
              ristra.allocate(get_4cs(a)).astype(np.float64),
              ristra.allocate(get_4cs(b)).astype(np.float64),
              ristra.allocate(get_4cs(c)).astype(np.float64),
              np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
              np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
              np.float64(mu_c), np.float64(sigma_c), np.float64(gamma_c),
              np.float64(tLL),np.float64(tUL),
              size_a, size_b, size_c,
              block = (BLOCK_SIZE,1,1),
              grid = (int(np.ceil(size_max/BLOCK_SIZE)),1,1)
            )



def acceptance_spline(
      time, #spline=None,
      b0=1, b1=1.3, b2=1.5, b3=1.8, b4=2.1, b5=2.3, b6=2.2, b7=2.1, b8=2.0,
      BLOCK_SIZE=32):
  coeffs = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
  time_d = ristra.allocate(time).astype(np.float64)
  spline_d = ristra.allocate(0*time).astype(np.float64)
  coeffs_d = ristra.allocate(get_4cs(coeffs)).astype(np.float64)
  n_evt    = len(time)
  __KERNELS.Spline(time_d, spline_d, coeffs_d, np.int32(n_evt),
                   block = (BLOCK_SIZE,1,1),
                   grid = (int(np.ceil(n_evt/BLOCK_SIZE)),1,1)
                  )
  return spline_d.get()



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
    C.append(-((b*u(-2+i)*pow(u(1+i),2))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))))+
        (a*pow(u(1+i),3))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))+
        (c*pow(u(-1+i),2)*u(1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        (b*u(-1+i)*u(1+i)*u(2+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        (c*u(-1+i)*u(i)*u(2+i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (b*u(i)*pow(u(2+i),2))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (d*pow(u(i),3))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))+
        (c*pow(u(i),2)*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
    C.append((2*b*u(-2+i)*u(1+i))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))-
        (3*a*pow(u(1+i),2))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))+
        (b*pow(u(1+i),2))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))-
        (c*pow(u(-1+i),2))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        (b*u(-1+i)*u(1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        (2*c*u(-1+i)*u(1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        (b*u(-1+i)*u(2+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        (b*u(1+i)*u(2+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        (c*u(-1+i)*u(i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (c*u(-1+i)*u(2+i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))+
        (2*b*u(i)*u(2+i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (c*u(i)*u(2+i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))+
        (b*pow(u(2+i),2))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (c*pow(u(i),2))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))+
        (3*d*pow(u(i),2))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))-
        (2*c*u(i)*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
    C.append(-((b*u(-2+i))/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*
        (-u(i)+u(1+i))))+(3*a*u(1+i))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))-
        (2*b*u(1+i))/
        ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))-
        (b*u(-1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        (2*c*u(-1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        (b*u(1+i))/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*
        (-u(-1+i)+u(2+i)))+(c*u(1+i))/
        ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        (b*u(2+i))/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*
        (-u(-1+i)+u(2+i)))+(c*u(-1+i))/
        ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (b*u(i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))+
        (c*u(i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        (2*b*u(2+i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))+
        (c*u(2+i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))+
        (2*c*u(i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))-
        (3*d*u(i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))+
        (c*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
    C.append(-(a/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))))+
        b/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))+
        b/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))-
        c/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i)))+
        b/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        c/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i)))-
        c/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i)))+
        d/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
    result.append(C)
  m = C[1] + 2*C[2]*u(n) + 3*C[3]*u(n)**2
  C = [C[0] + C[1]*u(n) + C[2]*u(n)**2 + C[3]*u(n)**3 - m*u(n),m,0,0]
  result.append(C)
  return np.array(result)
