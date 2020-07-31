# -*- coding: utf-8 -*-

import os
import builtins
import numpy as np
import ipanema
import warnings
from reikna.cluda import functions, dtypes
import platform
import cpuinfo
import re
import math
def get_sizes(size,BLOCK_SIZE=256):
    '''
    i need to check if this worls for 3d size and 3d block
    '''
    a = size % BLOCK_SIZE
    if a == 0:
      gs, ls = size, BLOCK_SIZE
    elif size < BLOCK_SIZE:
      gs, ls = size, 1
    else:
      a = np.ceil(size/BLOCK_SIZE)
      gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
    return int(gs), int(ls)



if __name__ == '__main__':
  PATH = '/home3/marcos.romero/phis-scq/badjanak'
  import ipanema
  ipanema.initialize('opencl',1,verbose=False)
else:
  PATH = os.path.dirname(os.path.abspath(__file__))

# Get builtins
BACKEND = builtins.BACKEND
DEVICE = builtins.DEVICE
CONTEXT = builtins.CONTEXT
QUEUE = builtins.THREAD



# Default compile flags --------------------------------------------------------
#     The compile_flags is a dict where each key substitutes a same-named
#     string in the kernel file by its value: #define KEY {KEY} <-- value
global config
config = dict(
debug =           0, # no prints
debug_evt =       0, # number of events to debug
fast_integral  = 1, #
sigma_t =         0.15,
knots =           [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00],
x_m =             [990, 1008, 1016, 1020, 1024, 1032, 1050],
tristan =         [1,1,1,0,0,0,1,0,0,0],
precision =       'double',
)



# Prepare flag to device code --------------------------------------------------
#    Bla bla bla
def flagger(verbose=False):
  if verbose:
    print(f"\n{80*'='}\n Badjanak kernel\n{80*'='}\n")
    print(f"{'Backend was set to':>20} : {BACKEND}")
    print(f"{' Using as host':>20} : {cpuinfo.get_cpu_info()['brand_raw']} with {platform.python_compiler()}")
    print(f"{' Using as device':>20} : {DEVICE.name} with CUDA 10.2")
    print(f" ")
  dict_flags = {}
  for key, value in config.items():
    if key == 'x_m':
      dict_flags['nmassbins'.upper()] = len(value)-1
      dict_flags['nmassknots'.upper()] = len(value)
    if key == 'knots':
      dict_flags['nknots'.upper()] = len(value)
      dict_flags['spl_bins'.upper()] = len(value)
      dict_flags['ntimebins'.upper()] = len(value)+1
    if key == 'tristan':
      dict_flags['nterms'.upper()] = len(value)
    if key != 'precision':
      dict_flags[key.upper()] = str(value).replace('[','{').replace(']','}')
  if verbose:
    for key, value in dict_flags.items():
      print(f'{key.upper():>20} : {value}')
  return dict_flags



# Compiler ---------------------------------------------------------------------
#     Compile kernel against given BACKEND

def compile(verbose=False, pedantic=False):
  kpath = os.path.join(PATH,'Kernel.cu')
  kstrg = open(kpath,"r").read()
  kstrg = kstrg.format(**{
            **flagger(verbose),
            "FUNCTIONS_CU":open(PATH+'/Functions.cu').read(),
            "TIMEANGULARDISTRIBUTION_CU":open(PATH+'/TimeAngularDistribution.cu').read(),
            "DECAYTIMEACCEPTANCE_CU":open(PATH+'/DecayTimeAcceptance.cu').read(),
            "DIFFERENTIALCROSSRATE_CU":open(PATH+'/DifferentialCrossRate.cu').read(),
            "ANGULARACCEPTANCE_CU":open(PATH+'/AngularAcceptance.cu').read(),
           })
  if config['precision'] == 'double':
    prog = THREAD.compile(kstrg,render_kwds={"ftype":dtypes.ctype(np.float64),
                 "ctype":dtypes.ctype(np.complex128)},keep=False)
  else:
    prog = THREAD.compile(kstrg,render_kwds={"ftype":dtypes.ctype(np.float32),
                 "ctype":dtypes.ctype(np.complex64)},keep=False)
  if pedantic:
    print(prog.source)
  if verbose:
    print('\nSuccesfully compiled.\n')
  return prog








# Get kernels ------------------------------------------------------------------
#    Hey ja
def get_kernels(verbose=False, pedantic=False):
  global __KERNELS__
  prog = compile(verbose, pedantic)
  items = ['pyDiffRate',
           'pyFcoeffs', 'pyAngularWeights', 'pyAngularCov',
           'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc', 'pySpline',
           'pyfaddeeva', 'pycerfc', 'pycexp', 'pyipacerfc']
  for item in items:
    setattr(prog, item[2:], prog.__getattr__(item))
    #print(item)
    #setattr(prog, item[2:], prog.get_function(item))
  __KERNELS__ = prog


# Get the kernels just after one import this module
get_kernels(verbose=False, pedantic=False)





# Functions
#     Here pythonic versions of KERNEL functions are defined. There are wrappers
#     that are simpler to interact with


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
                  eta_os, eta_ss,
                  p0_os,  p1_os, p2_os,
                  p0_ss,  p1_ss, p2_ss,
                  dp0_os, dp1_os, dp2_os,
                  dp0_ss, dp1_ss, dp2_ss,
                  # Time acceptance
                  timeacc,
                  # Angular acceptance
                  angacc,
                  # Flags
                  use_fk=1, use_angacc = 0, use_timeacc = 0,
                  use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
                  BLOCK_SIZE=256, **crap):
  """
  Look at kernel definition to see help
  The aim of this function is to be the fastest wrapper
  """
  g_size, l_size = get_sizes(output.shape[0],BLOCK_SIZE)
  __KERNELS__.DiffRate(
    # Input and output arrays
    input, output,
    # Differential cross-rate parameters
    np.float64(G), np.float64(DG), np.float64(DM),
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
    # Flags
    np.int32(use_fk), np.int32(len(CSP)), np.int32(use_angacc), np.int32(use_timeacc),
    np.int32(use_timeoffset), np.int32(set_tagging), np.int32(use_timeres),
    np.int32(len(output)),
    global_size=g_size, local_size=l_size)
    #grid=(int(np.ceil(output.shape[0]/BLOCK_SIZE)),1,1), block=(BLOCK_SIZE,1,1))



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
  delta_gamma5(
    vars, pdf,
    np.float64(Gs), np.float64(DGs), np.float64(DM),
    THREAD.to_device(CSP).astype(np.float64),
    THREAD.to_device(np.sqrt(ASlon)).astype(np.float64),
    THREAD.to_device(np.sqrt(APlon)).astype(np.float64),
    THREAD.to_device(np.sqrt(APpar)).astype(np.float64),
    THREAD.to_device(np.sqrt(APper)).astype(np.float64),
    np.float64(pSlon+pPlon),
    np.float64(pPlon), np.float64(pPpar+pPlon), np.float64(pPper+pPlon),
    THREAD.to_device(dSlon).astype(np.float64),
    np.float64(dPlon), np.float64(dPpar), np.float64(dPper),
    np.float64(lSlon*lPlon),
    np.float64(lPlon), np.float64(lPpar*lPlon), np.float64(lPper*lPlon),
    np.float64(0.3), np.float64(15),

    THREAD.to_device(get_4cs(coeffs)).astype(np.float64),
    np.int32(use_fk), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,len(CSP),1),
    grid = (int(np.ceil(vars.shape[0]/BLOCK_SIZE)),1,1))





def cross_rate_parser_new(
      Gd = 0.66137, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7, CSP = 1.0,
      # Time-dependent angular distribution
      fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
      dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
      pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
      lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
      # Time limits
      tLL = 0.0, tUL = 15.0,
      # Time resolution
      sigma_offset = 0.00, sigma_slope = 0.00, sigma_curvature = 0.00,
      mu = 0.00,
      # Flavor tagging
      eta_os = 0.00, eta_ss = 0.00,
      p0_os = 0.00,  p1_os = 0.00, p2_os = 0.00,
      p0_ss = 0.00,  p1_ss = 0.00, p2_ss = 0.00,
      dp0_os = 0.00, dp1_os = 0.00, dp2_os = 0.00,
      dp0_ss = 0.00, dp1_ss = 0.00, dp2_ss = 0.00,
      # Flags
      use_fk=1, use_angacc = 0, use_timeacc = 0,
      use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
      verbose = False,
      **p):

  r = {}
  r['mass_bins'] = len([ k for k in p.keys() if re.compile('CSP.*').match(k)])

  # Get all binned parameters and put them in ristras
  if r['mass_bins'] >= 1:
    CSP = [ p[k] for k in p.keys() if re.compile('CSP.*').match(k) ]
    r['CSP'] = THREAD.to_device(np.float64(CSP)).astype(np.float64)
    fSlon = [ p[k] for k in p.keys() if re.compile('fSlon.*').match(k) ]
    fSlon = THREAD.to_device(np.float64(fSlon)).astype(np.float64)
    dSlon = [ p[k] for k in p.keys() if re.compile('dSlon.*').match(k) ]
    dSlon = THREAD.to_device(np.float64(dSlon)).astype(np.float64)
  else:
    r['CSP'] = THREAD.to_device(np.float64([CSP])).astype(np.float64)
    fSlon = THREAD.to_device(np.float64([fSlon])).astype(np.float64)
    dSlon = THREAD.to_device(np.float64([dSlon])).astype(np.float64)

  # Parameters and parse Gs value
  r['DG'] = DGs
  r['DM'] = DM
  if 'Gs' in p:
    r['G'] = p['Gs']
  else:
    r['G'] = Gd + DGsd + DGd

  # Compute fractions of S and P wave objects
  FP = abs(1-fSlon)
  r['ASlon'] = ipanema.ristra.sqrt( fSlon )
  r['APlon'] = ipanema.ristra.sqrt( FP*fPlon )
  r['APper'] = ipanema.ristra.sqrt( FP*fPper )
  r['APpar'] = ipanema.ristra.sqrt( FP*abs(1-fPlon-fPper) )

  # Strong phases
  r['dPlon'] = dPlon
  r['dPper'] = dPper + r['dPlon']
  r['dPpar'] = dPpar + r['dPlon']
  r['dSlon'] = dSlon + r['dPper']

  # Weak phases
  r['pPlon'] = pPlon
  r['pSlon'] = pSlon + pPlon
  r['pPpar'] = pPpar + pPlon
  r['pPper'] = pPper + pPlon

  # Lambdas
  r['lPlon'] = lPlon
  r['lSlon'] = lSlon * lPlon
  r['lPpar'] = lPpar * lPlon
  r['lPper'] = lPper * lPlon

  # Time range
  r['tLL'] = tLL
  r['tUL'] = tUL

  # Time resolution
  r['sigma_offset'] = sigma_offset
  r['sigma_slope'] = sigma_slope
  r['sigma_curvature'] = sigma_curvature
  r['mu'] = mu

  # # Tagging
  r['eta_os'] = eta_os
  r['eta_ss'] = eta_ss
  r['p0_os'] = p0_os
  r['p1_os'] = p1_os
  r['p2_os'] = p2_os
  r['p0_ss'] = p0_ss
  r['p1_ss'] = p1_ss
  r['p2_ss'] = p2_ss
  r['dp0_os'] = dp0_os
  r['dp1_os'] = dp1_os
  r['dp2_os'] = dp2_os
  r['dp0_ss'] = dp0_ss
  r['dp1_ss'] = dp1_ss
  r['dp2_ss'] = dp2_ss

  # Time acceptance
  timeacc = [ p[k] for k in p.keys() if re.compile('c[0-9]+').match(k)]
  if timeacc:
    r['timeacc'] = THREAD.to_device(get_4cs(timeacc))
  else:
    r['timeacc'] = THREAD.to_device(np.float64([1]))

  # Angular acceptance
  angacc = [ p[k] for k in p.keys() if re.compile('w[0-9]+').match(k)]
  if angacc:
    r['angacc'] = THREAD.to_device(np.float64(angacc))
  else:
    r['angacc'] = THREAD.to_device(np.float64([1]))

  return r

################################################################################
# wrappers arround delta_gamma5 ################################################

def delta_gamma5_data(input, output, **pars):
  """
  delta_gamma5_data(input, output, **pars)
  This function is intended to be used with RD input arrays. It does use
  time acceptance and angular acceptance. The tagging and resolution will use
  calibration parameters.

    this functions is a wrap around badjanak.delta_gamma5

  In:
  0.123456789:
        input:  Input data with proper shape
                ipanema.ristra
       output:  Output array with proper shape to store pdf values
                ipanema.ristra
         pars:  Dictionary with parameters
                dict
  Out:
         void
  """
  p = badjanak.cross_rate_parser_new(**pars)
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 1, use_timeacc = 1,
                         use_timeoffset = 0, set_tagging = 1, use_timeres = 1,
                         BLOCK_SIZE=256, **p)



def delta_gamma5_mc(input, output, **pars):
  """
  delta_gamma5_mc(input, output, **pars)
  This function is intended to be used with MC input arrays. It doesn't use
  time acceptance nor angular acceptance. The tagging is set to perfect tagging
  and the time resolution is disabled.

    this functions is a wrap around badjanak.delta_gamma5

  In:
  0.123456789:
        input:  Input data with proper shape
                ipanema.ristra
       output:  Output array with proper shape to store pdf values
                ipanema.ristra
         pars:  Dictionary with parameters
                dict
  Out:
         void
  """
  p = badjanak.cross_rate_parser_new(**pars)
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 0, use_timeacc = 0,
                         use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
                         BLOCK_SIZE=256, **p)








################################################################################












def cross_rate_parser(parameters):
  print(parameters)

  pars = dict(
  Gd = +0.65789, DGsd = 0.08, DGs = 0.08, DGd=0, DM = 17.7, CSP   = 1,
  fSlon = 0.00, fPlon =  0.72,                 fPper = 0.50,
  dSlon = 3.07, dPlon =  0,      dPpar = 3.30, dPper = 3.07,
  pSlon = 0.00, pPlon = -0.03,   pPpar = 0.00, pPper = 0.00,
  lSlon = 1.00, lPlon =  1.00,   lPpar = 1.00, lPper = 1.00,
  fSlon1=0, #fSlon2=0, fSlon3=0,fSlon4=0,fSlon5=0, fSlon6=0, # binned mass
  dSlon1=0, #dSlon2=0, dSlon3=0,dSlon4=0,dSlon5=0, dSlon6=0, # binned mass
  CSP1=None, #CSP2=0, CSP3=0,CSP4=0,CSP5=0, CSP6=0,             # binned mass
  Gs = None,
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
    pars['Gs'] = pars['Gd']+pars['DGsd']+pars['DGd']
  #pars['nknots'] = len(pars['knots'])
  # for k, v in pars.items():
  #  print(f'{k:>10}:  {v}')
  return pars








def diff_cross_rate_full( data, pdf, use_fk=1, BLOCK_SIZE=32, **parameters):
  """
  Look at kernel definition to see help
  """
  #print('\n')
  p = cross_rate_parser(parameters)
  for k, v in p.items():
   print(f'{k:>10}:  {v}')
  print('\n\n')
  delta_gamma5(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64(p['Gs']),
    np.float64(p['DGs']),
    np.float64(p['DM']),
    THREAD.to_device(p['CSP']).astype(np.float64),
    THREAD.to_device(np.sqrt(p['ASlon'])).astype(np.float64),
    THREAD.to_device(np.sqrt(p['APlon'])).astype(np.float64),
    THREAD.to_device(np.sqrt(p['APpar'])).astype(np.float64),
    THREAD.to_device(np.sqrt(p['APper'])).astype(np.float64),
    np.float64(p['pSlon']+p['pPlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']+p['pPlon']),
    np.float64(p['pPper']+p['pPlon']),
    THREAD.to_device(p['dSlon']).astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']*p['lPlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']*p['lPlon']),
    np.float64(p['lPper']*p['lPlon']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']),
    np.float64(p['sigma_slope']),
    np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    #np.int32(p['nknots']),
    #THREAD.to_device(p['knots']).astype(np.float64),
    THREAD.to_device(get_4cs(p['coeffs'])).astype(np.float64),
    # Angular acceptance
    THREAD.to_device(p['w']).astype(np.float64)
    )












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

  __KERNELS__.DiffRate(
    # Input and output arrays
    data, pdf,
    # Differential cross-rate parameters
    np.float64(Gs),
    np.float64(DGs),
    np.float64(DM),
    THREAD.to_device(np.array(CSP)).astype(np.float64),
    THREAD.to_device(np.sqrt(ASlon)).astype(np.float64),
    THREAD.to_device(np.sqrt(APlon)).astype(np.float64),
    THREAD.to_device(np.sqrt(APpar)).astype(np.float64),
    THREAD.to_device(np.sqrt(APper)).astype(np.float64),
    np.float64(pSlon+pPlon),
    np.float64(pPlon),
    np.float64(pPpar+pPlon),
    np.float64(pPper+pPlon),
    THREAD.to_device(np.array(dSlon)).astype(np.float64),
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
    THREAD.to_device(knots).astype(np.float64),
    THREAD.to_device(get_4cs(coeffs)).astype(np.float64),
    # Angular acceptance
    THREAD.to_device(w).astype(np.float64),
    # Flags and cuda management
    np.int32(use_fk), np.int32(pdf.shape[0]),
    block = (BLOCK_SIZE,len(CSP),1),
    grid = (int(np.ceil(data.shape[0]/BLOCK_SIZE)),1,1))




































# angular acceptance functions ------------------------------------------------
#    These are several rules related to decay-time acceptance. The main one is
#    time_acceptance, which computes the spline coefficients of the
#    Bs2JpsiPhi acceptance.

def get_angular_cov(true, reco, weight, BLOCK_SIZE=32, **parameters):
  # Prepare kernel size
  g_size, l_size = get_sizes(true.shape[0],BLOCK_SIZE)
  # Parse parameters
  p = cross_rate_parser_new(**parameters)
  # Allocate some Variables
  terms = len(config['tristan'])
  ang_acc = ipanema.ristra.zeros(terms).astype(np.float64)
  cov_mat = THREAD.to_device(np.zeros([terms,terms])).astype(np.float64)
  scale = np.sum(ipanema.ristra.get(weight))
  # Filter some warnings
  warnings.filterwarnings('ignore')

  # Get angular weights values
  __KERNELS__.AngularWeights(
    true, reco, weight, ang_acc,
    # Differential cross-rate parameters
    np.float64(p['G']),
    np.float64(p['DG']),
    np.float64(p['DM']),
    p['CSP'].astype(np.float64),
    p['ASlon'].astype(np.float64),
    p['APlon'].astype(np.float64),
    p['APpar'].astype(np.float64),
    p['APper'].astype(np.float64),
    np.float64(p['pSlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']),
    np.float64(p['pPper']),
    p['dSlon'].astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']),
    np.float64(p['lPper']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']),
    np.float64(p['sigma_slope']),
    np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    THREAD.to_device(np.float64([1])).astype(np.float64),
    # Angular acceptance
    THREAD.to_device(np.float64([1])).astype(np.float64),
    np.int32(true.shape[0]),
    global_size=g_size, local_size=l_size
  )


  # Get angular weights covariance matrix
  __KERNELS__.AngularCov(
    true, reco, weight, ang_acc, cov_mat, np.float64(scale),
    # Differential cross-rate parameters
    np.float64(p['G']),
    np.float64(p['DG']),
    np.float64(p['DM']),
    p['CSP'].astype(np.float64),
    p['ASlon'].astype(np.float64),
    p['APlon'].astype(np.float64),
    p['APpar'].astype(np.float64),
    p['APper'].astype(np.float64),
    np.float64(p['pSlon']),
    np.float64(p['pPlon']),
    np.float64(p['pPpar']),
    np.float64(p['pPper']),
    p['dSlon'].astype(np.float64),
    np.float64(p['dPlon']),
    np.float64(p['dPpar']),
    np.float64(p['dPper']),
    np.float64(p['lSlon']),
    np.float64(p['lPlon']),
    np.float64(p['lPpar']),
    np.float64(p['lPper']),
    np.float64(p['tLL']), np.float64(p['tUL']),
    # Time resolution
    np.float64(p['sigma_offset']),
    np.float64(p['sigma_slope']),
    np.float64(p['sigma_curvature']),
    np.float64(p['mu']),
    # Flavor tagging
    np.float64(p['eta_os']), np.float64(p['eta_ss']),
    np.float64(p['p0_os']), np.float64(p['p1_os']), np.float64(p['p2_os']),
    np.float64(p['p0_ss']), np.float64(p['p1_ss']), np.float64(p['p2_ss']),
    np.float64(p['dp0_os']), np.float64(p['dp1_os']), np.float64(p['dp2_os']),
    np.float64(p['dp0_ss']), np.float64(p['dp1_ss']), np.float64(p['dp2_ss']),
    # Decay-time acceptance
    THREAD.to_device(np.float64([1])).astype(np.float64),
    # Angular acceptance
    THREAD.to_device(np.float64([1])).astype(np.float64),
    np.int32(true.shape[0]),
    global_size=g_size, local_size=l_size
      )

  # Arrays from device to host
  w = ang_acc.get(); cov = cov_mat.get()

  cov = cov + cov.T - np.eye(cov.shape[0])*cov         # fill the lower-triangle
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









# Time acceptance functions ---------------------------------------------------

def splinexerf(
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
  #print(b,mu,sigma,gamma)
  g_size, l_size = get_sizes(lkhd.shape[0],BLOCK_SIZE)
  __KERNELS__.SingleTimeAcc(
      time, lkhd, # input, output
      THREAD.to_device(get_4cs(b)).astype(np.float64),
      np.float64(mu), np.float64(sigma), np.float64(gamma),
      np.float64(tLL),np.float64(tUL),
      np.int32(lkhd.shape[0]),
      global_size=g_size, local_size=l_size
  )



def sbxscxerf(
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
  __KERNELS__.RatioTimeAcc(
    time_a, time_b, lkhd_a, lkhd_b,
    THREAD.to_device(get_4cs(a)).astype(np.float64),
    THREAD.to_device(get_4cs(b)).astype(np.float64),
    np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
    np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
    np.float64(tLL),np.float64(tUL),
    size_a, size_b,
    global_size=(len(output),)
  )


# --- #
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
  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_c  = np.int32(lkhd_c.shape[0])
  size_max = max(size_a,size_b,size_c)
  g_size, l_size = get_sizes(size_max,BLOCK_SIZE)
  __KERNELS__.FullTimeAcc(
    time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
    THREAD.to_device(get_4cs(a)).astype(np.float64),
    THREAD.to_device(get_4cs(b)).astype(np.float64),
    THREAD.to_device(get_4cs(c)).astype(np.float64),
    np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
    np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
    np.float64(mu_c), np.float64(sigma_c), np.float64(gamma_c),
    np.float64(tLL),np.float64(tUL),
    size_a, size_b, size_c,
    global_size=g_size, local_size=l_size
  )



def bspline(time, *coeffs, BLOCK_SIZE=32):
  time_d = THREAD.to_device(time).astype(np.float64)
  spline_d = THREAD.to_device(0*time).astype(np.float64)
  coeffs_d = THREAD.to_device(get_4cs(coeffs)).astype(np.float64)
  n_evt = len(time)
  __KERNELS__.Spline(
    time_d, spline_d, coeffs_d, np.int32(n_evt),
    global_size=(n_evt,)
  )
  return spline_d.get()


# def bspline(time, BLOCK_SIZE=32, *coeffs):
#   #coeffs = [b0, b1, b2, b3, b4, b5, b6, b7, b8]
#   print(coeffs)
#   time_d = THREAD.to_device(time).astype(np.float64)
#   spline_d = THREAD.to_device(0*time).astype(np.float64)
#   coeffs_d = THREAD.to_device(get_4cs(coeffs)).astype(np.float64)
#   n_evt = len(time)
#   __KERNELS__.Spline(
#     time_d, spline_d, coeffs_d, np.int32(n_evt),
#     global_size=(n_evt,)
#   )
#   return spline_d.get()



def get_knot(i, knots, n):
  if (i<=0):        i = 0
  elif (i>=n):      i = n
  return knots[i]



def get_4cs(listcoeffs):
  n = len(config['knots'])
  result = []                                           # list of bin coeffs C
  def u(j): return get_knot(j,config['knots'],len(config['knots'])-1)
  for i in range(0,len(config['knots'])-1):
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





# Other Functions -------------------------------------------------------------
#     Complex kernels running in device

def cexp(z):
  z_dev = THREAD.to_device(np.complex128(z))
  w_dev = THREAD.to_device(np.complex128(0*z))
  __KERNELS__.cexp(z_dev, w_dev, global_size=(len(z),))
  return w_dev.get()

def cerfc(z):
  z_dev = THREAD.to_device(np.complex128(z))
  w_dev = THREAD.to_device(np.complex128(0*z))
  __KERNELS__.cerfc(z_dev, w_dev, global_size=(len(z),))
  return w_dev.get()

def wofz(z):
  z_dev = THREAD.to_device(np.complex128(z))
  w_dev = THREAD.to_device(np.complex128(0*z))
  __KERNELS__.faddeeva(z_dev, w_dev, global_size=(len(z),))
  return w_dev.get()
