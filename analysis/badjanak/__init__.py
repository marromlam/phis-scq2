# -*- coding: utf-8 -*-

import os
import builtins
import numpy as np
import ipanema
from ipanema import ristra
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
fast_integral   = 1, # run integrals with approximation
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
            "FKHELPERS_CU":open(PATH+'/FkHelpers.cu').read(),
            "FUNCTIONS_CU":open(PATH+'/Functions.cu').read(),
            "TIMEANGULARDISTRIBUTION_CU":open(PATH+'/TimeAngularDistribution.cu').read(),
            "DECAYTIMEACCEPTANCE_CU":open(PATH+'/DecayTimeAcceptance.cu').read(),
            "DIFFERENTIALCROSSRATE_CU":open(PATH+'/DifferentialCrossRate.cu').read(),
            "TOY_CU":open(PATH+'/Toy.cu').read(),
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
           'pyFcoeffs',
           'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc', 'pySpline',
           'pyfaddeeva', 'pycerfc', 'pycexp', 'pyipacerfc',
           'dG5toy', 'integral_ijk_fx']
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

################################################################################
# delta_gamma5 #################################################################

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
  timeacc = [ p[k] for k in p.keys() if re.compile('c([0-9])([0-9])?(u|b)').match(k)]
  if timeacc:
    r['timeacc'] = THREAD.to_device(get_4cs(timeacc))
  else:
    r['timeacc'] = THREAD.to_device(np.float64([1]))

  # Angular acceptance
  angacc = [ p[k] for k in p.keys() if re.compile('w([0-9])([0-9])?(u|b)').match(k)]
  if angacc:
    r['angacc'] = THREAD.to_device(np.float64(angacc))
  else:
    r['angacc'] = THREAD.to_device(np.float64(config['tristan']))

  return r

################################################################################



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
  p = cross_rate_parser_new(**pars)
  delta_gamma5( input, output,
                         use_fk=1, use_angacc = 1, use_timeacc = 1,
                         use_timeoffset = 0, set_tagging = 1, use_timeres = 1,
                         BLOCK_SIZE=256, **p)



def delta_gamma5_mc(input, output, use_fk=1, **pars):
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
  p = cross_rate_parser_new(**pars)
  delta_gamma5( input, output,
                         use_fk=use_fk, use_angacc = 0, use_timeacc = 0,
                         use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
                         BLOCK_SIZE=256, **p)

################################################################################







################################################################################




# angular acceptance functions ------------------------------------------------
#    These are several rules related to decay-time acceptance. The main one is
#    time_acceptance, which computes the spline coefficients of the
#    Bs2JpsiPhi acceptance.

def get_angular_acceptance_weights(true, reco, weight, BLOCK_SIZE=256, **parameters):
  # Filter some warnings
  warnings.filterwarnings('ignore')

  # Compute weights scale and number of weights to compute
  scale = np.sum(ipanema.ristra.get(weight))
  terms = len(config['tristan'])

  # Define the computation core function
  def get_weights(true, reco, weight):
    pdf = THREAD.to_device(np.zeros(true.shape[0]))
    delta_gamma5_mc(true, pdf, use_fk=0, **parameters); num = pdf.get()
    pdf = THREAD.to_device(np.zeros(true.shape[0]))
    delta_gamma5_mc(true, pdf, use_fk=1, **parameters); den = pdf.get()
    fk = get_fk(reco.get()[:,0:3])
    ang_acc = fk*(weight.get()*num/den).T[::,np.newaxis]
    return ang_acc

  # Get angular weights with weight applied and sum all events
  w = get_weights(true, reco, weight).sum(axis=0)
  # Get angular weights without weight
  ones = THREAD.to_device(np.ascontiguousarray(np.ones(weight.shape[0]), dtype=np.float64))
  w10 = get_weights(true, reco, ones)

  # Get covariance matrix
  cov = np.zeros((terms,terms,weight.shape[0]))
  for i in range(0,terms):
    for j in range(i,terms):
      cov[i,j,:] = (w10[:,i]-w[i]/scale)*(w10[:,j]-w[j]/scale)*weight.get()**2
  cov = cov.sum(axis=2) # sum all per event cov

  # Handling cov matrix, and transform it to correlation matrix
  cov = cov + cov.T - np.eye(cov.shape[0])*cov         # fill the lower-triangle
  final_cov = np.zeros_like(cov); corr = np.zeros_like(cov)
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
  return w/w[0], np.sqrt(np.diagonal(final_cov)), final_cov, corr









# Time acceptance functions ---------------------------------------------------

def splinexerf(
      time, lkhd,
      coeffs,
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
  g_size, l_size = get_sizes(lkhd.shape[0],BLOCK_SIZE)
  #print(coeffs,mu,sigma,gamma)

  __KERNELS__.SingleTimeAcc(
      time, lkhd, # input, output
      THREAD.to_device(get_4cs(coeffs)).astype(np.float64),
      np.float64(mu), np.float64(sigma), np.float64(gamma),
      np.float64(tLL),np.float64(tUL),
      np.int32(lkhd.shape[0]),
      global_size=g_size, local_size=l_size
  )



def sbxscxerf(
      time_a, time_b, lkhd_a, lkhd_b,
      coeffs_a, coeffs_b,
      mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
      mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=256, **crap):
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
  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_max = max(size_a,size_b)
  __KERNELS__.RatioTimeAcc(
    time_a, time_b, lkhd_a, lkhd_b,
    THREAD.to_device(get_4cs(coeffs_a)).astype(np.float64),
    THREAD.to_device(get_4cs(coeffs_b)).astype(np.float64),
    np.float64(mu_a), np.float64(sigma_a), np.float64(gamma_a),
    np.float64(mu_b), np.float64(sigma_b), np.float64(gamma_b),
    np.float64(tLL),np.float64(tUL),
    size_a, size_b,
    global_size=(len(output),)
  )


# --- #
def saxsbxscxerf(
      time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
      coeffs_a, coeffs_b, coeffs_c,
      mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
      mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
      mu_c=0.0, sigma_c=0.04, gamma_c=0.6,
      tLL = 0.3, tUL = 15,
      BLOCK_SIZE=256, **crap):
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
  size_a  = np.int32(lkhd_a.shape[0]);
  size_b  = np.int32(lkhd_b.shape[0])
  size_c  = np.int32(lkhd_c.shape[0])
  size_max = max(size_a,size_b,size_c)
  g_size, l_size = get_sizes(size_max,BLOCK_SIZE)
  __KERNELS__.FullTimeAcc(
    time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
    THREAD.to_device(get_4cs(coeffs_a)).astype(np.float64),
    THREAD.to_device(get_4cs(coeffs_b)).astype(np.float64),
    THREAD.to_device(get_4cs(coeffs_c)).astype(np.float64),
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

def get_fk(z):
  z_dev = THREAD.to_device(np.ascontiguousarray(z, dtype=np.float64))
  w = np.zeros((z.shape[0],len(config['tristan'])))
  w_dev = THREAD.to_device(np.ascontiguousarray(w, dtype=np.float64))
  # print(z_dev)
  # print(w_dev)
  # print("get_fk z_dev.shape:",z.shape)
  # print("get_fk w_dev.shape:",w.shape)


  __KERNELS__.Fcoeffs(z_dev, w_dev, np.int32(len(z)), global_size=(len(z)))
  #print("-->",w_dev)
  return w_dev.get()
