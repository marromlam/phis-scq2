# Implements bindings between ocl/cuda and python by means of ipanema. Most
# important functions/p.d.f.s are binded here to their C counterparts.


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = [
    'delta_gamma5_data', 'get_kernels', 'compile', 'flagger', 'delta_gamma5',
    'parser_rateBs', 'parser_rateBd'
]

# Modules {{{

import os
import builtins
import numpy as np
import ipanema
from ipanema import ristra
# from ipanema.core.utils import get_sizes
import warnings
import platform
import cpuinfo
import re

import config as _config

if __name__ == '__main__':
    ipanema.initialize(_config.user['backend'], 1)


PATH = os.path.dirname(os.path.abspath(__file__))

# Get builtins (ipanema initialization exposes them)
BACKEND = builtins.BACKEND
DEVICE = builtins.DEVICE
CONTEXT = builtins.CONTEXT
QUEUE = builtins.THREAD


# read block size from config
BLOCK_SIZE = _config.user['block_size']


global __KERNELS__


def get_sizes(size, BLOCK_SIZE=BLOCK_SIZE):
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

# }}}


# Compile flags {{{
#     The compile_flags is a dict where each key substitutes a same-named
#     string in the kernel file by its value: #define KEY {KEY} <-- value

global config, real_type
config = dict(
    debug=0,  # no prints
    debug_evt=0,  # number of events to debug
    fast_integral=1,  # run integrals with approximation
    sigma_t=0.15,
    knots=[0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00],
    mHH=[990, 1008, 1016, 1020, 1024, 1032, 1050],
    tristan=[1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    flat_end=False,
    final_extrap=True,
    block_size=BLOCK_SIZE
)
real_type = np.float64


def flagger(verbose=False):
    if verbose:
        print(f"\n{80*'='}\nBadjanak kernel\n{80*'='}\n")
        print(f"{'Backend was set to':>20} : {BACKEND}")
        print( f"{' Using as host':>20} : {cpuinfo.get_cpu_info()['brand_raw']} with {platform.python_compiler()}")
        print(f"{' Using as device':>20} : {DEVICE.name} with CUDA 10.2")
        print(f" ")
    dict_flags = {}
    for key, value in config.items():
        if key == 'mHH':
            dict_flags['nmassbins'.upper()] = len(value)-1
            dict_flags['nmassknots'.upper()] = len(value)
        if key == 'knots':
            dict_flags['nknots'.upper()] = len(
                value) if config['final_extrap'] else len(value)
            dict_flags['spl_bins'.upper()] = len(
                value) if config['final_extrap'] else len(value)
            dict_flags['ntimebins'.upper()] = len(
                value)+1 if config['final_extrap'] else len(value)
        if key == 'tristan':
            dict_flags['nterms'.upper()] = len(value)
        if key != 'precision':
            dict_flags[key.upper()] = str(value).replace(
                '[', '{').replace(']', '}')
    if verbose:
        for key, value in dict_flags.items():
            print(f'{key.upper():>20} : {value}')
    return dict_flags

# }}}


# Compiler {{{

def compile(verbose=False, pedantic=False):
    """
    Compile badjanak kernels.

    Parameters
    ----------
    verbose : bool
        Will print the compilation process.
    pedantic : bool
        Will print the flagged-substituted kernel.cu code.

    """
    global real_type
    kstrg = open(os.path.join(PATH, 'kernel.cu'), "r").read()
    # some custom compiling options {{{
    opts = ''
    if BACKEND == 'cuda':
        opts = "-Xptxas -suppress-stack-size-warning"
        opts = ''
    # }}}
    if builtins.REAL == 'double':
        prog = ipanema.compile(
            kstrg,
            render_kwds={**{"USE_DOUBLE": "1"}, **flagger(verbose)},
            compiler_options=[f"-I{PATH} {opts}"],
            keep=False
        )
        real_type = np.float64
    else:
        prog = ipanema.compile(
            kstrg,
            render_kwds={**{"USE_DOUBLE": "0"}, **flagger(verbose)},
            compiler_options=[f"-I{PATH} {opts}"],
            keep=False
        )
        real_type = np.float32
    if pedantic:
        print(prog.source)
    if verbose:
        print('\nSuccesfully compiled.\n')
    return prog


def get_kernels(verbose=False, pedantic=False):
    """
    Get the compiled kernels and bind them to the the badjanak module.

    Parameters
    ----------
    verbose : bool
        verbose
    pedantic : bool
        pedantic
    """
    global __KERNELS__
    prog = compile(verbose, pedantic)
    items = ['pyrateBs', 'pyrateBd',  # 'pyrateBs_two_gamma',
             'pyFcoeffs',
             'pySingleTimeAcc', 'pyRatioTimeAcc', 'pyFullTimeAcc', 'pySpline',
             #  'py_tijk2weights'
             ]
    if BACKEND == 'cuda':
        items.append('dG5toy')
        items.append('integral_ijk_fx')
    for item in items:
        setattr(prog, item[2:], prog.__getattr__(item))
    __KERNELS__ = prog


# Get the kernels just after one import this module
get_kernels(verbose=False, pedantic=False)

# }}}


# Functions {{{
#     Here pythonic versions of KERNEL functions are defined. There are wrappers
#     that are simpler to interact with


# Cross-section parameters parser {{{

def parser_rateBs(
    Gd=0.66137, DGsd=0.08, DGs=0.08, DGd=0, DM=17.7, CSP=1.0,
    # Time-dependent angular distribution
    fSlon=0.00, fPlon=0.600,                 fPper=0.50,
    dSlon=3.07, dPlon=0,      dPpar=3.30, dPper=3.07,
    pSlon=0.00, pPlon=-0.03,   pPpar=0.00, pPper=0.00,
    lSlon=1.00, lPlon=1.00,   lPpar=1.00, lPper=1.00,
    # Time limits
    tLL=0.0, tUL=15.0,
    cosKLL=-1.0, cosKUL=1.0,
    cosLLL=-1.0, cosLUL=1.0,
    hphiLL=-np.pi, hphiUL=+np.pi,
    # Time resolution
    sigma_offset=0.00, sigma_slope=1.00, sigma_curvature=0.00,
    mu=0.00,
    # Flavor tagging
    eta_os=0.00, eta_ss=0.00,
    p0_os=0.00,  p1_os=0.00, p2_os=0.00,
    p0_ss=0.00,  p1_ss=0.00, p2_ss=0.00,
    dp0_os=0.00, dp1_os=0.00, dp2_os=0.00,
    dp0_ss=0.00, dp1_ss=0.00, dp2_ss=0.00,
    # Flags
    use_fk=1, use_angacc=0, use_timeacc=0,
    use_timeoffset=0, set_tagging=0, use_timeres=0,
    verbose=False, **p):

    r = {}
    r['mass_bins'] = len([k for k in p.keys() if re.compile('CSP.*').match(k)])
    # Get all binned parameters and put them in ristras
    if r['mass_bins'] >= 1:
        CSP = [p[k] for k in p.keys() if re.compile('CSP.*').match(k)]
        r['CSP'] = THREAD.to_device(real_type(CSP)).astype(real_type)
        fSlon = [p[k] for k in p.keys() if re.compile('fSlon.*').match(k)]
        fSlon = THREAD.to_device(real_type(fSlon)).astype(real_type)
        dSlon = [p[k] for k in p.keys() if re.compile('dSlon.*').match(k)]
        dSlon = THREAD.to_device(real_type(dSlon)).astype(real_type)
    else:
        r['CSP'] = THREAD.to_device(real_type([CSP])).astype(real_type)
        fSlon = THREAD.to_device(real_type([fSlon])).astype(real_type)
        dSlon = THREAD.to_device(real_type([dSlon])).astype(real_type)

    # Parameters and parse Gs value
    r['DG'] = DGs
    r['DM'] = DM
    if 'Gs' in p:
        r['G'] = p['Gs']
    else:
        r['G'] = Gd + DGsd + DGd
    if 'Gn' in p:  # nuisance gamma
        r['Gn'] = p['Gn']

    # Compute fractions of S and P wave objects
    FP = abs(1.-fSlon)
    r['ASlon'] = ipanema.ristra.sqrt(fSlon)
    r['APlon'] = ipanema.ristra.sqrt(FP*fPlon)
    r['APper'] = ipanema.ristra.sqrt(FP*fPper)
    r['APpar'] = ipanema.ristra.sqrt(FP*abs(1.-fPlon-fPper))

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
    r['cosKLL'] = cosKLL
    r['cosKUL'] = cosKUL
    r['cosLLL'] = cosLLL
    r['cosLUL'] = cosLUL
    r['hphiLL'] = hphiLL
    r['hphiUL'] = hphiUL

    # Time resolution
    r['sigma_offset'] = sigma_offset
    r['sigma_slope'] = sigma_slope
    r['sigma_curvature'] = sigma_curvature
    r['mu'] = mu

    # Tagging
    _eta_os = [k for k in p.keys() if re.compile("(eta_os|etaOS)((1|2)[0-9])?").match(k)]
    # print(_eta_os)
    r['eta_os'] = p[_eta_os[0]] if _eta_os else eta_os
    _eta_ss = [k for k in p.keys() if re.compile("(eta_ss|etaSS)((1|2)[0-9])?").match(k)]
    # print(_eta_ss)
    r['eta_ss'] = p[_eta_ss[0]] if _eta_ss else eta_ss
    _p0_os =  [k for k in p.keys() if re.compile("(p0_os|p0OS)((1|2)[0-9])?").match(k)]
    # print(_p0_os)
    r['p0_os'] =  p[_p0_os[0]] if _p0_os else p0_os
    _p1_os =  [k for k in p.keys() if re.compile("(p1_os|p1OS)((1|2)[0-9])?").match(k)]
    # print(_p1_os)
    r['p1_os'] =  p[_p1_os[0]] if _p1_os else p1_os
    _p2_os =  [k for k in p.keys() if re.compile("(p2_os|p2OS)((1|2)[0-9])?").match(k)]
    # print(_p2_os)
    r['p2_os'] =  p[_p2_os[0]] if _p2_os else p2_os
    _p0_ss =  [k for k in p.keys() if re.compile("(p0_ss|p0SS)((1|2)[0-9])?").match(k)]
    # print(_p0_ss)
    r['p0_ss'] =  p[_p0_ss[0]] if _p0_ss else p0_ss
    _p1_ss =  [k for k in p.keys() if re.compile("(p1_ss|p1SS)((1|2)[0-9])?").match(k)]
    # print(_p1_ss)
    r['p1_ss'] =  p[_p1_ss[0]] if _p1_ss else p1_ss
    _p2_ss =  [k for k in p.keys() if re.compile("(p2_ss|p2SS)((1|2)[0-9])?").match(k)]
    # print(_p2_ss)
    r['p2_ss'] =  p[_p2_ss[0]] if _p2_ss else p2_ss
    _dp0_os = [k for k in p.keys() if re.compile("(dp0_os|dp0OS)((1|2)[0-9])?").match(k)]
    # print(_dp0_os)
    r['dp0_os'] = p[_dp0_os[0]] if _dp0_os else dp0_os
    _dp1_os = [k for k in p.keys() if re.compile("(dp1_os|dp1OS)((1|2)[0-9])?").match(k)]
    # print(_dp1_os)
    r['dp1_os'] = p[_dp1_os[0]] if _dp1_os else dp1_os
    _dp2_os = [k for k in p.keys() if re.compile("(dp2_os|dp2OS)((1|2)[0-9])?").match(k)]
    # print(_dp2_os)
    r['dp2_os'] = p[_dp2_os[0]] if _dp2_os else dp2_os
    _dp0_ss = [k for k in p.keys() if re.compile("(dp0_ss|dp0SS)((1|2)[0-9])?").match(k)]
    # print(_dp0_ss)
    r['dp0_ss'] = p[_dp0_ss[0]] if _dp0_ss else dp0_ss
    _dp1_ss = [k for k in p.keys() if re.compile("(dp1_ss|dp1SS)((1|2)[0-9])?").match(k)]
    # print(_dp1_ss)
    r['dp1_ss'] = p[_dp1_ss[0]] if _dp1_ss else dp1_ss
    _dp2_ss = [k for k in p.keys() if re.compile("(dp2_ss|dp2SS)((1|2)[0-9])?").match(k)]
    # print(_dp2_ss)
    r['dp2_ss'] = p[_dp2_ss[0]] if _dp2_ss else dp2_ss

    # Time acceptance
    timeacc = [p[k] for k in p.keys() if re.compile(
        '(a|b|c)([0-9])([0-9])?(u|b)?((1|2)[0-9])?').match(k)]
    if timeacc:
        r['timeacc'] = THREAD.to_device(coeffs_to_poly(timeacc))
    else:
        r['timeacc'] = THREAD.to_device(real_type([1]))

    # Angular acceptance
    angacc = [p[k] for k in p.keys() if re.compile(
        'w([0-9])([0-9])?(u|b)?(\d{2})?').match(k)]
    if angacc:
        r['angacc'] = THREAD.to_device(real_type(angacc))
    else:
        r['angacc'] = THREAD.to_device(real_type(config['tristan']))

    return r


def parser_rateBd(
    Gd=0.66137, DGsd=0.08, DGs=0.08, DGd=0, DM=17.7, CSP=1.0,
    # Time-dependent angular distribution
    # cambiado ramon
    fSlon=0.00, fPlon=0.600,                 fPper=0.50,
    dSlon=3.07, dPlon=0,      dPpar=3.30, dPper=3.07,
    pSlon=0.00, pPlon=-0.03,   pPpar=0.00, pPper=0.00,
    lSlon=1.00, lPlon=1.00,   lPpar=1.00, lPper=1.00,
    # Time limits
    tLL=0.0, tUL=15.0,
    # Time resolution
    sigma_offset=0.00, sigma_slope=0.00, sigma_curvature=0.00,
    mu=0.00,
    # Flavor tagging
    eta_os=0.00, eta_ss=0.00,
    p0_os=0.00,  p1_os=0.00, p2_os=0.00,
    p0_ss=0.00,  p1_ss=0.00, p2_ss=0.00,
    dp0_os=0.00, dp1_os=0.00, dp2_os=0.00,
    dp0_ss=0.00, dp1_ss=0.00, dp2_ss=0.00,
    # Flags
    use_fk=1, use_angacc=0, use_timeacc=0,
    use_timeoffset=0, set_tagging=0, use_timeres=0,
    verbose=False,
        **p):

    r = {}
    r['mass_bins'] = len([k for k in p.keys() if re.compile('CSP.*').match(k)])
    # Get all binned parameters and put them in ristras
    if r['mass_bins'] >= 1:
        CSP = [p[k] for k in p.keys() if re.compile('CSP.*').match(k)]
        r['CSP'] = THREAD.to_device(real_type(CSP)).astype(real_type)
        fSlon = [p[k] for k in p.keys() if re.compile('fSlon.*').match(k)]
        fSlon = THREAD.to_device(real_type(fSlon)).astype(real_type)
        dSlon = [p[k] for k in p.keys() if re.compile('dSlon.*').match(k)]
        dSlon = THREAD.to_device(real_type(dSlon)).astype(real_type)
    else:
        r['CSP'] = THREAD.to_device(real_type([CSP])).astype(real_type)
        fSlon = THREAD.to_device(real_type([fSlon])).astype(real_type)
        dSlon = THREAD.to_device(real_type([dSlon])).astype(real_type)

    # Parameters and parse Gs value
    r['DG'] = DGs
    r['DM'] = DM
    r['G'] = Gd  # p['Gd']
    # Compute fractions of S and P wave objects
    FP = abs(1.-fSlon)
    r['ASlon'] = ipanema.ristra.sqrt(fSlon)
    r['APlon'] = ipanema.ristra.sqrt(FP*fPlon)
    r['APper'] = ipanema.ristra.sqrt(FP*fPper)
    r['APpar'] = ipanema.ristra.sqrt(FP*abs(1.-fPlon-fPper))

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
    timeacc = [p[k] for k in p.keys() if re.compile(
        'c([0-9])([0-9])?(u|b)?').match(k)]
    if timeacc:
        r['timeacc'] = THREAD.to_device(coeffs_to_poly(timeacc))
    else:
        r['timeacc'] = THREAD.to_device(real_type([1]))

    # Angular acceptance
    angacc = [p[k] for k in p.keys() if re.compile(
        'w([0-9])([0-9])?(u|b)?').match(k)]
    if angacc:
        r['angacc'] = THREAD.to_device(real_type(angacc))
    else:
        r['angacc'] = THREAD.to_device(real_type(config['tristan']))

    return r

# }}}


# Wrappers arround Bs cross rate {{{

def delta_gamma5(input, output, G, DG, DM, CSP, ASlon, APlon, APpar, APper,
                 pSlon, pPlon, pPpar, pPper, dSlon, dPlon, dPpar, dPper, lSlon,
                 lPlon, lPpar, lPper, tLL, tUL, cosKLL, cosKUL, cosLLL, cosLUL,
                 hphiLL, hphiUL, sigma_offset, sigma_slope, sigma_curvature,
                 mu, eta_os, eta_ss, p0_os,  p1_os, p2_os, p0_ss, p1_ss, p2_ss,
                 dp0_os, dp1_os, dp2_os, dp0_ss, dp1_ss, dp2_ss, timeacc,
                 angacc, use_fk=1, use_angacc=0, use_timeacc=0,
                 use_timeoffset=0, set_tagging=0, use_timeres=0, **crap):
    g_size, l_size = get_sizes(output.shape[0], config['block_size'])
    __KERNELS__.rateBs(
        input, output, real_type(G), real_type(DG), real_type(DM),
        CSP.astype(real_type), ASlon.astype(real_type),
        APlon.astype(real_type), APpar.astype(real_type),
        APper.astype(real_type), real_type(pSlon), real_type(pPlon),
        real_type(pPpar),                real_type(pPper),
        dSlon.astype(real_type),          real_type(dPlon),                 real_type(
            dPpar),                 real_type(dPper),
        real_type(lSlon),                 real_type(lPlon),                 real_type(
            lPpar),                 real_type(lPper),
        # Time range
        real_type(tLL), real_type(tUL),
        real_type(cosKLL), real_type(cosKUL),
        real_type(cosLLL), real_type(cosLUL),
        real_type(hphiLL), real_type(hphiUL),
        # Time resolution
        real_type(sigma_offset), real_type(
            sigma_slope), real_type(sigma_curvature),
        real_type(mu),
        # Flavor tagging
        real_type(eta_os), real_type(eta_ss),
        real_type(p0_os), real_type(p1_os), real_type(p2_os),
        real_type(p0_ss), real_type(p1_ss), real_type(p2_ss),
        real_type(dp0_os), real_type(dp1_os), real_type(dp2_os),
        real_type(dp0_ss), real_type(dp1_ss), real_type(dp2_ss),
        # Decay-time acceptance
        timeacc.astype(real_type),
        # Angular acceptance
        angacc.astype(real_type),
        # Flags
        np.int32(use_fk), np.int32(len(CSP)), np.int32(
            use_angacc), np.int32(use_timeacc),
        np.int32(use_timeoffset), np.int32(set_tagging), np.int32(use_timeres),
        np.int32(len(output)),
        global_size=g_size, local_size=l_size)
    # grid=(int(np.ceil(output.shape[0]/BLOCK_SIZE)),1,1), block=(BLOCK_SIZE,1,1))

def delta_gamma5_mod(input, output, G, Gn,  DG, DM, CSP, ASlon, APlon, APpar, APper,
                 pSlon, pPlon, pPpar, pPper, dSlon, dPlon, dPpar, dPper, lSlon,
                 lPlon, lPpar, lPper, tLL, tUL, cosKLL, cosKUL, cosLLL, cosLUL,
                 hphiLL, hphiUL, sigma_offset, sigma_slope, sigma_curvature,
                 mu, eta_os, eta_ss, p0_os,  p1_os, p2_os, p0_ss, p1_ss, p2_ss,
                 dp0_os, dp1_os, dp2_os, dp0_ss, dp1_ss, dp2_ss, timeacc,
                 angacc, use_fk=1, use_angacc=0, use_timeacc=0,
                 use_timeoffset=0, set_tagging=0, use_timeres=0, **crap):
    g_size, l_size = get_sizes(output.shape[0], config['block_size'])
    # print(Gn)
    __KERNELS__.rateBs_two_gamma(
        input, output, real_type(G), real_type(Gn), real_type(DG), real_type(DM),
        CSP.astype(real_type), ASlon.astype(real_type),
        APlon.astype(real_type), APpar.astype(real_type),
        APper.astype(real_type), real_type(pSlon), real_type(pPlon),
        real_type(pPpar),                real_type(pPper),
        dSlon.astype(real_type),          real_type(dPlon),                 real_type(
            dPpar),                 real_type(dPper),
        real_type(lSlon),                 real_type(lPlon),                 real_type(
            lPpar),                 real_type(lPper),
        # Time range
        real_type(tLL), real_type(tUL),
        real_type(cosKLL), real_type(cosKUL),
        real_type(cosLLL), real_type(cosLUL),
        real_type(hphiLL), real_type(hphiUL),
        # Time resolution
        real_type(sigma_offset), real_type(
            sigma_slope), real_type(sigma_curvature),
        real_type(mu),
        # Flavor tagging
        real_type(eta_os), real_type(eta_ss),
        real_type(p0_os), real_type(p1_os), real_type(p2_os),
        real_type(p0_ss), real_type(p1_ss), real_type(p2_ss),
        real_type(dp0_os), real_type(dp1_os), real_type(dp2_os),
        real_type(dp0_ss), real_type(dp1_ss), real_type(dp2_ss),
        # Decay-time acceptance
        timeacc.astype(real_type),
        # Angular acceptance
        angacc.astype(real_type),
        # Flags
        np.int32(use_fk), np.int32(len(CSP)), np.int32(
            use_angacc), np.int32(use_timeacc),
        np.int32(use_timeoffset), np.int32(set_tagging), np.int32(use_timeres),
        np.int32(len(output)),
        global_size=g_size, local_size=l_size)
    # grid=(int(np.ceil(output.shape[0]/BLOCK_SIZE)),1,1), block=(BLOCK_SIZE,1,1))

def delta_gamma5_data_mod(input, output, use_fk=1, use_angacc=1, use_timeacc=1,
                      use_timeoffset=0, set_tagging=1, use_timeres=1,
                      BLOCK_SIZE=256, **pars):
    """
    This function is intended to be used with MC input arrays. It does use
    time acceptance nor angular acceptance. The tagging is set to perfect
    tagging and the time resolution is disabled.


    Parameters
    ----------
    input : ipanema.ristra
      Input data with proper shape
    output : ipanema.ristra
      Output array with proper shape to store pdf values
    """
    p = parser_rateBs(**pars)
    delta_gamma5_mod(input, output, use_fk=use_fk, use_angacc=use_angacc,
                 use_timeacc=use_timeacc, use_timeoffset=use_timeoffset,
                 set_tagging=set_tagging, use_timeres=use_timeres,
                 BLOCK_SIZE=BLOCK_SIZE, **p)

def delta_gamma5_data(input, output, use_fk=1, use_angacc=1, use_timeacc=1,
                      use_timeoffset=0, set_tagging=1, use_timeres=1,
                      BLOCK_SIZE=256, **pars):
    """
    This function is intended to be used with MC input arrays. It does use
    time acceptance nor angular acceptance. The tagging is set to perfect
    tagging and the time resolution is disabled.


    Parameters
    ----------
    input : ipanema.ristra
      Input data with proper shape
    output : ipanema.ristra
      Output array with proper shape to store pdf values
    """
    p = parser_rateBs(**pars)
    delta_gamma5(input, output, use_fk=use_fk, use_angacc=use_angacc,
                 use_timeacc=use_timeacc, use_timeoffset=use_timeoffset,
                 set_tagging=set_tagging, use_timeres=use_timeres,
                 BLOCK_SIZE=BLOCK_SIZE, **p)


def delta_gamma5_mc(input, output, use_fk=1, set_tagging=0, **pars):
    """
    This function is intended to be used with MC input arrays. It does not use
    time acceptance nor angular acceptance. The tagging is set to perfect
    tagging and the time resolution is disabled.


    Parameters
    ----------
    input : ipanema.ristra
      Input data with proper shape
    output : ipanema.ristra
      Output array with proper shape to store pdf values
    """
    p = parser_rateBs(**pars)
    delta_gamma5(input, output, use_fk=use_fk, use_angacc=0, use_timeacc=0,
                 use_timeoffset=0, set_tagging=set_tagging, use_timeres=0,
                 BLOCK_SIZE=8, **p)

# }}}


# Wrappers arround Bd cross rate {{{

def delta_gamma5Bd(input, output,
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
                   use_fk=1, use_angacc=0, use_timeacc=0,
                   use_timeoffset=0, set_tagging=0, use_timeres=0,
                   BLOCK_SIZE=256, **crap):
    g_size, l_size = get_sizes(output.shape[0], BLOCK_SIZE)
    __KERNELS__.rateBd(
        # Input and output arrays
        input, output,
        # Differential cross-rate parameters
        real_type(G), CSP.astype(real_type),
        ASlon.astype(real_type), APlon.astype(real_type), APpar.astype(
            real_type), APper.astype(real_type),
        dSlon.astype(real_type),          real_type(dPlon),                 real_type(
            dPpar),                 real_type(dPper),
        # Time range
        real_type(tLL), real_type(tUL),
        # Angular acceptance
        angacc.astype(real_type),
        # Flags
        np.int32(use_fk),
        # BINS
        np.int32(len(CSP)), np.int32(use_angacc),
        # events
        np.int32(len(output)),
        global_size=g_size, local_size=l_size)
    # np.int32(use_angacc)


def delta_gamma5_data_Bd(input, output, use_fk=1, use_angacc=1, use_timeacc=1,
                         use_timeoffset=0, set_tagging=1, use_timeres=1,
                         BLOCK_SIZE=256, **pars):
    """
    This function is intended to be used with MC input arrays. It does use
    time acceptance nor angular acceptance. The tagging is set to perfect
    tagging and the time resolution is disabled.


    Parameters
    ----------
    input : ipanema.ristra
      Input data with proper shape
    output : ipanema.ristra
      Output array with proper shape to store pdf values
    """
    p = parser_rateBd(**pars)
    delta_gamma5Bd(input, output, use_fk=use_fk, use_angacc=use_angacc,
                   use_timeacc=use_timeacc, use_timeoffset=use_timeoffset,
                   set_tagging=set_tagging, use_timeres=use_timeres,
                   BLOCK_SIZE=BLOCK_SIZE, **p)


def delta_gamma5_mc_Bd(input, output, use_fk=1, use_angacc=0, **pars):
    """
    This function is intended to be used with MC input arrays. It doesn't use
    time acceptance nor angular acceptance. The tagging is set to perfect
    tagging and the time resolution is disabled.


    Parameters
    ----------
    input : ipanema.ristra
      Input data with proper shape
    output : ipanema.ristra
      Output array with proper shape to store pdf values
    """
    p = parser_rateBd(**pars)
    delta_gamma5Bd(input, output, use_fk=use_fk, use_angacc=use_angacc,
                   use_timeacc=0, use_timeoffset=0, set_tagging=0, use_timeres=0,
                   BLOCK_SIZE=256, **p)

# }}}


# angular acceptance functions {{{
#    These are several rules related to decay-time acceptance. The main one is
#    time_acceptance, which computes the spline coefficients of the
#    Bs2JpsiPhi acceptance.

def get_angular_acceptance_weights(true, reco, weight, tLL, tUL,
                                   **parameters):
    # Filter some warnings
    warnings.filterwarnings('ignore')
    # Compute weights scale and number of weights to compute
    scale = np.sum(ipanema.ristra.get(weight))
    terms = len(config['tristan'])

    # Define the computation core function
    def get_weights(true, reco, weight):
        pdf = THREAD.to_device(np.zeros(true.shape[0]))
        delta_gamma5_mc(true, pdf, use_fk=0, **parameters, tLL=tLL, tUL=tUL)
        num = pdf.get()
        pdf = THREAD.to_device(np.zeros(true.shape[0]))
        delta_gamma5_mc(true, pdf, use_fk=1, **parameters, tLL=tLL, tUL=tUL)
        den = pdf.get()
        fk = get_fk(reco.get()[:, 0:3])
        ang_acc = fk*(weight.get()*num/den).T[::, np.newaxis]
        return ang_acc

    ones = THREAD.to_device(np.ones(weight.shape[0]).astype(real_type))
    w = get_weights(true, reco, weight).sum(axis=0)
    w10 = get_weights(true, reco, ones)

    # Get covariance matrix
    cov = np.zeros((terms, terms, weight.shape[0]))
    for i in range(0, terms):
        for j in range(i, terms):
            cov[i, j, :] = (w10[:, i]-w[i]/scale) * \
                (w10[:, j]-w[j]/scale)*weight.get()**2
    cov = cov.sum(axis=2)  # sum all per event cov
    # fill the lower-triangle
    cov = cov + cov.T - np.eye(cov.shape[0])*cov

    # Handling cov matrix, and transform it to correlation matrix
    corr = np.zeros_like(cov)
    final_cov = np.zeros_like(cov)
    for i in range(0, cov.shape[0]):
        for j in range(0, cov.shape[1]):
            final_cov[i, j] = 1.0/(w[0]*w[0])*(
                w[i]*w[j]/(w[0]*w[0])*cov[0][0]+cov[i][j] -
                w[i]/w[0]*cov[0][j]-w[j]/w[0]*cov[0][i])
    final_cov[np.isnan(final_cov)] = 0
    cov = np.where(np.abs(final_cov) < 1e-12, 0, final_cov)

    # Correlation matrix
    for i in range(0, cov.shape[0]):
        for j in range(0, cov.shape[1]):
            corr[i, j] = cov[i][j]/np.sqrt(cov[i][i]*cov[j][j])
    return w/w[0], np.sqrt(np.diagonal(cov)), cov, corr


def analytical_angular_efficiency(angacc, cosK, cosL, hphi, project=None, order_cosK=4, order_cosL=4, order_hphi=4):
    """
    3d outputs by default
    """
    cosK_l = len(cosK)
    cosL_l = len(cosL)
    hphi_l = len(hphi)
    eff = ristra.zeros(cosK_l*cosL_l*hphi_l)
    try:
        _angacc = ristra.allocate(np.array(angacc))
    except:
        _angacc = ristra.allocate(np.array([a.n for a in angacc]))
    __KERNELS__.pyangular_efficiency(eff, _angacc.astype(real_type),
                                     cosK.astype(real_type), cosL.astype(
                                         real_type), hphi.astype(real_type),
                                     np.int32(cosK_l), np.int32(
                                         cosL_l), np.int32(hphi_l),
                                     np.int32(order_cosK), np.int32(
                                         order_cosL), np.int32(order_hphi),
                                     global_size=(cosL_l, hphi_l, cosK_l))
    # global_size=(1,))
    res = ristra.get(eff).reshape(cosL_l, hphi_l, cosK_l)
    if project == 1:
        # return np.trapz(res.T, cosKd.get())[:,5]
        return np.sum(res, (0, 1))
        # return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4).T, cosLd.get()), np.pi*hphid.get())
    if project == 2:
        # return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4), cosKd.get()), np.pi*hphid.get())
        return np.sum(res, (2, 1))
    if project == 3:
        # return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4).T, cosLd.get()).T, cosKd.get())
        return np.sum(res, (2, 0))
    return res


def angular_efficiency_weights(angacc, cosK, cosL, hphi, project=None):
    eff = ristra.zeros_like(cosK)
    try:
        _angacc = ristra.allocate(np.array(angacc))
    except:
        _angacc = ristra.allocate(np.array([a.n for a in angacc]))
    __KERNELS__.kangular_efficiency_weights(
        eff, cosK, cosL, hphi, _angacc, global_size=(eff.shape[0],))
    n = round(eff.shape[0]**(1/3))
    res = ristra.get(eff).reshape(n, n, n)
    if project == 1:
        return np.sum(res, (1, 0))
    if project == 2:
        return np.sum(res, (1, 2))
    if project == 3:
        return np.sum(res, (2, 0))
    return res


def get_angular_acceptance_weights_Bd(true, reco, weight, BLOCK_SIZE=256, **parameters):
    # Filter some warnings
    warnings.filterwarnings('ignore')

    # Compute weights scale and number of weights to compute
    scale = np.sum(ipanema.ristra.get(weight))
    terms = len(config['tristan'])
    # Define the computation core function

    def get_weights_Bd(true, reco, weight):
        pdf = THREAD.to_device(np.zeros(true.shape[0]))
        delta_gamma5_mc_Bd(true, pdf, use_fk=0, use_angacc=0, **parameters)
        num = pdf.get()
        pdf = THREAD.to_device(np.zeros(true.shape[0]))
        delta_gamma5_mc_Bd(true, pdf, use_fk=1, use_angacc=0, **parameters)
        den = pdf.get()
        fk = get_fk(reco.get()[:, 0:3])
        ang_acc = fk*(weight.get()*num/den).T[::, np.newaxis]
        return ang_acc

    # Get angular weights with weight applied and sum all events
    w = get_weights_Bd(true, reco, weight).sum(axis=0)
    # Get angular weights without weight
    ones = THREAD.to_device(np.ascontiguousarray(
        np.ones(weight.shape[0]), dtype=real_type))
    w10 = get_weights_Bd(true, reco, ones)

    # Get covariance matrix
    cov = np.zeros((terms, terms, weight.shape[0]))
    for i in range(0, terms):
        for j in range(i, terms):
            cov[i, j, :] = (w10[:, i]-w[i]/scale) * \
                (w10[:, j]-w[j]/scale)*weight.get()**2
    cov = cov.sum(axis=2)  # sum all per event cov

    # Handling cov matrix, and transform it to correlation matrix
    # fill the lower-triangle
    cov = cov + cov.T - np.eye(cov.shape[0])*cov
    final_cov = np.zeros_like(cov)
    corr = np.zeros_like(cov)
    # Cov relative to f0
    for i in range(0, cov.shape[0]):
        for j in range(0, cov.shape[1]):
            final_cov[i, j] = 1.0/(w[0]*w[0])*(
                w[i]*w[j]/(w[0]*w[0])*cov[0][0]+cov[i][j] -
                w[i]/w[0]*cov[0][j]-w[j]/w[0]*cov[0][i])
    final_cov[np.isnan(final_cov)] = 0
    final_cov = np.where(np.abs(final_cov) < 1e-12, 0, final_cov)
    # Correlation matrix
    for i in range(0, cov.shape[0]):
        for j in range(0, cov.shape[1]):
            corr[i, j] = final_cov[i][j] / \
                np.sqrt(final_cov[i][i]*final_cov[j][j])
    return w/w[0], np.sqrt(np.diagonal(final_cov)), final_cov, corr

# }}}


# Time acceptance functions {{{

def splinexerf(time, lkhd, coeffs, mu=0.0, sigma=0.04, gamma=0.6, tLL=0.3,
               tUL=15, BLOCK_SIZE=256):
    """
    Spline interpolation of the time acceptance function.

    Parameters
    ----------
    time:
      1D gpuarray with time to be fitted
    lkhd:
      1D gpuarray where likelihood is being stored

    Notes
    -----
    Look at pySingleTimeAcc kernel definition to see more help.
    """
    g_size, l_size = get_sizes(lkhd.shape[0], BLOCK_SIZE)

    __KERNELS__.SingleTimeAcc(
        time, lkhd,  # input, output
        THREAD.to_device(coeffs_to_poly(coeffs)).astype(real_type),
        real_type(mu), real_type(sigma), real_type(gamma),
        real_type(tLL), real_type(tUL),
        np.int32(lkhd.shape[0]),
        global_size=g_size, local_size=l_size
    )


def saxsbxscxerf(
        time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
        coeffs_a, coeffs_b, coeffs_c,
        mu_a=0.0, sigma_a=0.04, gamma_a=0.6,
        mu_b=0.0, sigma_b=0.04, gamma_b=0.6,
        mu_c=0.0, sigma_c=0.04, gamma_c=0.6,
        tLL=0.3, tUL=15,
        BLOCK_SIZE=256, **crap):
    """
    Two splines
    """
    size_a = np.int32(lkhd_a.shape[0])
    size_b = np.int32(lkhd_b.shape[0])
    size_c = np.int32(lkhd_c.shape[0])
    size_max = max(size_a, size_b, size_c)
    g_size, l_size = get_sizes(size_max, BLOCK_SIZE)
    __KERNELS__.FullTimeAcc(
        time_a, time_b, time_c, lkhd_a, lkhd_b, lkhd_c,
        THREAD.to_device(coeffs_to_poly(coeffs_a)).astype(real_type),
        THREAD.to_device(coeffs_to_poly(coeffs_b)).astype(real_type),
        THREAD.to_device(coeffs_to_poly(coeffs_c)).astype(real_type),
        real_type(mu_a), real_type(sigma_a), real_type(gamma_a),
        real_type(mu_b), real_type(sigma_b), real_type(gamma_b),
        real_type(mu_c), real_type(sigma_c), real_type(gamma_c),
        real_type(tLL), real_type(tUL),
        size_a, size_b, size_c,
        global_size=g_size, local_size=l_size
    )


def bspline(time, coeffs, BLOCK_SIZE=256):
    if isinstance(time, np.ndarray):
        time_d = THREAD.to_device(time).astype(real_type)
        spline_d = THREAD.to_device(0*time).astype(real_type)
        deallocate = True
    else:
        time_d = time
        spline_d = THREAD.to_device(0*time).astype(real_type)
        deallocate = False
    coeffs_d = THREAD.to_device(coeffs_to_poly(coeffs)).astype(np.float64)
    __KERNELS__.Spline(
        time_d, spline_d, coeffs_d, np.int32(len(time)),
        global_size=(len(time),)
    )
    return ristra.get(spline_d) if deallocate else spline_d


# Time accepntace helpers {{{

def get_knot(i, knots, n):
    if (i <= 0):
        i = 0
    elif (i >= n):
        i = n
    return knots[i]


def coeffs_to_poly(listcoeffs: list) -> np.ndarray:
    n = len(config['knots'])
    result = []                                          # list of bin coeffs C

    def u(j):
        return get_knot(j, config['knots'], n-1)

    for i in range(0, n-1):
        a, b, c, d = listcoeffs[i:i+4]                     # bspline coeffs b_i
        C = []                                    # each bin 4 coeffs c_{bin,i}
        C.append(-((b*u(-2+i)*pow(u(1+i), 2)) /
                   ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))) +
                 (a*pow(u(1+i), 3)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) +
                 (c*pow(u(-1+i), 2)*u(1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 (b*u(-1+i)*u(1+i)*u(2+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 (c*u(-1+i)*u(i)*u(2+i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (b*u(i)*pow(u(2+i), 2)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (d*pow(u(i), 3))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) +
                 (c*pow(u(i), 2)*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
        C.append((2*b*u(-2+i)*u(1+i)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) -
                 (3*a*pow(u(1+i), 2)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) +
                 (b*pow(u(1+i), 2)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) -
                 (c*pow(u(-1+i), 2)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 (b*u(-1+i)*u(1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 (2*c*u(-1+i)*u(1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 (b*u(-1+i)*u(2+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 (b*u(1+i)*u(2+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 (c*u(-1+i)*u(i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (c*u(-1+i)*u(2+i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) +
                 (2*b*u(i)*u(2+i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (c*u(i)*u(2+i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) +
                 (b*pow(u(2+i), 2)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (c*pow(u(i), 2))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) +
                 (3*d*pow(u(i), 2))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) -
                 (2*c*u(i)*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
        C.append(-((b*u(-2+i))/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i)) *
                                (-u(i)+u(1+i))))+(3*a*u(1+i)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) -
                 (2*b*u(1+i)) /
                 ((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) -
                 (b*u(-1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 (2*c*u(-1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 (b*u(1+i))/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i)) *
                             (-u(-1+i)+u(2+i)))+(c*u(1+i)) /
                 ((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 (b*u(2+i))/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i)) *
                             (-u(-1+i)+u(2+i)))+(c*u(-1+i)) /
                 ((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (b*u(i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) +
                 (c*u(i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 (2*b*u(2+i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) +
                 (c*u(2+i))/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) +
                 (2*c*u(i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) -
                 (3*d*u(i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) +
                 (c*u(3+i))/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
        C.append(-(a/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i)))) +
                 b/((-u(-2+i)+u(1+i))*(-u(-1+i)+u(1+i))*(-u(i)+u(1+i))) +
                 b/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) -
                 c/((-u(-1+i)+u(1+i))*(-u(i)+u(1+i))*(-u(-1+i)+u(2+i))) +
                 b/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 c/((-u(i)+u(1+i))*(-u(-1+i)+u(2+i))*(-u(i)+u(2+i))) -
                 c/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))) +
                 d/((-u(i)+u(1+i))*(-u(i)+u(2+i))*(-u(i)+u(3+i))))
        result.append(C)

    # usually we do a linear extrapolation from out last knot to the final tUL
    # If we *do not* want this to be done, we just need to diable the linear
    # extrapolation in the general badjanak.config
    if config['final_extrap']:
        if config['flat_end']:
            # if instead of linear extrapolation we just want the extrapolation
            # to be completely flat (no slope at all)
            m = 0
        else:
            # linear extrapolation in the last bin till tUL
            m = C[1] + 2 * C[2] * u(n) + 3 * C[3] * u(n)**2
        C = [C[0] + C[1]*u(n) + C[2]*u(n)**2 + C[3]*u(n)**3 - m*u(n), m, 0, 0]
        result.append(C)
    return np.array(result)

# }}}

# }}}


# Toy generators {{{

def dG5toys(output,
            G, DG, DM,
            CSP,
            ASlon, APlon, APpar, APper,
            pSlon, pPlon, pPpar, pPper,
            dSlon, dPlon, dPpar, dPper,
            lSlon, lPlon, lPpar, lPper,
            # Time limits
            tLL, tUL, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL,
            # Time resolution
            sigma_offset, sigma_slope, sigma_curvature, mu,
            # Flavor tagging
            eta_os, eta_ss,
            p0_os,  p1_os, p2_os,
            p0_ss,  p1_ss, p2_ss,
            dp0_os, dp1_os, dp2_os,
            dp0_ss, dp1_ss, dp2_ss,
            # Time acceptance
            timeacc,
            # Angular acceptance
            angacc, order_cosK=2, order_cosL=4, order_hphi=0,
            # Flags
            use_fk=1, use_angacc=0, use_timeacc=0,
            use_timeoffset=0, set_tagging=0, use_timeres=0,
            prob_max=2.7,
            BLOCK_SIZE=256, seed=False, **crap):
    """
    Generate toy Bs MC sample from given parameters.
    """
    if not seed:
        seed = int(1e10*np.random.rand())
    g_size, l_size = get_sizes(output.shape[0], BLOCK_SIZE)
    __KERNELS__.dG5toy(
        # Input and output arrays
        output,
        # Differential cross-rate parameters
        real_type(G), real_type(DG), real_type(DM),
        CSP.astype(real_type),
        ASlon.astype(real_type),
        APlon.astype(real_type),
        APpar.astype(real_type),
        APper.astype(real_type),
        real_type(pSlon),
        real_type(pPlon), real_type(pPpar), real_type(pPper),
        dSlon.astype(real_type),
        real_type(dPlon), real_type(dPpar), real_type(dPper),
        real_type(lSlon),
        real_type(lPlon), real_type(lPpar), real_type(lPper),
        # Time range
        real_type(tLL), real_type(tUL),
        real_type(cosKLL), real_type(cosKUL),
        real_type(cosLLL), real_type(cosLUL),
        real_type(hphiLL), real_type(hphiUL),
        # Time resolution
        real_type(sigma_offset),
        real_type(sigma_slope),
        real_type(sigma_curvature),
        real_type(mu),
        # Flavor tagging
        real_type(eta_os), real_type(eta_ss),
        real_type(p0_os), real_type(p1_os), real_type(p2_os),
        real_type(p0_ss), real_type(p1_ss), real_type(p2_ss),
        real_type(dp0_os), real_type(dp1_os), real_type(dp2_os),
        real_type(dp0_ss), real_type(dp1_ss), real_type(dp2_ss),
        # Decay-time acceptance
        timeacc.astype(real_type),
        # Angular acceptance
        angacc.astype(real_type),
        np.int32(order_cosK), np.int32(order_cosL), np.int32(order_hphi),
        # Flags
        np.int32(use_fk), np.int32(len(CSP)),
        np.int32(use_angacc), np.int32(use_timeacc),
        np.int32(use_timeoffset), np.int32(set_tagging), np.int32(use_timeres),
        real_type(prob_max), np.int32(seed),  np.int32(len(output)),
        global_size=g_size, local_size=l_size)

# }}}


# Other useful bindings {{{

def get_fk(z):
    z_dev = THREAD.to_device(np.ascontiguousarray(z, dtype=real_type))
    w = np.zeros((z.shape[0], len(config['tristan'])))
    w_dev = THREAD.to_device(np.ascontiguousarray(w, dtype=real_type))
    __KERNELS__.Fcoeffs(z_dev, w_dev, np.int32(len(z)), global_size=(len(z)))
    return ristra.get(w_dev)

# }}}


# }}}


# vim:foldmethod=marker
