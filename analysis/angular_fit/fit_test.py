#!/home3/marcos.romero/conda3/envs/ipanema3/bin/python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']




################################################################################
# %% Modules ###################################################################

import numpy as np
import pandas as pd
import uproot
import os
import sys
import hjson
import pandas

from ipanema import initialize
initialize('cuda',2)
from ipanema import Sample, Parameters, Parameter, ristra, optimize
from pycuda.compiler import SourceModule


#import pycuda.driver as cuda
#import pycuda.autoinit
#import pycuda.gpuarray as cu_array




"""
prog = SourceModule(open('kernel.cu',"r").read())
"""



# get bsjpsikk and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 774
badjanak.get_kernels(True)
mass = badjanak.config['mHH']

################################################################################



################################################################################
################################################################################

# %% Load samples --------------------------------------------------------------

# branches to allocate
real = ['cosK','cosL','hphi','time','X_M','sigmat','tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']

data = {}
data['2016'] = {}
# Get sample
data['2016']['biased'] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root', cuts="Jpsi_Hlt1DiMuonHighMassDecision_TOS==0")
data['2016']['unbiased'] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root', cuts="Jpsi_Hlt1DiMuonHighMassDecision_TOS==1")
# Time acceptance
timeacc = Parameters.load(f'time_acceptance/params/2016/Bd2JpsiKstar/v0r0_biased.json')
data['2016']['biased'].timeacc = timeacc.build(timeacc,timeacc.find('c.*'))
timeacc = Parameters.load(f'time_acceptance/params/2016/Bd2JpsiKstar/v0r0_unbiased.json')
data['2016']['unbiased'].timeacc = timeacc.build(timeacc,timeacc.find('c.*'))
# Angular acceptance
angacc = Parameters.load(f'angular_acceptance/params/2016/Bs2JpsiPhi/v0r0_biased.json')
data['2016']['biased'].angacc = angacc
angacc = Parameters.load(f'angular_acceptance/params/2016/Bs2JpsiPhi/v0r0_unbiased.json')
data['2016']['unbiased'].angacc = angacc
# Time resolution
resolution = Parameters.load(f'time_resolution/time_resolution.json')
data['2016']['biased'].resolution = resolution
resolution = Parameters.load(f'time_resolution/time_resolution.json')
data['2016']['unbiased'].resolution = resolution
# CSP
csp_factors = Parameters.load(f'csp_factors/CSP.json')
data['2016']['biased'].csp_factors = csp_factors.build(csp_factors,csp_factors.find('CSP.*'))
csp_factors = Parameters.load(f'csp_factors/CSP.json')
data['2016']['unbiased'].csp_factors = csp_factors.build(csp_factors,csp_factors.find('CSP.*'))
# Weights and allocations
sw = np.zeros_like(data['2016']['biased'].df['sw'])
for l,h in zip(mass[:-1],mass[1:]):
  pos = data['2016']['biased'].df.eval(f'X_M>={l} & X_M<{h}')
  this_sw = data['2016']['biased'].df.eval(f'sw*(X_M>={l} & X_M<{h})')
  sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
data['2016']['biased'].df['sWeight'] = sw
data['2016']['biased'].allocate(input=real,weight='sWeight',output='0*time')
sw = np.zeros_like(data['2016']['unbiased'].df['sw'])
for l,h in zip(mass[:-1],mass[1:]):
  pos = data['2016']['unbiased'].df.eval(f'X_M>={l} & X_M<{h}')
  this_sw = data['2016']['unbiased'].df.eval(f'sw*(X_M>={l} & X_M<{h})')
  sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
data['2016']['unbiased'].df['sWeight'] = sw
data['2016']['unbiased'].allocate(input=real,weight='sWeight',output='0*time')





# Prepare parameters 9.536740132617429e-07
SWAVE = 1
DGZERO = 0
pars = Parameters()
list_of_parameters = [#
# Parameter(name='fSlon1', value=0.1+SWAVE*0*0.4260, min=0.00, max=0.80, free=SWAVE),
# Parameter(name='fSlon2', value=0.1+SWAVE*0*0.0590, min=0.00, max=0.80, free=SWAVE),
# Parameter(name='fSlon3', value=0.1+SWAVE*0*0.0101, min=0.00, max=0.80, free=SWAVE),
# Parameter(name='fSlon4', value=0.1+SWAVE*0*0.0103, min=0.00, max=0.80, free=SWAVE),
# Parameter(name='fSlon5', value=0.1+SWAVE*0*0.0490, min=0.00, max=0.80, free=SWAVE),
# Parameter(name='fSlon6', value=0.1+SWAVE*0*0.1930, min=0.00, max=0.80, free=SWAVE),
Parameter(name='fSlon1',          value=SWAVE*+0.0009765623447890**2,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon2',          value=SWAVE*+0.0009765623447890**2,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon3',          value=SWAVE*+0.0009765623447890**2,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon4',          value=SWAVE*+0.0009765623447890**2,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon5',          value=SWAVE*+0.0009765623447890**2,          min=SWAVE*0.00,   max=0.80,   free=SWAVE),
Parameter(name='fSlon6',          value=SWAVE*+0.0009765623447890**2,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
#
Parameter(name="fPlon", value=0.5241, min=0.4, max=0.6, latex=r'f_0'),
Parameter(name="fPper", value=0.25, min=0.2, max=0.3, latex=r'f_{\perp}'),
#
Parameter(name="pSlon", value= 0.00, min=-0.5, max=0.5, free=False),
Parameter(name="pPlon", value=-0.03, min=-0.5, max=0.5),
Parameter(name="pPpar", value= 0.00, min=-0.5, max=0.5, free=False),
Parameter(name="pPper", value= 0.00, min=-0.5, max=0.5, free=False),
#
Parameter(name='dSlon1', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon2', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon3', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon4', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon5', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon6', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
#
Parameter(name="dPlon", value=0.00, min=-2*3.14, max=2*3.14, free = False),
Parameter(name="dPpar", value=3.26, min=-2*3.14, max=2*3.14),
Parameter(name="dPper", value=3.1, min=-2*3.14, max=2*3.14),
#
Parameter(name="lSlon", value=1., min=0.7, max=1.6, free=False),
Parameter(name="lPlon", value=1., min=0.7, max=1.6),
Parameter(name="lPpar", value=1., min=0.7, max=1.6, free=False),
Parameter(name="lPper", value=1., min=0.7, max=1.6, free=False),
# 17.768 17.757    +0.0917
Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0, free=False),
Parameter(name="DGs", value= (1-DGZERO)*0.08, min= 0.0, max= 0.2, free=1-DGZERO),
Parameter(name="DGsd", value= 0.03*0,   min=-0.1, max= 0.1),
Parameter(name="DM", value=17.757,   min=17.0, max=18.0),
#
Parameter(name="tLL", value= 0.3,free=False),
Parameter(name="tUL", value=15,free=False),
#
# Parameter(name='CSP1', value=0.8569*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
# Parameter(name='CSP2', value=0.8569*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
# Parameter(name='CSP3', value=0.8478*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
# Parameter(name='CSP4', value=0.8821*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
# Parameter(name='CSP5', value=0.9406*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
# Parameter(name='CSP6', value=0.9711*SWAVE+1-SWAVE, min=0.0, max=1.0, free=False),
Parameter(name='CSP1',            value=0.8463000000000001*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP2',            value=0.8756*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP3',            value=0.8478*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP4',            value=0.8833*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP5',            value=0.9415*SWAVE,         min=0.0,    max=1.0,    free=False),
Parameter(name='CSP6',            value=0.9756*SWAVE,         min=0.0,    max=1.0,    free=False),
#
# Time resolution parameters
Parameter(name='mu',              value=0.0,                  min=0.0,    max=1.0,    free=False),
Parameter(name='sigma_offset',    value=0.01297,              min=0.0,    max=1.0,    free=False),
Parameter(name='sigma_slope',     value=0.8446,               min=0.0,    max=1.0,    free=False),
Parameter(name='sigma_curvature', value=0,                    min=0.0,    max=1.0,    free=False),
# Flavor tagging parameters
Parameter(name='eta_os',           value=0.3602,               min=0.0,    max=1.0,    free=False),
Parameter(name='eta_ss',           value=0.4167,               min=0.0,    max=1.0,    free=False),
Parameter(name='p0_os',            value=0.389,                min=0.0,    max=1.0,    free=False),
Parameter(name='p0_ss',            value=0.4325,               min=0.0,    max=1.0,    free=False),
Parameter(name='p1_os',            value=0.8486,               min=0.0,    max=1.0,    free=False),
Parameter(name='p1_ss',            value=0.9241,               min=0.0,    max=1.0,    free=False),
Parameter(name='dp0_os',           value=0.009 ,               min=0.0,    max=1.0,    free=False),
Parameter(name='dp0_ss',           value=0,                    min=0.0,    max=1.0,    free=False),
Parameter(name='dp1_os',           value=0.0143,               min=0.0,    max=1.0,    free=False),
Parameter(name='dp1_ss',           value=0,                    min=0.0,    max=1.0,    free=False)
]
pars.add(*list_of_parameters);




#@profile
def wrapper_fcn(input, output, **pars):
  p = badjanak.cross_rate_parser_new(**pars)
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 1, use_timeacc = 1,
                         use_timeoffset = 0, set_tagging = 1, use_timeres = 1,
                         BLOCK_SIZE=256, **p)

#wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
#wrapper_fcn(data['2016']['unbiased'].input,data['2016']['unbiased'].output,**pars.valuesdict(),**data['2016']['unbiased'].timeacc.valuesdict(),**data['2016']['unbiased'].angacc.valuesdict())
#exit()



#@profile
def fcn_data(parameters, data):
  pars_dict = parameters.valuesdict()
  chi2 = []
  for y, dy in data.items():
    for dt in [dy['biased'],dy['unbiased']]:
      wrapper_fcn(dt.input, dt.output, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict())
      chi2.append( -2.0 * (ristra.log(dt.output) * dt.weight).get() );
  return np.concatenate(chi2)


result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
         verbose=True, tol=0.5, strategy=1)
print(result)

for p in ['DGsd', 'DGs', 'fPper', 'fPlon', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DM', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6']:
  print(f"{p:>12} : {result.params[p].value:+.8f}")
