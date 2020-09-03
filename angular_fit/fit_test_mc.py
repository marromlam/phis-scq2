#!/home3/marcos.romero/conda3/envs/ipanema3/bin/python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']




################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import pandas as pd
import uproot
import os
import sys
import hjson
import pandas

from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import Sample, Parameters, Parameter, ristra, optimize

# get bsjpsikk and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 5
badjanak.config['debug_evt'] = 774



################################################################################





def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--samples',
    default = ['/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root'],
    help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--csp',
    default = ['csp_factors/CSP.json'],
    help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--params',
    default = 'output/angular_fit/params/2016/Bs2JpsiPhi/v0r1_Yearly.json',
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--tables',
    default = 'output/angular_fit/tables/2016/Bs2JpsiPhi/v0r1_Yearly.tex',
    help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--year',
    default = '2016',
    help='Year of data-taking')
  parser.add_argument('--version',
    default = 'v0r1',
    help='Year of data-taking')

  return parser






################################################################################
################################################################################



#args = vars(argument_parser().parse_args(''))
args = vars(argument_parser().parse_args())
YEARS = [int(y) for y in args['year'].split(',')] # years are int
VERSION = args['version']

for k,v in args.items():
  print(f'{k}: {v}')



# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\n",
      "Loading samples",
      f"\n{80*'='}\n")

# Lists of data variables to load and build arrays
real  = ['helcosthetaK','helcosthetamu','helphi','time']                        # angular variables
real += ['X_M','0*B_ID']                                     # mass and sigmat
#real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging
real += ['B_ID','B_ID', '0*B_ID', '0*B_ID']  # tagging
weight_rd='(time/time)'


data = {}
mass = badjanak.config['x_m']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[f'{y}'] = Sample.from_root(args['samples'].split(',')[i], treename='T', cuts="time>=0.3 & time<=15 & X_M>=990 & X_M<=1050")
  csp = Parameters.load(args['csp'].split(',')[i])  # <--- WARNING
  data[f'{y}'].csp = csp.build(csp,csp.find('CSP.*'))
  print(f" *  Allocating {y} arrays in device ")
  sw = np.ones_like(data[f'{y}'].df.eval(weight_rd))
  for l,h in zip(mass[:-1],mass[1:]):
      pos = data[f'{y}'].df.eval(f'X_M>={l} & X_M<{h}')
      this_sw = data[f'{y}'].df.eval(f'{weight_rd}*(X_M>={l} & X_M<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
  data[f'{y}'].df['sWeight'] = sw
  data[f'{y}'].allocate(input=real,weight='sWeight',output='0*time')



# Prepare parameters
SWAVE = False
DGZERO = False
POLDEP = False
BLIND = True

pars = Parameters()
list_of_parameters = [#
# S wave fractions
Parameter(name='fSlon1', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{1}'),
Parameter(name='fSlon2', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{2}'),
Parameter(name='fSlon3', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{3}'),
Parameter(name='fSlon4', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{4}'),
Parameter(name='fSlon5', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{5}'),
Parameter(name='fSlon6', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{6}'),
# P wave fractions
Parameter(name="fPlon", value=0.5241, min=0.4, max=0.6,
          free=True, latex=r'f_0'),
Parameter(name="fPper", value=0.25, min=0.1, max=0.3,
          free=True, latex=r'f_{\perp}'),
# Weak phases
Parameter(name="pSlon", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_S - \phi_0",
          blindstr="BsPhisSDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
Parameter(name="pPlon", value=-0.03, min=-5.0, max=5.0,
          free=True, latex=r"\phi_0",
          blindstr="BsPhiszeroFullRun2" if POLDEP else "BsPhisFullRun2",
          blind=BLIND, blindscale=2.0 if POLDEP else 1.0, blindengine="root"),
Parameter(name="pPpar", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_{\parallel} - \phi_0",
          blindstr="BsPhisparaDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
Parameter(name="pPper", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_{\perp} - \phi_0",
          blindstr="BsPhisperpDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
# S wave strong phases
Parameter(name='dSlon1', value=+np.pi/4*SWAVE, min=-0.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{1} - \delta_{\perp}"),
Parameter(name='dSlon2', value=+np.pi/4*SWAVE, min=-0.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{2} - \delta_{\perp}"),
Parameter(name='dSlon3', value=+np.pi/4*SWAVE, min=-0.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{3} - \delta_{\perp}"),
Parameter(name='dSlon4', value=-np.pi/4*SWAVE, min=-3.0, max=+0.0,
          free=SWAVE, latex="\delta_S^{4} - \delta_{\perp}"),
Parameter(name='dSlon5', value=-np.pi/4*SWAVE, min=-3.0, max=+0.0,
          free=SWAVE, latex="\delta_S^{5} - \delta_{\perp}"),
Parameter(name='dSlon6', value=-np.pi/4*SWAVE, min=-3.0, max=+0.0,
          free=SWAVE, latex="\delta_S^{6} - \delta_{\perp}"),
# P wave strong phases
Parameter(name="dPlon", value=0.00, min=-2*3.14, max=2*3.14,
          free=False, latex="\delta_0"),
Parameter(name="dPpar", value=3.26, min=-2*3.14, max=2*3.14,
          free=True, latex="\delta_{\parallel} - \delta_0"),
Parameter(name="dPper", value=3.1, min=-2*3.14, max=2*3.14,
          free=True, latex="\delta_{\perp} - \delta_0"),
# lambdas
Parameter(name="lSlon", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_S/\lambda_0"),
Parameter(name="lPlon", value=1., min=0.7, max=1.6,
          free=True,  latex="\lambda_0"),
Parameter(name="lPpar", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_{\parallel}/\lambda_0"),
Parameter(name="lPper", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_{\perp}/\lambda_0"),
# lifetime parameters
Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
          free=False, latex=r"\Gamma_d"),
Parameter(name="DGs", value= (1-DGZERO)*0.1, min= 0.0, max= 1.7,
          free=1-DGZERO, latex=r"\Delta\Gamma_s",
          blindstr="BsDGsFullRun2",
          blind=BLIND, blindscale=1.0, blindengine="root"),
Parameter(name="DGsd", value= 0.03*0,   min=-0.1, max= 0.1,
          free=True, latex=r"\Gamma_s - \Gamma_d"),
Parameter(name="DM", value=17.757,   min=15.0, max=20.0,
          free=True, latex=r"\Delta m"),
]

pars.add(*list_of_parameters);
print(pars)

# compile the kernel
#    so if knots change when importing parameters, the kernel is compiled
badjanak.get_kernels(True)


#@profile
def wrapper_fcn(input, output, **pars):
  p = badjanak.cross_rate_parser_new(**pars)
  # for k,v in p.items():
  #     print(f"{k} : {v}")
  # exit()
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 0, use_timeacc = 0,
                         use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
                         BLOCK_SIZE=256, **p)
  exit()


def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():

      wrapper_fcn(dy.input, dy.output, **pars_dict,
                  **dy.csp.valuesdict(),
                  #**dy.flavor.valuesdict(),
                  tLL=0.3, tUL=0.15)

      chi2.append( -2.0 * (ristra.log(dy.output) * 1.0 ).get() );
  print(np.concatenate(chi2).sum())
  return np.concatenate(chi2)

################################################################################
#%% Run and get the job done ###################################################

print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
                  verbose=True, timeit=True, tol=0.5 , strategy=1)

print(result)

for p in ['DGsd', 'DGs', 'fPper', 'fPlon', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DM', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6']:
  if args['year'] == '2015,2016':
    print(f"{p:>12} : {result.params[p].value:+.4f}  {result.params[p]._getval(False):+.4f}")
  else:
    print(f"{p:>12} : {result.params[p].value:+.4f} +/- {result.params[p].stdev:+.4f}")



# Dump json file
result.params.dump(args['params'])
# Write latex table
with open(args['tables'], "w") as tex_file:
  tex_file.write( result.params.dump_latex(caption="Physics parameters.") )
