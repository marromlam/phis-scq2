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
badjanak.config['fast_integral'] = 1
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 774

# Parse arguments for this script
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



args = vars(argument_parser().parse_args())
YEARS = [int(y) for y in args['year'].split(',')] # years are int
VERSION = args['version']



# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\n", "Loading samples", f"\n{80*'='}\n")

# Lists of data variables to load and build arrays
real  = ['helcosthetaK','helcosthetaL','helphi','time']                        # angular variables
real += ['X_M','0*B_ID']                                     # mass and sigmat
real += ['B_ID','B_ID', '0*B_ID', '0*B_ID']  # tagging

real  = ['gencosK','gencosL','genhphi','gentime']                        # angular variables
real += ['mHH','0*genidB']                                     # mass and sigmat
real += ['genidB','genidB', '0*genidB', '0*genidB']  # tagging
weight_rd='(gentime/gentime)'
#real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging


data = {}
mass = badjanak.config['mHH']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  tree = uproot.open(args['samples'].split(',')[i]).keys()[0]
  data[f'{y}'] = Sample.from_root(args['samples'].split(',')[i], treename=tree, cuts="gentime>=0.0 & gentime<=15 & mHH>=990 & mHH<=1050")
  csp = Parameters.load(args['csp'].split(',')[i])
  print(csp)
  data[f'{y}'].csp = csp.build(csp,csp.find('CSP.*'))
  print(f" *  Allocating {y} arrays in device ")
  sw = np.ones_like(data[f'{y}'].df.eval(weight_rd))
  for l,h in zip(mass[:-1],mass[1:]):
      pos = data[f'{y}'].df.eval(f'mHH>={l} & mHH<{h}')
      this_sw = data[f'{y}'].df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
  data[f'{y}'].df['sWeight'] = sw
  data[f'{y}'].allocate(input=real,weight='sWeight',output='0*gentime')

# Prepare parameters
SWAVE = True
DGZERO = False
POLDEP = False
BLIND = False

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
Parameter(name="DGs", value= (1-DGZERO)*0.08, min= 0.0, max= 1.7,
          free=1-DGZERO, latex=r"\Delta\Gamma_s",
          blindstr="BsDGsFullRun2",
          blind=BLIND, blindscale=1.0, blindengine="root"),
Parameter(name="DGsd", value= 0.03,   min=-0.5, max= 0.5,
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
def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():
    badjanak.delta_gamma5_mc(dy.input, dy.output, **pars_dict,
                             **dy.csp.valuesdict(), tLL=0.00, tUL=15.0)
    chi2.append( -2.0 * (ristra.log(dy.output) * 1.0 ).get() );
  return np.concatenate(chi2)

################################################################################



################################################################################
#%% Run and get the job done ###################################################

print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
                  verbose=False, timeit=True, tol=0.05 , strategy=1)
print(result)

# Dump json file
result.params.dump(args['params'])
# Write latex table
with open(args['tables'], "w") as tex_file:
  tex_file.write( result.params.dump_latex(caption="Physics parameters.") )

################################################################################


"""
SWAVE FRACTIONS
I_f0_p1/(I_f0_p1+I_phi_p1)  = 0.433
I_f0_p2/(I_f0_p2+I_phi_p2)  = 0.042
I_f0_p3/(I_f0_p3+I_phi_p3)  = 0.005
I_f0_p4/(I_f0_p4+I_phi_p4)  = 0.007
I_f0_p5/(I_f0_p5+I_phi_p5)  = 0.034
I_f0_p6/(I_f0_p6+I_phi_p6)  = 0.122

SWAVE DELTAS
Csp (0.990,1.008) = 0.864
Tsp (0.990,1.008) = 5.092
Csp (1.008,1.016) = 0.927
Tsp (1.008,1.016) = 4.908
Csp (1.016,1.020) = 0.905
Tsp (1.016,1.020) = 0.950
Csp (1.020,1.024) = 0.949
Tsp (1.020,1.024) = 6.056
Csp (1.024,1.032) = 0.974
Tsp (1.024,1.032) = 5.583
Csp (1.032,1.050) = 0.985
Tsp (1.032,1.050) = 5.465
"""

"""
        DGsd : +0.0054 +/- +0.0013
         DGs : +0.0822 +/- +0.0040
       fPper : +0.2563 +/- +0.0020
       fPlon : +0.5158 +/- +0.0015
       dPpar : +3.2578 +/- +0.0126
       dPper : +3.0801 +/- +0.0110
       pPlon : -0.0319 +/- +0.0034
       lPlon : +0.9992 +/- +0.0022
          DM : +17.8024 +/- +0.0036
      fSlon1 : +0.4149 +/- +0.0162
      fSlon2 : +0.0443 +/- +0.0021
      fSlon3 : +0.0043 +/- +0.0003
      fSlon4 : +0.0065 +/- +0.0005
      fSlon5 : +0.0340 +/- +0.0017
      fSlon6 : +0.1191 +/- +0.0041
      dSlon1 : +1.8648 +/- +0.0253
      dSlon2 : +1.6819 +/- +0.0273
      dSlon3 : +0.8173 +/- +0.0405
      dSlon4 : -0.3776 +/- +0.0366
      dSlon5 : -0.7915 +/- +0.0263
      dSlon6 : -0.9282 +/- +0.0195
"""
