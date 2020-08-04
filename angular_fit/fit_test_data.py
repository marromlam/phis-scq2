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
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 774



################################################################################





def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--samples',
    default = ['/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root'],
    help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--angacc-biased',
    default = ['angular_acceptance/params/2016/Bs2JpsiPhi/v0r0_biased.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angacc-unbiased',
    default = ['angular_acceptance/params/2016/Bs2JpsiPhi/v0r0_unbiased.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-biased',
    default = ['time_acceptance/params/2016/Bd2JpsiKstar/v0r0_biased.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-unbiased',
    default = ['time_acceptance/params/2016/Bd2JpsiKstar/v0r0_unbiased.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--csp',
    default = ['csp_factors/CSP.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--time-resolution',
    default = ['time_resolution/time_resolution.json'],
    help='Bs2JpsiPhi MC sample')
  parser.add_argument('--flavor-tagging',
    default = ['/scratch03/marcos.romero/test_phis_scq/phis-scq/flavor_tagging/flavor_tagging.json'],
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
real  = ['cosK','cosL','hphi','time']                        # angular variables
real += ['X_M','sigmat']                                     # mass and sigmat
real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging
#real += ['0*B_ID','0*B_ID', '0*B_ID', '0*B_ID']  # tagging
weight_rd='(sw)'


data = {}
mass = badjanak.config['x_m']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[f'{y}'] = {}
  csp = Parameters.load(args['csp'].split(',')[i])  # <--- WARNING
  csp = csp.build(csp,csp.find('CSP.*'))
  flavor = Parameters.load(args['flavor_tagging'].split(',')[i])
  resolution = Parameters.load(args['time_resolution'].split(',')[i])
  for t, T in zip(['biased','unbiased'],[0,1]):
    print(f" *  Loading {y} sample in {t} category\n    {args['samples'].split(',')[i]}")
    this_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={T}) & (time>=0.3) & (time<=15)'
    data[f'{y}'][f'{t}'] = Sample.from_root(args['samples'].split(',')[i], cuts=this_cut)
    data[f'{y}'][f'{t}'].csp = csp
    data[f'{y}'][f'{t}'].flavor = flavor
    data[f'{y}'][f'{t}'].resolution = resolution
    print(data[f'{y}'][f'{t}'].csp)
    print(data[f'{y}'][f'{t}'].flavor)
    print(data[f'{y}'][f'{t}'].resolution)
  for t, coeffs in zip(['biased','unbiased'],[args['timeacc_biased'],args['timeacc_unbiased']]):
    print(f" *  Associating {y}-{t} time acceptance[{i}] from\n    {coeffs.split(',')[i]}")
    c = Parameters.load(coeffs.split(',')[i])
    print(c)
    knots = np.array(Parameters.build(c,c.fetch('k.*'))).tolist()
    badjanak.config['knots'] = knots
    print(knots)
    data[f'{y}'][f'{t}'].timeacc = Parameters.build(c,c.fetch('c.*'))
    data[f'{y}'][f'{t}'].tLL = c['tLL'].value
    data[f'{y}'][f'{t}'].tUL = c['tUL'].value
  for t, weights in zip(['biased','unbiased'],[args['angacc_biased'],args['angacc_unbiased']]):
    print(f" *  Associating {y}-{t} angular weights from\n    {weights.split(',')[i]}")
    w = Parameters.load(weights.split(',')[i])
    print(w)
    data[f'{y}'][f'{t}'].angacc = Parameters.build(w,w.fetch('w.*'))
  print(f" *  Allocating {y} arrays in device ")
  for d in [data[f'{y}']['biased'],data[f'{y}']['unbiased']]:
    sw = np.zeros_like(d.df['sw'])
    for l,h in zip(mass[:-1],mass[1:]):
      pos = d.df.eval(f'X_M>={l} & X_M<{h}')
      this_sw = d.df.eval(f'sw*(X_M>={l} & X_M<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
    d.df['sWeight'] = sw
    d.allocate(input=real,weight='sWeight',output='0*time')




# Prepare parameters
SWAVE = 1
DGZERO = 0
pars = Parameters()
list_of_parameters = [#
# S wave fractions
Parameter(name='fSlon1', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{1}'),
Parameter(name='fSlon2', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{2}'),
Parameter(name='fSlon3', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{3}'),
Parameter(name='fSlon4', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{4}'),
Parameter(name='fSlon5', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{5}'),
Parameter(name='fSlon6', value=SWAVE*0.0009765623447890**2, min=SWAVE*0.00, max=0.80, free=SWAVE, latex=r'f_S^{6}'),
# P wave fractions
Parameter(name="fPlon", value=0.5241, min=0.4, max=0.6, latex=r'f_0'),
Parameter(name="fPper", value=0.25, min=0.1, max=0.3, latex=r'f_{\perp}'),
# Weak phases
Parameter(name="pSlon", value= 0.00, min=-1.0, max=1.0, free=False, blind="BsPhisSDelFullRun2",    blindscale=1.0, blindengine="root", latex=r"\phi_S - \phi_0"),
Parameter(name="pPlon", value=-0.03, min=-1.0, max=1.0, free=True , blind="BsPhiszeroFullRun2",    blindscale=1.0, blindengine="root", latex=r"\phi_0" ),
Parameter(name="pPpar", value= 0.00, min=-1.0, max=1.0, free=False, blind="BsPhisparaDelFullRun2", blindscale=1.0, blindengine="root", latex=r"\phi_{\parallel} - \phi_0"),
Parameter(name="pPper", value= 0.00, min=-1.0, max=1.0, free=False, blind="BsPhisperpDelFullRun2", blindscale=1.0, blindengine="root", latex=r"\phi_{\perp} - \phi_0"),
# S wave strong phases
Parameter(name='dSlon1', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon2', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon3', value=+np.pi/4*SWAVE, min=-0.0,   max=+3.0,   free=SWAVE),
Parameter(name='dSlon4', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon5', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
Parameter(name='dSlon6', value=-np.pi/4*SWAVE, min=-3.0,   max=+0.0,   free=SWAVE),
# P wave strong phases
Parameter(name="dPlon", value=0.00, min=-2*3.14, max=2*3.14, free = False),
Parameter(name="dPpar", value=3.26, min=-2*3.14, max=2*3.14),
Parameter(name="dPper", value=3.1, min=-2*3.14, max=2*3.14),
# lambdas
Parameter(name="lSlon", value=1., min=0.7, max=1.6, free=False, latex="\lambda_S/\lambda_0"),
Parameter(name="lPlon", value=1., min=0.7, max=1.6, free=True,  latex="\lambda_0"),
Parameter(name="lPpar", value=1., min=0.7, max=1.6, free=False, latex="\lambda_{\parallel}/\lambda_0"),
Parameter(name="lPper", value=1., min=0.7, max=1.6, free=False, latex="\lambda_{\perp}/\lambda_0"),
# life parameters
Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0, free=False, latex=r"\Gamma_d"),
Parameter(name="DGs", value= (1-DGZERO)*0.08, min= 0.0, max= 0.7, free=1-DGZERO, blind="BsDGsFullRun2", blindengine="root", latex=r"\Delta\Gamma_s"),
Parameter(name="DGsd", value= 0.03*0,   min=-0.1, max= 0.1, latex=r"\Gamma_s - \Gamma_d"),
Parameter(name="DM", value=17.757,   min=15.0, max=20.0, latex=r"\Delta m"),
#
]
pars.add(*list_of_parameters);
print(pars)


# compile the kernel
#    so if knots change when importing parameters, the kernel is compiled
badjanak.get_kernels(True)


#@profile
def wrapper_fcn(input, output, **pars):
  p = badjanak.cross_rate_parser_new(**pars)
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 1, use_timeacc = 1,
                         use_timeoffset = 0, set_tagging = 1, use_timeres = 1,
                         BLOCK_SIZE=256, **p)


# test here crap
#wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
#wrapper_fcn(data['2016']['unbiased'].input,data['2016']['unbiased'].output,**pars.valuesdict(),**data['2016']['unbiased'].timeacc.valuesdict(),**data['2016']['unbiased'].angacc.valuesdict())
#exit()



#@profile
def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():
    for dt in [dy['biased'],dy['unbiased']]:
      wrapper_fcn(dt.input, dt.output, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  **dt.flavor.valuesdict(), tLL=dt.tLL, tUL=dt.tUL)
      chi2.append( -2.0 * (ristra.log(dt.output) * dt.weight).get() );
  return np.concatenate(chi2)

################################################################################
#%% Run and get the job done ###################################################

print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
                  verbose=False, timeit=True, tol=0.5, strategy=1)

print(result)

for p in ['DGsd', 'DGs', 'fPper', 'fPlon', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DM', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6']:
  if args['year'] == '2015,2016':
    print(f"{p:>12} : {result.params[p].value:+.8f}  {result.params[p]._getval(False):+.8f}")
  else:
    print(f"{p:>12} : {result.params[p].value:+.8f}")


# Dump json file
result.params.dump(args['params'])
# Write latex table
with open(args['tables'], "w") as tex_file:
  tex_file.write( result.params.dump_latex(caption="Physics parameters.") )
