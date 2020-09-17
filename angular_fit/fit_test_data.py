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

    '''
    data[f'{y}'][f'{t}'].flavor['p0_os'].free = True
    data[f'{y}'][f'{t}'].flavor['p0_os'].min = 0.0
    data[f'{y}'][f'{t}'].flavor['p0_os'].max = 1.0

    data[f'{y}'][f'{t}'].flavor['dp0_os'].free = True
    data[f'{y}'][f'{t}'].flavor['dp0_os'].min = -0.1
    data[f'{y}'][f'{t}'].flavor['dp0_os'].max = +0.1

    data[f'{y}'][f'{t}'].flavor['p1_os'].free = True
    data[f'{y}'][f'{t}'].flavor['p1_os'].min = 0.5
    data[f'{y}'][f'{t}'].flavor['p1_os'].max = 1.5

    data[f'{y}'][f'{t}'].flavor['dp1_os'].free = True
    data[f'{y}'][f'{t}'].flavor['dp1_os'].min = -0.1
    data[f'{y}'][f'{t}'].flavor['dp1_os'].max = +0.1

    data[f'{y}'][f'{t}'].flavor['p0_ss'].free = True
    data[f'{y}'][f'{t}'].flavor['p0_ss'].min = 0.0
    data[f'{y}'][f'{t}'].flavor['p0_ss'].max = 2.0

    data[f'{y}'][f'{t}'].flavor['dp0_ss'].free = True
    data[f'{y}'][f'{t}'].flavor['dp0_ss'].min = -0.1
    data[f'{y}'][f'{t}'].flavor['dp0_ss'].max = +0.1

    data[f'{y}'][f'{t}'].flavor['p1_ss'].free = True
    data[f'{y}'][f'{t}'].flavor['p1_ss'].min = 0.0
    data[f'{y}'][f'{t}'].flavor['p1_ss'].max = 2.0

    data[f'{y}'][f'{t}'].flavor['dp1_ss'].free = True
    data[f'{y}'][f'{t}'].flavor['dp1_ss'].min = -0.1
    data[f'{y}'][f'{t}'].flavor['dp1_ss'].max = +0.1
    '''

    print(data[f'{y}'][f'{t}'].csp)
    print(data[f'{y}'][f'{t}'].flavor)
    print(data[f'{y}'][f'{t}'].resolution)
  for t, coeffs in zip(['biased','unbiased'],[args['timeacc_biased'],args['timeacc_unbiased']]):
    print(f" *  Associating {y}-{t} time acceptance[{i}] from\n    {coeffs.split(',')[i]}")
    c = Parameters.load(coeffs.split(',')[i])
    print(c)
    knots = np.array(Parameters.build(c,c.fetch('k.*'))).tolist()
    badjanak.config['knots'] = knots
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
SWAVE = True
DGZERO = False
POLDEP = False
BLIND = True

pars = Parameters()
list_of_parameters = [#
# S wave fractions
Parameter(name='fSlon1', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{1}'),
Parameter(name='fSlon2', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{2}'),
Parameter(name='fSlon3', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{3}'),
Parameter(name='fSlon4', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{4}'),
Parameter(name='fSlon5', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{5}'),
Parameter(name='fSlon6', value=SWAVE*0.0009765623447890**2, min=0.00, max=0.80,
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
          free=POLDEP, blindstr="BsPhisperpDelFullRun2", blind=BLIND, blindscale=2.0, blindengine="root", latex=r"\phi_{\perp} - \phi_0"),
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
          free=1-DGZERO, blindstr="BsDGsFullRun2", blind=BLIND, blindscale=1.0, blindengine="root", latex=r"\Delta\Gamma_s"),
Parameter(name="DGsd", value= 0.03*0,   min=-0.1, max= 0.1, latex=r"\Gamma_s - \Gamma_d"),
Parameter(name="DM", value=17.757,   min=15.0, max=20.0, latex=r"\Delta m"),
Parameter("eta_os", value = data['2016']['unbiased'].flavor['eta_os'].value,
          free = False),
Parameter("eta_ss", value = data['2016']['unbiased'].flavor['eta_ss'].value, free = False),
Parameter("p0_os",  value = data['2016']['unbiased'].flavor['p0_os'].value,  free = True, min =  0.0, max = 1.0, latex = "p^{\rm OS}_{0}"),
Parameter("p1_os",  value = data['2016']['unbiased'].flavor['p1_os'].value,  free = True, min =  0.5, max = 1.5, latex = "p^{\rm OS}_{1}"),
Parameter("p0_ss",  value = data['2016']['unbiased'].flavor['p0_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{0}"),
Parameter("p1_ss",  value = data['2016']['unbiased'].flavor['p1_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{1}"),
Parameter("dp0_os", value = data['2016']['unbiased'].flavor['dp0_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{0}"),
Parameter("dp1_os", value = data['2016']['unbiased'].flavor['dp1_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{1}"),
Parameter("dp0_ss", value = data['2016']['unbiased'].flavor['dp0_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{0}"),
Parameter("dp1_ss", value = data['2016']['unbiased'].flavor['dp1_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{1}"),
]

pars.add(*list_of_parameters);
print(pars)
'''
tagging_pars = Parameters()
list_tagging_parameters = [
  # tagging parameters - currently set to the same values for all years!!!
  Parameter("eta_os", value = data['2016']['unbiased'].flavor['eta_os'].value, free = False),
  Parameter("eta_ss", value = data['2016']['unbiased'].flavor['eta_ss'].value, free = False),
  Parameter("p0_os",  value = data['2016']['unbiased'].flavor['p0_os'].value,  free = True, min =  0.0, max = 1.0, latex = "p^{\rm OS}_{0}"),
  Parameter("p1_os",  value = data['2016']['unbiased'].flavor['p1_os'].value,  free = True, min =  0.5, max = 1.5, latex = "p^{\rm OS}_{1}"),
  Parameter("p0_ss",  value = data['2016']['unbiased'].flavor['p0_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{0}"),
  Parameter("p1_ss",  value = data['2016']['unbiased'].flavor['p1_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{1}"),
  Parameter("dp0_os", value = data['2016']['unbiased'].flavor['dp0_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{0}"),
  Parameter("dp1_os", value = data['2016']['unbiased'].flavor['dp1_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{1}"),
  Parameter("dp0_ss", value = data['2016']['unbiased'].flavor['dp0_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{0}"),
  Parameter("dp1_ss", value = data['2016']['unbiased'].flavor['dp1_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{1}"),
]
tagging_pars.add(*list_tagging_parameters)
print(tagging_pars)
'''
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
# wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
# wrapper_fcn(data['2016']['unbiased'].input,data['2016']['unbiased'].output,**pars.valuesdict(),**data['2016']['unbiased'].timeacc.valuesdict(),**data['2016']['unbiased'].angacc.valuesdict())
#
# wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
# wrapper_fcn(data['2016']['unbiased'].input,data['2016']['unbiased'].output,**pars.valuesdict(),**data['2016']['unbiased'].timeacc.valuesdict(),**data['2016']['unbiased'].angacc.valuesdict())
# exit()

#Calculate tagging constraints - currently using one value for all years only!!!
def taggingConstraints(data):
  rhoOS = data['2016']['unbiased'].flavor['rho01_os'].value
  rhoSS = data['2016']['unbiased'].flavor['rho01_ss'].value

  pOS = np.matrix([data['2016']['unbiased'].flavor['p0_os'].value,
                  data['2016']['unbiased'].flavor['p1_os'].value])
  pSS = np.matrix([data['2016']['unbiased'].flavor['p0_ss'].value,
                  data['2016']['unbiased'].flavor['p1_ss'].value])

  p0OS_err = data['2016']['unbiased'].flavor['p0_os'].stdev
  p1OS_err = data['2016']['unbiased'].flavor['p1_os'].stdev
  p0SS_err = data['2016']['unbiased'].flavor['p0_ss'].stdev
  p1SS_err = data['2016']['unbiased'].flavor['p1_ss'].stdev

  covOS = np.matrix([[p0OS_err**2, p0OS_err*p1OS_err*rhoOS],
                     [p0OS_err*p1OS_err*rhoOS, p1OS_err**2]])
  covSS = np.matrix([[p0SS_err**2, p0SS_err*p1SS_err*rhoSS],
                     [p0SS_err*p1SS_err*rhoSS, p1SS_err**2]])

  covOSInv = covOS.I
  covSSInv = covSS.I

  dictOut = {'pOS': pOS, 'pSS': pSS, 'covOS': covOS, 'covSS': covSS, 'covOSInv': covOSInv, 'covSSInv': covSSInv}

  return dictOut

tagConstr = taggingConstraints(data)

#@profile
def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2TagConstr = 0.

  chi2TagConstr += (pars_dict['dp0_os']-data['2016']['unbiased'].flavor['dp0_os'].value)**2/data['2016']['unbiased'].flavor['dp0_os'].stdev**2
  chi2TagConstr += (pars_dict['dp1_os']-data['2016']['unbiased'].flavor['dp1_os'].value)**2/data['2016']['unbiased'].flavor['dp1_os'].stdev**2
  chi2TagConstr += (pars_dict['dp0_ss']-data['2016']['unbiased'].flavor['dp0_ss'].value)**2/data['2016']['unbiased'].flavor['dp0_ss'].stdev**2
  chi2TagConstr += (pars_dict['dp1_ss']-data['2016']['unbiased'].flavor['dp1_ss'].value)**2/data['2016']['unbiased'].flavor['dp1_ss'].stdev**2

  tagcvOS = np.matrix([pars_dict['p0_os'], pars_dict['p1_os']]) - tagConstr['pOS']
  tagcvSS = np.matrix([pars_dict['p0_ss'], pars_dict['p1_ss']]) - tagConstr['pSS']

  Y_OS = np.dot(tagcvOS, tagConstr['covOSInv'])
  '''
  print("Inputs:")
  print(Y_OS)
  print('----')
  print(tagcvOS.T)
  '''

  chi2TagConstr += np.dot(Y_OS, tagcvOS.T)

  '''
  print("Result:")
  print(np.dot(Y_OS, tagcvOS.T))
  '''

  Y_SS = np.dot(tagcvSS, tagConstr['covSSInv'])
  chi2TagConstr += np.dot(Y_SS, tagcvSS.T)

  chi2 = []
  for y, dy in data.items():
    trigCats = [dy['biased'],dy['unbiased']]
    for dt in trigCats:

      wrapper_fcn(dt.input, dt.output, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  #**dt.flavor.valuesdict(),
                  tLL=dt.tLL, tUL=dt.tUL)

      chi2.append( -2.0 * (ristra.log(dt.output) * dt.weight).get() );

  chi2conc =  np.concatenate(chi2)
  #chi2conc = chi2conc + np.array(len(chi2conc)*[chi2TagConstr[0][0]/float(len(chi2conc))])

  chi2TagConstr = float(chi2TagConstr[0][0]/len(chi2conc))
  #for i in range(len(chi2conc)): chi2conc[i] += chi2TagConstr

  #print(chi2TagConstr)
  return chi2conc + chi2TagConstr#np.concatenate(chi2)

################################################################################



################################################################################
#%% Run and get the job done ###################################################

print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
                  verbose=False, timeit=True, tol=0.05, strategy=2)
print(result)

# Dump json file
result.params.dump(args['params'])
# Write latex table
with open(args['tables'], "w") as tex_file:
  tex_file.write( result.params.dump_latex(caption="Physics parameters.") )

################################################################################
