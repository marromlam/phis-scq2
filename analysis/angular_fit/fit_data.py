DESCRIPTION = """
    fit data
"""

__author__ = ['Marcos Romero Lamas']
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
badjanak.config['fast_integral'] = 1
badjanak.config['debug_evt'] = 774

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and
from utils.helpers import  version_guesser, timeacc_guesser, trigger_scissors

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

################################################################################





def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--samples', help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--csp', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--tables', help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--year', help='Year of data-taking')
  parser.add_argument('--version', help='Year of data-taking')
  parser.add_argument('--flag',help='Year of data-taking')
  parser.add_argument('--blind', default=0, help='Year of data-taking')
  return parser






################################################################################
################################################################################



#args = vars(argument_parser().parse_args(''))
#args = vars(argument_parser().parse_args())
#YEARS = [int(y) for y in args['year'].split(',')] # years are int
#VERSION = args['version']


args = vars(argument_parser().parse_args())
VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
YEARS = args['year'].split(',')
MODE = 'Bs2JpsiPhi'
FIT, ANGACC, TIMEACC = args['flag'].split('_')


# Prepare the cuts -----------------------------------------------------------
CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')

for k,v in args.items():
  print(f'{k}: {v}\n')

# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\nLoading samples\n{80*'='}\n")

# Lists of data variables to load and build arrays
real  = ['cosK','cosL','hphi','time','mHH','sigmat']
real += ['tagOSdec','tagSSdec', 'tagOSeta', 'tagSSeta']      # tagging

weight_rd='(sw)'


data = {}
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[y] = {}
  csp = Parameters.load(args['csp'].split(',')[i])
  mass = np.array(csp.build(csp, csp.find('mKK.*')))
  csp = csp.build(csp,csp.find('CSP.*'))
  flavor = Parameters.load(args['flavor_tagging'].split(',')[i])
  resolution = Parameters.load(args['time_resolution'].split(',')[i])
  badjanak.config['x_m'] = mass.tolist()
  for t in ['biased','unbiased']:
    tc = trigger_scissors(t, CUT)
    data[y][t] = Sample.from_root(args['samples'].split(',')[i], cuts=tc)
    data[y][t].name = f"Bs2JpsiPhi-{y}-{t}"
    data[y][t].csp = csp
    data[y][t].flavor = flavor
    data[y][t].resolution = resolution
    # Time acceptance
    c = Parameters.load(args[f'timeacc_{t}'].split(',')[i])
    knots = np.array(Parameters.build(c,c.fetch('k.*')))
    print(knots)
    badjanak.config['knots'] = knots.tolist()
    # Angular acceptance
    data[y][t].timeacc = Parameters.build(c,c.fetch('c.*'))
    w = Parameters.load(args[f'angacc_{t}'].split(',')[i])
    data[y][t].angacc = Parameters.build(w,w.fetch('w.*'))
    print(data[y][t])
    # Normalize sWeights per bin
    sw = np.zeros_like(data[y][t].df['sw'])
    for l,h in zip(mass[:-1],mass[1:]):
      pos = data[y][t].df.eval(f'mHH>={l} & mHH<{h}')
      this_sw = data[y][t].df.eval(f'sw*(mHH>={l} & mHH<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
    data[y][t].df['sWeight'] = sw
    print(sw)
    data[y][t].allocate(input=real,weight='sWeight',output='0*time')

# Compile the kernel
#    so if knots change when importing parameters, the kernel is compiled
badjanak.get_kernels(True)

# Prepare parameters
SWAVE = True
if 'Pwave' in FIT:
  SWAVE = False
DGZERO = False
if 'DGzero' in FIT:
  DGZERO = True
POLDEP = False
if 'Poldep' in FIT:
  POLDEP = True
BLIND = args['blind']

pars = Parameters()
list_of_parameters = [#
# S wave fractions
Parameter(name='fSlon1', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{1}'),
Parameter(name='fSlon2', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{2}'),
Parameter(name='fSlon3', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{3}'),
Parameter(name='fSlon4', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{4}'),
Parameter(name='fSlon5', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{5}'),
Parameter(name='fSlon6', value=SWAVE*0.2, min=0.00, max=0.90,
          free=SWAVE, latex=r'f_S^{6}'),
# P wave fractions
Parameter(name="fPlon", value=0.5241, min=0.4, max=0.6,
          free=True, latex=r'f_0'),
Parameter(name="fPper", value=0.25, min=0.1, max=0.3,
          free=True, latex=r'f_{\perp}'),
# Weak phases
Parameter(name="pSlon", value= 0.00, min=-3.0, max=3.0,
          free=POLDEP, latex=r"\phi_S - \phi_0",
          blindstr="BsPhisSDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
Parameter(name="pPlon", value=-0.03, min=-3.0, max=3.0,
          free=True, latex=r"\phi_0",
          blindstr="BsPhiszeroFullRun2" if POLDEP else "BsPhisFullRun2",
          blind=BLIND, blindscale=2.0 if POLDEP else 1.0, blindengine="root"),
Parameter(name="pPpar", value= 0.00, min=-3.0, max=3.0,
          free=POLDEP, latex=r"\phi_{\parallel} - \phi_0",
          blindstr="BsPhisparaDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
Parameter(name="pPper", value= 0.00, min=-3.0, max=3.0,
          free=POLDEP, blindstr="BsPhisperpDelFullRun2", blind=BLIND, 
          blindscale=2.0, blindengine="root", latex=r"\phi_{\perp} - \phi_0"),
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
Parameter(name="lSlon", value=1., min=0.4, max=1.6,
          free=POLDEP, latex="\lambda_S/\lambda_0"),
Parameter(name="lPlon", value=1., min=0.4, max=1.6,
          free=True,  latex="\lambda_0"),
Parameter(name="lPpar", value=1., min=0.4, max=1.6,
          free=POLDEP, latex="\lambda_{\parallel}/\lambda_0"),
Parameter(name="lPper", value=1., min=0.4, max=1.6,
          free=POLDEP, latex="\lambda_{\perp}/\lambda_0"),
# lifetime parameters
Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
          free=False, latex=r"\Gamma_d"),
Parameter(name="DGs", value= (1-DGZERO)*0.1, min= 0.0, max= 1.7,
          free=1-DGZERO,  latex=r"\Delta\Gamma_s",
          blindstr="BsDGsFullRun2", 
          blind=BLIND, blindscale=1.0, blindengine="root"),
Parameter(name="DGsd", value= 0.03*0,   min=-0.1, max= 0.1, 
          free=True, latex=r"\Gamma_s - \Gamma_d"),
Parameter(name="DM", value=17.757,   min=15.0, max=20.0, 
          free=True, latex=r"\Delta m"),
Parameter("eta_os", value = data[str(YEARS[0])]['unbiased'].flavor['eta_os'].value,
          free = False),
Parameter("eta_ss", value = data[str(YEARS[0])]['unbiased'].flavor['eta_ss'].value, free = False),
Parameter("p0_os",  value = data[str(YEARS[0])]['unbiased'].flavor['p0_os'].value,  free = True, min =  0.0, max = 1.0, latex = "p^{\rm OS}_{0}"),
Parameter("p1_os",  value = data[str(YEARS[0])]['unbiased'].flavor['p1_os'].value,  free = True, min =  0.5, max = 1.5, latex = "p^{\rm OS}_{1}"),
Parameter("p0_ss",  value = data[str(YEARS[0])]['unbiased'].flavor['p0_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{0}"),
Parameter("p1_ss",  value = data[str(YEARS[0])]['unbiased'].flavor['p1_ss'].value,  free = True, min =  0.0, max = 2.0, latex = "p^{\rm SS}_{1}"),
Parameter("dp0_os", value = data[str(YEARS[0])]['unbiased'].flavor['dp0_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{0}"),
Parameter("dp1_os", value = data[str(YEARS[0])]['unbiased'].flavor['dp1_os'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm OS}_{1}"),
Parameter("dp0_ss", value = data[str(YEARS[0])]['unbiased'].flavor['dp0_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{0}"),
Parameter("dp1_ss", value = data[str(YEARS[0])]['unbiased'].flavor['dp1_ss'].value, free = True, min = -0.1, max = 0.1, latex = "\Delta p^{\rm SS}_{1}"),
]

pars.add(*list_of_parameters);
print(pars)



#@profile
def wrapper_fcn(input, output, **pars):
  p = badjanak.cross_rate_parser_new(**pars)
  badjanak.delta_gamma5( input, output,
                         use_fk=1, use_angacc = 1, use_timeacc = 1,
                         use_timeoffset = 0, set_tagging = 1, use_timeres = 1,
                         BLOCK_SIZE=256, **p)
# test here crap
# wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
# wrapper_fcn(data[str(YEARS[0])]['unbiased'].input,data[str(YEARS[0])]['unbiased'].output,**pars.valuesdict(),**data[str(YEARS[0])]['unbiased'].timeacc.valuesdict(),**data[str(YEARS[0])]['unbiased'].angacc.valuesdict())
#
# wrapper_fcn(data['2016']['biased'].input,data['2016']['biased'].output,**pars.valuesdict(),**data['2016']['biased'].timeacc.valuesdict(),**data['2016']['biased'].angacc.valuesdict())
# wrapper_fcn(data[str(YEARS[0])]['unbiased'].input,data[str(YEARS[0])]['unbiased'].output,**pars.valuesdict(),**data[str(YEARS[0])]['unbiased'].timeacc.valuesdict(),**data[str(YEARS[0])]['unbiased'].angacc.valuesdict())
# exit()

#Calculate tagging constraints - currently using one value for all years only!!!
def taggingConstraints(data):
  rhoOS = data[str(YEARS[0])]['unbiased'].flavor['rho01_os'].value
  rhoSS = data[str(YEARS[0])]['unbiased'].flavor['rho01_ss'].value

  pOS = np.matrix([data[str(YEARS[0])]['unbiased'].flavor['p0_os'].value,
                  data[str(YEARS[0])]['unbiased'].flavor['p1_os'].value])
  pSS = np.matrix([data[str(YEARS[0])]['unbiased'].flavor['p0_ss'].value,
                  data[str(YEARS[0])]['unbiased'].flavor['p1_ss'].value])

  p0OS_err = data[str(YEARS[0])]['unbiased'].flavor['p0_os'].stdev
  p1OS_err = data[str(YEARS[0])]['unbiased'].flavor['p1_os'].stdev
  p0SS_err = data[str(YEARS[0])]['unbiased'].flavor['p0_ss'].stdev
  p1SS_err = data[str(YEARS[0])]['unbiased'].flavor['p1_ss'].stdev

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

  chi2TagConstr += (pars_dict['dp0_os']-data[str(YEARS[0])]['unbiased'].flavor['dp0_os'].value)**2/data[str(YEARS[0])]['unbiased'].flavor['dp0_os'].stdev**2
  chi2TagConstr += (pars_dict['dp1_os']-data[str(YEARS[0])]['unbiased'].flavor['dp1_os'].value)**2/data[str(YEARS[0])]['unbiased'].flavor['dp1_os'].stdev**2
  chi2TagConstr += (pars_dict['dp0_ss']-data[str(YEARS[0])]['unbiased'].flavor['dp0_ss'].value)**2/data[str(YEARS[0])]['unbiased'].flavor['dp0_ss'].stdev**2
  chi2TagConstr += (pars_dict['dp1_ss']-data[str(YEARS[0])]['unbiased'].flavor['dp1_ss'].value)**2/data[str(YEARS[0])]['unbiased'].flavor['dp1_ss'].stdev**2

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
    for t, dt in dy.items():
      badjanak.delta_gamma5_data(
          dt.input, dt.output, **pars_dict,
          **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
          **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
          #**dt.flavor.valuesdict(),
          tLL=tLL, tUL=tUL,
          use_fk=1, use_angacc=1, use_timeacc=1,
          use_timeoffset=0, set_tagging=1, use_timeres=1,
          BLOCK_SIZE=256
      )
      chi2.append( -2.0 * (ristra.log(dt.output) * dt.weight).get() );

  chi2conc =  np.concatenate(chi2)
  #chi2conc = chi2conc + np.array(len(chi2conc)*[chi2TagConstr[0][0]/float(len(chi2conc))])

  chi2TagConstr = float(chi2TagConstr[0][0]/len(chi2conc))
  #for i in range(len(chi2conc)): chi2conc[i] += chi2TagConstr

  #print(chi2TagConstr)
  #print( np.nan_to_num(chi2conc + chi2TagConstr, 0, 100, 100).sum() )
  return chi2conc + chi2TagConstr #np.concatenate(chi2)









################################################################################


# print csp factors
lb = [data[y]['biased'].csp.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
print(f"\nCSP factors\n{80*'='}")
for l in zip(*lb):
  print(*l, sep="| ")

# print flavor tagging parameters
lb = [data[y]['biased'].flavor.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
print(f"\nFlavor tagging parameters\n{80*'='}")
for l in zip(*lb):
  print(*l, sep="| ")

# print time resolution
lb = [data[y]['biased'].resolution.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
print(f"\nResolution parameters\n{80*'='}")
for l in zip(*lb):
  print(*l, sep="| ")

# print time acceptances
lb = [data[y]['biased'].timeacc.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
lu = [data[y]['unbiased'].timeacc.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
print(f"\nBiased time acceptance\n{80*'='}")
for l in zip(*lb):
  print(*l, sep="| ")

print(f"\nUnbiased time acceptance\n{80*'='}")
for l in zip(*lu):
  print(*l, sep="| ")

# print angular acceptance
lb = [data[y]['biased'].angacc.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
lu = [data[y]['unbiased'].angacc.__str__(
    ['value']).splitlines() for i, y in enumerate(YEARS)]
print(f"\nBiased angular acceptance\n{80*'='}")
for l in zip(*lb):
  print(*l, sep="| ")
print(f"\nUnbiased angular acceptance\n{80*'='}")
for l in zip(*lu):
  print(*l, sep="| ")
print(f"\n")









################################################################################
#%% Run and get the job done ###################################################

print(f"\n{80*'='}\n", "Simultaneous minimization procedure", f"\n{80*'='}\n")
result = optimize(fcn_data, method='Nelder', params=pars, fcn_kwgs={'data':data},
                  verbose=False, timeit=True, tol=0.1, strategy=2)



for p in  ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DGs',
                'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
  if args['year'] == '2015,2016':
    #print(f"{p:>12} : {result.params[p].value:+.4f}  {result.params[p]._getval(False):+.4f}")
    print(f"{p:>12} : {result.params[p]._getval(False):+.4f} +/- {result.params[p].stdev:+.4f}")
  else:
    print(f"{p:>12} : {result.params[p].value:+.4f} +/- {result.params[p].stdev:+.4f}")

print(result)



# Dump json file
result.params.dump(args['params'])
# Write latex table
with open(args['tables'], "w") as tex_file:
  tex_file.write( result.params.dump_latex(caption="Physics parameters.") )
