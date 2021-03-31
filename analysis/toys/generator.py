DESCRIPTION = """
    This script generated a Toy file
"""

__author__ = ['Marcos Romero Lamas']
__email__  = ['mromerol@cern.ch']
__all__ = []



################################################################################
# %% Modules ###################################################################

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings

import argparse
import numpy as np
import pandas as pd
import uproot3 as uproot
import os
import hjson

from ipanema import initialize
#initialize(os.environ['IPANEMA_BACKEND'],1)
initialize('cuda',1)
from ipanema import Sample, Parameters, ristra

# get bsjpsikk and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 1
badjanak.config['debug_evt'] = 0

# import some phis-scq utils
from utils.strings import cuts_and
from utils.helpers import  version_guesser, trigger_scissors
from analysis.toys.timeacc_generator import randomize_timeacc   
from analysis.toys.angacc_generator import randomize_angacc   

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

################################################################################





def argument_parser():
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  parser.add_argument('--sample', help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--csp-factors', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--fitted-params', help='Bs2JpsiPhi MC sample')
  # output tuple
  parser.add_argument('--output', help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--year', help='Year of data-taking')
  parser.add_argument('--version', help='Year of data-taking')
  parser.add_argument('--randomize-timeacc', help='Year of data-taking')
  parser.add_argument('--randomize-angacc', help='Year of data-taking')
  return parser








args = vars(argument_parser().parse_args())
VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
YEAR = args['year']
MODE = 'TOY_Bs2JpsiPhi'

# Prepare the cuts -----------------------------------------------------------
CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')

print(args['randomize_timeacc'], args['randomize_angacc'])
RANDOMIZE_TIMEACC = True if args['randomize_timeacc']=='True' else False
RANDOMIZE_ANGACC = True if args['randomize_angacc']=='True' else False
print(RANDOMIZE_TIMEACC, RANDOMIZE_ANGACC)

# % Load sample and parameters -------------------------------------------------
print(f"\n{80*'='}\nLoading samples and gather information\n{80*'='}\n")

data = {}
sample = Sample.from_root(args['sample'], cuts=CUT)
#print(sample)
csp = Parameters.load(args['csp_factors'])
mKK = np.array(csp.build(csp,csp.find('mKK.*')))
csp = csp.build(csp,csp.find('CSP.*'))
flavor = Parameters.load(args['flavor_tagging'])
resolution = Parameters.load(args['time_resolution'])
for t, T in zip(['biased','unbiased'],[0,1]):
  data[t] = {}
  ccut = trigger_scissors(t,CUT)
  tacc = Parameters.load(args[f'timeacc_{t}'])
  aacc = Parameters.load(args[f'angacc_{t}'])
  knots = np.array(Parameters.build(tacc,tacc.fetch('k.*')))
  if RANDOMIZE_TIMEACC:
      _arr = np.array(Parameters.build(tacc,tacc.fetch('c.*')))
      print(f"Time acc. randomizer from {_arr}")
      tacc = randomize_timeacc(tacc)
      _arr = np.array(Parameters.build(tacc,tacc.fetch('c.*')))
      print(f"                     to   {_arr}")
  if RANDOMIZE_ANGACC:
      aacc = randomize_timeacc(aacc)
  for bin in range(0,len(mKK)-1):
    data[t][bin] = {}
    ml = mKK[bin]; mh = mKK[bin+1]
    noe = sample.cut( cuts_and(ccut,f'mHH>={ml} & mHH<{mh}') ).eval('sw')
    noe = noe * np.sum(noe) / np.sum(noe*noe)
    noe = int(np.sum(noe.values))
    #print(noe)
    data[t][bin]['output'] = ristra.allocate(np.float64(noe*[10*[0.5*(ml+mh)]]))
    data[t][bin]['csp'] = csp
    data[t][bin]['flavor'] = flavor
    data[t][bin]['resolution'] = resolution
    data[t][bin]['timeacc'] = Parameters.build(tacc,tacc.fetch('c.*'))
    data[t][bin]['angacc'] = Parameters.build(aacc,aacc)
    data[t][bin]['params'] = Parameters.load(args['fitted_params'])
# Just recompile the kernel attenting to the gathered information
badjanak.config['knots'] = knots.tolist()
badjanak.config['mHH'] = mKK.tolist()
badjanak.get_kernels(True)



# Printout information ---------------------------------------------------------
print(f"\n{80*'='}\nParameters used to generate the toy\n{80*'='}\n")

print('CSP factors')
print(data['biased'][0]['csp'])
print('Flavor tagging')
print(data['biased'][0]['flavor'])
print('Time resolution')
print(data['biased'][0]['resolution'])
print('Time acceptance biased')
print(data['biased'][0]['timeacc'])
print('Time acceptance unbiased')
print(data['unbiased'][0]['timeacc'])
print('Angular acceptance biased')
print(data['biased'][0]['angacc'])
print('Angular acceptance unbiased')
print(data['unbiased'][0]['angacc'])
print('Physics parameters')
print(data['unbiased'][0]['params'])


"""


G                  : +0.63189553
DG                 : +0.05305606
DM                 : +17.65194843
CSP                : +0.85000000
ASlon              : +0.81193878
APlon              : +0.42209199
APpar              : +0.28717885
APper              : +0.28305843
pSlon              : +0.06029618
pPlon              : +0.06029618
pPpar              : +0.06029618
pPper              : +0.06029618
dSlon              : +4.70008991
dPlon              : +4.70008850
dPper              : +3.24543821
dPpar              : +3.58484457
lSlon              : +1.26486074
lPlon              : +1.26486074
lPper              : +1.26486074
lPpar              : +1.26486074
tLL                : +0.30000000
sigma_offset       : +0.00000000
sigma_slope        : +0.76837129
sigma_curvature    : +0.01165766

"""

"""
p = {'mass_bins': 6,
        'CSP': ristra.allocate(np.float64([0.85  , 0.8722, 0.8213, 0.8648, 0.9398, 0.9742])),
        'DG': 0.05305606,
        'DM': 17.65194843,
        'G': 0.63189553,
        'ASlon': ristra.allocate(np.float64([0.81193878, 0.20448014, 0.03363752, 0.07614789, 0.2978678, 0.42100032])),
        'APlon': ristra.allocate(np.float64([0.42209199, 0.70665867, 0.72150363, 0.71981611, 0.6891425, 0.654818])),
        'APper': ristra.allocate(np.float64([0.28305843, 0.48226981, 0.49240097, 0.4912493, 0.47031563, 0.44689036])),
        'APpar': ristra.allocate(np.float64([0.28717885, 0.47564401, 0.48563599, 0.48450014, 0.46385407, 0.44075064])),
        'dPlon': 0,
        'dPper': 3.24543821,
        'dPpar': 3.58484457,
        'dSlon': ristra.allocate(np.float64([ 4.15410839,  3.95558913,  3.38647787, -0.27749857,  0.22512674, 1.55636049])),
        'pPlon': +0.06029618,
        'pSlon': +0.06029618,
        'pPpar': +0.06029618,
        'pPper': +0.06029618,
        'lPlon': 1.26486074,
        'lSlon': 1.26486074,
        'lPpar': 1.26486074,
        'lPper': 1.26486074,
        'tLL': 0.3, 'tUL': 15,
        'cosKLL': -1.0, 'cosKUL': 1.0,
        'cosLLL': -1.0, 'cosLUL': 1.0,
        'hphiLL': -3.141592653589793, 'hphiUL': 3.141592653589793,
        'sigma_offset': 0.01165766,
        'sigma_slope': 0.76837129,
        'sigma_curvature': 0.0,
        'mu': 0.0,
        'eta_os': 0.3602,
        'eta_ss': 0.4167,
        'p0_os': 0.389,
        'p1_os': 0.8486,
        'p2_os': 0.0,
        'p0_ss': 0.4325,
        'p1_ss': 0.9241,
        'p2_ss': 0.0,
        'dp0_os': 0.009,
        'dp1_os': 0.0143,
        'dp2_os': 0.0,
        'dp0_ss': 0,
        'dp1_ss': 0,
        'dp2_ss': 0.0,
        'timeacc': ristra.allocate(np.float64([[-3.64255079e-01,  6.39857578e+00, -6.92783207e+00, 2.52545305e+00],
                          [ 1.52970475e+00,  1.54752165e-01, -6.64874429e-02, 1.21400036e-02],
                          [ 1.63678065e+00, -9.13951947e-03,  1.71307636e-02,-2.08077984e-03],
[+2.291307,    -0.332478,       +0.104473,       -0.008989],
[+2.291307,    -0.332478,       +0.104473,       -0.008989],
[+2.291307,    -0.332478,       +0.104473,       -0.008989],
                          [ 3.28296581e+00, -2.06415276e-01,  0.00000000e+00, 0.00000000e+00]])),
        'angacc': ristra.allocate(np.float64([ 1.000000e+00,  1.036546e+00,  1.036646e+00, -1.026400e-02,  3.754000e-03,  3.091000e-03,  1.020005e+00, -4.250000e-03, -8.620000e-04, -2.872100e-02]
))
}
"""

# Printout information ---------------------------------------------------------
print(f"\n{80*'='}\nGeneration\n{80*'='}\n")
for t, trigger in data.items():
  for b, bin in trigger.items():
    print(f"Generating {bin['output'].shape[0]:>6} events for",
          f"{YEAR}-{t:>8} at {b+1:>2} mass bin")
    pars  = bin['csp'] + bin['flavor'] + bin['resolution']
    #pars += bin['timeacc'] + bin['params']
    pars += bin['timeacc'] + bin['angacc'] + bin['params']
    p = badjanak.parser_rateBs(**pars.valuesdict(False), tLL=tLL, tUL=tUL)
    print(p)
    #p['angacc'] = ristra.allocate(np.array(bin['angacc']))
    # for k,v in p.items():
    #   print(f"{k:>20}: {v}")
    #print(p['angacc'])
    badjanak.dG5toys(bin['output'], **p,
                     use_angacc=1, use_timeacc=1, use_timeres=1,
                     set_tagging=1, use_timeoffset=0,
                     seed=int(1e10*np.random.rand()) )
    genarr = ristra.get(bin['output'])
    hlt1b = np.ones_like(genarr[:,0])
    gendic = {
      'cosK'        :  genarr[:,0],
      'cosL'        :  genarr[:,1],
      'hphi'        :  genarr[:,2],
      'time'        :  genarr[:,3],
      'gencosK'     :  genarr[:,0],
      'gencosL'     :  genarr[:,1],
      'genhphi'     :  genarr[:,2],
      'gentime'     :  genarr[:,3],
      'mHH'         :  genarr[:,4],
      'sigmat'      :  genarr[:,5],
      'idB'         :  genarr[:,6],
      'genidB'      :  genarr[:,7],
      'tagOSdec'    :  genarr[:,6],
      'tagSSdec'    :  genarr[:,7],
      'tagOSeta'    :  genarr[:,8],
      'tagSSeta'    :  genarr[:,9],
      'polWeight'   :  np.ones_like(genarr[:,0]),
      'sw'          :  np.ones_like(genarr[:,0]),
      'gb_weights'  :  np.ones_like(genarr[:,0]),
      'hlt1b'       :  hlt1b if t=='biased' else 0*hlt1b
    }
    #exit()
    bin['df'] = pd.DataFrame.from_dict(gendic)

# Printout information ---------------------------------------------------------
print(f"\n{80*'='}\nSave tuple\n{80*'='}\n")
dfl = [data[t][b]['df'] for t in data for b in data[t]]

df = pd.concat(dfl)
print(df)
df = df.reset_index(drop=True)
print(df)

with uproot.recreate(args['output']) as fp:
  fp['DecayTree'] = uproot.newtree({var:'float64' for var in df})
  fp['DecayTree'].extend(df.to_dict(orient='list'))
fp.close()

