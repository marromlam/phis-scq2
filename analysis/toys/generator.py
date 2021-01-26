DESCRIPTION = """
    This script generated a Toy file
"""

__author__ = ['Marcos Romero Lamas']
__email__  = ['mromerol@cern.ch']
__all__ = []



################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import pandas as pd
import uproot
import os
import hjson

from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import Sample, Parameters, ristra

# get bsjpsikk and compile it with corresponding flags
import badjanak
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 1
badjanak.config['debug_evt'] = 774

# import some phis-scq utils
from utils.strings import cuts_and
from utils.helpers import  version_guesser, trigger_scissors

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
  return parser








args = vars(argument_parser().parse_args())
VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
YEAR = args['year']
MODE = 'TOY_Bs2JpsiPhi'

# Prepare the cuts -----------------------------------------------------------
CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')




# % Load sample and parameters -------------------------------------------------
print(f"\n{80*'='}\nLoading samples and gather information\n{80*'='}\n")

data = {}
sample = Sample.from_root(args['sample'], cuts=CUT)
print(sample)
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
  for bin in range(0,len(mKK)-1):
    data[t][bin] = {}
    ml = mKK[bin]; mh = mKK[bin+1]
    noe = sample.cut( cuts_and(ccut,f'mHH>={ml} & mHH<{mh}') ).shape[0]
    data[t][bin]['output'] = ristra.allocate(np.float64(noe*[10*[0.5*(ml+mh)]]))
    data[t][bin]['csp'] = csp
    data[t][bin]['flavor'] = flavor
    data[t][bin]['resolution'] = resolution
    data[t][bin]['timeacc'] = Parameters.build(tacc,tacc.fetch('c.*'))
    data[t][bin]['angacc'] = Parameters.build(aacc,aacc.fetch('w.*'))
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



# Printout information ---------------------------------------------------------
print(f"\n{80*'='}\nGeneration\n{80*'='}\n")
for t, trigger in data.items():
  for b, bin in trigger.items():
    print(f"Generating {bin['output'].shape[0]:>6} events for",
          f"{YEAR}-{t:>8} at {b+1:>2} mass bin")
    pars  = bin['csp'] + bin['flavor'] + bin['resolution']
    pars += bin['timeacc'] + bin['angacc'] + bin['params']
    p = badjanak.parser_rateBs(**pars.valuesdict(), tLL=tLL, tUL=tUL)
    # for k,v in p.items():
    #   print(f"{k:>20}: {v}")
    badjanak.dG5toys(bin['output'], **p,
                     use_angacc=0, use_timeacc=0, use_timeres=1,
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

