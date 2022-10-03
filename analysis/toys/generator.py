from analysis.toys.angacc_generator import randomize_angacc
from analysis.toys.timeacc_generator import randomize_timeacc
from utils.helpers import version_guesser, trigger_scissors
from utils.strings import cuts_and
import badjanak
from ipanema import Sample, Parameters, ristra
from ipanema import initialize
import hjson
import os
import uproot3 as uproot
import pandas as pd
import numpy as np
import argparse
from warnings import simplefilter
DESCRIPTION = """
    This script generated a Toy file
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


################################################################################
# %% Modules ###################################################################

simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings


# initialize(os.environ['IPANEMA_BACKEND'],1)
initialize('cuda', 1)

# get bsjpsikk and compile it with corresponding flags
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 1
badjanak.config['debug_evt'] = 0

# import some phis-scq utils

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
CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

print(args['randomize_timeacc'], args['randomize_angacc'])
RANDOMIZE_TIMEACC = True if args['randomize_timeacc'] == 'True' else False
RANDOMIZE_ANGACC = True if args['randomize_angacc'] == 'True' else False
print(RANDOMIZE_TIMEACC, RANDOMIZE_ANGACC)

# % Load sample and parameters -------------------------------------------------
print(f"\n{80*'='}\nLoading samples and gather information\n{80*'='}\n")

data = {}
sample = Sample.from_root(args['sample'], cuts=CUT)
# print(sample)
csp = Parameters.load(args['csp_factors'])
mKK = np.array(csp.build(csp, csp.find('mKK.*')))
csp = csp.build(csp, csp.find('CSP.*'))
flavor = Parameters.load(args['flavor_tagging'])
resolution = Parameters.load(args['time_resolution'])
for t, T in zip(['biased', 'unbiased'], [0, 1]):
  data[t] = {}
  ccut = trigger_scissors(t, CUT)
  tacc = Parameters.load(args[f'timeacc_{t}'])
  aacc = Parameters.load(args[f'angacc_{t}'])
  knots = np.array(Parameters.build(tacc, tacc.fetch('k.*')))
  if RANDOMIZE_TIMEACC:
    _arr = np.array(Parameters.build(tacc, tacc.fetch('c.*')))
    print(f"Time acc. randomizer from {_arr}")
    tacc = randomize_timeacc(tacc)
    _arr = np.array(Parameters.build(tacc, tacc.fetch('c.*')))
    print(f"                     to   {_arr}")
  if RANDOMIZE_ANGACC:
    aacc = randomize_timeacc(aacc)
  for bin in range(0, len(mKK) - 1):
    data[t][bin] = {}
    ml = mKK[bin]
    mh = mKK[bin + 1]
    noe = sample.cut(cuts_and(ccut, f'mHH>={ml} & mHH<{mh}')).eval('sw')
    noe = noe * np.sum(noe) / np.sum(noe * noe)
    noe = int(np.sum(noe.values))
    # print(noe)
    data[t][bin]['output'] = ristra.allocate(np.float64(noe * [10 * [0.5 * (ml + mh)]]))
    data[t][bin]['csp'] = csp
    data[t][bin]['flavor'] = flavor
    data[t][bin]['resolution'] = resolution
    data[t][bin]['timeacc'] = Parameters.build(tacc, tacc.fetch('c.*'))
    data[t][bin]['angacc'] = Parameters.build(aacc, aacc)
    data[t][bin]['params'] = Parameters.load(args['fitted_params'])
# Just recompile the kernel attenting to the gathered information
badjanak.config['knots'] = knots.tolist()
badjanak.config['mHH'] = mKK.tolist()
badjanak.get_kernels()


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


# Printout information --------------------------------------------------------
print(f"\n{80*'='}\nGeneration\n{80*'='}\n")
for t, trigger in data.items():
  for b, bin in trigger.items():
    print(f"Generating {bin['output'].shape[0]:>6} events for",
          f"{YEAR}-{t:>8} at {b+1:>2} mass bin")
    pars = bin['csp'] + bin['flavor'] + bin['resolution']
    #pars += bin['timeacc'] + bin['params']
    pars += bin['timeacc'] + bin['angacc'] + bin['params']
    p = badjanak.parser_rateBs(**pars.valuesdict(False), tLL=tLL, tUL=tUL)
    print(p)
    #p['angacc'] = ristra.allocate(np.array(bin['angacc']))
    # for k,v in p.items():
    #   print(f"{k:>20}: {v}")
    # print(p['angacc'])
    badjanak.dG5toys(bin['output'], **p,
                     use_angacc=1, use_timeacc=1, use_timeres=1,
                     set_tagging=1, use_timeoffset=0,
                     seed=int(1e10 * np.random.rand()))
    genarr = ristra.get(bin['output'])
    hlt1b = np.ones_like(genarr[:, 0])
    gendic = {
        'cosK': genarr[:, 0],
        'cosL': genarr[:, 1],
        'hphi': genarr[:, 2],
        'time': genarr[:, 3],
        'gencosK': genarr[:, 0],
        'gencosL': genarr[:, 1],
        'genhphi': genarr[:, 2],
        'gentime': genarr[:, 3],
        'mHH': genarr[:, 4],
        'sigmat': genarr[:, 5],
        'idB': genarr[:, 6],
        'genidB': genarr[:, 7],
        'tagOSdec': genarr[:, 6],
        'tagSSdec': genarr[:, 7],
        'tagOSeta': genarr[:, 8],
        'tagSSeta': genarr[:, 9],
        'polWeight': np.ones_like(genarr[:, 0]),
        'sw': np.ones_like(genarr[:, 0]),
        'gb_weights': np.ones_like(genarr[:, 0]),
        'hlt1b': hlt1b if t == 'biased' else 0 * hlt1b
    }
    # exit()
    bin['df'] = pd.DataFrame.from_dict(gendic)

# Printout information ---------------------------------------------------------
print(f"\n{80*'='}\nSave tuple\n{80*'='}\n")
dfl = [data[t][b]['df'] for t in data for b in data[t]]

df = pd.concat(dfl)
print(df)
df = df.reset_index(drop=True)
print(df)

with uproot.recreate(args['output']) as fp:
  fp['DecayTree'] = uproot.newtree({var: 'float64' for var in df})
  fp['DecayTree'].extend(df.to_dict(orient='list'))
fp.close()
