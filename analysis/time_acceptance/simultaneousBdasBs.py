__all__ = ['wildcards_parser']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


# Modules {{{

import argparse
from typing import Optional, Any
# import os
# import hjson

# load ipanema
from ipanema import initialize
from ipanema import Parameters, optimize, Sample, ristra

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and, printsec, printsubsec
from utils.helpers import version_guesser, timeacc_guesser
from utils.helpers import swnorm, trigger_scissors
from trash_can.knot_generator import create_time_bins
from analysis.reweightings.kinematic_weighting import reweight

import config
# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
import numpy as np
import math
def arccos(x):
    return math.acos(x)

# Parse arguments for this script
def argument_parser():
  #p.add_argument('--contour', help='Different flag to ... ')
  return p

# }}}


def wildcards_parser(version:Optional[str]=None, year:Optional[str]=None,
                     mode:Optional[str]=None, timeacc:Optional[str]=None,
                     trigger: Optional[str]=None, angacc:Optional[str]=None) -> Any:
  """
  Fully parses all wildcards from snakemake.

  Parameters
  ----------
  version: str or None
  """
  ans = {
    "version": {} if version else None,
    "year": {} if year else None,
    "mode": {} if mode else None,
    "timeacc": {} if timeacc else None,
    "angacc": {} if angacc else None,
    # "version": {},
    "trigger": {} if trigger else None
  }

  if version:
    VERSION, SHARE, EVT, MAG, FULLCUT, _, _ = version_guesser(version)
    ans['version'] = {
      'name': VERSION,
      'share': SHARE,
      'evt': EVT,
      'magnet': MAG,
      'cut': FULLCUT
    }

  if timeacc:
    TIMEACC = timeacc_guesser(timeacc)
    if version:
      TIMEACC['use_upTime'] = TIMEACC['use_upTime'] | ('UT' in 'version')
      TIMEACC['use_lowTime'] = TIMEACC['use_lowTime'] | ('LT' in 'version')
    ans['timeacc'] = TIMEACC
    # Prepare the cuts
    if TIMEACC['use_transverse_time']:
      ans['timeacc']['time'] = 'timeT'
    else:
      ans['timeacc']['time'] = 'time'

    if TIMEACC['use_truetime']:
      ans['timeacc']['time'] = f"gen{ans['timeacc']['time']}"

    if TIMEACC['use_upTime']:
      tLL = config.general['upper_time_lower_limit']
    else:
      tLL = config.general['time_lower_limit']
    if TIMEACC['use_lowTime']:
      tUL = config.general['lower_time_upper_limit']
    else:
      tUL = config.general['time_upper_limit']

    knots = create_time_bins(int(TIMEACC['nknots']), tLL, tUL).tolist()
    tLL, tUL = knots[0], knots[-1]
    ans['timeacc']['time_lower_limit'] = tLL
    ans['timeacc']['time_upper_limit'] = tUL
    ans['timeacc']['knots'] = knots

  if trigger:
    ans['trigger'] = {
      "name": trigger,
      "abrv": trigger[0],
      "categories": ['biased', 'unbiased'] if trigger == 'combined' else [trigger],
      "cut": trigger_scissors(trigger)
    }

  if year:
    ans['year'] = {
      "name": str(year),
      "value": int(year),
      "categories": config.years[year],
    }

  if mode:
    ans['mode'] = {
      "original": mode,
      "name": mode,
    }

  print(ans)
  return ans


# command line interface {{{

if __name__ == '__main__':

  # Parse argulments {{{
  DESCRIPTION = """
  Computes angular acceptance coefficients using half BdMC sample as BsMCdG0
  and half BdRD sample as BdRD.
  """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--resolutions', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Different flag to ... ')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--minimizer', default='minuit', help='Different flag to ... ')
  args = vars(p.parse_args())

  # }}}

  this_minimizer = args['minimizer']

  this_conf = wildcards_parser(version=args['version'], year=args['year'],
                               trigger=args['trigger'], timeacc=args['timeacc'],
                               # mode=args['mode']
                               )

  # Get badjanak model and configure it
  initialize(config.user['backend'], 1)
  import analysis.time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = this_conf['trigger']['cut']
  tLL = this_conf['timeacc']['time_lower_limit']
  tUL = this_conf['timeacc']['time_upper_limit']
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  splitter = '(evtN%2)==0' # this is Bd as Bs
  if this_conf['timeacc']['cuts'] == 'mKstar':
    splitter = cuts_and(splitter, f"mHH>890")
    splitter = f"mHH>890"
  elif this_conf['timeacc']['cuts'] == 'alpha':
    splitter = cuts_and(splitter, f"alpha<0.025")
    splitter = "alpha<0.025"
  elif this_conf['timeacc']['cuts'] == 'deltat':
    splitter = cuts_and(splitter, f"sigmat<0.04")

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  # print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  # print(f"{'trigger':>15}: {TRIGGER:50}")
  # print(f"{'cuts':>15}: {CUT:50}")
  # print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  # print(f"{'minimizer':>15}: {this_minimizer:50}")
  # print(f"{'splitter':>15}: {splitter:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')
  # oparams = [Parameters.load(i) for i in oparams]
  time_offset = args['resolutions'].split(',')
  time_offset = [Parameters.load(i) for i in time_offset]

  # }}}

  # Get data into categories {{{
  printsubsec(f"Loading categories")

  cats = {}
  for i,m in enumerate(['MC_Bd2JpsiKstar','MC_Bd2JpsiKstar','Bd2JpsiKstar']):
    if m == 'MC_Bd2JpsiKstar':
      if this_conf['timeacc']['corr']:
        # FIXME: we need to craete a kbdWeight for this test
        # weight = f'kbdWeight*polWeight*pdfWeight*sWeight'
        weight = f'sWeight'
      else:
        weight = f'sWeight'
      mode = 'BdMC'; c = 'b'
    elif m == 'Bd2JpsiKstar':
      weight = f'sWeight'
      mode = 'BdRD'; c = 'c'
    else:
      print("simulBdasBs only accepts Bd tuples")
      exit()
    print(weight)

    F = 1 if i%2==0 else 0
    f = 'A' if F else 'B'
    f = 'B' if m == 'Bd2JpsiKstar' else f
    F = 0 if m == 'Bd2JpsiKstar' else F
    F = f'({splitter}) == {F}'

    if not mode in cats:
      cats[mode] = {}

    # for f, F in zip(['A', 'B'], [f'({splitter}) == 1', f'({splitter}) == 0']):
    cats[mode][f] = Sample.from_root(samples[i], share=this_conf['version']['share'])
    tmp = Sample.from_root(samples[i].replace('.root', '_sWeight.root'), share=this_conf['version']['share']).df
    angle_str = "arccos((hplus_PX*hminus_PX+hplus_PY*hminus_PY+hplus_PZ*hminus_PZ)/(hplus_P*hminus_P))"
    angle_arr = tmp.eval(angle_str)
    cats[mode][f].df['alpha'] = angle_arr
    cats[mode][f].name = f"{mode}-{f}"
    cats[mode][f].chop( cuts_and(CUT,F) )
    print(cats[mode][f])
    print(cats[mode][f].df)

    # allocate arrays
    cats[mode][f].allocate(time='time', lkhd='0*time')
    cats[mode][f].allocate(weight=weight)
    cats[mode][f].weight = swnorm(cats[mode][f].weight)

    # Add knots and time limits
    cats[mode][f].knots = Parameters()
    for j, v in enumerate(this_conf['timeacc']['knots'][:-1]):
      cats[mode][f].knots.add({
                                'name':f'k{j}',
                                'value':v,
                                'latex':f'k_{j}',
                                'free':False
      })
    cats[mode][f].knots.add({'name':f'tLL', 'value':tLL,
                          'latex':'t_{ll}', 'free':False})
    cats[mode][f].knots.add({'name':f'tUL', 'value':tUL,
                          'latex':'t_{ul}', 'free':False})

    # Add coeffs parameters
    cats[mode][f].params = Parameters()
    for j in range(len(this_conf['timeacc']['knots'][:-1])+2):
      cats[mode][f].params.add({
        'name':f"{c}{f}{j}{this_conf['trigger']['abrv']}", "value":1.0,
        'latex': f"{c}_{{{f},{j}}}^{this_conf['trigger']['abrv']}",
        'free':True if j > 0 else False,  # first coeff is always 1
         #'min':0.10, 'max':5.0,
    })
    cats[mode][f].params.add({'name':f'gamma_{f}{c}',
                            'value':Gdvalue+resolutions[m]['DGsd'],
                            'latex':rf'\Gamma_{{{f}{c}}}', 'free':False})
    cats[mode][f].params.add({'name':f'mu_{f}{c}',
                            'value':resolutions[m]['mu'],
                            'latex':rf'\mu_{{{f}{c}}}', 'free':False})
    cats[mode][f].params.add({'name':f'sigma_{f}{c}',
                            'value':resolutions[m]['sigma'],
                            'latex':rf'\sigma_{{{f}{c}}}', 'free':False})
    #print(cats[mode][f].knots)
    #print(cats[mode][f].params)

    # Attach labels and paths
    cats[mode][f].label = mode_tex(mode)
    _i = len([k for K in cats.keys() for k in cats[K].keys()]) -1
    #_i = len(cats) + len(cats[mode]) - 2
    if _i in (0,1,2):
      cats[mode][f].pars_path = oparams[_i if _i<3 else 2]
    else:
      print('\t\tThis sample is NOT being used, only for check purposes!')
  
  print(cats)
  #del cats['BdRD']['A'] # remove this one

  if this_conf['timeacc']['corr']:
    print("Reweighting...")
    # kbdWeight = reweight(original=cats['BdMC']['A'].df[['pTB', 'pB', 'mHH']],
    #                      target=cats['BdRD']['B'].df[['pTB', 'pB', 'mHH']],
    #                      original_weight=cats['BdMC']['A'].weight.get(),
    #                      target_weight=cats['BdRD']['B'].weight.get())
    # cats['BdMC']['A'].weight = cats['BdMC']['A'].weight * ristra.allocate(kbdWeight)
    # cats['BdMC']['A'].weight = swnorm(cats['BdMC']['A'].weight)
    kbdWeight = reweight(original=cats['BdMC']['B'].df[['pTB', 'pB', 'mHH']],
                         target=cats['BdRD']['B'].df[['pTB', 'pB', 'mHH']],
                         original_weight=cats['BdMC']['B'].weight.get(),
                         target_weight=cats['BdRD']['B'].weight.get())
    cats['BdMC']['B'].weight = cats['BdMC']['B'].weight * ristra.allocate(kbdWeight)
    cats['BdMC']['B'].weight = swnorm(cats['BdMC']['B'].weight)
  # Configure kernel
  fcns.badjanak.config['knots'] = this_conf['timeacc']['knots'][:-1]
  fcns.badjanak.get_kernels()

  # Time to fit acceptance {{{

  printsubsec(f"Simultaneous minimization procedure")

  fcn_pars = cats['BdMC']['A'].params
  fcn_pars += cats['BdMC']['B'].params
  fcn_pars += cats['BdRD']['B'].params
  fcn_kwgs = {
    'data': [cats['BdMC']['A'].time,
             cats['BdMC']['B'].time,
             cats['BdRD']['B'].time
    ],
    'prob': [cats['BdMC']['A'].lkhd,
             cats['BdMC']['B'].lkhd,
             cats['BdRD']['B'].lkhd
    ],
    'weight': [cats['BdMC']['A'].weight,
               cats['BdMC']['B'].weight,
               cats['BdRD']['B'].weight
    ]
  }

  if this_minimizer.lower() in ("minuit", "minos"):
    result = optimize(fcn_call=fcns.saxsbxscxerf,
                      params=fcn_pars,
                      fcn_kwgs=fcn_kwgs,
                      method=this_minimizer,
                      verbose=False, timeit=True, strategy=1, tol=0.05);
  else:
    result = optimize(fcn_call=fcns.saxsbxscxerf,
                      params=fcn_pars,
                      fcn_kwgs=fcn_kwgs,
                      method=this_minimizer,
                      verbose=False, timeit=True)

  print(result)

  # }}}

  # Writing results {{{

  printsec(f"Dumping parameters")

  for cat in [c for C in cats.values() for c in C.values()]:
    list_params = cat.params.find(r'(bA|bB|cB)(\d{1})(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])


    print(f"Dumping json parameters to {cat.pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cat.pars_path)

  # }}}

# }}}

# vim: fdm=marker
# that's all folks!
