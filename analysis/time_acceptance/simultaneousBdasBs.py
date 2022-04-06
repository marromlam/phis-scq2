DESCRIPTION = """
    Computes angular acceptance coefficients using half BdMC sample as BsMCdG0
    and half BdRD sample as BdRD.
"""

__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import hjson

# load ipanema
from ipanema import initialize, plotting
from ipanema import ristra, Parameters, optimize, Sample, plot_conf2d, Optimizer

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and
from utils.helpers import version_guesser, timeacc_guesser
from utils.helpers import swnorm, trigger_scissors


import config
# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
tLL = config.general['tLL']
tUL = config.general['tUL']


# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Different flag to ... ')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--minimizer', default='minuit', help='Different flag to ... ')
  #p.add_argument('--contour', help='Different flag to ... ')
  return p

if __name__ != '__main__':
  import badjanak

################################################################################



################################################################################
# Run and get the job done #####################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  TRIGGER = args['trigger']
  MODE = 'Bd2JpsiKstar'
  TIMEACC = timeacc_guesser(args['timeacc'])
  MINER = args['minimizer']

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else 1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = '' #bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  splitter = '(evtN%2)==0' # this is Bd as Bs
  if TIMEACC['cuts'] == 'mKstar':
    splitter = cuts_and(splitter, f"mHH>890")
  elif TIMEACC['cuts'] == 'alpha':
    splitter = cuts_and(splitter, f"alpha<0.025")
  elif TIMEACC['cuts'] == 'deltat':
    splitter = cuts_and(splitter, f"sigmat<0.04")

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'minimizer':>15}: {MINER:50}")
  print(f"{'splitter':>15}: {splitter:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = all_knots[str(TIMEACC['nknots'])]
  kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  sw = f'sw_{VAR}' if VAR else 'sw'



  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\nLoading categories\n{80*'='}\n")

  cats = {}
  for i,m in enumerate(['MC_Bd2JpsiKstar','MC_Bd2JpsiKstar','Bd2JpsiKstar']):
    if m=='MC_Bd2JpsiKstar':
      if TIMEACC['corr']:
        weight = f'kinWeight*polWeight*pdfWeight*sw'
      else:
        weight = f'sw'
      mode = 'BdMC'; c = 'b'
    elif m=='Bd2JpsiKstar':
      weight = f'sw'
      mode = 'BdRD'; c = 'c'
    print(weight)
    
    F = 1 if i%2==0 else 0
    f = 'A' if F else 'B'
    f = 'B' if m == 'Bd2JpsiKstar' else f
    F = 0 if m == 'Bd2JpsiKstar' else F
    F = f'({splitter}) == {F}'
    
    if not mode in cats:
      cats[mode] = {}

    # for f, F in zip(['A', 'B'], [f'({splitter}) == 1', f'({splitter}) == 0']):
    cats[mode][f] = Sample.from_root(samples[i], share=SHARE)
    cats[mode][f].name = f"{mode}-{f}"
    cats[mode][f].chop( cuts_and(CUT,F) )
    print(cats[mode][f])

    # allocate arrays
    cats[mode][f].allocate(time='time', lkhd='0*time')
    cats[mode][f].allocate(weight=weight)
    cats[mode][f].weight = swnorm(cats[mode][f].weight)

    # Add knots
    cats[mode][f].knots = Parameters()
    cats[mode][f].knots.add(*[
                  {'name':f'k{j}', 'value':v, 'latex':f'k_{j}', 'free':False}
                  for j,v in enumerate(knots[:-1])
                ])
    cats[mode][f].knots.add({'name':f'tLL', 'value':knots[0],
                          'latex':'t_{ll}', 'free':False})
    cats[mode][f].knots.add({'name':f'tUL', 'value':knots[-1],
                          'latex':'t_{ul}', 'free':False})

    # Add coeffs parameters
    cats[mode][f].params = Parameters()
    cats[mode][f].params.add(*[
                  {'name':f'{c}{f}{j}{TRIGGER[0]}', 'value':1.0,
                   'latex': f'{c}_{{{f},{j}}}^{TRIGGER[0]}',
                   'free':True if j>0 else False, #'min':0.10, 'max':5.0,
                  } for j in range(len(knots[:-1])+2)
    ])
    cats[mode][f].params.add({'name':f'gamma_{f}{c}',
                            'value':Gdvalue+resolutions[m]['DGsd'],
                            'latex':f'\Gamma_{{{f}{c}}}', 'free':False})
    cats[mode][f].params.add({'name':f'mu_{f}{c}',
                            'value':resolutions[m]['mu'],
                            'latex':f'\mu_{{{f}{c}}}', 'free':False})
    cats[mode][f].params.add({'name':f'sigma_{f}{c}',
                            'value':resolutions[m]['sigma'],
                            'latex':f'\sigma_{{{f}{c}}}', 'free':False})
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

  #del cats['BdRD']['A'] # remove this one


  # Configure kernel
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)

  # Time to fit acceptance
  print(f"\n{80*'='}\nSimultaneous minimization procedure\n{80*'='}\n")

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

  if MINER.lower() in ("minuit","minos"):
    result = optimize(fcn_call=fcns.saxsbxscxerf,
                      params=fcn_pars,
                      fcn_kwgs=fcn_kwgs,
                      method=MINER,
                      verbose=False, timeit=True, strategy=1, tol=0.05);
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    result = optimize(fcn_call=fcns.saxsbxscxerf,
                      params=fcn_pars,
                      fcn_kwgs=fcn_kwgs,
                      method=MINER,
                      verbose=False, timeit=True)

  print(result)

  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\nDumping parameters\n{80*'='}\n")

  for cat in [c for C in cats.values() for c in C.values()]:
    list_params = cat.params.find('(bA|bB|cB)(\d{1})(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])


    print(f"Dumping json parameters to {cat.pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cat.pars_path)



################################################################################
# that's all folks!
