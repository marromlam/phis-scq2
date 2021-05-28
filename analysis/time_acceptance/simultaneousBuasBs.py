DESCRIPTION = """
    Computes angular acceptance coefficients using half BdMC sample as udG0
    and half BdRD sample as BdRD.
"""

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
from reweightings.kinematic_weighting import computekinWeight

# binned variables
bdconfig = hjson.load(open('config.json'))['time_acceptance_bdtconfig']
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
all_knots = hjson.load(open('config.json'))['time_acceptance_knots']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Different flag to ... ')
  p.add_argument('--timeacc', help='Different flag to ... ')
  #p.add_argument('--contour', help='Different flag to ... ')
  return p

if __name__ != '__main__':
  import badjanak

################################################################################



################################################################################
# Run and get the job done #####################################################
if __name__ == '__main__':

  # Parse arguments -----------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  TRIGGER = args['trigger']
  MODE = 'Bu2JpsiKplus'
  TIMEACC, NKNOTS, CORR, FLAT, LIFECUT, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else 1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # splitter = '(evtN%2)==0' # this is Bd as Bs
  # if LIFECUT == 'mKstar':
  #   splitter = cuts_and(splitter, f"mHH>890")
  # elif LIFECUT == 'alpha':
  #   splitter = cuts_and(splitter, f"alpha<0.025")
  # elif LIFECUT == 'deltat':
  #   splitter = cuts_and(splitter, f"sigmat<0.04")

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}")
  # print(f"{'splitter':>15}: {splitter:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')
  otables = args['tables'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = all_knots[str(NKNOTS)]
  kinWeight = f'Weight_{VAR}' if VAR else 'Weight'
  sw = f'sw_{VAR}' if VAR else 'sw'
  if CORR:
    # to be implemented
    0



  # Get data into categories --------------------------------------------------
  print(f"\n{80*'='}\nLoading categories\n{80*'='}\n")

  cats = {}
  for i, m in enumerate(['MC_Bu2JpsiKplus', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar']):
    # Correctly apply weight and name for diffent samples
    if m == 'MC_Bu2JpsiKplus':
      if CORR:
        weight = f'kin{kinWeight}*polWeight*{sw}'
      else:
        weight = f'polWeight*{sw}'
      mode = 'BuMC'; c = 'a'
    elif m == 'MC_Bd2JpsiKstar':
      if CORR:
        weight = f'kbu{kinWeight}*polWeight*pdfWeight*{sw}'
      else:
        weight = f'pdfWeight*polWeight*{sw}'
      mode = 'BdMC'; c = 'b'
    elif m == 'Bd2JpsiKstar':
      if CORR:
        weight = f'kbu{kinWeight}*{sw}'
      else:
        weight = f'{sw}'
      mode = 'BdRD'; c = 'c'

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time', lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)
    print(cats[mode])

    # Add knots
    cats[mode].knots = Parameters()
    cats[mode].knots.add(*[
        {'name': f'k{j}', 'value': v, 'latex': f'k_{j}', 'free': False}
        for j, v in enumerate(knots[:-1])
    ])
    cats[mode].knots.add({'name': f'tLL', 'value': tLL,
                          'latex': 't_{ll}', 'free': False})
    cats[mode].knots.add({'name': f'tUL', 'value': tUL,
                          'latex': 't_{ul}', 'free': False})

    # Add coeffs parameters
    cats[mode].params = Parameters()
    cats[mode].params.add(*[
        {'name': f'{c}{j}{TRIGGER[0]}', 'value': 1.0,
         'latex': f'{c}_{j}^{TRIGGER[0]}',
         'free': False if j == 0 else True,  # 'min':0.10, 'max':5.0
         } for j in range(len(knots[:-1])+2)
    ])
    cats[mode].params.add({'name': f'gamma_{c}',
                           'value': Gdvalue+resolutions[m]['DGsd'],
                           'latex': f'\Gamma_{c}', 'free': False})
    cats[mode].params.add({'name': f'mu_{c}',
                           'value': resolutions[m]['mu'],
                           'latex': f'\mu_{c}', 'free': False})
    cats[mode].params.add({'name': f'sigma_{c}',
                           'value': resolutions[m]['sigma'],
                           'latex': f'\sigma_{c}', 'free': False})

    # Attach labels and paths
    cats[mode].label = mode_tex(mode)
    cats[mode].pars_path = oparams[i]
    cats[mode].tabs_path = otables[i]



  # Configure kernel ----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)



  # Time to fit acceptance ----------------------------------------------------
  print(f"\n{80*'='}\nSimultaneous minimization procedure\n{80*'='}\n")
  fcn_call = fcns.saxsbxscxerf
  fcn_pars = cats['BuMC'].params+cats['BdMC'].params+cats['BdRD'].params
  fcn_kwgs = {
      'data': [cats['BuMC'].time, cats['BdMC'].time, cats['BdRD'].time],
      'prob': [cats['BuMC'].lkhd, cats['BdMC'].lkhd, cats['BdRD'].lkhd],
      'weight': [cats['BuMC'].weight, cats['BdMC'].weight, cats['BdRD'].weight]
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)

  if MINER.lower() in ("minuit", "minos"):
    result = mini.optimize(method='minuit', verbose=False, tol=0.1)
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    _res = optimize(method='nelder', verbose=False)
    result = mini.optimize(method=MINER, params=_res.params, verbose=False)
  elif MINER.lower() in ('nelder'):
    result = mini.optimize(method='nelder', verbose=False)
  elif MINER.lower() in ('emcee'):
    _res = mini.optimize(method='minuit', verbose=False, tol=0.05)
    result = mini.optimize(method='emcee', verbose=False, params=_res.params,
                           steps=1000, nwalkers=100, behavior='chi2')
  print(result)



  # Writing results -----------------------------------------------------------
  print(f"\n{80*'='}\nDumping parameters\n{80*'='}\n")

  for name, cat in zip(cats.keys(), cats.values()):
    list_params = cat.params.find('(a|b|c)(\d{1})(\d{1})?(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])

    print(f"Dumping tex table to {cats[name].tabs_path}")
    with open(cat.tabs_path, "w") as text:
      text.write(cat.params.dump_latex(caption=f"Time acceptance for the $\
      {mode_tex(f'{MODE}')}$ ${YEAR}$ {TRIGGER} category in simultaneous fit\
      using $B_u^+$ as $B_s^0$."))

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cats[name].pars_path)

################################################################################
# that's all folks!
