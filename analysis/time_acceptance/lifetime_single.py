DESCRIPTION = """
    This file contains 3 fcn functions to be minimized under ipanema3 framework
    those functions are, actually functions of badjanak kernels.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import hjson
import numpy as np

# load ipanema
from ipanema import initialize, plotting
from ipanema import ristra, Parameters, optimize, Sample, plot_conf2d, Optimizer

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and, printsec, printsubsec
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
  return p

if __name__ != '__main__':
  initialize(os.environ['IPANEMA_BACKEND'],1)
  import time_acceptance.fcn_functions as fcns

################################################################################



################################################################################
#%% Run and get the job done ###################################################
if __name__ == '__main__':

  # Parse arguments {{{
  p = argparse.ArgumentParser(description='Compute single decay-time acceptance.')
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--biased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--unbiased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  args = vars(p.parse_args())

  printsec("Lifetime (single) determination")

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year'].split(',')
  MODE = args['mode']
  TRIGGER = args['trigger']
  TIMEACC= timeacc_guesser(args['timeacc'])
  MINER = config.base['minimizer']

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'],1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = cuts_and(f'time>={tLL} & time<={tUL}')

  # Print settings
  printsubsec(f"Settings")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'minimizer':>15}: {MINER:50}")

  if TRIGGER == 'combined':
    TRIGGER = ['biased', 'unbiased']
  else:
    TRIGGER = [TRIGGER]

  sWeight = "sw"

  # }}}


  # Get data into categories {{{

  printsubsec(f"Loading samples")

  cats = {}
  for i, y in enumerate(YEAR):
    cats[y] = {}
    for t in TRIGGER:
      cats[y][t] = Sample.from_root(args['samples'].split(',')[i], share=SHARE)
      cats[y][t].chop( trigger_scissors(t, CUT) )
      cats[y][t].allocate(time='time', lkhd='0*time')
      cats[y][t].allocate(weight=sWeight)
      cats[y][t].weight = swnorm(cats[y][t].weight)
      print(cats[y][t])

      # Add coeffs parameters
      c = Parameters.load(args[f'{t}_params'].split(',')[i])
      print(c)
      knots = Parameters.build(c, c.find('k.*'))
      list_c = c.find('(a|b|c).*') + c.find('(mu|sigma)_(a|b|c)')
      cats[y][t].params = Parameters.build(c, list_c)

  # }}}


  # Configure kernel {{{

  fcns.badjanak.config['knots'] = np.array(knots).tolist()
  fcns.badjanak.get_kernels(True)

  # }}}


  # Time to fit {{{

  printsubsec(f"Minimization procedure")

  # create a common gamma parameter for biased and unbiased
  if "Bs" in MODE:
    q = 's'
  elif "Bd" in MODE:
    q = 'd'
  elif "Bu" in MODE:
    q = 'u'
  else:
    print("WTF. Exit")
    exit()

  lfpars = Parameters()
  lfpars.add(dict(name='gamma', value=0.5, min=0.0, max=1.0,
                  latex=rf'\Gamma_{q}'))

  # join and print parameters before the lifetime fit
  for y in cats:
    for t in cats[y]:
      for p, par in cats[y][t].params.items():
        if p[0] == 'c' or p[0] == 'b' or p[0] == 'a':
          print(p)
          _p = list(p); _p[0] = 'c'; _p = ''.join(_p)  # basic renaming
          lfpars.add({"name": f"{_p}_{y[2:]}",
                      "value": par.value, "stdev": par.stdev,
                      "latex": f"{par.latex}{{}}^{y[2:]}",
                      "min": par.min, "max": par.max, "free": par.free
                     })
        else:
          lfpars.add({"name": p[:-2]})
          lfpars[p[:-2]] = par
  lfpars.lock(); lfpars.unlock('gamma')
  print(lfpars)

  # lifetime fit
  if MINER.lower() in ("minuit","minos"):
    lifefit = optimize(fcn_call=fcns.splinexerfconstr_single, params=lfpars,
                       fcn_kwgs={'cats':cats, 'weight':True},
                       method=MINER, verbose=False, strategy=1, tol=0.05);
    print(lifefit)
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    0 # fix me!
  else:
    print("Unrecognized method. Halted!")
    exit()


  # }}}


  # Writing results {{{

  printsubsec(f"Lifetime estimation")
  print(f"\\tau(B_{q}) = {1/lifefit.params['gamma'].uvalue:.2uL}")

  print(f"Dumping json parameters to {args['output_params']}")
  lifefit.params.dump(args['output_params'])

  # }}}

# }}}


# vim:foldmethod=marker
# that's all folks!
