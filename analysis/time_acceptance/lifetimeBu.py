import config
from utils.helpers import swnorm, trigger_scissors
from utils.helpers import version_guesser, timeacc_guesser
from utils.strings import cuts_and, printsec, printsubsec
from ipanema import Parameters, optimize, Sample
from ipanema import initialize
import uncertainties as unc
import hjson
import numpy as np
import os
import argparse
DESCRIPTION = """
    Computes the lifetime of half a Bu RD sample using spline coefficients
    taken from the other halve. Runs over YEARS variable tuples.
"""

__author__ = ['Marcos Romero Lamas']
__all__ = []
__email__ = ['mromerol@cern.ch']


# Modules {{{


# load ipanema

# import some phis-scq utils

# phis-scq config
resolutions = config.timeacc['constants']
all_knots = config.timeacc['knots']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']
tLL = config.general['tLL']
tUL = config.general['tUL']

# }}}


# CMDline interface {{{

if __name__ == '__main__':

  # Parse arguments {{{

  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  p.add_argument('--biased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--unbiased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--trigger', help='Different flag to ... ')
  p.add_argument('--minimizer', default='minuit', help='Different flag to ... ')
  args = vars(p.parse_args())
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = 'Bu2JpsiKplus'
  TRIGGER = args['trigger']
  TIMEACC = timeacc_guesser(args['timeacc'])
  MINER = args['minimizer']

  if TRIGGER == 'combined':
    TRIGGER = ['biased', 'unbiased']
  else:
    TRIGGER = [TRIGGER]

  # }}}

  # Settings {{{

  # Get badjanak model and configure it
  initialize(config.user['backend'], 1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = ""  # bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'year(s)':>15}: {YEAR:50}")
  print(f"{'timeacc':>15}: {TIMEACC['acc']:50}")
  print(f"{'minimizer':>15}: {MINER:50}\n")

  # }}}

  # Get data into categories {{{

  printsubsec("Loading categories")

  sw = 'sw'
  cats = {}
  for i, m in enumerate(YEAR.split(',')):
    cats[m] = {}
    for t in TRIGGER:
      cats[m][t] = Sample.from_root(args['sample'].split(',')[i], share=SHARE)
      cats[m][t].name = f"BdRD-{m}-{t}"
      cats[m][t].chop(trigger_scissors(t, CUT))

      # allocate arrays
      cats[m][t].allocate(time='time', lkhd='0*time')
      cats[m][t].allocate(weight=f'{sw}')
      cats[m][t].weight = swnorm(cats[m][t].weight)
      print(cats[m][t])

      # Add coeffs parameters
      c = Parameters.load(args[f'{t}_params'].split(',')[i])
      knots = Parameters.build(c, c.find('k.*'))
      cats[m][t].params = Parameters.build(c, c.find('c.*') + ['mu_c', 'sigma_c'])

      # Update kernel with the corresponding knots
      fcns.badjanak.config['knots'] = np.array(knots).tolist()

  # }}}

  # Fit {{{

  printsubsec("Minimization procedure")

  # recompile kernel (just in case)
  fcns.badjanak.get_kernels()

  # create a common gamma parameter for biased and unbiased
  lfpars = Parameters()
  lfpars.add(dict(name='gamma', value=0.5, min=-1.0, max=1.0,
                  latex="\Gamma_u-\Gamma_d"))

  # join and print parameters before the lifetime fit
  for y in cats:
    for t in cats[y]:
      for p, par in cats[y][t].params.items():
        if p[0] == 'c':
          lfpars.add({"name": f"{p.replace('cB', 'cA')}_{y[2:]}",
                      "value": par.value, "stdev": par.stdev,
                      "latex": f"{par.latex.replace('cB', 'cA')}{{}}^{y[2:]}",
                      "min": par.min, "max": par.max, "free": par.free
                      })
        else:
          lfpars.add({"name": p.replace('Bc', 'Ac')})
          lfpars[p.replace('Bc', 'Ac')] = par
  #lfpars.lock(); lfpars.unlock('gamma')
  print(lfpars)

  # lifetime fit
  lifefit = optimize(fcn_call=fcns.splinexerfconstr, params=lfpars,
                     fcn_kwgs={'cats': cats, 'weight': True},
                     method=MINER, verbose=False, strategy=1, tol=0.05)

  # }}}

  # Saving results {{{

  printsubsec(f"Lifetime estimation")
  print(lifefit)
  tauBd = unc.ufloat(1.520, 0.004)
  tauBu = 1 / lifefit.params['gamma'].uvalue
  print(f"\\tau(B_u^+) = {tauBu:.2uL}")
  print(f"\\tau(B_d^0) = {tauBd:.2uL} WA")
  print(f"\\tau(B_u^+)/\\tau(B_d^0) = {tauBu/tauBd:.2uL}")

  print(f"Dumping json parameters to {args['output_params']}")
  lifefit.params.dump(args['output_params'])

  # }}}

# }}}


# vim:foldmethod=marker
