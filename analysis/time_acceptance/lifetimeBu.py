DESCRIPTION = """
    Computes the lifetime of half a Bd RD sample using spline coefficients 
    taken from the other halve. Runs over YEARS variable tuples.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import numpy as np
import hjson
import uncertainties as unc

# load ipanema
from ipanema import initialize
from ipanema import Parameters, optimize, Sample

# import some phis-scq utils
from utils.strings import cuts_and
from utils.helpers import  version_guesser, timeacc_guesser
from utils.helpers import  swnorm, trigger_scissors

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  parser.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--biased-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--unbiased-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--year', help='Year to fit')
  parser.add_argument('--version', help='Version of the tuples to use')
  parser.add_argument('--timeacc', help='Different flag to ... ')
  return parser

if __name__ != '__main__':
  import badjanak

################################################################################



################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':

  #%% Parse arguments ----------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = 'Bu2JpsiKplus'
  TIMEACC, NKNOTS, CORR, FLAT, LIFECUT, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else 1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')
  
  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'year(s)':>15}: {YEAR:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}\n")
  
  # final arrangemets
  samples = args['sample'].split(',')
  sw = f'sw_{VAR}' if VAR else 'sw'


  # Get data into categories ---------------------------------------------------
  print(f"\n{80*'='}\nLoading categories\n{80*'='}\n")

  cats = {}  
  for i, m in enumerate(YEAR.split(',')):
    cats[m] = {}
    for t in ['biased', 'unbiased']:
      cats[m][t] = Sample.from_root(samples[i], share=SHARE)
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
      cats[m][t].params = Parameters.build(c,c.find('c.*')+['mu_c','sigma_c'])

      # Update kernel with the corresponding knots
      fcns.badjanak.config['knots'] = np.array(knots).tolist()
  
  # recompile kernel (just in case)
  fcns.badjanak.get_kernels(True)

  

  # Time to fit lifetime -------------------------------------------------------
  print(f"\n{80*'='}\nMinimization procedure\n{80*'='}\n")
  
  # create a common gamma parameter for biased and unbiased
  lfpars = Parameters()
  lfpars.add(dict(name='gamma', value=0.5, min=-1.0, max=1.0, latex="\Gamma_u-\Gamma_d"))
  
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
  if MINER.lower() in ("minuit","minos"):
    lifefit = optimize(fcn_call=fcns.splinexerfconstr, params=lfpars,
                       fcn_kwgs={'cats':cats, 'weight':True},
                       method=MINER, verbose=False, strategy=1, tol=0.05);
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    0 # fix me!

  print(lifefit)
  print(f"\n{80*'='}\nLifetime estimation\n{80*'='}\n")
  tauBd = unc.ufloat(1.520,0.004)
  tauBu = 1/lifefit.params['gamma'].uvalue
  print(f"\\tau(B_u^+) = {tauBu:.2uL}")
  print(f"\\tau(B_d^0) = {tauBd:.2uL} WA")
  print(f"\\tau(B_u^+)/\\tau(B_d^0) = {tauBu/tauBd:.2uL}")



  # Writing results ------------------------------------------------------------
  print(f"\n{80*'='}\nDumping parameters\n{80*'='}\n")

  print(f"Dumping json parameters to {args['output_params']}")
  lifefit.params.dump(args['output_params'])

  print(f"Dumping tex table to {args['output_tables']}")
  with open(args['output_tables'], "w") as text:
    text.write(lifefit.params.dump_latex(caption=f"""Trigger-combinned fit 
    parameters for $B_d^0$ lifetime check using {YEAR}."""))

################################################################################
# that's all folks!
