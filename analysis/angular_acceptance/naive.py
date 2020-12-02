DESCRIPTION = """
    Computes angular acceptance without any corrections.
"""

__author__ = ['Marcos Romero Lamas']
__email__  = ['mromerol@cern.ch']
__all__ = []



################################################################################
# Modules ######################################################################

import argparse
import os
from ipanema.samples import cuts_and
import numpy as np
import hjson

# load ipanema
from ipanema import initialize
from ipanema import ristra, Sample, Parameters
initialize(os.environ['IPANEMA_BACKEND'],1)

# import some phis-scq utils
from utils.helpers import version_guesser, trigger_scissors
from utils.strings import printsec
from utils.plot import mode_tex

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description='Compute angular acceptance.')
  p.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Configuration')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--trigger', help='Trigger(s) to fit [comb/(biased)/unbiased]')
  return p

################################################################################



################################################################################
# Run and get the job done #####################################################

if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  ANGACC = args['angacc']
  MODE = args['mode']
  TRIGGER = args['trigger']

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  
  knots = [0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15.0]
  if ANGACC != 'naive':
    timebin = int(ANGACC.split('Time')[1])
    tLL = knots[timebin-1]
    tUL = knots[timebin]
    CUT = cuts_and(CUT,f'time>={tLL} & time<{tUL}')

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {trigger_scissors(TRIGGER, CUT):50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  
  if VERSION == 'v0r1':
    args['input_params'] = args['input_params'].replace('generator','generator_old')



  # Load samples ---------------------------------------------------------------
  printsec('Loading category')

  mc = Sample.from_root(args['sample'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'].replace('TOY','MC'))
  mc.chop(trigger_scissors(TRIGGER, CUT))
  print(mc.params)

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  weight = f'polWeight*sw_{VAR}' if VAR else 'polWeight*sw'
  if 'Bs2JpsiPhi' in MODE:
    weight += '/gb_weights'
  print(weight)

  # Allocate some arrays with the needed branches
  try:
    mc.allocate(reco=reco+['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH'])
    mc.allocate(true=true+['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH'])
  except:
    print('Guessing you are working with TOY files. No X_M provided')
    mc.allocate(reco=reco+['0*time', '0*time', 'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    mc.allocate(true=true+['0*time', '0*time', 'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    weight = 'time/time'
  mc.allocate(pdf='0*time', weight=weight)


  # Compute angWeights without corrections -------------------------------------
  #     Let's start computing the angular weights in the most naive version, w/o
  #     any corrections
  printsec('Compute angWeights without correcting MC sample')

  if 'Bd2JpsiKstar' in MODE:
    badjanak.config["x_m"] = [826, 861, 896, 931, 966]
  badjanak.get_kernels(True)

  print('Computing angular weights')
  w, uw, cov, corr = badjanak.get_angular_acceptance_weights(
              mc.true, mc.reco, mc.weight, **mc.params.valuesdict(), tLL=tLL, tUL=tUL
            )
  pars = Parameters()
  for i in range(0,len(w)):
    correl = {f'w{j}{TRIGGER[0]}': corr[i][j]
              for j in range(0, len(w)) if i > 0 and j > 0}
    pars.add({'name': f'w{i}{TRIGGER[0]}', 'value': w[i], 'stdev': uw[i], 
              'correl': correl, 'free': False, 'latex': f'w_{i}^{TRIGGER[0]}'})


  # Writing results ------------------------------------------------------------
  print('Dumping parameters')
  pars.dump(args['output_params'])
  # Export parameters in tex tables
  print('Saving table of params in tex')
  with open(args['output_tables'], "w") as tex_file:
    tex_file.write(
      pars.dump_latex( caption="""
      Naive angular weights for {YEAR} {TRIGGER} ${mode_tex(MODE)}$
      category.""")
    )
  tex_file.close()
  print(f"Naive angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")

################################################################################
# that's all folks!
