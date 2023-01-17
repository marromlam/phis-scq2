from analysis import badjanak
import config
from utils.strings import printsec
from utils.helpers import version_guesser, trigger_scissors
from ipanema import Sample, Parameters
from ipanema import initialize
import os
import argparse
from warnings import simplefilter
from utils.strings import cuts_and


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

simplefilter(action='ignore', category=FutureWarning)
# from ipanema.samples import cuts_and

# load ipanema
initialize(config.user['backend'], 1)

# import some phis-scq utils

# binned variables
# bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = config.timeacc['constants']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']

# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

# }}}


# Run and get the job done {{{

if __name__ == '__main__':

  # Parse arguments {{{

  DESCRIPTION = """
    Computes angular acceptance without any corrections.
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Configuration')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--trigger', help='Trigger(s) to fit')
  args = vars(p.parse_args())

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(
      args['version'])
  YEAR = args['year']
  ANGACC = args['angacc']
  MODE = args['mode']
  TRIGGER = args['trigger']

  tLL = config.general['time_lower_limit']
  tUL = config.general['time_upper_limit']

  # Prepare the cuts
  CUT = ''

  if ANGACC != 'naive':
    # BUG: here we should import the new knot generator
    nknots, timebin = ANGACC.split('knots')
    tLL = all_knots[nknots[5:]][int(timebin) - 1]
    tUL = all_knots[nknots[5:]][int(timebin)]
    CUT = cuts_and(CUT, f'time>={tLL} & time<{tUL}')

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {trigger_scissors(TRIGGER, CUT):50}")
  print(f"{'angacc':>15}: {ANGACC:50}")

  if VERSION == 'v0r0':
    args['input_params'] = args['input_params'].replace(
        'generator', 'generator_old')

  # }}}

  # Load samples {{{

  printsec('Loading category')

  mc = Sample.from_root(args['sample'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'].replace('TOY', 'MC'))
  mc.chop(trigger_scissors(TRIGGER, CUT))
  print(mc.params)

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  weight = 'polWeight*sw'
  weight = 'polWeight*sWeight'

  # Allocate some arrays with the needed branches
  # WARNING: please add directly sigma and mHH to TOY samples
  #          so we can skip the following if
  try:
    mc.allocate(reco=reco + ['mHH', '0*mHH', 'genidB',
                'genidB', '0*mHH', '0*mHH'])
    mc.allocate(true=true + ['mHH', '0*mHH', 'genidB',
                'genidB', '0*mHH', '0*mHH'])
  except:
    print('Guessing you are working with TOY files. No X_M provided')
    mc.allocate(reco=reco + ['0*time', '0*time',
                'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    mc.allocate(true=true + ['0*time', '0*time',
                'B_ID', 'B_ID', '0*B_ID', '0*B_ID'])
    weight = 'time/time'
  mc.allocate(pdf='0*time', weight=weight)

  # }}}

  # Compute angWeights without corrections {{{
  #     Let's start computing the angular weights in the most naive version,
  #     w/o any corrections
  printsec('Compute angWeights without correcting MC sample')

  if 'Bd2JpsiKstar' in MODE:
    badjanak.config["x_m"] = [826, 861, 896, 931, 966]
  badjanak.get_kernels()

  print('Computing angular weights')
  w, uw, cov, corr = badjanak.get_angular_acceptance_weights(
      mc.true, mc.reco, mc.weight, **mc.params.valuesdict(), tLL=tLL,
      tUL=tUL)
  pars = Parameters()
  for i in range(0, len(w)):
    correl = {f'w{j}{TRIGGER[0]}': corr[i][j]
              for j in range(0, len(w)) if i > 0 and j > 0}
    pars.add({'name': f'w{i}{TRIGGER[0]}', 'value': w[i], 'stdev': uw[i],
              'correl': correl, 'free': False,
              'latex': f'w_{i}^{TRIGGER[0]}'})

  print('Dumping parameters')
  pars.dump(args['output_params'])
  print(f"Naive angular weights for {MODE}-{YEAR}-{TRIGGER} sample are:")
  print(f"{pars}")

  # }}}

# }}}


# vim: fdm=marker
