DESCRIPTION = """
    Computes angular acceptance with corrections in mHH, pB, pTB variables
    using an a reweight.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []



################################################################################
# % Modules ####################################################################

import argparse
import os
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

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data', help='Bs2JpsiPhi data sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC generator parameters')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC angular acceptance')
  p.add_argument('--output-tables', help='Bs2JpsiPhi MC angular acceptance tex')
  p.add_argument('--output-weights-file', help='angWeights file')
  p.add_argument('--mode', help='Mode to compute angular acceptance with')
  p.add_argument('--year', help='Year to compute angular acceptance with')
  p.add_argument('--version', help='Version of the tuples')
  p.add_argument('--trigger', help='Trigger to compute angular acceptance with')
  return p




################################################################################



################################################################################
# Run and get the job done #####################################################

if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {trigger_scissors(TRIGGER, CUT):50}")
  print(f"{'angacc':>15}: {'corrected':50}")
  print(f"{'bdtconfig':>15}: {list(bdconfig.values())}\n")



  # %% Load samples ------------------------------------------------------------
  printsec("Loading categories")

  # Load Monte Carlo samples
  mc = Sample.from_root(args['sample_mc'], share=SHARE, name=MODE)
  mc.assoc_params(args['input_params'])
  kinWeight = np.zeros_like(list(mc.df.index)).astype(np.float64)
  mc.chop(trigger_scissors(TRIGGER, CUT))
  # Load corresponding data sample
  rd = Sample.from_root(args['sample_data'], share=SHARE, name='data')
  rd.chop(trigger_scissors(TRIGGER, CUT))

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  weight_rd = f'(sw_{VAR})' if VAR else '(sw)'
  weight_mc = f'(polWeight*{weight_rd}/gb_weights)'
  print(weight_mc,weight_rd)

  # Allocate some arrays with the needed branches
  mc.allocate(reco=reco+['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH'])
  mc.allocate(true=true+['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH'])
  mc.allocate(pdf='0*time')
  mc.allocate(weight=weight_mc)



  # Compute standard kinematic weights -----------------------------------------
  #     This means compute the kinematic weights using 'mHH','pB' and 'pTB'
  #     variables
  printsec('Compute angWeights correcting MC sample in kinematics')
  print(f" * Computing kinematic GB-weighting in pTB, pB and mHH")

  reweighter.fit(original        = mc.df[['mHH','pB','pTB']],
                 target          = rd.df[['mHH','pB','pTB']],
                 original_weight = mc.df.eval(weight_mc),
                 target_weight   = rd.df.eval(weight_rd));
  angWeight = reweighter.predict_weights(mc.df[['mHH', 'pB', 'pTB']])
  kinWeight[list(mc.df.index)] = angWeight
  
  print(f"{'idx':>3} | {'sw':>11} | {'polWeight':>11} | {'angWeight':>11} ")
  for i in range(0,100):
    if kinWeight[i] != 0:
      print(f"{str(i):>3} | {mc.df.eval('sw/gb_weights')[i]:+.8f} |",
            f"{mc.df['polWeight'][i]:+.8f} | {kinWeight[i]:+.8f} ")

  np.save(args['output_weights_file'], kinWeight)



  # Compute angWeights correcting with kinematic weights -----------------------
  #     This means compute the kinematic weights using 'mHH','pB' and 'pTB'
  #     variables

  angacc = badjanak.get_angular_acceptance_weights(mc.true, mc.reco, 
                                     mc.weight*ristra.allocate(angWeight),
                                     **mc.params.valuesdict())
  w, uw, cov, corr = angacc
  pars = Parameters()
  for i in range(0,len(w)):
    correl = {f'w{j}{TRIGGER[0]}': corr[i][j]
              for j in range(0, len(w)) if i > 0 and j > 0}
    pars.add({'name': f'w{i}{TRIGGER[0]}', 'value': w[i], 'stdev': uw[i], 
              'correl': correl, 'free': False, 'latex': f'w_{i}^{TRIGGER[0]}'})
  print(f" * Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")

  print(f"{pars}")



  # Writing results ------------------------------------------------------------
  #    Exporting computed results
  printsec("Dumping parameters")
  # Dump json file
  print(f"Dumping json parameters to {args['output_params']}")
  pars.dump(args['output_params'])
  # Export parameters in tex tables
  print(f"Dumping tex table to {args['output_tables']}")
  with open(args['output_tables'], "w") as tex_file:
    tex_file.write(
      pars.dump_latex(caption=f"""Kinematically corrected angular weights for
      {YEAR} {TRIGGER} {mode_tex(MODE)}  category."""
      )
    )
  tex_file.close()

################################################################################
# that's all folks!
