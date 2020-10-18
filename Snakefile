# Main Workflow - phis-scq
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors:
#       Marcos Romero Lamas, mromerol@cern.ch

import hjson
CONFIG = hjson.load(open('config.json'))

from utils.helpers import tuples, version_guesser, send_mail



# Main constants ---------------------------------------------------------------
#    VERSION: is the version Bs2JpsiPhi-FullRun2 pipeline was run against, and
#             should be matched with this constant.
#    MAIN_PATH: the path where all eos-samples will be synced, make sure there
#               is enough free space there
#    SAMPLES_PATH: where all ntuples for a given VERSION will be stored

SAMPLES_PATH = CONFIG['path']
MAILS = CONFIG['mail']

MINERS = "(Minos|BFGS|LBFGSB|CG|Nelder)"



# Some wildcards options ( this is not actually used )
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0',#'MC_Bs2JpsiPhi',
         'Bd2JpsiKstar', 'MC_Bd2JpsiKstar'];
YEARS = {#
  '2011'  : ['2011'],
  '2012'  : ['2012'],
  'Run1'  : ['2011','2012'],
  '2015'  : ['2015'],
  '2016'  : ['2016'],
  'Run2a' : ['2015','2016'],
  '2017'  : ['2017'],
  '2018'  : ['2018'],
  'Run2b' : ['2017','2018'],
  'Run2'  : ['2015','2016','2017','2018']
};



# Rule orders
#ruleorder: sync_ntuples > reduce_ntuples



# Set pipeline-wide constraints ------------------------------------------------
#     Some wilcards will only have some well defined values.

wildcard_constraints:
  trigger = "(biased|unbiased|combined)",
  year = "(2015|2016|run2a|2017|2018|run2b|run2|2020)"



# Including Snakefiles ---------------------------------------------------------
#     dfdf

include: 'analysis/samples/Snakefile'
include: 'analysis/reweightings/Snakefile'
include: 'analysis/time_acceptance/Snakefile'
include: 'analysis/flavor_tagging/Snakefile'
include: 'analysis/csp_factors/Snakefile'
include: 'analysis/time_resolution/Snakefile'
include: 'analysis/angular_acceptance/Snakefile'
include: 'analysis/angular_fit/Snakefile'
include: 'analysis/bundle/Snakefile'
include: 'analysis/params/Snakefile'
