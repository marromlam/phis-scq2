# Main Workflow - phis-scq
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors: Marcos Romero

# Main constants ---------------------------------------------------------------
#    VERSION: is the version Bs2JpsiPhi-FullRun2 pipeline was run against, and
#             should be matched with this constant.
#    MAIN_PATH: the path where all eos-samples will be synced, make sure there
#               is enough free space there
#    SAMPLES_PATH: where all ntuples for a given VERSION will be stored
VERSION = 'v0r2'
MAIN_PATH = '/scratch17/marcos.romero/phis_samples/'
SAMPLES_PATH = MAIN_PATH + VERSION + '/'

# Some wildcards options ( this is not actually used )
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0',#'MC_Bs2JpsiPhi',
         'Bd2JpsiKstar', 'MC_Bd2JpsiKstar'];
years = ['2011','2012','Run1',
         '2015','2016','Run2a','2017','2018','Run2b','Run2'];
trigger = ['combined','biased','unbiased']



# Rule orders
ruleorder: sync_ntuples > reduce_ntuples


# Including Snakefiles
include: 'samples/Snakefile'
include: 'reweightings/Snakefile'
include: 'time_acceptance/Snakefile'
