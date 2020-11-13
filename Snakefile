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
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0','MC_Bs2JpsiPhi',
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
#include: 'analysis/toys/Snakefile'


# Final rule (compile slides) --------------------------------------------------

rule all:
  input:
    "output/b2cc_all.pdf"

rule compile_slides:
  input:
    rwp1 = expand(rules.reweightings_plot_time_acceptance.output,
                  version='v0r5',
                  mode=['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar','Bd2JpsiKstar'],
                  branch=['B_P','B_PT','X_M'],
                  year=['2015','2016','2017','2018']),
    tap1 = expand(rules.time_acceptance_plot.output,
                  #version=['v0r5@cutpt1','v0r5@cutpt2','v0r5@cutpt3','v0r5@cutpt4'],
                  version=['v0r5','v0r5@cutpt1','v0r5@cutpt2','v0r5@cutpt3','v0r5@cutpt4','v0r5+v0r5@cutpt1+v0r5@cutpt2+v0r5@cutpt3+v0r5@cutpt4','v0r5@cutsigmat1','v0r5@cutsigmat2','v0r5@cutsigmat3','v0r5+v0r5@cutsigmat1+v0r5@cutsigmat2+v0r5@cutsigmat3','v0r5@cuteta1','v0r5@cuteta2','v0r5@cuteta3','v0r5+v0r5@cuteta1+v0r5@cuteta2+v0r5@cuteta3'],
                  mode=['MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar','Bd2JpsiKstar'],
                  timeacc=['simul'],
                  year=['2015','2016','2017','2018'],
                  plot=['fitlog','spline'],
                  trigger=['biased','unbiased']),
    # rwp2 = expand(rules.reweightings_plot_angular_acceptance.output,
    #               version=['v0r5'],
    #               mode=['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0'],
    #               branch=['B_P','B_PT','X_M','hplus_PT','hplus_P','hminus_PT','hminus_P'],
    #               angacc=['yearly'],
    #               timeacc=['repo'],
    #               weight=['sWeight','kinWeight','kkpWeight'],
    #               year=['2015']),
    #               #year=['2015','2016','2017','2018']),
  output:
    "output/b2cc_{date}.pdf"
  log:
    'output/log/bundle/compile_slides/{date}.log'
  run:
    date = f"{wildcards.date}"
    import os
    if not os.path.isfile(f"slides/main_{date}.tex"):
      print(f"Creating main_{date}.tex from main.tex template")
      os.system(f"cp slides/containers/main.tex slides/main_{date}.tex")
    shell(f"cd slides/; latexmk -xelatex main_{date}.tex")
    shell(f"cp slides/main_{date}.pdf output/b2cc_{date}.pdf")
    shell(f"cd slides/; latexmk -c -silent main_{date}.tex")
    shell(f"rm slides/*.xdv")
