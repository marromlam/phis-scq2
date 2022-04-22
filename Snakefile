# WORKFLOW
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors:
#       Marcos Romero Lamas, mromerol@cern.ch
#       Ramón Ángel Ruiz Fernández, rruizfer@cern.ch


# Modules {{{

import os, shutil
import time
import yaml
import config as settings
from string import Template
from utils.helpers import (tuples, angaccs, csps, flavors, timeaccs, timeress,
                           version_guesser, send_mail)
configfile: "config/base.json"

from snakemake.remote.XRootD import RemoteProvider as XRootDRemoteProvider
XRootD = XRootDRemoteProvider(stay_on_remote=True)

# }}}


# Main constants {{{

SAMPLES_PATH = settings.user['path']
SAMPLES = settings.user['path']
NOTE = settings.user['note']
MAILS = settings.user['mail']
YEARS = settings.years

# }}}


# Set pipeline-wide constraints {{{
#     Some wilcards will only have some well defined values.

wildcard_constraints:
  trigger = "(biased|unbiased|combined)",
  year = "(2015|2016|run2a|2017|2018|run2b|run2|2020|2021)",
  strip_sim = "str.*",
  version = '[A-Za-z0-9@~]+',
  polarity = '(Up|Down)'

MINERS = "(Minos|BFGS|LBFGSB|CG|Nelder)"

# Some wildcards options ( this is not actually used )
modes = ['Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi',
         'MC_Bs2JpsiKK_Swave', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar',
         'Bu2JpsiKplus', 'MC_Bu2JpsiKplus']

#{Bs2JpsiPhi,MC_Bs2JpsiPhi_dG0,MC_Bs2JpsiPhi,Bd2JpsiKstar,MC_Bd2JpsiKstar,Bu2JpsiKplus,MC_Bu2JpsiKplus}

# }}}


# Including Snakefiles {{{

if config['run_selection']:
    include: 'selection/Snakefile'
include: 'selection/sweights/Snakefile'
include: 'tagging/Snakefile'
include: 'analysis/samples/Snakefile'
include: 'analysis/reweightings/Snakefile'
include: 'analysis/velo_weights/Snakefile'
include: 'analysis/time_acceptance/Snakefile'
include: 'analysis/lifetime/Snakefile'
include: 'analysis/flavor_tagging/Snakefile'
include: 'analysis/csp_factors/Snakefile'
include: 'analysis/time_resolution/Snakefile'
include: 'analysis/angular_acceptance/Snakefile'
include: 'analysis/angular_fit/Snakefile'
# include: 'analysis/bundle/Snakefile'
include: 'analysis/params/Snakefile'
include: 'analysis/what_the_hell/Snakefile'
include: 'analysis/toys/Snakefile'
include: 'packandgo/Snakefile'

# }}}


# Final rule (compile slides) {{{

rule help:
    """
    Print list of all targets with help.
    """
    run:
        for rule in workflow.rules:
            print(rule.name)
            print(rule.docstring)


rule all:
  input:
    "output/b2cc_all.pdf"


rule slides_compile:
  input:
    # TABLES {{{
    # time acceptance tables {{{
    # baseline time acceptance {{{ 
    f"output/tables/time_acceptance/run2/MC_Bs2JpsiPhi_dG0/{config['version']}_simul3.tex",
    f"output/tables/time_acceptance/run2/MC_Bd2JpsiKstar/{config['version']}_simul3.tex",
    f"output/tables/time_acceptance/run2/Bd2JpsiKstar/{config['version']}_simul3.tex",
    # }}}
    # baseline with dG!=0 time acceptance {{{ 
    f"output/tables/time_acceptance/run2/MC_Bs2JpsiPhi/{config['version']}_simul3DGn0.tex",
    f"output/tables/time_acceptance/run2/MC_Bd2JpsiKstar/{config['version']}_simul3DGn0.tex",
    f"output/tables/time_acceptance/run2/Bd2JpsiKstar/{config['version']}_simul3DGn0.tex",
    # }}}
    # }}}
    # lifetimes {{{
    # single (each mode independently fitted) {{{
    f"output/tables/lifetime/run2/Bs2JpsiPhi/{config['version']}_lifesingle_combined.tex",
    f"output/tables/lifetime/run2/Bd2JpsiKstar/{config['version']}_lifesingle_combined.tex",
    f"output/tables/lifetime/run2/Bu2JpsiKplus/{config['version']}_lifesingle_combined.tex",
    f"output/tables/lifetime/run2/Bs2JpsiPhi/{config['version']}_lifesingle_unbiased.tex",
    f"output/tables/lifetime/run2/Bd2JpsiKstar/{config['version']}_lifesingle_unbiased.tex",
    f"output/tables/lifetime/run2/Bu2JpsiKplus/{config['version']}_lifesingle_unbiased.tex",
    f"output/tables/lifetime/run2/Bs2JpsiPhi/{config['version']}_lifesingle_biased.tex",
    f"output/tables/lifetime/run2/Bd2JpsiKstar/{config['version']}_lifesingle_biased.tex",
    f"output/tables/lifetime/run2/Bu2JpsiKplus/{config['version']}_lifesingle_biased.tex",
    # }}}
    # cross-checks {{{
    f"output/tables/lifetime/run2/Bd2JpsiKstar/{config['version']}@evtEven_simul3BdasBs_combined.tex",
    f"output/tables/lifetime/run2/Bu2JpsiKplus/{config['version']}_simul3BuasBs_combined.tex",
    # }}}
    # }}}
    # angular acceptance {{{
    # baseline
    f"output/tables/angular_acceptance/run2/Bs2JpsiPhi/{config['version']}_run2_vgc_amsrd_simul3_amsrd.tex",
    # yearly
    f"output/tables/angular_acceptance/run2/Bs2JpsiPhi/{config['version']}_yearly_vgc_amsrd_simul3_amsrd.tex",
    # }}}
    # physics parameters {{{
    # HERE nominal {{{
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # trigger cross-checks {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@Trigger_run2_run2_vgc_amsrd_simul3_amsrd.tex",
    # }}}
    # magnet cross-checks {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@Magnet_run2_run2_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # yearly cross-checks {{{
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_yearly_yearly_vgc_amsrd_simul3Noncorr_amsrd_combined.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3DGn0_amsrd_combined.tex",
    # }}}
    # pT cross-check {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@pTB_run2_run2_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # pT cross-check using Bu as control channel {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@pTB_run2_run2_vgc_amsrd_simul3BuasBd_amsrd_combined.tex",
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3BuasBd_amsrd_combined.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3BuasBd_amsrd_combined.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3BuasBd_amsrd_combined.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@pTB_yearly_yearly_vgc_amsrd_simul3BuasBd_amsrd_combined.tex",
    # }}}
    # etaB cross-check {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@etaB_run2_run2_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@etaB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@etaB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@etaB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@etaB_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # sigmat cross-check {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@sigmat_run2_run2_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@sigmat_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@sigmat_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@sigmat_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@sigmat_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # time acceptance variations {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3DGn0_amsrd_combined.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3Noncorr_amsrd_combined.tex",
    # }}}
    # }}}
    # }}}
    # FIGURES {{{
    #
    # reweighting plots
    # expand(rules.reweightings_plot_time_acceptance.output,
    #        version = 'v0r5',
    #        mode = ['MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar',
    #                'Bd2JpsiKstar'],
    #        branch = ['B_P', 'B_PT', 'X_M'],
    #        year = ['2015', '2016', '2017', '2018']),
    # time acceptance plot - nominal case only
    expand(rules.time_acceptance_simultaneous_plot.output,
           version=config['version'],
           mode=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
           timeacc=['simul3', 'simul3Noncorr'],
           year=['2015', '2016', '2017', '2018'],
           plot=['fitlog', 'splinelog'],
           trigger=['biased', 'unbiased']),
    # lifetime trend plots {{{
    expand(rules.lifetime_trend.output,
           version=config['version'],
           mode=['Bs2JpsiPhi', 'Bu2JpsiKplus', 'Bd2JpsiKstar'],
           timeacc=['single', 'singleNoncorr'],
           year=['run2']),
    # }}}
    # time acceptance plots - binned variables
    # expand(rules.time_acceptance_plot.output,
    #        version=['v0r5+v0r5@pTB1+v0r5@pTB2+v0r5@pTB3+v0r5@pTB4',
    #                 'v0r5+v0r5@sigmat1+v0r5@sigmat2+v0r5@sigmat3',
    #                 'v0r5+v0r5@etaB1+v0r5@etaB2+v0r5@etaB3'],
    #        mode=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
    #        timeacc=['simul3'],
    #        year=['2015', '2016', '2017', '2018'],
    #        plot=['splinelog'],
    #        trigger=['biased', 'unbiased']),
    # time acceptance plot - different knots + w/o kinWeight
    # expand(rules.time_acceptance_plot.output,
    #        version=config['version'],
    #        mode=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
    #        timeacc=['simul3+simul6', 'simul3+simul3Noncorr'],
    #        year=['2015', '2016', '2017', '2018'],
    #        plot=['splinelog'],
    #        trigger=['biased', 'unbiased']),
    # rwp2 = expand(rules.reweightings_plot_angular_acceptance.output,
    #               version=['v0r5'],
    #               mode=['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0'],
    #               branch=['B_P','B_PT','X_M','hplus_PT','hplus_P','hminus_PT','hminus_P'],
    #               angacc=['yearly'],
    #               timeacc=['repo'],
    #               weight=['sWeight','kinWeight','kkpWeight'],
    #               year=['2015']),
    #               #year=['2015','2016','2017','2018']),
    # }}}
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

# }}}


# vim:foldmethod=marker
