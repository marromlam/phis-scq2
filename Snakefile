# WORKFLOW
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors:
#       Marcos Romero Lamas, mromerol@cern.ch
#       Ramón Ángel Ruiz Fernández, rruizfer@cern.ch
#
# Info:
#       Tuple sizes: A full tuple, directly dowloded from EOS or selected within
#       pipeline is ~8GB. The largest reduced tuple, v@cut.root sizes ~1GB.
#
#


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

def temporal(arg):
    if config['delete_temporal']:
        return temp(arg)
    else:
        return arg

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
         'GUN_Bs2JpsiPhi', 'GUN_Bs2JpsiKK_Swave',
         'Bs2DsPi', 'MC_Bs2DsPi',
         'Bs2DsPi_Prompt', 'MC_Bs2DsPi_Prompt',
         'MC_Bs2JpsiKK_Swave', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar',
         'Bu2JpsiKplus', 'MC_Bu2JpsiKplus']

#{Bs2JpsiPhi,MC_Bs2JpsiPhi_dG0,MC_Bs2JpsiPhi,Bd2JpsiKstar,MC_Bd2JpsiKstar,Bu2JpsiKplus,MC_Bu2JpsiKplus}

# }}}


# Including Snakefiles {{{

if config['run_selection']:
    include: 'selection/Snakefile'
include: 'selection/sweights/Snakefile'
if config['run_tagging']:
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
include: 'analysis/systematics/Snakefile'
include: 'packandgo/Snakefile'

# }}}



rule help:
    """
    Print list of all targets with help.
    """
    run:
        all_rules = 0
        for rule in workflow.rules:
            print(rule.name)
            print(rule.docstring)
            all_rules += 1
        print(f"Total number of rules is: {all_rules}")


rule all:
  input:
    "output/b2cc_all.pdf"


rule pack_thesis:
  input:
    #
    #
    #
    # TABLES {{{
    # All tables needed for the theis or analysis note
    #
    # stat tuple{{{
    expand("output/tables/samples_stat_tuple/{myear}/{mmode}/{mversion}_{mstat}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mmode = ["Bu2JpsiKplus", "Bd2JpsiKstar", "Bs2JpsiPhi"],
        myear = ["run2"],
        mstat = ["stat"],
    ),
    # }}}
    #
    # time acceptance tables {{{
    #
    # baseline time acceptance
    expand([
        "output/tables/time_acceptance/run2/MC_Bs2JpsiPhi_dG0/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/MC_Bd2JpsiKstar/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/Bd2JpsiKstar/{mversion}_{mtimeacc}.tex",
        ],
        mtimeacc = [
            f"{config['timeacc']}",  # baseline
            f"{config['timeacc']}".replace('3', '6'),  # using 6 knots
            f"{config['timeacc']}Noncorr"  # wihtout timeacc corerctions
        ],
        mversion = [f"{config['version']}@LcosK"]
    ),
    # using Bs signal MC instead of DG0 one
    expand([
        "output/tables/time_acceptance/run2/MC_Bs2JpsiPhi/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/MC_Bd2JpsiKstar/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/Bd2JpsiKstar/{mversion}_{mtimeacc}.tex",
        ],
        mtimeacc = [ f"{config['timeacc']}DGn0" ],
        mversion = [f"{config['version']}@LcosK"]
    ),
    # using Bu as control channel instead of Bd
    expand([
        "output/tables/time_acceptance/run2/MC_Bs2JpsiPhi_dG0/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/MC_Bu2JpsiKplus/{mversion}_{mtimeacc}.tex",
        "output/tables/time_acceptance/run2/Bu2JpsiKplus/{mversion}_{mtimeacc}.tex",
        ],
        mtimeacc = [
            f"{config['timeacc']}NoncorrBuasBd",  # using Bu as control channel
        ],
        mversion = [f"{config['version']}@LcosK"]
    ),
    #
    # }}}
    #
    #
    # lifetimes {{{
    #
    # single (each mode independently fitted)
    expand(
        "output/tables/lifetime/run2/{mmode}/{mversion}_{mtimeacc}_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mtimeacc = [f"{config['timeacc']}".replace('simul', 'single')],
        mtrigger = ["combined", "biased", "unbiased"],
        # mmode = ["Bu2JpsiKplus", "Bd2JpsiKstar", "Bs2JpsiPhi"],
        mmode = ["Bd2JpsiKstar", "Bs2JpsiPhi"],
    ),
    #
    # simul
    expand(
        "output/tables/lifetime/run2/{mmode}/{mversion}_{mtimeacc}_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mtimeacc = [f"{config['timeacc']}".replace('simul', 'single')],
        mtrigger = ["combined", "biased", "unbiased"],
        # mmode = ["Bu2JpsiKplus", "Bd2JpsiKstar"],
        mmode = ["Bd2JpsiKstar"],
    ),
    #
    # Bd as Bs lifetime measurement
    expand(
        "output/tables/lifetime/{myear}/{mmode}/{mversion}_{mtimeacc}BdasBs_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosKevtEven"],
        mtimeacc = [f"{config['timeacc']}"],
        mtrigger = ["combined", "biased", "unbiased"],
        mmode = ["Bd2JpsiKstar"],
        myear = ['run2']
    ),
    #
    # }}}
    #
    #
    # csp factors {{{
    #
    # PUT PATH HERE
    #
    # }}}
    #
    #
    # angular acceptance {{{
    #
    expand(
        "output/tables/angular_acceptance/run2/Bs2JpsiPhi/{mversion}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mangacc = [
            f"{config['angacc']}",
            f"{config['angacc']}".replace('run2', 'yearly'),
        ],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}"],
        mtimeres = [f"{config['timeres']}"],
    ),
    #
    # }}}
    #
    #
    # physics parameters {{{
    # f"output/tables/lifetime/run2/Bd2JpsiKstar/{config['version']}@evtEven_simul3BdasBs_combined.tex",
    # f"output/tables/lifetime/run2/Bu2JpsiKplus/{config['version']}_simul3BuasBs_combined.tex",
    #
    # cuts on the tuple using Bd as control channel
    expand(
        "output/tables/physics_params/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [
            f"{config['version']}@LcosK+{config['version']}",
            f"{config['version']}@LcosK+{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4",
            f"{config['version']}@LcosK+{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3",
            f"{config['version']}@LcosK+{config['version']}@LcosKPV1+{config['version']}@LcosKPV2+{config['version']}@LcosKPV3",
        ],
        myear = [f"{config['year']}", "2015", "2016", "2017", "2018"],
        mfit = [f"{config['fit']}"],
        mangacc = [ f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}"],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    #
    expand(
        "output/tables/physics_params/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [
            f"{config['version']}@LcosK+{config['version']}@LcosKPV1+{config['version']}@LcosKPV2+{config['version']}@LcosKPV3",
            f"{config['version']}@LcosK+{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
            f"{config['version']}@LcosK+{config['version']}@LcosKLSB+{config['version']}@LcosKLSB",
            # f"{config['version']}@LcosK+{config['version']}@LcosKT1+{config['version']}@LcosKT2+{config['version']}@LcosKT3",
            f"{config['version']}@LcosK+{config['version']}@LcosKbkgcat60",
            # f"{config['version']}@LcosK+{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
        ],
        myear = [f"{config['year']}"],
        mfit = [f"{config['fit']}"],
        mangacc = [ f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}"],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    # syst
    expand(
        "output/tables/systematics/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [
            f"{config['version']}@LcosKPV1+{config['version']}@LcosKPV2+{config['version']}@LcosKPV3",
            f"{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
            f"{config['version']}@LcosKLSB+{config['version']}@LcosKRSB",
            # f"{config['version']}@LcosKT1+{config['version']}@LcosKT2+{config['version']}@LcosKT3",
            f"{config['version']}@LcosKbkgcat60",
            # f"{config['version']}@LcosK+{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
        ],
        myear = [f"{config['year']}"],
        mfit = [f"{config['fit']}"],
        mangacc = [ f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}"],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    expand(
        "output/tables/systematics/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [
            # f"{config['version']}@LcosKPV1+{config['version']}@LcosKPV2+{config['version']}@LcosKPV3",
            # f"{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
            # f"{config['version']}@LcosKLSB+{config['version']}@LcosKLSB",
            f"{config['version']}@LcosKT1+{config['version']}@LcosKT2+{config['version']}@LcosKT3",
            # f"{config['version']}@LcosKbkgcat60",
            # f"{config['version']}@LcosK+{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS",
        ],
        myear = [f"{config['year']}"],
        mfit = [f"{config['fit']}"],
        mangacc = [ f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}".replace('3', '2')],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    #
    # cuts on the tuple using Bd as control channel
    expand(
        "output/tables/physics_params/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [
            f"{config['version']}@LcosK",
            f"{config['version']}",
            # f"{config['version']}@LcosK+{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4",
            # f"{config['version']}@LcosK+{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3",
        ],
        myear = [f"{config['year']}", "2015+2016+2017+2018"],
        mfit = [f"{config['fit']}"],
        mangacc = [ f"{config['angacc']}".replace('run2', 'yearly')],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [f"{config['timeacc']}"],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    #
    # time acceptance variations
    expand(
        "output/tables/systematics/run2/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mfit = [f"{config['fit']}"],
        mangacc = [f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [
            f"{config['timeacc']}DGn0",  # using DG!=0 Bs MC
            f"{config['timeacc'].replace('3', '6')}",  # using 6 knots
            f"{config['timeacc']}BuasBd",  # using Bu as control channel
            f"{config['timeacc']}Noncorr"  # wihtout timeacc corerctions
        ],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    #
    # angular acceptance variations
    expand(
        "output/tables/systematics/run2/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mfit = [f"{config['fit']}"],
        mangacc = [f"analytic2knots1Dual+analytic2knots2Dual+analytic2knots3Dual"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [ f"{config['timeacc']}" ],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    expand(
        "output/tables/physics_params/run2/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}.tex",
        mversion = [f"{config['version']}@LcosK"],
        mfit = [f"{config['fit']}+{config['fit']}Poldep"],
        mangacc = [f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [ f"{config['timeacc']}" ],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    #
    # }}}
    #
    #
    # trigger cross-checks {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosK_auto_run2Dual_vgc_amsrd_simul3_amsrd_biased+unbiased.tex",
    # }}}
    # magnet cross-checks {{{
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKmagUp+{config['version']}@LcosKmagDown_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # yearly cross-checks {{{
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_yearly_yearly_vgc_amsrd_simul3_amsrd_combined.tex",
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_yearly_yearly_vgc_amsrd_simul3Noncorr_amsrd_combined.tex",
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3DGn0_amsrd_combined.tex",
    # }}}
    # pT and etaB cross-check with Bs and Bd as control {{{
    # run2
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_{config['angacc']}_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_{config['angacc']}_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    # yearly
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}_{config['timeres']}_{config['trigger']}.tex",
    # tag
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKonlyOST+{config['version']}@LcosKonlySST+{config['version']}@LcosKonlyOSS_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.tex",
    f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKPV1+{config['version']}@LcosKPV2+{config['version']}@LcosKPV3_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.tex",
    # }}}
    # pT and etaB cross-check with Bs and Bu as control {{{
    # run2
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_{config['angacc']}_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_{config['angacc']}_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # yearly
    # f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2015/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2016/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2017/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # f"output/tables/physics_params/2018/Bs2JpsiPhi/{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3_{config['fit']}_yearlyDual_{config['csp']}_{config['flavor']}_{config['timeacc']}BuasBd_{config['timeres']}_{config['trigger']}.tex",
    # }}}
    # time acceptance variations {{{
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3DGn0_amsrd_combined.tex",
    # f"output/tables/physics_params/run2/Bs2JpsiPhi/{config['version']}_run2_run2_vgc_amsrd_simul3Noncorr_amsrd_combined.tex",
    # }}}
    # }}}
    #
    #
    #
    # FIGURES {{{
    #
    # sweights {{{
    expand(rules.sweights_add.output.plots,
           version = [ f"{config['version']}@LcosK" ],
           mode = ['Bs2JpsiPhi', 'Bd2JpsiKstar',
                   # 'Bu2JpsiKplus'
            ],
           year = ['2015', '2016', '2017', '2018'],
           sweight=['sWeight'],
    ),
    expand(rules.mass_fit_bs_lb.output.plots,
           version = [ f"{config['version']}" ],
           mode = ['Bs2JpsiPhi_Lb'],
           year = ['2015', '2016', '2017', '2018'],
           sweight=['sWeight'],
           massbin=['all'],
           massmodel=['ipatiaChebyshev'],
           trigger=['kminus', 'kplus'],
    ),
    # }}}
    #
    # reweighting plots {{{
    #
    expand(rules.reweightings_plot_time_acceptance.output,
           version = [
               f"{config['version']}@LcosK",
           ],
           mode = ['MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar',
                   'Bd2JpsiKstar'],
           branch = ['pTB', 'pB', 'mHH'],
           year = ['2015', '2016', '2017', '2018'],
           trigger=['biased', 'unbiased'],
    ),
    #
    # }}}
    #
    # time resolution {{{
    expand(
       "output/figures/time_resolution_fit/{myear}/{mmode}/{mversion}_{mtimeres}1of1_{mtrigger}",
        mtimeres = [ f"triple" ],
        mmode = [ 'MC_Bs2JpsiPhi_Prompt', 'Bs2JpsiPhi_Prompt'],
        myear = [ '2015', '2016', '2017', '2018' ],
        mversion = [ f"{config['version']}" ],
        mtrigger = ['combined'],
    ),
    expand(
       "output/figures/time_resolution/{myear}/{mmode}/{mversion}_{mtimeres}",
        mtimeres = [ f"double10" ],
        mmode = [ 'Bs2JpsiPhi'],
        myear = [ '2015', '2016', '2017', '2018' ],
        mversion = [ f"{config['version']}" ],
    ),
    # }}}
    #
    # time acceptance plots {{{
    #
    # baseline time acceptance
    expand(
        rules.time_acceptance_simultaneous_plot.output,
        mtimeacc = [
            f"{config['timeacc']}",  # baseline
            # f"{config['timeacc']}".replace('3', '6'),  # using 6 knots
            # f"{config['timeacc']}Noncorr"  # wihtout timeacc corerctions
        ],
        mmode = [ 'MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar' ],
        myear = [ '2015', '2016', '2017', '2018' ],
        mversion = [
            f"{config['version']}@LcosK",
            f"{config['version']}@LcosK+{config['version']}",
            f"{config['version']}@LcosKpTB1+{config['version']}@LcosKpTB2+{config['version']}@LcosKpTB3+{config['version']}@LcosKpTB4",
            f"{config['version']}@LcosKetaB1+{config['version']}@LcosKetaB2+{config['version']}@LcosKetaB3",
        ],
        plot = ['fit', 'fitlog', 'spline', 'splinelog'],
        trigger = ['biased', 'unbiased'],
    ),
    expand(
        rules.time_acceptance_simultaneous_plot.output,
        mtimeacc = [
            f"{config['timeacc']}+{config['timeacc'].replace('3', '6')}",  # using 6 knots
            # f"{config['timeacc']}BuasBd",  # using Bu as control channel
            f"{config['timeacc']}+{config['timeacc']}Noncorr",   # wihtout timeacc corerctions
        ],
        mmode = [ 'MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar' ],
        myear = [ '2015', '2016', '2017', '2018' ],
        mversion = [ f"{config['version']}@LcosK", ],
        plot = ['fit', 'fitlog', 'spline', 'splinelog'],
        trigger = ['biased', 'unbiased'],
    ),
    #
    # time acceptance plots with DGn0
    # this rule cannot handle these plots yet!
    # expand(
    #     rules.time_acceptance_simultaneous_plot.output,
    #     mtimeacc = [
    #         f"{config['timeacc']}DGn0",  # using DG!=0 Bs MC
    #         f"{config['timeacc']}DGn0Noncorr"  # wihtout timeacc corerctions
    #     ],
    #     mmode = [ 'MC_Bs2JpsiPhi', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar' ],
    #     myear = [ '2015', '2016', '2017', '2018' ],
    #     mversion = [
    #         f"{config['version']}@LcosK",
    #         # f"{config['version']}@LcosK+{config['version']}"
    #     ],
    #     plot = ['fit', 'fitlog', 'spline', 'splinelog'],
    #     trigger = ['biased', 'unbiased'],
    # ),
    #
    # }}}
    #
    # 
    # csp factors {{{
    expand("output/figures/mass_efficiencies/{myear}/{mmode}/{mversion}_{mcsp}.pdf",
           mversion=["v4r0~mX6@LcosK"],
           myear=[2015, 2016, 2017, 2018],
           mmode=['MC_Bs2JpsiPhi'],
           mcsp = ['yearly4', 'yearly6']
           ),
    # }}}
    #
    # angular acceptance plots {{{
    #
    expand("output/figures/angular_acceptance/{year}/Bs2JpsiPhi/{version}_{angacc}Timedep_{csp}_{flavor}_{timeacc}_{timeres}_{trigger}.pdf",
            version=[f"{config['version']}@LcosK"],
            trigger=["biased", "unbiased"],
            timeacc=["simul3"],
            timeres=["amsrd"],
            flavor=["amsrd"],
            csp=["vgc"],
            angacc=["run2Dual"],
            year=[ 2015, 2016, 2017, 2018],
    ),
    expand("output/figures/angular_acceptance/{year}/{mode}/{version}_analytic2knots_{trigger}_{branch}.pdf",
            version=[f"{config['version']}@LcosK"],
            mode=["MC_Bs2JpsiPhi_dG0", "MC_Bs2JpsiPhi"],
            trigger=["biased", "unbiased"],
            branch=["cosL", "cosK", "hphi"],
            year=[ 2015, 2016, 2017, 2018],
    ),
    expand("output/figures/angular_acceptance/{year}/{mode}/{version}_analytic_{trigger}",
            version=[f"{config['version']}@LcosK"],
            mode=["MC_Bs2JpsiPhi_dG0", "MC_Bs2JpsiPhi"],
            trigger=["biased", "unbiased"],
            # branch=["cosL", "cosK", "hphi"],
            year=[ 2015, 2016, 2017, 2018],
    ),
    #
    # rwp2 = expand(rules.reweightings_plot_angular_acceptance.output,
    #               version=['v0r5'],
    #               mode=['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0'],
    #               branch=['B_P','B_PT','X_M','hplus_PT','hplus_P','hminus_PT','hminus_P'],
    #               angacc=['yearly'],
    #               timeacc=['repo']
    #               weight=['sWeight','kinWeight','kkpWeight'],
    #               year=['2015']),
    #               #year=['2015','2016','2017','2018']),
    # }}}
    #
    #
    # physics plots {{{
    #
    expand("output/figures/physics_params/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}/{mbranch}.pdf",
        mversion = [f"{config['version']}@LcosK"],
        mfit = [f"{config['fit']}"],
        myear = [f"{config['year']}"],
        mangacc = [f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [ f"{config['timeacc']}" ],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}", "biased", "unbiased"],
        mbranch = ["time", "cosK", "cosL", "hphi"],
    ),
    # expand("output/figures/angular_acceptance/{year}/{mode}/{version}_analytic2knots_{trigger}_{branch}.pdf",
    #         version=[f"{config['version']}@LcosK"],
    #         mode=["MC_Bs2JpsiPhi_dG0", "MC_Bs2JpsiPhi"],
    #         trigger=["biased", "unbiased"],
    #         branch=["cosL", "cosK", "hphi"],
    #         year=[ 2015, 2016, 2017, 2018],
    # ),
    #
    # rwp2 = expand(rules.reweightings_plot_angular_acceptance.output,
    #               version=['v0r5'],
    #               mode=['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0'],
    #               branch=['B_P','B_PT','X_M','hplus_PT','hplus_P','hminus_PT','hminus_P'],
    #               angacc=['yearly'],
    #               timeacc=['repo']
    #               weight=['sWeight','kinWeight','kkpWeight'],
    #               year=['2015']),
    #               #year=['2015','2016','2017','2018']),
    # }}}
    #
    # scans {{{
    expand("output/figures/physics_params/{myear}/Bs2JpsiPhi/{mversion}_{mfit}_{mangacc}_{mcsp}_{mflavor}_{mtimeacc}_{mtimeres}_{mtrigger}/scans",
        mversion = [f"{config['version']}@LcosK"],
        mfit = [f"{config['fit']}"],
        myear = [f"{config['year']}"],
        mangacc = [f"{config['angacc']}"],
        mcsp = [f"{config['csp']}"],
        mflavor = [f"{config['flavor']}"],
        mtimeacc = [ f"{config['timeacc']}" ],
        mtimeres = [f"{config['timeres']}"],
        mtrigger = [f"{config['trigger']}"],
    ),
    # }}}
    #
    #
    # }}}
    #
    #
    #
  output:
    f"output/{config['version']}.zip"
  run:
    import os
    import shutil
    # import tqdm
    from tqdm import tqdm
    v = config['version']

    all_files = []
    for this_input in input:
      if isinstance(this_input, list):
        print('its a list')
        for item in this_input:
          all_files.append(item)
      else:
        if os.path.isdir(this_input):
            _files = [f for f in os.listdir(this_input) if not f.startswith('.')]
            for f in _files:
                print(f)
                all_files.append(os.path.join(this_input, f))
        else:
            all_files.append(this_input)

    # print(all_files)
    # print(output)
    cpath = f'{output}'
    cpath = os.path.abspath(os.path.dirname(cpath)) + f"/{v}/"
    # print(cpath)

    # Remove directory if it exists
    if os.path.isdir(cpath):
        # print(f"rm -rf {cpath}")
        os.system(f"rm -rf {cpath}")
        # os.system(f"rm -rf {output[:-4]}")
    # exit()
    # Loop over all input files and make a copy of all pdfs
    with tqdm(total=len(all_files)) as pbar:
      for i, fn in enumerate(all_files):
        # print(i, fn)
        _fn = os.path.abspath(fn)
        _fn = _fn.replace(f"/{v}", "/thetuple")
        if _fn.endswith('.json'):
          out_path = _fn.replace('output/params/', f'output/{v}/jsons/')
        elif _fn.endswith('.tex'):
          out_path = _fn.replace('output/tables/', f'output/{v}/tables/')
        elif _fn.endswith('.pdf'):
          out_path = _fn.replace('output/figures/', f'output/{v}/plots/')
        else:
          print('Problems')
          out_path = None # add other methods if needed
        if out_path:
          # print(f"Copying {fn} to {out_path}...")
          os.system(f"mkdir -p {os.path.dirname(out_path)}") # create dir
          os.system(f"cp {fn} {out_path}")                 # copy file
          #shutil.copy2(f"{file}", f"{file.replace('output/',out_path)}")
        pbar.update(1)

    shutil.make_archive(f'{cpath}', 'zip', f'{cpath}')




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
           mversion=config['version'],
           mmode=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
           mtimeacc=['simul3', 'simul3Noncorr'],
           myear=['2015', '2016', '2017', '2018'],
           plot=['fitlog', 'splinelog'],
           trigger=['biased', 'unbiased']),
    # lifetime trend plots {{{
    expand(rules.lifetime_trend.output,
           version=config['version'],
           mode=['Bs2JpsiPhi', 'Bu2JpsiKplus', 'Bd2JpsiKstar'],
           timeacc=['single', 'singleNoncorr'],
           year=['run2']),
    # expand(rules.time_acceptance_simultaneous_plot.output,
    #        mversion=config['version'],
    #        mode=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
    #        mtimeacc=['simul3', 'simul3Noncorr'],
    #        myear=['2015', '2016', '2017', '2018'],
    #        plot=['fitlog', 'splinelog'],
    #        trigger=['biased', 'unbiased']),
    # # lifetime trend plots {{{
    # expand(rules.lifetime_trend.output,
    #        version=config['version'],
    #        mode=['Bs2JpsiPhi', 'Bu2JpsiKplus', 'Bd2JpsiKstar'],
    #        timeacc=['single', 'singleNoncorr'],
    #        year=['run2']),
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
