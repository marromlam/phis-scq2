# optimize_bdt_cut
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import os
import json
import argparse
import yaml
import numpy 
import cppyy
import math
from array import array

import numpy as np
import multiprocessing
import ipanema
import complot

from selection.gb_weights.background_subtraction import mass_fitter


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-file', help='Name of the input file')
    p.add_argument('--input-tree-name', default='DecayTree', help='Name of the input tree')
    p.add_argument('--input-branch', default='B_ConstJpsi_M_1', help='Name of the branch to be fitted')
    p.add_argument('--bdt-branch', default='bdtg3', help='Name of the BDT branch to be cut')
    p.add_argument('--input-params', help='Location in which to save the plots')
    p.add_argument('--output-figures', help='Location in which to save the plots')
    p.add_argument('--output-file', default='FOM_Bs2JpsiPhi_2016.pdf', help='name of output FOM plot')
    p.add_argument('--mode', help='Name of the selection in yaml')
    p.add_argument('--version', help='Name of the selection in yaml')
    p.add_argument('--year', required=True, help='Year of data taking')
    p.add_argument('--bdtcut', default='-1.0', help='pre-bdtg3 cut')
    p.add_argument('--cuts', default='1', help='pre-selection cut')
    p.add_argument('--mass-model', default='hypatia', help='pre-selection cut')
    p.add_argument('--mass-weight', default=False, help='pre-selection cut')
    p.add_argument('--mass-branch', default='B_ConstJpsi_M_1', help='pre-selection cut')
    args = vars(p.parse_args())

    bdt_branch = args['bdt_branch']
    cuts = args['cuts']
    

    # branches
    branches = [args['mass_branch'], args['bdt_branch']]
    branches = branches + ["B_ConstJpsi_MERR_1"]

    if args["input_params"] and args['mass_model'] != 'dgauss':
        input_pars = ipanema.Parameters.load(args["input_params"])
    else:
        input_pars = False


    if args["mass_weight"]:
        mass_weight = args["mass_weight"]
        branches += [mass_weight]
    else:
        mass_weight = f"{branches[0]}/{branches[0]}"

    sample = ipanema.Sample.from_root(args["input_file"], branches=branches)

    # allocate arrays for the curve
    number_of_bins = 100
    x_min, x_max = np.min(sample.df[bdt_branch]), np.max(sample.df[bdt_branch])
    x_min, x_max = np.floor(x_min), np.ceil(x_max)
    print(x_min, x_max)
    x_arr = np.linspace(x_min, x_max, number_of_bins+1)
    y_arr = 0 * x_arr

    for xi, vx in enumerate(x_arr[:-1]):
        # ccut = f"({cuts}) & ({bdt_branch}>{vx:.2f})"
        ccut = f"({bdt_branch}>{vx:.2f})"
        print("\n\n ----------------------------------------")
        print(f"Fitting {xi+1} bin: {ccut}")

        pars, sw = mass_fitter(
            sample.df,
            mass_range=False,  # we compute it from the mass branch range
            mass_branch=args['mass_branch'],  # branch to fit
            mass_weight=mass_weight,  # weight to apply to the likelihood
            figs=args["output_figures"],  # where to save the fit plots
            model=args["mass_model"],  # mass model to use
            cut=ccut,  # extra cuts, if required
            sweights=True,  #  whether to comput or not the sWeights
            input_pars=input_pars,  # whether to use prefit tail parameters or not
            verbose=False,  # level of verobisty
            # prefit=False,
            free_tails=False,
            mode=args['mode']
        )
        # exit()

        # get fraction of signal
        psig = ipanema.Parameters.find(pars, "fsig.*")[0]
        pbkg = 'fcomb'
        fsig = pars[psig].uvalue
        fbkg = pars['fcomb'].uvalue
        print(f"{bdt_branch}>{vx:.2f} : fsig={fsig:.2uP} & fbkg={fbkg:.2uP}")
        sw_sig = sw[psig]
        sw_bkg = sw[pbkg]
        print(f"{bdt_branch}>{vx:.2f} : sum signal sw={sw_sig.sum():.2f}")
        print(f"{bdt_branch}>{vx:.2f} : sum backgr sw={sw_bkg.sum():.2f}")
        fom = np.sum(sw_sig)**2 / np.sum(sw_sig**2)
        print(f"{bdt_branch}>{vx:.2f} : FOM={fom:.2f}\n")
        y_arr[xi] = fom

    # find best fom
    data = ( np.array((y_arr, x_arr)).T ).tolist()
    best_fom = max(data, key=lambda x: (x[0], -x[1]))
    print("Best fom:", best_fom)
    best_y, best_x = best_fom

    # create an illustrative plot
    hbdt = complot.hist(sample.df[bdt_branch], number_of_bins,
                        range=(x_min, x_max))
    fig, axplot = complot.axes_plot()
    # plot histogram
    axplot.fill_between(hbdt.bins, hbdt.counts, 0, step='post', facecolor='k', alpha=0.3)
    axplot.set_ylabel("Candidates")
    axplot2 = axplot.twinx()
    axplot2.plot(x_arr, y_arr, '-')
    axplot2.plot(x_arr, y_arr, '.', color='C0')
    axplot2.plot(best_x, best_y, 'o', color='C2',
                 label=f'Optimal cut at {best_x:.2f}')
    axplot.set_xlabel(bdt_branch.replace('_', '-'))
    axplot2.set_ylabel(r"f.o.m = $\frac{(\sum w_i)^2}{\sum_i w_i^2}$")
    axplot.set_xlim(np.min(hbdt.bins), np.max(hbdt.bins))
    axplot.set_yscale('log')
    from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
    # fig.savefig("bdtshit.pdf")
    v_mark = args['version'].split('@')[0]  # watermark plots
    tag_mark = ''
    # if v_mark[0] == 'b' or v_mark == 'v1r1':
    v_mark = 'LHC$b$'  # watermark plots
    tag_mark = 'THIS THESIS' 
    watermark(axplot2, version=v_mark, tag=tag_mark, scale=1.3)
    axplot2.legend(loc='lower left')
    fig.savefig(f"{args['output_figures']}/fom.pdf")

    # save best bdt_branch cut
    with open(args['output_file'], 'w') as f:
        f.write(f"{best_x:.2f}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
