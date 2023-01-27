# plot_mva
#
#

__all__ = []
__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@CERN.CH"]


import uproot3 as uproot
from ipanema import Sample, Parameters, Parameter
import complot
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from utils.helpers import trigger_scissors
from utils.plot import mode2tex, get_range, watermark, mode_tex, get_var_in_latex



#To be inputs of the script
years = ["2016", "2017", "2018"]
target_mode = "Bs2JpsiPhi"
control_mode = "MC_Bs2JpsiPhi"
version = "v4r0"


control_state = "selected"
target_state = "selected"

if "MC_" in target_mode:
  weight_target = "gb_weights"
if "MC_" in control_mode:
  weight_control = "gb_weights"

if "MC_" not in target_mode:
  if "Bs" in target_mode:
    weight_target = "sigBsSW"
  if "Bd" in target_mode:
    weight_target = "sigBdSW"
  target_state = "sWeight"

if "MC_" not in control_mode:
  if "Bs" in control_mode:
    weight_control = "sigBsSW"
  if "Bd" in control_mode:
    weight_control = "sigBdSW"
  control_state = "sWeight"


limits = {
	          'max_K_TRCHI2_mva' :  [0, 5.],
	          'log_min_K_PNNk_mva' : [-6., 0.],
	          'max_mu_TRCHI2_mva' : [0., 4.],
	          'log_min_mu_PNNmu_mva' : [-6., 0.],
	          'log_Jpsi_VX_CHI2NDOF_mva' : [-35., 2.],
	          'X_PT_mva' : [0, 3.5e4],
          	'B_VX_CHI2NDOF_mva' : [0., 20.],
          	'log_B_DTF_CHI2NDOF_mva' : [-5., 8.],
          	'log_B_IPCHI2_mva' : [-13., 13.],
          	'B_PT_mva' : [0, 8e4]
}

for year in years:
  # #The script begins
  target_path = f"/scratch49/marcos.romero/sidecar/{year}/{target_mode}/{version}_{target_state}.root"
  control_path = f"/scratch49/marcos.romero/sidecar/{year}/{control_mode}/{version}_{control_state}.root"

  branches_toplot = [i.decode() for i in uproot.open(f"{target_path}")["DecayTree"].keys() if "mva" in i.decode()]
  branches_target = branches_toplot + list(weight_target.split(","))
  branches_control = branches_toplot + list(weight_control.split(","))


  target = Sample.from_root(target_path, branches=branches_target).df
  control = Sample.from_root(control_path, branches=branches_control).df

  for b in branches_toplot:
    x, y, pull = complot.compare_hist(target[f"{b}"], control[f"{b}"],
								        target.eval(f"{weight_target}"), control.eval(f"{weight_control}"),
												range = limits[f'{b}'],
								        bins=100 , density=True)

    fig, axplot, axpull = complot.axes_plotpull()
  
    axplot.errorbar(x.bins, x.counts,
					         yerr=x.yerr[0], xerr=x.xerr,
					         color='r', fmt='.',
                   label=f"{target_mode} {year}")

    axplot.errorbar(y.bins, y.counts,
				            yerr=y.yerr[0], xerr=y.xerr,
					          color='k', fmt='.',
										label=f"{control_mode} {year}")


    if ( ("log" in b) and ("PNN" in b) ):
      axplot.set_yscale('log')
      axplot.set_ylim(1.e-7, 1.2)

    axplot.set_xlim(limits[f'{b}'])
    axpull.fill_between(y.bins, x.counts/y.counts, 1, facecolor='r')
    axpull.set_xlabel(f"{b.replace('_mva', '')}")
    axpull.set_ylabel("ratio")
    axpull.set_ylim(0., 2)
    axpull.set_yticks([0.5,1, 1.5])
    axplot.legend()
    os.system(f"mkdir -p plot_mva/{target_mode}_vs_{control_mode}/{year}/")
    fig.savefig(f"plot_mva/{target_mode}_vs_{control_mode}/{year}/{b}.pdf")
#
#
#
# # vim: fdm=marker ts=2 sw=2 sts=2 sr noet
