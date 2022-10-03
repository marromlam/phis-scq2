DESCRIPTION = """
    adfghjghkgfdgdhfjgghfgfdgsdhfjhgfdgh
"""

__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


# Modules {{{

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import numpy as np
import uncertainties as unc
from uncertainties import unumpy as unp
import complot

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
from utils.helpers import version_guesser, trigger_scissors

# get bsjpsikk and compile it with corresponding flags
from analysis import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

from analysis.angular_acceptance.new_merger import merge_std_dg0

# }}}


def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-std',help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-dg0',help='Bs2JpsiPhi MC sample')
  p.add_argument('--kkpweight-std',help='Bs2JpsiPhi MC sample')
  p.add_argument('--kkpweight-dg0',help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-std',help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-dg0',help='Bs2JpsiPhi MC sample')
  p.add_argument('--timeacc',help='Bs2JpsiPhi MC sample')
  p.add_argument('--angacc',help='Bs2JpsiPhi MC sample')
  p.add_argument('--figure',help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode',help='Configuration')
  p.add_argument('--year',help='Year of data-taking')
  p.add_argument('--version',help='Year of data-taking')
  p.add_argument('--trigger',help='Trigger(s) to fit')
  p.add_argument('--shit',help='Save a fit report with the results')

  return p

"""
YEAR = 2016
VERSION = 'v0r5'
MODE = 'Bs2JpsiPhi'
TRIGGER = 'unbiased'

std_params = f'analysis/params/generator/2016/MC_Bs2JpsiPhi.json'
dg0_params = f'analysis/params/generator/2016/MC_Bs2JpsiPhi_dG0.json'

std_sample = f'/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi/v0r5.root'
dg0_sample = f'/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5.root'

std_weights = f'/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi/v0r5_run2_simul_kkpWeight.root'
dg0_weights = f'/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5_run2_simul_kkpWeight.root'

#angular_acceptance = f'output/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_run2_simul_{TRIGGER}.json'


timeaccs = {
  'biased': f'output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul_biased.json',
  'unbiased': f'output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul_unbiased.json'
}


nominals = {
  'biased': f'output/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_run2_simul_biased.json',
  'unbiased': f'output/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_run2_simul_unbiased.json'
}

figspath = 'output/figures/angular_acceptance/2016/Bs2JpsiPhi/v0r5_run2Bintime_simul_unbiased.pdf'
"""

#print(input_params_path,sample_path,output_tables_path)
"""
YEAR = args['year']
VERSION = args['version']
MODE = args['mode']
TRIGGER = args['trigger']
input_params_path = args['input_params']
sample_path = args['sample']
output_tables_path = args['output_tables']
output_params_path = args['output_params']
"""
#print(input_params_path,sample_path,output_tables_path)

################################################################################
################################################################################
################################################################################


def plot_comparison_binned_angular_weights(pars, knots, version):
  print(pars)
  nterms = len(pars[0])   # triangle number of the number of amplitudes T(4)=10
  pcols = 3
  prows = (nterms-1)//pcols + (1 if (nterms-1)%pcols else 0)

  fig, ax = plt.subplots(prows, pcols, sharex=True, figsize=[prows*4.8,3*4.8])
  for i,wi in enumerate(list(pars[0].keys())[1:]):
    #print(f"{i} -> {i//pcols ,i%pcols}")
    ax[i//pcols ,i%pcols].fill_between(
      [knots[0], knots[-1]],
      2*[pars[0][wi].value+pars[0][wi].stdev],
      2*[pars[0][wi].value-pars[0][wi].stdev],
      alpha=0.5, label="base")
    ax[i//pcols ,i%pcols].set_ylabel(f"$w_{i+1}$")
    for bin in range( len(knots)-1 ):
      ax[i//pcols ,i%pcols].errorbar(
        (knots[bin]+knots[bin+1])*0.5,
        pars[int(bin+1)][wi].value,
        xerr=knots[bin+1]-(knots[bin]+knots[bin+1])*0.5,
        yerr=pars[bin+1][wi].stdev, fmt='.', color='k')
    #ax[i//pcols ,i%pcols].legend()
  watermark(ax[0,0], version=f"${version}$", scale=1.01)
  [make_square_axes(ax[ix,iy]) for ix,iy in np.ndindex(ax.shape)]
  [tax.set_xlabel(f"$t\,\mathrm{{[ps]}}$") for tax in ax[prows-1,:]]
  return fig, ax


if __name__ == '__main__':
  args = vars(argument_parser().parse_args())
  TRIGGER = args['trigger']
  std_params = args['params_std']
  dg0_params = args['params_dg0']

  # Load DGn0 Monte Carlo sample
  stdmc = Sample.from_root(args['sample_std'])
  kin = Sample.from_root(args['kkpweight_std'], treename='DecayTree')
  stdmc.df = pd.concat([stdmc.df,kin.df],axis=1)
  del kin
  stdmc.assoc_params(std_params.replace('TOY','MC').replace('2021','2018'))

  # Load DG0 Monte Carlo sample
  dg0mc = Sample.from_root(args['sample_dg0'])
  kin = Sample.from_root(args['kkpweight_dg0'], treename='DecayTree')
  dg0mc.df = pd.concat([dg0mc.df,kin.df],axis=1)
  del kin
  dg0mc.assoc_params(dg0_params.replace('TOY','MC').replace('2021','2018'))

  # Calculating the weight
  n = len(stdmc.find('kkp.*')) # get last iteration number
  strweight = f'angWeight*kkpWeight{n}*polWeight*sWeight'
  stdmc.df['weight'] = stdmc.df.eval(strweight).values
  dg0mc.df['weight'] = dg0mc.df.eval(strweight).values

  # Load acceptances
  timeacc = Parameters.load(args['timeacc'])
  knots = np.array( Parameters.build(timeacc,timeacc.find('k.*')+['tUL'])  )

  stdmc.subdfs = []
  for i in range(0,len(knots)-1):
    stdmc.subdfs.append(stdmc.df.query(
        trigger_scissors(TRIGGER,f'time >= {knots[i]} & time < {knots[i+1]}') ))

  dg0mc.subdfs = []
  for i in range(0,len(knots)-1):
    dg0mc.subdfs.append(dg0mc.df.query(
        trigger_scissors(TRIGGER,f'time >= {knots[i]} & time < {knots[i+1]}') ))

  # Variables and branches to be used
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  reco += ['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH']
  true += ['mHH', '0*mHH', 'genidB', 'genidB', '0*mHH', '0*mHH']

  #%% Compute angWeights without corrections -----------------------------------
  #     Let's start computing the angular weights in the most naive version, w/
  #     any corrections
  print(f"\nCompute angWeights for std MC\n{80*'='}\n")
  angacc_std = []
  for i in range(0,len(knots)-1):
    tLL = knots[i]
    tUL = knots[i+1]
    badjanak.config['knots'] = [tLL, tUL]
    print(f'Computing angular weights for time >= {knots[i]} & time < {knots[i+1]}')
    vt = ristra.allocate( np.stack( stdmc.subdfs[i].eval(true), axis=-1) )
    vr = ristra.allocate( np.stack( stdmc.subdfs[i].eval(reco), axis=-1) )
    vw = ristra.allocate(stdmc.subdfs[i]['weight'].values)
    ans = badjanak.get_angular_acceptance_weights(vt, vr, vw, tLL=tLL, tUL=tUL, **stdmc.params.valuesdict() )
    w, uw, cov, corr = ans
    temp = Parameters()
    for k in range(0,len(w)):
      correl = {f'w{j}{TRIGGER[0]}': corr[k][j] for j in range(0, len(w)) if k > 0 and j > 0}
      temp.add({'name': f'w{k}{TRIGGER[0]}', 'value': w[k], 'stdev': uw[k],
                'free': False, 'latex': f'w_{k}^{TRIGGER[0]}', 'correl': correl})
    angacc_std.append( temp )
    print(temp)

  print(f"\nCompute angWeights for DG0 MC\n{80*'='}\n")
  angacc_dg0 = []
  for i in range(0,len(knots)-1):
    tLL = knots[i]
    tUL = knots[i+1]
    badjanak.config['knots'] = [tLL, tUL]
    print(f'Computing angular weights for time >= {knots[i]} & time < {knots[i+1]}')
    vt = ristra.allocate( np.stack( dg0mc.subdfs[i].eval(true), axis=-1) )
    vr = ristra.allocate( np.stack( dg0mc.subdfs[i].eval(reco), axis=-1) )
    vw = ristra.allocate(dg0mc.subdfs[i]['weight'].values)
    ans = badjanak.get_angular_acceptance_weights(vt, vr, vw, tLL=tLL, tUL=tUL, **dg0mc.params.valuesdict() )
    w, uw, cov, corr = ans
    temp = Parameters()
    for k in range(0,len(w)):
      correl = {f'w{j}{TRIGGER[0]}': corr[k][j] for j in range(0, len(w)) if k > 0 and j > 0}
      temp.add({'name': f'w{k}{TRIGGER[0]}', 'value': w[k], 'stdev': uw[k],
                'free': False, 'latex': f'w_{k}^{TRIGGER[0]}', 'correl': correl})
    angacc_dg0.append( temp )
    print(temp)

  print(f"\nCombine MC angular acceptances in one\n{80*'='}\n")
  angaccs = []
  for i in range(0,len(knots)-1):
    angaccs.append( merge_std_dg0(angacc_std[i], angacc_dg0[i]) )
    # print(args['angacc'].replace(f"_run2_",f"_run2Time{i+1}_"))
    # angaccs[-1].dump(args['angacc'].replace(f"_run2_",f"_run2Time{i+1}_"))

  print(f"\nCreate figure\n{80*'='}\n")
  base = Parameters.load(args['angacc'])
  print(base)
  [print(a) for a in angaccs]
  fig, ax = plot_comparison_binned_angular_weights([base,*angaccs], knots,
                                                   args['version'].split('@')[0])
  fig.savefig(args['figure'])
