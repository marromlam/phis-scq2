#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']




################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import uncertainties as unc
from uncertainties import unumpy as unp
from scipy.stats import chi2
from timeit import default_timer as timer

# reweighting config
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from hep_ml.metrics_utils import ks_2samp_weighted

# threading
import logging
import threading
import time
import multiprocessing

# load ipanema
from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import ristra, Sample, Parameters, Parameter, optimize

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# reweighting config
from hep_ml import reweight
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
print(bdconfig)
reweighter = reweight.GBReweighter(**bdconfig)
#reweighter = reweight.GBReweighter(n_estimators=40, learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={'subsample': 1})
#reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 1})

def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  if checker == 0:
    return True
  return False

# Parse arguments for this script
def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample-mc-std',
                      default = 'samples/MC_Bs2JpsiPhi_dG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-mc-dg0',
                      default = 'samples/MC_Bs2JpsiPhi_dG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-data',
                      default = 'samples/MC_Bs2JpsiPhi_dG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--params-mc-std',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--params-mc-dg0',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angular-weights-mc-std',
                      default = 'output/time_acceptance/parameters/2016/Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angular-weights-mc-dg0',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-weights-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-weights-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-coeffs-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-coeffs-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-csp',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-time-resolution',
                      default = 'output/time_resolution/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-flavor-tagging',
                      default = 'output/time_resolution/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-weights-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-weights-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables-biased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-tables-unbiased',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_dG0/v0r1_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--weigths-branch',
                      default = 'yearly_base',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'test',
                      help='Year of data-taking')

  return parser


# Parse arguments
#try:
args = vars(argument_parser().parse_args())
YEARS = [int(y) for y in args['year'].split(',')] # years are int
VERSION = args['version']

samples_std   = args['sample_mc_std'].split(',')
samples_dg0   = args['sample_mc_dg0'].split(',')
samples_data  = args['sample_data'].split(',')

input_std_params = args['params_mc_std'].split(',')
input_dg0_params = args['params_mc_dg0'].split(',')

angWeight_std = args['angular_weights_mc_std'].split(',')
angWeight_dg0 = args['angular_weights_mc_dg0'].split(',')

w_biased      = args['input_weights_biased'].split(',')
w_unbiased    = args['input_weights_unbiased'].split(',')

coeffs_biased      = args['input_coeffs_biased'].split(',')
coeffs_unbiased    = args['input_coeffs_unbiased'].split(',')

csp_factors    = args['input_csp'].split(',')
time_resolution = args['input_time_resolution'].split(',')
flavor_tagging = args['input_flavor_tagging'].split(',')

params_biased      = args['output_weights_biased'].split(',')
params_unbiased    = args['output_weights_unbiased'].split(',')
tables_biased      = args['output_tables_biased'].split(',')
tables_unbiased    = args['output_tables_unbiased'].split(',')

# If version is v0r1, you will be running over old tuples, I guess you
# pursuit to reproduce HD-fitter results. So I will change a little the config
if VERSION == 'v0r1':
  reweighter = reweight.GBReweighter(n_estimators=40,learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={"subsample": 1})
  input_std_params = args['params_mc_std'].replace("generator","generator_old").split(',')
  input_dg0_params = args['params_mc_dg0'].replace("generator","generator_old").split(',')


print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
print(f"{'version':>15}: {VERSION:50}")
print(f"{'year(s)':>15}: {YEARS:50}")
print(f"{'cuts':>15}: {'time>=0.3 & time<=15':50}")
print(f"{'bin cuts':>15}: {CUT if CUT else 'None':50}")
print(f"{'bdtconfig':>15}: {list(bdconfig.values()).join(':'):50}\n")





# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\n", "Loading samples", f"\n{80*'='}\n")

# Lists of MC variables to load and build arrays
reco = ['cosK', 'cosL', 'hphi', 'time']
true = ['true'+i+'_GenLvl' for i in reco]
reco += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']
true += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']
weight_mc='(polWeight*sw/gb_weights)'

# Lists of data variables to load and build arrays
real  = ['cosK','cosL','hphi','time']                        # angular variables
real += ['X_M','sigmat']                                     # mass and sigmat
real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging
#real += ['0*B_ID','0*B_ID', '0*B_ID', '0*B_ID']  # tagging
weight_rd='(sw)'

# Load Monte Carlo samples
mc = {}
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y} MC samples')
  mc[f'{y}'] = {}
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[samples_std,samples_dg0]):
    print(f' *  Loading {m}-{y} sample from\n    {v[i]}')
    mc[f'{y}'][f'{m}'] = Sample.from_root(v[i])
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[input_std_params,input_dg0_params]):
    print(f' *  Associating {m}-{y} parameters from\n    {v[i]}')
    mc[f'{y}'][f'{m}'].assoc_params(v[i])
  for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[angWeight_std,angWeight_dg0]):
    print(f' *  Attaching {m}-{y} kinWeight from\n    {v[i]}')
    mc[f'{y}'][f'{m}'].kinWeight = uproot.open(v[i])['DecayTree'].array('kinWeight')
    mc[f'{y}'][f'{m}'].path_to_weights = v[i]
    print(f"    {mc[f'{y}'][f'{m}'].kinWeight}")

for y, modes in mc.items():
  for m, v in modes.items():
    print(f' *  Allocating arrays in device for {m}-{y}')
    mc[f'{y}'][f'{m}'].allocate(reco=reco)
    mc[f'{y}'][f'{m}'].allocate(true=true)
    mc[f'{y}'][f'{m}'].allocate(pdf='0*time', ones='time/time', zeros='0*time')
    mc[f'{y}'][f'{m}'].allocate(weight=weight_mc)
    mc[f'{y}'][f'{m}'].allocate(biased='Jpsi_Hlt1DiMuonHighMassDecision_TOS==0')
    mc[f'{y}'][f'{m}'].allocate(unbiased='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
    mc[f'{y}'][f'{m}'].angular_weights = {'biased':0, 'unbiased':0}
    mc[f'{y}'][f'{m}'].kkpWeight = {}
    mc[f'{y}'][f'{m}'].pdfWeight = {}


# Load corresponding data sample
data = {}
mass = badjanak.config['x_m']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[f'{y}'] = {}
  data[f'{y}']['combined'] = Sample.from_root(samples_data[i])
  csp = Parameters.load(csp_factors[i])
  csp = csp.build(csp,csp.find('CSP.*'))
  resolution = Parameters.load(time_resolution[i])
  flavor = Parameters.load(flavor_tagging[i])
  data[f'{y}'][f'combined'].csp = csp
  data[f'{y}'][f'combined'].flavor = flavor
  data[f'{y}'][f'combined'].resolution = resolution
  for t, T in zip(['biased','unbiased'],[0,1]):
    print(f' *  Loading {y} sample in {t} category\n    {samples_data[i]}')
    this_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={T}) & (time>=0.3) & (time<=15)'
    data[f'{y}'][f'{t}'] = Sample.from_root(samples_data[i], cuts=this_cut)
    data[f'{y}'][f'{t}'].csp = csp
    data[f'{y}'][f'{t}'].flavor = flavor
    data[f'{y}'][f'{t}'].resolution = resolution
    print(csp)
    print(resolution)
    print(flavor)
  for t, coeffs in zip(['biased','unbiased'],[coeffs_biased,coeffs_unbiased]):
    print(f' *  Associating {y}-{t} time acceptance[{i}] from\n    {coeffs[i]}')
    c = Parameters.load(coeffs[i])
    print(c)
    data[f'{y}'][f'{t}'].timeacc = Parameters.build(c,c.fetch('c.*'))
    data[f'{y}'][f'{t}'].tLL = c['tLL'].value
    data[f'{y}'][f'{t}'].tUL = c['tUL'].value
  for t, weights in zip(['biased','unbiased'],[w_biased,w_unbiased]):
    print(f' *  Associating {y}-{t} angular weights from\n    {weights[i]}')
    w = Parameters.load(weights[i])
    data[f'{y}'][f'{t}'].angacc = Parameters.build(w,w.fetch('w.*'))
    data[f'{y}'][f'{t}'].angular_weights = [Parameters.build(w,w.fetch('w.*'))]
  for t, path in zip(['biased','unbiased'],[params_biased,params_unbiased]):
    data[f'{y}'][f'{t}'].params_path = path[i]
    print(path[i])
  for t, path in zip(['biased','unbiased'],[tables_biased,tables_unbiased]):
    data[f'{y}'][f'{t}'].tables_path = path[i]
    print(path[i])
  print(f' *  Allocating {y} arrays in device ')
  for d in [data[f'{y}']['biased'],data[f'{y}']['unbiased']]:
    sw = np.zeros_like(d.df['sw'])
    for l,h in zip(mass[:-1],mass[1:]):
      pos = d.df.eval(f'X_M>={l} & X_M<{h}')
      this_sw = d.df.eval(f'sw*(X_M>={l} & X_M<{h})')
      sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
    d.df['sWeight'] = sw
    d.allocate(data=real,weight='sWeight',lkhd='0*time')
#exit()
















#%% define likelihood


pars = Parameters()
list_of_parameters = [#
# S wave fractions
Parameter(name='fSlon1', value=0.480, min=0.00, max=0.90,
          free=True, latex=r'f_S^{1}'),
Parameter(name='fSlon2', value=0.040, min=0.00, max=0.90,
          free=True, latex=r'f_S^{2}'),
Parameter(name='fSlon3', value=0.004, min=0.00, max=0.90,
          free=True, latex=r'f_S^{3}'),
Parameter(name='fSlon4', value=0.009, min=0.00, max=0.90,
          free=True, latex=r'f_S^{4}'),
Parameter(name='fSlon5', value=0.059, min=0.00, max=0.90,
          free=True, latex=r'f_S^{5}'),
Parameter(name='fSlon6', value=0.130, min=0.00, max=0.90,
          free=True, latex=r'f_S^{6}'),
# P wave fractions
Parameter(name="fPlon", value=0.5240, min=0.4, max=0.6,
          free=True, latex=r'f_0'),
Parameter(name="fPper", value=0.2500, min=0.1, max=0.3,
          free=True, latex=r'f_{\perp}'),
# Weak phases
Parameter(name="pSlon", value= 0.00, min=-1.0, max=1.0,
          free=False, latex=r"\phi_S - \phi_0"),
Parameter(name="pPlon", value= 0.07, min=-1.0, max=1.0,
          free=True , latex=r"\phi_0" ),
Parameter(name="pPpar", value= 0.00, min=-1.0, max=1.0,
          free=False, latex=r"\phi_{\parallel} - \phi_0"),
Parameter(name="pPper", value= 0.00, min=-1.0, max=1.0,
          free=False, latex=r"\phi_{\perp} - \phi_0"),
# S wave strong phases
Parameter(name='dSlon1', value=+2.34, min=-0.0, max=+3.0,
          free=True, latex=r"\delta_S^{1} - \delta_{\perp}"),
Parameter(name='dSlon2', value=+1.64, min=-0.0, max=+3.0,
          free=True, latex=r"\delta_S^{2} - \delta_{\perp}"),
Parameter(name='dSlon3', value=+1.09, min=-0.0, max=+3.0,
          free=True, latex=r"\delta_S^{3} - \delta_{\perp}"),
Parameter(name='dSlon4', value=-0.25, min=-3.0, max=+0.0,
          free=True, latex=r"\delta_S^{4} - \delta_{\perp}"),
Parameter(name='dSlon5', value=-0.48, min=-3.0, max=+0.0,
          free=True, latex=r"\delta_S^{5} - \delta_{\perp}"),
Parameter(name='dSlon6', value=-1.18, min=-3.0, max=+0.0,
          free=True, latex=r"\delta_S^{6} - \delta_{\perp}"),
# P wave strong phases
Parameter(name="dPlon", value=0.000, min=-2*3.14, max=2*3.14,
          free=False, latex=r"\delta_0"),
Parameter(name="dPpar", value=3.260, min=-2*3.14, max=2*3.14,
          free=True, latex=r"\delta_{\parallel} - \delta_0"),
Parameter(name="dPper", value=3.026, min=-2*3.14, max=2*3.14,
          free=True, latex=r"\delta_{\perp} - \delta_0"),
# lambdas
Parameter(name="lSlon", value=1.0, min=0.7, max=1.6,
          free=False, latex="\lambda_S/\lambda_0"),
Parameter(name="lPlon", value=1.0, min=0.7, max=1.6,
          free=True,  latex="\lambda_0"),
Parameter(name="lPpar", value=1.0, min=0.7, max=1.6,
          free=False, latex="\lambda_{\parallel}/\lambda_0"),
Parameter(name="lPper", value=1.0, min=0.7, max=1.6,
          free=False, latex="\lambda_{\perp}/\lambda_0"),
# life parameters
Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
          free=False, latex=r"\Gamma_d"),
Parameter(name="DGs", value= 0.0917, min= 0.03, max= 0.15,
          free=True, latex=r"\Delta\Gamma_s"),
Parameter(name="DGsd", value= 0.03, min=-0.2, max= 0.2,
          free=True, latex=r"\Gamma_s - \Gamma_d"),
Parameter(name="DM", value=17.768, min=16.0, max=20.0,
          free=True, latex=r"\Delta m"),
#
]
pars.add(*list_of_parameters);


print(pars)


















def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():
    for dt in [dy['unbiased'],dy['biased']]:
      bsjpsikk.delta_gamma5_data(dt.data, dt.lkhd, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  **dt.flavor.valuesdict(), tLL=dt.tLL, tUL=dt.tUL)
      #dt.df['pdf'] = dt.lkhd.get()
      #dt.df.eval("check = (time > 5*sigmat)", inplace=True)
      chi2.append( -2.0 * (ristra.log(dt.lkhd) * dt.weight).get() );
  # shit = pd.concat([dy['biased'].df, dy['unbiased'].df]) .sort_values(by=['entry']) [['time','sigmat','check','pdf']]
  # print(shit.query("check==False"))
  # my_pdf = np.array(shit['pdf'])
  # peilian_pdf = np.genfromtxt('/home3/marcos.romero/log_pdf_2015',delimiter=',')
  # peilian_pdf = peilian_pdf[np.argsort(peilian_pdf[:, 0])]
  # my_n = 29804
  # print(peilian_pdf[:my_n,0])
  # peilian_pdf = peilian_pdf[:my_n,1]
  # my_pdf= my_pdf[10:my_n+10]
  # print(peilian_pdf)
  # print(my_pdf)
  # print(f"{'idx':>8} | marcos | peilian")
  # for i in range(0,my_n):
  #   if (np.abs(my_pdf[i]-peilian_pdf[i]) < 1e0) & (np.abs(my_pdf[i]-peilian_pdf[i]) > 1e-5):
  #     print(f"{i+10:>8} | {my_pdf[i]:.6f} | {peilian_pdf[i]:.6f}")
  # print(np.max(np.abs(my_pdf-peilian_pdf)))
  # import matplotlib.pyplot as plt
  # plt.plot(np.abs(my_pdf-peilian_pdf))
  # plt.show()
  # exit()
  return np.concatenate(chi2)











def KS_test(original, target, original_weight, target_weight):
  vars = ['hminus_PT','hplus_PT','hminus_P','hplus_P']
  for i in range(0,4):
    xlim = np.percentile(np.hstack([target[:,i]]), [0.01, 99.99])
    print(f'   KS({vars[i]:>10}) =',
          ks_2samp_weighted(original[:,i], target[:,i],
                            weights1=original_weight, weights2=target_weight))




def kkp_weighting(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter.fit(original = original_v, target = target_v,
                 original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter.predict_weights(original_v), 0)
  # reweighter_fold.fit(original = original_v, target = target_v,
  #                original_weight = original_w, target_weight = target_w );
  # kkpWeight = np.where(original_w!=0, reweighter_fold.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{t}.npy',kkpWeight)
  print(f" * GB-weighting {m}-{y}-{t} sample is done")
  KS_test(original_v, target_v, original_w*kkpWeight, target_w)
  #print(f" * GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")


def kkp_weighting_bins(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter_bin.fit(original = original_v, target = target_v,
                     original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter_bin.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{t}.npy',kkpWeight)
  #print(f" * GB-weighting {m}-{y}-{t} sample is done")
  print(f" * GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")


def get_angular_acceptance(mc,t):
  # Select t
  if t == 'biased':
    trigger = mc.biased
  elif t == 'unbiased':
    trigger = mc.unbiased
  ang_acc = bsjpsikk.get_angular_cov(
              mc.true, mc.reco,
              trigger*mc.weight*ristra.allocate(mc.kkpWeight[i]*mc.kinWeight),
              **mc.params.valuesdict()
            )
  # Create parameters
  w, uw, cov, corr = ang_acc
  mc.angular_weights[t] = Parameters()
  for k in range(0,len(w)):
    correl = {f'w{j}':cov[k][j] for j in range(0,len(w)) if k>0 and j>0}
    mc.angular_weights[t].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                           'free': False, 'latex': f'w_{k}', 'correl': correl})
  #print(f"{  np.array(mc.angular_weights[t])}")




################################################################################
#%% Iterative procedure computing angWeights with corrections ##################

print(f"\n{80*'='}\n",
      "Iterative fitting procedure with pdf and kinematic-weighting",
      f"\n{80*'='}\n")

lb = [ data[f'{y}']['biased'].angular_weights[-1].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
lu = [ data[f'{y}']['unbiased'].angular_weights[-1].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
print(f"\n{80*'-'}\n","Biased angular acceptance")
for l in zip(*lb):
  print(*l, sep="| ")
print("\n","Unbiased angular acceptance")
for l in zip(*lu):
  print(*l, sep="| ")
print(f"\n{80*'-'}\n")

from ipanema import optimize




################################################################################
#%% Run and get the job done ###################################################




for i in range(1,15):
  checker = []                  # here we'll store if weights do converge or not
  itstr = f"[iteration #{i}]"

  # 1st step: fit data ---------------------------------------------------------
  print(f'Simultaneous fit Bs2JpsiPhi {"&".join(list(mc.keys()))} {itstr}')
  result = optimize(fcn_data,
                    method='minuit', params=pars, fcn_kwgs={'data':data},
                    verbose=False, timeit=True, tol=0.05, strategy=2)
  #print(result)
  pars = Parameters.clone(result.params)
  if f"{'&'.join(list(mc.keys()))}" == '2015' or f"{'&'.join(list(mc.keys()))}" == '2016':
    for p in [ 'fPper', 'fPlon', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DGs', 'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
      print(f"{p:>12} : {result.params[p].value:+.8f} +/- {result.params[p].stdev:+.8f}")
  #exit()


  # 2nd step: pdf weights ------------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0)
  print(f'\nPDF weighting MC samples to match Bs2JpsiPhi data {itstr}')
  t0 = timer()
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      print(f' * Calculating pdfWeight for {m}-{y} sample')
      bsjpsikk.delta_gamma5_mc(v.true, v.pdf, use_fk=1,
                               **v.params.valuesdict(), tLL=0.3)
      original_pdf_h = v.pdf.get()
      bsjpsikk.delta_gamma5_mc(v.true, v.pdf, use_fk=0,
                               **v.params.valuesdict(), tLL=0.3)
      original_pdf_h /= v.pdf.get()
      print(original_pdf_h)
      bsjpsikk.delta_gamma5_mc(v.true, v.pdf, use_fk=1,
                               **pars.valuesdict(),
                               **data[y]['combined'].csp.valuesdict(), tLL=0.3)
      target_pdf_h = v.pdf.get()
      bsjpsikk.delta_gamma5_mc(v.true, v.pdf, use_fk=0, **pars.valuesdict(),
                               **data[y]['combined'].csp.valuesdict(), tLL=0.3)
      target_pdf_h /= v.pdf.get()
      #print(target_pdf_h)
      v.pdfWeight[i] = np.nan_to_num(target_pdf_h/original_pdf_h)
      print(f"   pdfWeight[{i}]: {v.pdfWeight[i]}")
      #exit()
  tf = timer()-t0
  print(f'PDF weighting took {tf:.3f} seconds.')

  # 3rd step: kinematic weights ------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0).
  #   As a matter of fact, it's important to have data[y][combined] sample,
  #   the GBweighter gives different results when having those 0s or having
  #   nothing after cutting the sample.
  print(f'\nKinematic reweighting MC samples in K momenta {itstr}')
  threads = list()
  logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                      datefmt="%H:%M:%S")
  t0 = timer()
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      #v.kkpWeight = np.zeros_like(v.pdfWeight)
      for t, t_cut in zip(['biased','unbiased'],[0,1]):
        t_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={t_cut})*'
        #print(t_cut)
        original_v = v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']].values
        original_w = v.df.eval(t_cut+'polWeight*sw/gb_weights')*v.pdfWeight[i]*v.kinWeight
        target_v = data[f'{y}']['combined'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']].values
        target_w = data[f'{y}']['combined'].df.eval(t_cut+'sw').values
        # Run in single core (REALLY SLOW 10+ h)
        # kkp_weighting(original_v, original_w, target_v, target_w, v.kkpWeight[f'{t}'], y, m, t, 0)
        # Run in multithread mode (still has some problems with memory)
        # job = threading.Thread(target=kkp_weighting,
        #                        args=(original_v, original_w, target_v, target_w,
        #                        v.kkpWeight[f'{t}'], y, m, t))
        # Run multicore (about 12 minutes per iteration)
        job = multiprocessing.Process(target=kkp_weighting,
                               args=(original_v, original_w, target_v, target_w,
                               v.path, y, m, t, len(threads) ))
        threads.append(job); job.start()

  # Wait all processes to finish
  print(f' * There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]
  tf = timer()-t0
  print(f'Kinematic weighting took {tf:.3f} seconds.')

  # 4th step: angular weights --------------------------------------------------
  print(f'\nExtract angular weights {itstr}')
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      # write down code to save kkpWeight to root files
      # <CODE>
      b = np.load(os.path.dirname(v.path)+f'/kkpWeight_biased.npy')
      u = np.load(os.path.dirname(v.path)+f'/kkpWeight_unbiased.npy')
      v.kkpWeight[i] = b+u
      #print(f' kkpWeight[{i}] = {v.kkpWeight[i][:20]}')
      for trigger in ['biased','unbiased']:
        #print(f"Current angular weights for {m}-{y}-{trigger} sample are:")
        get_angular_acceptance(v,trigger)
    print(f'kkpWeight[{i}] for {y}')
    print(f"{'MC_Bs2JpsiPhi':<10} | {'MC_Bs2JpsiPhi_dG0':<10}")
    for evt in range(0,20):
      print(f"{dy['MC_BsJpsiPhi'].kkpWeight[i][evt]:>+.8f} | {dy['MC_BsJpsiPhi_dG0'].kkpWeight[i][evt]:>+.8f}")


  # 5th step: merge MC std and dg0 results -------------------------------------
  print(f'\nCombining MC_BsJpsiPhi and MC_BsJpsiPhi_dG0 {itstr}')
  for y, dy in mc.items(): # loop over years
    for trigger in ['biased','unbiased']:
      # Get angular weights for each MC
      std = dy['MC_BsJpsiPhi'].angular_weights[trigger]
      dg0 = dy['MC_BsJpsiPhi_dG0'].angular_weights[trigger]

      # Create w and cov arrays
      std_w = np.array([std[f'w{i}'].value for i in range(1,len(std))])
      dg0_w = np.array([dg0[f'w{i}'].value for i in range(1,len(dg0))])
      std_cov = std.correl_mat()[1:,1:];
      dg0_cov = dg0.correl_mat()[1:,1:];

      # Some matrixes
      std_covi = np.linalg.inv(std_cov)
      dg0_covi = np.linalg.inv(dg0_cov)
      cov_comb_inv = np.linalg.inv( std_cov + dg0_cov )
      cov_comb = np.linalg.inv( std_covi + dg0_covi )

      # Check p-value
      chi2_value = (std_w-dg0_w).dot(cov_comb_inv.dot(std_w-dg0_w));
      dof = len(std_w)
      prob = chi2.sf(chi2_value,dof)

      # Combine angular weights
      w = np.ones((dof+1))
      w[1:] = cov_comb.dot( std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T)  )

      # Combine uncertainties
      uw = np.zeros_like(w)
      uw[1:] = np.sqrt(np.diagonal(cov_comb))

      # Build correlation matrix
      corr = np.zeros((dof+1,dof+1))
      for k in range(1,cov_comb.shape[0]):
        for j in range(1,cov_comb.shape[1]):
          corr[k,j] = cov_comb[k][j]/np.sqrt(cov_comb[k][k]*cov_comb[j][j])

      # Create parameters
      merged_w = Parameters()
      for k in range(0,len(w)):
        correl = {f'w{j}':corr[k][j] for j in range(0,len(w)) if k>0 and j>0}
        merged_w.add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                      'free': False, 'latex': f'w_{k}', 'correl': correl})
      print(f"Current angular weights for Bs2JpsiPhi-{y}-{trigger} sample are:")
      print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8}")
      for _i in range(len(merged_w.keys())):
        print(f"{np.array(std)[_i]:+1.5f} | {np.array(dg0)[_i]:+1.5f} | {merged_w[f'w{_i}'].uvalue:+1.4uP}")
      data[f'{y}'][trigger].angacc = merged_w
      data[f'{y}'][trigger].angular_weights.append(merged_w)
      qwe = check_for_convergence( data[f'{y}'][trigger].angular_weights[-2], data[f'{y}'][trigger].angular_weights[-1] )
      checker.append( qwe )
      merged_w.dump(f'output/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_Iteration{i}_{trigger}.json')
      print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}\n')

  # Check if they are the same as previous iteration
  lb = [ data[f'{y}']['biased'].angular_weights[-1].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  lu = [ data[f'{y}']['unbiased'].angular_weights[-1].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\n{80*'-'}\nBiased angular acceptance")
  for l in zip(*lb):
    print(*l, sep="| ")
  print("\nUnbiased angular acceptance")
  for l in zip(*lu):
    print(*l, sep="| ")
  print(f"\n{80*'-'}\n")

  if all(checker):
    print(f"\nDone! Convergence was achieved within {i} iterations")
    for y, dy in data.items(): # loop over years
      for trigger in ['biased','unbiased']:
        pars = data[f'{y}'][trigger].angular_weights[-1]
        print('Saving table of params in tex')
        pars.dump(data[f'{y}'][trigger].params_path)
        print('Saving table of params in tex')
        with open(data[f'{y}'][trigger].tables_path, "w") as tex_file:
          tex_file.write(
            pars.dump_latex( caption="""
            Baseline angular weights for \\textbf{%s} \\texttt{\\textbf{%s}}
            category using combined MC samples.""" % (y,trigger)
            )
          )
        tex_file.close()
    break


# Storing some weights in disk -------------------------------------------------
#     For future use of computed weights created in this loop, these should be
#     saved to the path where samples are stored.
#     GBweighting is slow enough once!
print('Storing weights in root file')
for y, dy in mc.items(): # loop over years
  for m, v in dy.items(): # loop over mc_std and mc_dg0
    pool = {}
    for iter, wvalues in v.pdfWeight.items():
      pool.update({f'pdfWeight{iter}': wvalues})
    for iter, wvalues in v.kkpWeight.items():
      pool.update({f'kkpWeight{iter}': wvalues})
    print(pool)
    with uproot.recreate(v.path_to_weights,compression=None) as f:
      this_treename = args['weigths_branch']
      f['DecayTree'] = uproot.newtree({'kinWeight':np.float64})
      f['DecayTree'].extend({'kinWeight':v.kinWeight})
      f[this_treename] = uproot.newtree({var:np.float64 for var in pool})
      f[this_treename].extend(pool)
print(f' * Succesfully writen')
