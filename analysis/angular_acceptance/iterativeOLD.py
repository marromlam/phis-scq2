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

# threading
import logging
import threading
import time
import multiprocessing

from iminuit import Minuit as minuit

# load ipanema
from ipanema import initialize
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import ristra, Sample, Parameters, Parameter, Optimizer

bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']


# get bsjpsikk and compile it with corresponding flags
import  bsjpsikk
bsjpsikk.config['debug'] = 0
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
bsjpsikk.config['use_perftag'] = 1
bsjpsikk.config['use_truetag'] = 0
bsjpsikk.get_kernels()

# reweighting config
from hep_ml import reweight
reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 1})
#reweighter = reweight.GBReweighter(n_estimators=40, learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={'subsample': 1})
#reweighter = reweight.GBReweighter(n_estimators=500, learning_rate=0.1, max_depth=2, min_samples_leaf=1000, gb_args={'subsample': 1})

def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  if checker == 0:
    return True
  return False

################################################################################
################################################################################
#%% ############################################################################



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
input_data_params = f'angular_acceptance/params/{2016}/Bs2JpsiPhi.json'

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





# %% Load samples --------------------------------------------------------------
print(f"\n{80*'='}\n",
      "Loading samples",
      f"\n{80*'='}\n")

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
    this_pars = hjson.load(open(v[i]))
    #mc[f'{y}'][f'{m}'].params = Parameters()
    #mc[f'{y}'][f'{m}'].params.add(*[ {"name":k, "value":v} for k,v in this_pars.items()])  # WARNING
    # this is what I will use in the future
    mc[f'{y}'][f'{m}'].assoc_params(v[i])
    print(mc[f'{y}'][f'{m}'].params)
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
mass = bsjpsikk.config['x_m']
for i, y in enumerate(YEARS):
  print(f'Fetching elements for {y}[{i}] data sample')
  data[f'{y}'] = {}
  data[f'{y}']['combined'] = Sample.from_root(samples_data[i])
  csp = Parameters.load(csp_factors[i])
  csp = csp.build(csp,csp.find('CSP.*'))
  resolution = Parameters.load(time_resolution[i])
  flavor = Parameters.load(flavor_tagging[i])
  for t, T in zip(['biased','unbiased'],[0,1]):
    print(f' *  Loading {y} sample in {t} category\n    {samples_data[i]}')
    this_cut = f'(Jpsi_Hlt1DiMuonHighMassDecision_TOS=={T}) & (time>=0.3) & (time<=15)'
    data[f'{y}'][f'{t}'] = Sample.from_root(samples_data[i], cuts=this_cut)
    data[f'{y}'][f'{t}'].csp = csp
    data[f'{y}'][f'{t}'].flavor = flavor
    data[f'{y}'][f'{t}'].resolution = resolution
    print(csp)
    print(resolution)
  for t, coeffs in zip(['biased','unbiased'],[coeffs_biased,coeffs_unbiased]):
    print(f' *  Associating {y}-{t} time acceptance[{i}] from\n    {coeffs[i]}')
    c = Parameters.load(coeffs[i])
    print(c)
    data[f'{y}'][f'{t}'].timeacc = np.array(Parameters.build(c,c.fetch('c.*')))
    data[f'{y}'][f'{t}'].tLL = c['tLL'].value
    data[f'{y}'][f'{t}'].tUL = c['tUL'].value
  for t, weights in zip(['biased','unbiased'],[w_biased,w_unbiased]):
    print(f' *  Associating {y}-{t} angular weights from\n    {weights[i]}')
    w = Parameters.load(weights[i])
    print(w)
    data[f'{y}'][f'{t}'].angacc = np.array(Parameters.build(w,w.fetch('w.*')))
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
    print(sw[774])
    d.allocate(data=real,weight='sWeight',lkhd='0*time')
    d.angacc_dev = ristra.allocate(d.angacc)
    d.timeacc_dev = ristra.allocate(bsjpsikk.get_4cs(d.timeacc))
#exit()


# data[f'2016']['unbiased'].angacc = np.array([
# 1.0,
# 1.026188051680966,
# 1.025919907820129,
# -0.0008080722059112641,
# 0.0007686171894090263,
# 0.0002055097707937297,
# 1.00647239083432,
# 0.000455221821734608,
# 0.0001466163291351719,
# -0.000731595571913009
# ])
# data[f'2016']['unbiased'].angacc = np.array([
# 1,
# 1.036616501815053,
# 1.036438881647181,
# -0.000767902108603735,
# 0.0002722187888972826,
# 0.0002375659824330591,
# 1.009448663033953,
# 2.073630681856376e-05,
# 4.092379705098602e-05,
# -0.003959703563916721
# ])
# data[f'2016']['biased'].angacc = np.array([
# 1.0,
# 1.020419588928056,
# 1.020502754804629,
# 0.002631350622172166,
# 0.003125427462874503,
# -0.0003293730619200012,
# 1.011599141342973,
# 0.0002557661696621679,
# 4.612016290721501e-06,
# -0.001331697639192716
# ])
# data[f'2016']['biased'].angacc = np.array([
# 1,
# 1.034440015714541,
# 1.034642153098812,
# 0.00272584738881403,
# 0.003038166631007048,
# -0.0002781312683095018,
# 1.020346829061547,
# 0.0001065078746602566,
# 6.226895891636155e-05,
# 0.001126252400056541
# ])














#%% define likelihood



pars = Parameters.load('angular_acceptance/Bs2JpsiPhi.json')
hey = hjson.load(open('angular_acceptance/params/2016/Bs2JpsiPhi.json'))
#hey = hjson.load(open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_dG0_'+str(0)+'.json'))
for k,v in hey.items():
  try:
    pars[k].set(value=v)
    pars[k].set(init=v)
  except:
    0
# pars['CSP1'].value = csp_dev.get()[0]
# pars['CSP2'].value = csp_dev.get()[1]
# pars['CSP3'].value = csp_dev.get()[2]
# pars['CSP4'].value = csp_dev.get()[3]
# pars['CSP5'].value = csp_dev.get()[4]
# pars['CSP6'].value = csp_dev.get()[5]
print(pars)

#pars = pars+Parameters.load(f'output/{VERSION}/params/flavor_tagging/{y}/Bs2JpsiPhi/200506a.json')+Parameters.load(f'output/{VERSION}/params/time_resolution/{y}/Bs2JpsiPhi/200506a.json')

bsjpsikk.config['debug']           = 0
bsjpsikk.config['debug_evt']       = 0
bsjpsikk.config['use_time_acc']    = 1
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res']    = 1
bsjpsikk.config['use_perftag']     = 0
bsjpsikk.config['use_truetag']     = 0
bsjpsikk.get_kernels()





def fcn_data(parameters, data, weight = False, lkhd0=False):
  pars_dict = parameters.valuesdict()
  likelihood = []
  for y, dy in data.items():
    for dt in [dy['unbiased'],dy['biased']]:
      bsjpsikk.diff_cross_rate_full(dt.data, dt.lkhd,
                                    w = dt.angacc,
                                    coeffs = dt.timeacc,
                                    **dt.csp.valuesdict(),
                                    **dt.resolution.valuesdict(),
                                    **pars_dict)
      if weight:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd) * dt.weight).get() );
      else:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd)            ).get() );
  if lkhd0:
    return np.sum(np.concatenate(likelihood)) - (lkhd0-100)
  return np.sum(np.concatenate(likelihood))


#print( fcn_data(pars, data=data, weight=True) )


def fcn_data_opt(parameters, data, weight = False, lkhd0=False):
  pars = parameters.valuesdict()
  likelihood = []
  for dy in data.values():
    for dt in [dy['biased'],dy['unbiased']]:
      bsjpsikk.new_diff_rate(dt.data, dt.lkhd,
                             angacc = dt.angacc_dev, timeacc = dt.timeacc_dev,
                             CSP = dt.csp_dev,
                             tLL = dt.tLL, tUL = dt.tUL,
                             **pars)
      if weight:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd) * dt.weight).get() );
      else:
        likelihood.append( (-2.0 * ristra.log(dt.lkhd)            ).get() );
  if lkhd0:
    return np.sum(np.concatenate(likelihood)) - (lkhd0-100)
  return np.sum(np.concatenate(likelihood))


# from timeit import default_timer as timer
# #
# t0 = timer()
# for i in range(10):
#   fcn_data(pars, data=data, weight=True)
# tf = timer()-t0
# print(tf)
# t0 = timer()
# for i in range(10):
#   fcn_data_opt(pars, data=data, weight=True)
# tf = timer()-t0
# print(tf)
#
#
#
#
#
#
#
# exit()

def fcn_data_final(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():
    for dt in [dy['unbiased'],dy['biased']]:
      wrapper_fcn(dt.input, dt.output, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  **dt.flavor.valuesdict(), tLL=dt.tLL, tUL=dt.tUL)
      chi2.append( -2.0 * (ristra.log(dt.output) * dt.weight).get() );
  return np.concatenate(chi2)











#%% Prepate fitter






def minuit_fit(pars, data):
  # Set model to fit data
  bsjpsikk.config['debug']           = 0
  bsjpsikk.config['debug_evt']       = 0#774
  bsjpsikk.config['use_time_acc']    = 1
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res']    = 1
  bsjpsikk.config['use_perftag']     = 0
  bsjpsikk.config['use_truetag']     = 0
  bsjpsikk.get_kernels()

  lkhd0 = fcn_data(pars, data=data, weight=True)
  # Minuit wrapper
  def wrapper_minuit(*fvars):
    for name, val in zip(list_of_pars, fvars):
      pars[name].value = val
    result =  fcn_data(pars, data=data, weight=True, lkhd0=lkhd0)
    #exit()
    return result

  def configure_minuit( pars, pars_list, **kwgs):
    def parameter_minuit_config(par):
      out = {par.name: par.value}
      lims = [None,None]
      if abs(par.min) != np.inf: lims[0] = par.min
      if abs(par.max) != np.inf: lims[1] = par.max
      if not par.free:
        out.update ({"fix_" + par.name: True})
      out.update ({"limit_" + par.name: tuple(lims)})
      out.update ({"error_" + par.name: 1e-6})
      return out

    config = {}
    for par in pars.keys():
      if par in pars_list:
        config.update(parameter_minuit_config(pars[par]))
    config.update(kwgs)
    # for k,v in config.items():
    #   print(f'{k:>15} : {v}')
    return config

  # Get info for minuit
  #shit = Optimizer(fcn_data, params=pars); shit.prepare_fit()
  #list_of_pars = np.copy(shit.result.param_vary).tolist()
  list_of_pars = ['fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 'fPlon', 'fPper', 'pPlon', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6', 'dPpar', 'dPper', 'lPlon', 'DGs', 'DGsd', 'DM']
  dict_of_conf = configure_minuit(pars,list_of_pars)

  # Gor for it!
  print('Fit is starting...')
  crap = minuit(wrapper_minuit, forced_parameters=list_of_pars, **dict_of_conf, print_level=-1, errordef=1, pedantic=False)
  crap.strategy = 2
  crap.tol = 0.05
  crap.errordef = 1.0
  crap.migrad()
  crap.hesse()
  if not crap.migrad_ok():
    for i in range(0,10):
      crap.migrad()
      crap.hesse()
      if crap.migrad_ok():
        break
    if not crap.migrad_ok():
      print("Can't do better, sorry. Be aware of the precision loss")
      crap.hesse()
  print('Fit is finished! Cross your fingers and pray Simon')

  # Update pars
  #pars_fitted = Parameters.clone(pars)
  for name, val in zip(list_of_pars, crap.values.values()):
    pars[name].value = val
  for name, val in zip(list_of_pars, crap.errors.values()):
    pars[name].stdev = val
  #return pars_fitted

from hep_ml.metrics_utils import ks_2samp_weighted



def KS_test(original, target, original_weight, target_weight):
  vars = ['hminus_PT','hplus_PT','hminus_P','hplus_P']
  for i in range(0,4):
    xlim = np.percentile(np.hstack([target[:,i]]), [0.01, 99.99])
    print(f'KS over {vars[i]} ', ' = ', ks_2samp_weighted(original[:,i], target[:,i],
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

for i in range(1,15):
  checker = []                  # here we'll store if weights do converge or not

  # 1st step: fit data ---------------------------------------------------------
  print(f'Simultaneous fit Bs2JpsiPhi {"&".join(list(mc.keys()))} [iteration #{i}]')
  t0 = timer()
  minuit_fit(pars, data)
  tf = timer()-t0
  print(f'Fit took {tf:.3f} seconds.')
  pars = Parameters.clone(pars)
  print(pars)
  # fit-final
  # result = optimize(fcn_data, method='minuit', params=pars, fcn_kwgs={'data':data},
  #                   verbose=False, timeit=True, tol=0.5, strategy=1)
  # pars = Parameters.clone(result.pars)


  pars_loaded = hjson.load( open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_dG0_'+str(i-1)+'.json') )
  # for k, v in pars.items():
  #   print(f'{k:>10}: {v}')

  pars_comparison = Parameters.clone( pars )
  print(pars_comparison)

  for k, v in pars.items():
    try:
      pars_comparison[k].value = pars_loaded[k]
    except:
      0
    if v.free:
      try:
        print(f'{k:>10}: {v.value:+4.8f}   {pars_comparison[k].value:+4.8f}   {(v.value-pars_comparison[k].value):+2.2e}    {100*( (v.value-pars_comparison[k].value)/pars_comparison[k].value ):+4.4f}%')
      except:
        print(f'{k:>10}: {v.value:+4.8f}   ')

  print(f'FCN with SCQ parameters: {fcn_data(pars, data=data, weight=True):.16f}')
  print(f' FCN with HD parameters: {fcn_data(pars_comparison, data=data, weight=True):.16f}')


  # 2nd step: pdf weights ------------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0)
  print(f'\nPDF weighting MC samples to match Bs2JpsiPhi data [iteration #{i}]')
  bsjpsikk.config['debug']    = 0
  bsjpsikk.config['debug_evt'] = 0
  bsjpsikk.config['use_time_acc']    = 0
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res']    = 0
  bsjpsikk.config['use_perftag']     = 1
  bsjpsikk.config['use_truetag']     = 0
  bsjpsikk.get_kernels()

  t0 = timer()
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      print(f'* Calculating pdfWeight for {m}-{y} sample')
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **v.params.valuesdict(), mass_bins=1)
      original_pdf_h = v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **v.params.valuesdict(), mass_bins=1)
      original_pdf_h /= v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=1, **pars.valuesdict())
      target_pdf_h = v.pdf.get()
      bsjpsikk.diff_cross_rate_full(v.true, v.pdf, use_fk=0, **pars.valuesdict())
      target_pdf_h /= v.pdf.get()
      v.pdfWeight[i] = np.nan_to_num(target_pdf_h/original_pdf_h)
      print(f"  pdfWeight[{i}]: {v.pdfWeight[i]}")
  tf = timer()-t0
  print(f'PDF weighting took {tf:.3f} seconds.')

  # 3rd step: kinematic weights ------------------------------------------------
  #   We need to change bsjpsikk to handle MC samples and then we compute the
  #   desired pdf weights for a given set of fitted pars in step 1. This implies
  #   looping over years and MC samples (std and dg0).
  #   As a matter of fact, it's important to have data[y][combined] sample,
  #   the GBweighter gives different results when having those 0s or having
  #   nothing after cutting the sample.
  print(f'\nKinematic reweighting MC samples in K momenta [iteration #{i}]')
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
  print(f'\nExtract angular weights [iteration #{i}]')
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
  print(f'\nCombining MC_BsJpsiPhi and MC_BsJpsiPhi_dG0 [iteration #{i}]')
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
        print(f"{np.array(std)[_i]:+1.5f} | {np.array(dg0)[_i]:+1.5f} | {merged_w[f'w{_i}'].uvalue:+1.2uP}")
      data[f'{y}'][trigger].angacc = np.array(merged_w)
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
      this_treename = '&'.join(map(str,YEARS))
      f['DecayTree'] = uproot.newtree({'kinWeight':np.float64})
      f['DecayTree'].extend({'kinWeight':v.kinWeight})
      f[this_treename] = uproot.newtree({var:np.float64 for var in pool})
      f[this_treename].extend(pool)
print(f' * Succesfully writen')
