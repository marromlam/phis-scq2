#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE
-- """

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

# reweighting config
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters

import time



# get bsjpsikk and compile it with corresponding flags
import bsjpsikk
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
# reweighter = reweight.GBReweighter(n_estimators     = 40,
#                                    learning_rate    = 0.25,
#                                    max_depth        = 5,
#                                    min_samples_leaf = 500,
#                                    gb_args          = {'subsample': 1})

#30:0.3:4:500
#20:0.3:3:1000


import threading
import time
import multiprocessing



################################################################################



################################################################################

def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample-mc',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-data',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--input-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-tables',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-weights-file',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
                      default = 'baseline',
                      help='Configuration')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'test',
                      help='Year of data-taking')
  parser.add_argument('--trigger',
                      default = 'biased',
                      help='Trigger(s) to fit [comb/(biased)/unbiased]')

  return parser


# Parse arguments
try:
  args = vars(argument_parser().parse_args())
except:
  1+1

"""
YEAR = 2016
VERSION = 'v0r0'
MODE = 'MC_Bs2JpsiPhi'
TRIGGER = 'biased'
input_params_path = f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json'
sample_mc_path = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/{MODE}/{VERSION}.root'
sample_data_path = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/Bs2JpsiPhi/{VERSION}.root'
output_tables_path = f'output/v0r4/tables/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
output_params_path = f'output/v0r4/params/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
output_weight_file = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/{MODE}/{VERSION}_angWeight.root'
"""


YEAR = args['year']
VERSION = args['version']
MODE = args['mode']
TRIGGER = args['trigger']
input_params_path = args['input_params']
sample_mc_path = args['sample_mc']
sample_data_path = args['sample_data']
output_tables_path = args['output_tables']
output_params_path = args['output_params']
output_weight_file = args['output_weights_file']

print(output_weight_file)

################################################################################
#%% ############################################################################



################################################################################
################################################################################
################################################################################


# %% Load samples --------------------------------------------------------------

# Load Monte Carlo samples
std = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/v0r1.root')
dg0 = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r1.root')
std.assoc_params(f'angular_acceptance/params/2016/MC_Bs2JpsiPhi.json')
dg0.assoc_params(f'angular_acceptance/params/2016/MC_Bs2JpsiPhi_dG0.json')

print(std.branches)
# Load corresponding data sample
data = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/v0r1.root')

# Variables and branches to be used
reco = ['helcosthetaK', 'helcosthetaL', 'helphi', 'time']
true = ['truehelcosthetaK_GenLvl', 'truehelcosthetaL_GenLvl', 'truehelphi_GenLvl', '1000*B_TRUETAU_GenLvl']
#true = ['true'+i+'_GenLvl' for i in reco]
weight_mc='(polWeight*sw/gb_weights)'
weight_data='(sw)'


# Select trigger
# if TRIGGER == 'biased':
#   trigger = 'biased';
#   weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
#   weight_data += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
# elif TRIGGER == 'unbiased':
#   trigger = 'unbiased';
#   weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
#   weight_data += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
# elif TRIGGER == 'comb':
#   trigger = 'comb';

# Allocate some arrays with the needed branches
std.allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
std.allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
std.allocate(pdf='0*time', ones='time/time', zeros='0*time')
std.allocate(weight=weight_mc)
dg0.allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
dg0.allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
dg0.allocate(pdf='0*time', ones='time/time', zeros='0*time')
dg0.allocate(weight=weight_mc)


################################################################################
################################################################################
################################################################################



#%% Compute standard kinematic weights -------------------------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
"""
Name        Value     Stdev    Min   Max  Free
w0            1.0      None   -inf   inf  False
w1         1.0204    0.0015   -inf   inf  False
w2         1.0205    0.0015   -inf   inf  False
w3         0.0026    0.0012   -inf   inf  False
w4        0.00313   0.00072   -inf   inf  False
w5       -0.00033   0.00071   -inf   inf  False
w6         1.0116    0.0010   -inf   inf  False
w7        0.00026   0.00091   -inf   inf  False
w8        0.00000   0.00092   -inf   inf  False
w9        -0.0013    0.0019   -inf   inf  False
"""

def kkp_weighting(original_v, original_w, target_v, target_w, path, y,m,t,i ,reweighter):
  reweighter.fit(original = original_v, target = target_v,
                 original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kinWeight_{t}.npy',kkpWeight)
  #print(f"* GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")

def fcn_to_get_bdtconf(args):

  estim = int(np.round(args[0]))
  learn = np.round(args[1]*100)/100
  depth = int(np.round(args[2]))
  leafs = 100*int(np.round(args[3]))
  print(f'bdt_conf = {estim}:{learn}:{depth}:{leafs}')

  reweighter = reweight.GBReweighter(n_estimators   = estim,
                                   learning_rate    = learn,
                                   max_depth        = depth,
                                   min_samples_leaf = leafs,
                                   gb_args          = {'subsample': 1})

  threads = [];
  original_vars_std  = std.df[['X_M','B_P','B_PT']]
  original_vars_dg0  = dg0.df[['X_M','B_P','B_PT']]
  target_vars        = data.df[['X_M','B_P','B_PT']]

  # UNBIASED STUFF
  target_weight   = data.df.eval(weight_data+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)')

  original_weight = std.df.eval(weight_mc+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)')
  job = multiprocessing.Process(
          target=kkp_weighting,
          args=(original_vars_std, original_weight, target_vars, target_weight,
                '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/',
                2016, 'MC_Bs2JpsiPhi', 'unbiased', 0 , reweighter))
  threads.append(job)

  original_weight = dg0.df.eval(weight_mc+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)')
  job = multiprocessing.Process(
          target=kkp_weighting,
          args=(original_vars_dg0, original_weight, target_vars, target_weight,
                '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/',
                2016, 'MC_Bs2JpsiPhi_dG0', 'unbiased', 1 , reweighter))
  threads.append(job)


  # BIASED STUFF
  target_weight   = data.df.eval(weight_data+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)')

  original_weight = std.df.eval(weight_mc+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)')
  job = multiprocessing.Process(
          target=kkp_weighting,
          args=(original_vars_std, original_weight, target_vars, target_weight,
                '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/',
                2016, 'MC_Bs2JpsiPhi', 'biased', 2 , reweighter))
  threads.append(job)

  original_weight = dg0.df.eval(weight_mc+'*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)')
  job = multiprocessing.Process(
          target=kkp_weighting,
          args=(original_vars_dg0, original_weight, target_vars, target_weight,
                '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/',
                2016, 'MC_Bs2JpsiPhi_dG0', 'biased', 3 , reweighter))
  threads.append(job)

  #print(f'There are {len(threads)} jobs running in parallel')
  for thread in threads:
    thread.start()
  time.sleep(13*60)
  for thread in threads:
    thread.join()

  # Collect weights
  std_k_u = np.load('/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/kinWeight_unbiased.npy')
  std_k_b = np.load('/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/kinWeight_biased.npy')
  dg0_k_u = np.load('/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/kinWeight_unbiased.npy')
  dg0_k_b = np.load('/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/kinWeight_biased.npy')

  ang_acc = bsjpsikk.get_angular_cov(std.true, std.reco, std.weight*ristra.allocate(std_k_u), **std.params.valuesdict() )
  std_w_u, uw, std_cov_u, corr = ang_acc
  ang_acc = bsjpsikk.get_angular_cov(std.true, std.reco, std.weight*ristra.allocate(std_k_b), **std.params.valuesdict() )
  std_w_b, uw, std_cov_b, corr = ang_acc
  ang_acc = bsjpsikk.get_angular_cov(dg0.true, dg0.reco, dg0.weight*ristra.allocate(dg0_k_u), **dg0.params.valuesdict() )
  dg0_w_u, uw, dg0_cov_u, corr = ang_acc
  ang_acc = bsjpsikk.get_angular_cov(dg0.true, dg0.reco, dg0.weight*ristra.allocate(dg0_k_b), **dg0.params.valuesdict() )
  dg0_w_b, uw, dg0_cov_b, corr = ang_acc

  w_final = []
  for std_w, dg0_w, std_cov, dg0_cov in [ [std_w_u[1:], dg0_w_u[1:], std_cov_u[1:,1:], dg0_cov_u[1:,1:]] , [std_w_b[1:], dg0_w_b[1:], std_cov_b[1:,1:], dg0_cov_b[1:,1:]] ]:
      # Some matrixes
      std_covi = np.linalg.inv(std_cov)
      dg0_covi = np.linalg.inv(dg0_cov)
      cov_comb_inv = np.linalg.inv( std_cov + dg0_cov )
      cov_comb = np.linalg.inv( std_covi + dg0_covi )
      # Combine angular weights
      w_final.append( cov_comb.dot( std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T)  ) )


  print( w_final[0])
  print( w_final[1])
  res0  = (  w_final[0] - np.array([1.02613,1.02581,0.00084,0.00076, 0.00023,1.00647,0.00045,0.00015,0.00076])  )**2
  #print(res0)
  res1  = (  w_final[1] - np.array([1.0209, 1.0209 ,0.0027 ,0.00312,-0.00037,1.0118 ,0.00018,0.00012,-0.0010])  )**2
  #print(res1)
  res = res0.sum() + res1.sum()
  print(f'residual = {res}\n\n')
  if res == 0:
    print(f'USE     bdt_conf = {estim}:{learn}:{depth}:{leafs}   FTW')
    print(f'USE     bdt_conf = {estim}:{learn}:{depth}:{leafs}   FTW')
    print(f'USE     bdt_conf = {estim}:{learn}:{depth}:{leafs}   FTW')
    exit()
  return res



from scipy.optimize import minimize
from scipy.optimize import optimize
from scipy.optimize import differential_evolution, brute

"""
x0 = [20,0.3,3,10]
fcn_to_get_bdtconf(x0)
x0 = [40,0.25,5,5]
fcn_to_get_bdtconf(x0)
x0 = [50,0.25,3,5]
fcn_to_get_bdtconf(x0)
x0 = [30,0.3,3,5]
fcn_to_get_bdtconf(x0)
x0 = [40,0.3,4,5]
fcn_to_get_bdtconf(x0)
x0 = [35,0.3,3,5]
fcn_to_get_bdtconf(x0)
x0 = [50,0.3,2,5]
fcn_to_get_bdtconf(x0)
x0 = [30,0.3,2,5]
fcn_to_get_bdtconf(x0)
x0 = [30,0.3,1,5]
fcn_to_get_bdtconf(x0)

print('\nEND of MANUAL TESTING\n')
"""

bounds = [(1,100), (0, 1), (1, 10), (1, 10)]
#res = minimize(fcn_to_get_bdtconf, x0, method='Nelder-Mead', tol=1e-6)
res = differential_evolution(fcn_to_get_bdtconf, bounds, updating='deferred', workers=1)
#res = brute(fcn_to_get_bdtconf, ranges=bounds, finish=optimize.fmin, workers=1)
print(res.x)

#fcn_to_get_bdtconf(*[40,0.25,5,500])




exit()




def f(x,p):
  return (p[0]-2)*x**2+p[1]

from scipy.optimize import minimize

X = np.linspace(-10,10,100)
x0 = [1.3, 0.7]
f(X,x0)
res = minimize(lambda p: np.sum(f(X,p)**2), x0, method='Nelder-Mead', tol=1e-6)
res.x

BREAK





#%% Iterative procedure computing angWeights with corrections --------------------
#     dfdfdf Weight MC to match data in the iterative variables namely
#              p and pT of K+ and K-



print('STEP 4: Launching iterative procedure, pdf and kinematic-weighting')
for i in range(1,5):
  for k, v in mc.items():
    # fit MC  ------------------------------------------------------------------
    print('\tFitting %s in %s iteration' % (k,str(i)))
    tparams_pdf = hjson.load(
                    open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_'+str(i-1)+'.json')
                  )

    # do the pdf-weighting -----------------------------------------------------
    print('\tCalculating pdfWeight in %s iteration' % str(i))
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **v['sample'].params.valuesdict())
    original_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **v['sample'].params.valuesdict())
    original_pdf_h /= v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **tparams_pdf)
    target_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **tparams_pdf)
    target_pdf_h /= v['sample'].pdf.get()
    v[f'pdfWeight{i}'] = np.nan_to_num(target_pdf_h/original_pdf_h)
    print(f"\tpdfWeight{i}:",v[f'pdfWeight{i}'])

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f'\tCalculating p and pT of K+ and K- weight in {i} iteration')
    reweighter.fit(original        = v['sample'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   target          = data.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   original_weight = v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{i}']*v['kinWeight'].get(),
                   target_weight   = data.df.eval(trig_cut+'sw')
                  );
    kkpWeight = reweighter.predict_weights(v['sample'].df[kin_vars])
    v[f'kkpWeight{i}'] = ristra.allocate(np.where(oweight!=0, kkpWeight, 0))
    print(f"\tkkpWeight{i} = {v[f'kkpWeight{i}']}")

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f"\tAngular weights for {k} category in {i} iteration")
    v[f'w_kkpweighted{i}'] = bsjpsikk.get_angular_weights(
                v['sample'].true,
                v['sample'].reco,
                v['sample'].weight*v['kinWeight']*v[f'kkpWeight{i}'],
                v['sample'].params.valuesdict()
                )
    v[f'w_kkpweighted{i}'] /= v[f'w_kkpweighted{i}'][0]
    print(10*"\t%+.8lf\n" % tuple(v[f'w_kkpweighted{i}']) )



foo = uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')





"""
i = 1
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03782 +- 0.000949792 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03762 +- 0.000926573 (unbinned)
int f4 dcostheta dphi dcospsi = -0.00103561 +- 0.000740913 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000329971 +- 0.000447396 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000272403 +- 0.000457952 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01036 +- 0.000640444 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000336074 +- 0.000567024 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000553363 +- 0.000584319 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00322722 +- 0.00121307 (unbinned)

i = 2
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949104 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03758 +- 0.000926344 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966926 +- 0.00073987 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383365 +- 0.000446895 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456986 (unbinned)
int f7 dcostheta dphi dcospsi = 1.0103 +- 0.000640177 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000283837 +- 0.000566376 (unbinned)
int f9 dcostheta dphi dcospsi = 0.00058618 +- 0.000583348 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00280243 +- 0.00121186 (unbinned)

i = 3
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949098 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03757 +- 0.000926338 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966919 +- 0.000739864 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383704 +- 0.000446893 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456984 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01029 +- 0.000640172 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000284358 +- 0.000566372 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000586183 +- 0.000583344 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00278764 +- 0.00121186 (unbinned)








v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights') * v['kinWeight'].get() * v[f'kkpWeight{i}'].get()

v['sample'].weight*v[f'kkpWeight{1}']*v['kinWeight']
*

0.47/2


data = uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root')['DecayTree'].arrays(['sw','gb_weights'])

swg = data[b'sw']/data[b'gb_weights']
pol = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].array('PolWeight')

pdf_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step0'].array('pdfWeight')
pdf_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step1'].array('pdfWeight')
kin_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step0'].array('kinWeight')
kin_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step1'].array('kinWeight')


kin_0

kin_1

swg*pol*kin_0*kin_1

mc['MC_Bs2JpsiPhi'].df.eval(trig_cut+'polWeight*sw/gb_weights') *
mc['MC_Bs2JpsiPhi']['kinWeight'].get()
v[f'kkpWeight{1}'].get()




swg*pol
mc_std.df.eval('polWeight*(sw/gb_weights)')




























int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03714 +- 0.000948856 (unbinned)
int f3 dcostheta dphi dcospsi = 1.0369 +- 0.000926164 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000945802 +- 0.000738975 (unbinned)
int f5 dcostheta dphi dcospsi = 0.00035262 +- 0.000447158 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000285595 +- 0.000457371 (unbinned)
int f7 dcostheta dphi dcospsi = 1.00986 +- 0.00063955 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000366054 +- 0.000566205 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000567317 +- 0.000583275 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00220241 +- 0.00121173 (unbinned)

"""





v[f'kkpWeight{1}']


mc_std.df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{1}']*v['kinWeight'].get()


tweight


v['kkpWeight1']
v['kinWeight']

simon['MC_Bs2JpsiPhi']['w_kkpweighted1']



simon['MC_Bs2JpsiPhi']['kkpWeight1']



v['sample'].weight*v['kinWeight']*v[f'kkpWeight1']

os.listdir('/scratch08/marcos.romero/SideCar/')

a.keys()



################################################################################
################################################################################
################################################################################



################################################################################
# compare with Simon ###########################################################
################################################################################
simon = {}
for mode in ['MC_Bs2JpsiPhi']:
  d = {}
  for i in range(-1,10):
    # Get angular weights
    f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{i}.root')['fi']
    if i==-1:
      d[f'w_uncorrected'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    elif i==0:
      d[f'w_kinweighted'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    else:
      d[f'w_kkpweighted{i}'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')[f'kinWeight_Unbiased_Step{i}']
    if i==0:
      d[f'kinWeight'] = f.array('kinWeight')
    elif i>0:
      d[f'kkpWeight{i}'] = f.array('kinWeight')
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{i}"]
      d[f'pdfWeight{i+1}'] = f.array('pdfWeight')
  simon[mode] = d
s = simon['MC_Bs2JpsiPhi']

s['w_uncorrected']

f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{-1}.root')['fi']
mat = np.zeros((10,10))
for j1 in range(1,10):
  for j2 in range(1,10):
    mat[j1,j2] = f.array(f'cf{j1+1}{j2+1}')[0]

mat

f.arrays('cf*')
scale = mc['MC_Bs2JpsiPhi']['unbiased']['weight'].get().sum()
scale
0.8*mc['MC_Bs2JpsiPhi']['unbiased']['cov']/(scale*scale)
uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')
