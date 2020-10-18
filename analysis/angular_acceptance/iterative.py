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
initialize(os.environ['IPANEMA_BACKEND'],2)
from ipanema import ristra, Sample, Parameters, Parameter, optimize

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and
from utils.helpers import  version_guesser, timeacc_guesser, trigger_scissors

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000


def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  if checker == 0:
    return True
  return False

def pdf_reweighting(mcsample, mcparams, rdparams):
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                           **mcparams.valuesdict(), tLL=0.3)
  original_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                           **mcparams.valuesdict(), tLL=0.3)
  original_pdf_h /= mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                           **rdparams.valuesdict(), tLL=0.3)
  target_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                           **rdparams.valuesdict(), tLL=0.3)
  target_pdf_h /= mcsample.pdf.get()
  #print(f"   pdfWeight[{i}]: { np.nan_to_num(target_pdf_h/original_pdf_h) }")
  return np.nan_to_num(target_pdf_h/original_pdf_h)

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
  np.save(path.replace('.root',f'_{t}.npy'),kkpWeight)
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
  ang_acc = badjanak.get_angular_cov(
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


def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []

  for y, dy in data.items():
    for dt in [dy['unbiased'],dy['biased']]:
      badjanak.delta_gamma5_data(dt.data, dt.lkhd, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  **dt.flavor.valuesdict(), tLL=dt.tLL, tUL=dt.tUL)
      chi2.append( -2.0 * (ristra.log(dt.lkhd) * dt.weight).get() );

  return np.concatenate(chi2)



# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  p.add_argument('--sample-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data',   help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--angular-weights-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--angular-weights-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-weights-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-weights-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-coeffs-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-coeffs-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-csp', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-time-resolution', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-flavor-tagging', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angular-weights-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angular-weights-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  return p

################################################################################
#%% Run and get the job done ###################################################

if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = [int(y) for y in args['year'].split(',')]        # years as list of int
  MODE = 'Bs2JpsiPhi'
  ANGACC = args['angacc']

  # Get badjanak model and configure it ----------------------------------------
  #initialize(os.environ['IPANEMA_BACKEND'], 1 if YEARS in (2015,2017) else -1)
  from time_acceptance.fcn_functions import trigger_scissors

  # Prepare the cuts -----------------------------------------------------------
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  #CUT = trigger_scissors(TRIGGER, CUT)         # place cut attending to trigger
  CUT = cuts_and(CUT,'time>=0.3 & time<=15')

  # List samples, params and tables --------------------------------------------
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

  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')
  kkpWeight_dg0 = args['output_angular_weights_mc_dg0'].split(',')

  # If version is v0r1, you will be running over old tuples, I guess you
  # pursuit to reproduce HD-fitter results. So I will change a little the config
  if VERSION == 'v0r1':
    reweighter = reweight.GBReweighter(n_estimators=40,learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={"subsample": 1})
    input_std_params = args['params_mc_std'].replace("generator","generator_old").split(',')
    input_dg0_params = args['params_mc_dg0'].replace("generator","generator_old").split(',')

  # Print settings
  print(f"\n{80*'='}\n", "Settings", f"\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdconfig.values()):50}\n")



  # %% Load samples ------------------------------------------------------------
  print(f"\n{80*'='}\n", "Loading samples", f"\n{80*'='}\n")

  # Lists of MC variables to load and build arrays
  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  #true = [f'true{i}_GenLvl' for i in reco]
  reco += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']
  true += ['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*time', '0*time']
  #weight_mc = '(polWeight*sw/gb_weights)'

  # Load Monte Carlo samples ---------------------------------------------------
  mc = {}
  mcmodes = ['MC_BsJpsiPhi', 'MC_BsJpsiPhi_dG0']
  
  for i, y in enumerate(YEARS):
    print(f'\nFetching elements for {y} MC samples')
    mc[f'{y}'] = {}
    for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[samples_std,samples_dg0]):
      print(f'\n *  Loading {m}-{y} sample')
      mc[f'{y}'][f'{m}'] = Sample.from_root(v[i], share=SHARE)
    for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[input_std_params,input_dg0_params]):
      print(f' *  Associating {m}-{y} parameters from\n    {v[i]}')
      mc[f'{y}'][f'{m}'].assoc_params(v[i])
    for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[angWeight_std,angWeight_dg0]):
      print(f' *  Attaching {m}-{y} kinWeight from\n    {v[i]}')
      mc[f'{y}'][f'{m}'].kinWeight = uproot.open(v[i])['DecayTree'].array('kinWeight')
      print("len angWeight = ",len(mc[f'{y}'][f'{m}'].kinWeight))
    for m, v in zip(['MC_BsJpsiPhi','MC_BsJpsiPhi_dG0'],[kkpWeight_std,kkpWeight_dg0]):
      mc[f'{y}'][f'{m}'].path_to_weights = v[i]
      print(f"    {mc[f'{y}'][f'{m}'].path_to_weights}")

  for y, modes in mc.items():
    for m, v in modes.items():
      print(f' *  Allocating arrays in device for {m}-{y}')
      mc[f'{y}'][f'{m}'].allocate(reco=reco)
      mc[f'{y}'][f'{m}'].allocate(true=true)
      mc[f'{y}'][f'{m}'].allocate(pdf='0*time', ones='time/time', zeros='0*time')

      mc[f'{y}'][f'{m}'].allocate(weight=weight_mc)
      mc[f'{y}'][f'{m}'].allocate(biased=f'Jpsi_Hlt1DiMuonHighMassDecision_TOS==0')
      mc[f'{y}'][f'{m}'].allocate(unbiased=f'Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
      mc[f'{y}'][f'{m}'].angular_weights = {'biased':0, 'unbiased':0}
      mc[f'{y}'][f'{m}'].kkpWeight = {}
      mc[f'{y}'][f'{m}'].pdfWeight = {}

  # Lists of data variables to load and build arrays ---------------------------
  real  = ['cosK','cosL','hphi','time']                        # angular variables
  real += ['X_M','sigmat']                                     # mass and sigmat
  real += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']  # tagging
  #real += ['0*B_ID','0*B_ID', '0*B_ID', '0*B_ID']  # tagging
  weight_rd = f'sw_{VAR}' if VAR else 'sw'

  # Load corresponding data sample ---------------------------------------------
  data = {}
  mass = badjanak.config['x_m'] # warning here!
  for i, y in enumerate(YEARS):
    print(f'Fetching elements for {y}[{i}] data sample')
    data[f'{y}'] = {}
    data[f'{y}']['combined'] = Sample.from_root(samples_data[i], share=SHARE)
    csp = Parameters.load(csp_factors[i]);
    csp = csp.build(csp,csp.find('CSP.*'))
    resolution = Parameters.load(time_resolution[i])
    flavor = Parameters.load(flavor_tagging[i])
    data[f'{y}'][f'combined'].csp = csp
    data[f'{y}'][f'combined'].flavor = flavor
    data[f'{y}'][f'combined'].resolution = resolution
    for t, T in zip(['biased','unbiased'],[0,1]):
      print(f'\n *  Loading {y} sample in {t} category')
      this_cut = cuts_and(f'Jpsi_Hlt1DiMuonHighMassDecision_TOS=={T}', CUT)
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
    print(weight_rd)
    for d in [data[f'{y}']['biased'],data[f'{y}']['unbiased']]:
      sw = np.zeros_like(d.df[f'{weight_rd}'])
      for l,h in zip(mass[:-1],mass[1:]):
        pos = d.df.eval(f'X_M>={l} & X_M<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(X_M>={l} & X_M<{h})')
        sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
      d.df['sWeight'] = sw
      d.allocate(data=real,weight='sWeight',lkhd='0*time')


  # %% Prepare dict of parameters ----------------------------------------------
  print(f"\n{80*'='}\nParameters and initial status\n{80*'='}\n")
  pars = Parameters()
  # S wave fractions
  pars.add(dict(name='fSlon1', value=0.480, min=0.00, max=0.90,
            free=True, latex=r'f_S^{1}'))
  pars.add(dict(name='fSlon2', value=0.040, min=0.00, max=0.90,
            free=True, latex=r'f_S^{2}'))
  pars.add(dict(name='fSlon3', value=0.004, min=0.00, max=0.90,
            free=True, latex=r'f_S^{3}'))
  pars.add(dict(name='fSlon4', value=0.009, min=0.00, max=0.90,
            free=True, latex=r'f_S^{4}'))
  pars.add(dict(name='fSlon5', value=0.059, min=0.00, max=0.90,
            free=True, latex=r'f_S^{5}'))
  pars.add(dict(name='fSlon6', value=0.130, min=0.00, max=0.90,
            free=True, latex=r'f_S^{6}'))
  # P wave fractions
  pars.add(dict(name="fPlon", value=0.5240, min=0.4, max=0.6,
            free=True, latex=r'f_0'))
  pars.add(dict(name="fPper", value=0.2500, min=0.1, max=0.3,
            free=True, latex=r'f_{\perp}'))
  # Weak phases
  pars.add(dict(name="pSlon", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_S - \phi_0"))
  pars.add(dict(name="pPlon", value= 0.07, min=-1.0, max=1.0,
            free=True , latex=r"\phi_0" ))
  pars.add(dict(name="pPpar", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_{\parallel} - \phi_0"))
  pars.add(dict(name="pPper", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_{\perp} - \phi_0"))
  # S wave strong phases
  pars.add(dict(name='dSlon1', value=+2.34, min=-0.0, max=+3.0,
            free=True, latex=r"\delta_S^{1} - \delta_{\perp}"))
  pars.add(dict(name='dSlon2', value=+1.64, min=-0.0, max=+3.0,
            free=True, latex=r"\delta_S^{2} - \delta_{\perp}"))
  pars.add(dict(name='dSlon3', value=+1.09, min=-0.0, max=+3.0,
            free=True, latex=r"\delta_S^{3} - \delta_{\perp}"))
  pars.add(dict(name='dSlon4', value=-0.25, min=-3.0, max=+0.0,
            free=True, latex=r"\delta_S^{4} - \delta_{\perp}"))
  pars.add(dict(name='dSlon5', value=-0.48, min=-3.0, max=+0.0,
            free=True, latex=r"\delta_S^{5} - \delta_{\perp}"))
  pars.add(dict(name='dSlon6', value=-1.18, min=-3.0, max=+0.0,
            free=True, latex=r"\delta_S^{6} - \delta_{\perp}"))
  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.000, min=-2*3.14, max=2*3.14,
            free=False, latex=r"\delta_0"))
  pars.add(dict(name="dPpar", value=3.260, min=-2*3.14, max=2*3.14,
            free=True, latex=r"\delta_{\parallel} - \delta_0"))
  pars.add(dict(name="dPper", value=3.026, min=-2*3.14, max=2*3.14,
            free=True, latex=r"\delta_{\perp} - \delta_0"))
  # lambdas
  pars.add(dict(name="lSlon", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_S/\lambda_0"))
  pars.add(dict(name="lPlon", value=1.0, min=0.7, max=1.6,
            free=True,  latex="\lambda_0"))
  pars.add(dict(name="lPpar", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\parallel}/\lambda_0"))
  pars.add(dict(name="lPper", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\perp}/\lambda_0"))
  # life parameters
  pars.add(dict(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
            free=False, latex=r"\Gamma_d"))
  pars.add(dict(name="DGs", value= 0.0917, min= 0.03, max= 0.15,
            free=True, latex=r"\Delta\Gamma_s"))
  pars.add(dict(name="DGsd", value= 0.03, min=-0.2, max= 0.2,
            free=True, latex=r"\Gamma_s - \Gamma_d"))
  pars.add(dict(name="DM", value=17.768, min=16.0, max=20.0,
            free=True, latex=r"\Delta m"))
  print(pars)


  # print time acceptances
  lb = [ data[f'{y}']['biased'].timeacc.__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  lu = [ data[f'{y}']['unbiased'].timeacc.__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\nBiased time acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  print(f"\nUnbiased time acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")

  # print csp factors
  lb = [ data[f'{y}']['biased'].csp.__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\nCSP factors\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print flavor tagging parameters
  lb = [ data[f'{y}']['biased'].flavor.__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\nFlavor tagging parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print time resolution
  lb = [ data[f'{y}']['biased'].resolution.__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\nResolution parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print angular acceptance
  lb = [ data[f'{y}']['biased'].angaccs[0].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  lu = [ data[f'{y}']['unbiased'].angaccs[0].__str__(['value']).splitlines() for i,y in enumerate(YEARS) ]
  print(f"\nBiased angular acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")
  print(f"\nUnbiased angular acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")
  print(f"\n")

  for i in range(1,30):
    print(f"\n{80*'='}\nIteration {i} of the procedure\n{80*'='}\n")
    checker = []                  # here we'll store if weights do converge or not
    itstr = f"[iteration #{i}]"

    # 1st step: fit data -------------------------------------------------------
    for v in pars.values():
      v.init = v.value # start where we left
    print(f'Simultaneous fit Bs2JpsiPhi {"&".join(list(mc.keys()))} {itstr}')
    result = optimize(fcn_data,
                      method='minuit', params=pars, fcn_kwgs={'data':data},
                      verbose=False, timeit=True, tol=0.05, strategy=2)
    #print(result) # parameters are not blinded, so we dont print the result
    likelihoods.append(result.chi2)

    pars = Parameters.clone(result.params)
    if not '2018' in data.keys() and not '2017' in data.keys():
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DGs',
                'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
        print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
    else:
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'lPlon', 'DGsd',
                'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
        print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")


    # 2nd step: pdf weights ----------------------------------------------------
    #   We need to change badjanak to handle MC samples and then we compute the
    #   desired pdf weights for a given set of fitted pars in step 1. This
    #   implies looping over years and MC samples (std and dg0)
    print(f'\nPDF weighting MC samples to match Bs2JpsiPhi data {itstr}')
    t0 = timer()
    for y, dy in mc.items(): # loop over years
      for m, dm in dy.items(): # loop over mc_std and mc_dg0
        for t, v in dm.items(): # loop over mc_std and mc_dg0
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
          v.pdfWeight[i] = pdf_reweighting(v,v.params,pars+data[y][t].csp)
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_BsJpsiPhi']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi']['unbiased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['unbiased'].pdfWeight[i][evt]:>+.8f}")

    tf = timer()-t0
    print(f'PDF weighting took {tf:.3f} seconds.')



    # 3rd step: kinematic weights ----------------------------------------------
    #    We need to change badjanak to handle MC samples and then we compute the
    #    desired pdf weights for a given set of fitted pars in step 1. This
    #    implies looping over years and MC samples (std and dg0).
    #    As a matter of fact, it's important to have data[y][combined] sample,
    #    the GBweighter gives different results when having those 0s or having
    #    nothing after cutting the sample.
    print(f'\nKinematic reweighting MC samples in K momenta {itstr}')
    threads = list()
    t0 = timer()
    for y, dy in mc.items(): # loop over years
      for m, dm in dy.items(): # loop over mc_std and mc_dg0
        for t, v in dm.items():
          # original variables + weight (mc)
          ov  = v.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']]
          ow  = v.df.eval(f'angWeight*polWeight*{weight_rd}/gb_weights')
          ow *= v.pdfWeight[i]
          # target variables + weight (real data)
          tv = data[y][t].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']]
          tw = data[y][t].df.eval(f'{weight_rd}')

          # Run in single core (REALLY SLOW 10+ h)
          # kkp_weighting(ov, ow, tv, tw, v.kkpWeight[t], y, m, t, 0)
          
          # Run in multithread mode (still has some problems with memory)
          # job = threading.Thread(target=kkp_weighting,
          #                        args=(ov, ow, tv, tw,
          #                        v.kkpWeight[t], y, m, t))
          
          # Run multicore (about 15 minutes per iteration)
          job = multiprocessing.Process(target=kkp_weighting, args=(
                                ov.values, ow.values, tv.values, tw.values,
                                 v.path_to_weights, y, m, t, len(threads) ))
          threads.append(job); job.start()

    # Wait all processes to finish
    print(f' * There are {len(threads)} jobs running in parallel')
    [thread.join() for thread in threads]
    tf = timer()-t0
    print(f'Kinematic weighting took {tf:.3f} seconds.')



    # 4th step: angular weights ------------------------------------------------
    print(f'\nExtract angular weights {itstr}')
    for y, dy in mc.items(): # loop over years
      for m, dm in dy.items(): # loop over mc_std and mc_dg0
        for t, v in dm.items(): # loop over biased and unbiased triggers
          path_to_weights = v.path_to_weights.replace('.root',f'_{t}.npy')
          v.kkpWeight[i] = np.load(path_to_weights)
          os.remove(path_to_weights)
          get_angular_acceptance(v, kkpWeight=True)
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0,10):
        print(f"{dy['MC_BsJpsiPhi']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['unbiased'].kkpWeight[i][evt]:>+.8f}")



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
          print(f"{np.array(std)[_i]:+1.5f} | {np.array(dg0)[_i]:+1.5f} | {merged_w[f'w{_i}'].uvalue:+1.2uP}")
        if i>50:
          print( list(merged_w.values()) )
          print( list(data[f'{y}'][trigger].angular_weights[-1].values()) )
          for wk in merged_w.keys():
            new = merged_w[wk].value
            old = data[f'{y}'][trigger].angular_weights[-1][wk].value
            print(f"  = > {new}+{old} ")
            merged_w[wk].set(value = 0.5*(new+old))
          print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8} -- mod")
          for _i in range(len(merged_w.keys())):
            print(f"{np.array(std)[_i]:+1.5f} | {np.array(dg0)[_i]:+1.5f} | {merged_w[f'w{_i}'].uvalue:+1.2uP}")
        data[y][trigger].angacc = merged_w
        data[y][trigger].angaccs[i] = merged_w
        _check = []
        for oi in range(0,i):
          _check.append( check_for_convergence( data[y][trigger].angaccs[oi], data[y][trigger].angaccs[i] )) 
        my_checker.append(_check)
        #print("my_checker",my_checker)
        #print("_check", my_checker)

        qwe = check_for_convergence( data[y][trigger].angaccs[i-1], data[y][trigger].angaccs[i] )
        checker.append( qwe )
        merged_w.dump(data[f'{y}'][trigger].params_path.replace('.json',f'i{i}.json'))
        print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}\n')

    # Check if they are the same as previous iteration
    lb = [ data[f'{y}']['biased'].angaccs[i].__str__(['value']).splitlines() for i__,y in enumerate(YEARS) ]
    lu = [ data[f'{y}']['unbiased'].angaccs[i].__str__(['value']).splitlines() for i__,y in enumerate(YEARS) ]
    print(f"\n{80*'-'}\nBiased angular acceptance")
    for l in zip(*lb):
      print(*l, sep="| ")
    print("\nUnbiased angular acceptance")
    for l in zip(*lu):
      print(*l, sep="| ")
    print(f"\n{80*'-'}\n")


    print("CHECK: ", checker)
    if all(checker):
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items(): # loop over years
        for trigger in ['biased','unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in tex')
          pars.dump(data[y][trigger].params_path)
          print('Saving table of params in tex')
          with open(data[y][trigger].tables_path, "w") as tex_file:
            tex_file.write(
              pars.dump_latex( caption="""
              Angular acceptance for \\textbf{%s} \\texttt{\\textbf{%s}}
              category.""" % (y,trigger)
              )
            )
          tex_file.close()
      break

  # plot likelihood evolution
  lkhd_x = [i+1 for i, j in enumerate(likelihoods)]
  lkhd_y = [j+0 for i, j in enumerate(likelihoods)]
  import termplotlib
  lkhd_p = termplotlib.figure()
  lkhd_p.plot(lkhd_x, lkhd_y, xlabel='iteration', label='likelihood',
              xlim=(0, 30), ylim=(0.9*min(lkhd_y), 1.1*max(lkhd_y)))
  lkhd_p.show()

  # Storing some weights in disk -------------------------------------------------
  #     For future use of computed weights created in this loop, these should be
  #     saved to the path where samples are stored.
  #     GBweighting is slow enough once!
  print('Storing weights in root file')
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      pool = {}
      for i in v['biased'].pdfWeight.keys(): # loop over iterations
        wb = np.zeros((v['biased'].olen))
        wu = np.zeros((v['unbiased'].olen))
        wb[list(v['biased'].df.index)] = v['biased'].pdfWeight[i]
        wu[list(v['unbiased'].df.index)] = v['unbiased'].pdfWeight[i]
        pool.update({f'pdfWeight{i}': wb + wu})
      for i in v['biased'].kkpWeight.keys():  # loop over iterations
        wb = np.zeros((v['biased'].olen))
        wu = np.zeros((v['unbiased'].olen))
        wb[list(v['biased'].df.index)] = v['biased'].kkpWeight[i]
        wu[list(v['unbiased'].df.index)] = v['unbiased'].kkpWeight[i]
        print(len(wb + wu))
        pool.update({f'kkpWeight{i}': wb + wu})
      wb = np.zeros((v['biased'].olen))
      wu = np.zeros((v['unbiased'].olen))
      wb[list(v['biased'].df.index)] = v['biased'].df['angWeight'].values
      wu[list(v['unbiased'].df.index)] = v['unbiased'].df['angWeight'].values
      pool.update({f'angWeight{i}': wb + wu})
      with uproot.recreate(v['biased'].path_to_weights) as f:
        f['DecayTree'] = uproot.newtree({var:np.float64 for var in pool.keys()})
        f['DecayTree'].extend(pool)
  print(f' * Succesfully writen')
