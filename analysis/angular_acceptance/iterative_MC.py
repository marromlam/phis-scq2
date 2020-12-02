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
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)


def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  if checker == 0:
    return True
  return False


def pdf_reweighting(mcsample, mcparams, rdparams):
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=1,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=0,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h /= mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=1,
                           **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
  target_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=0,
                           **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
  target_pdf_h /= mcsample.pdf.get()
  #print(f"   pdfWeight[{i}]: { np.nan_to_num(target_pdf_h/original_pdf_h) }")
  return np.nan_to_num(target_pdf_h/original_pdf_h)



def KS_test(original, target, original_weight, target_weight):
  vars = ['pTHm','pTHp','pHm','pHp']
  for i in range(0,4):
    xlim = np.percentile(np.hstack([target[:,i]]), [0.01, 99.99])
    print(f'   KS({vars[i]:>10}) =',
          ks_2samp_weighted(original[:,i], target[:,i],
                            weights1=original_weight, weights2=target_weight))


def kkp_weighting(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter.fit(original = original_v, target = target_v,
                 original_weight = original_w, target_weight = target_w );
  kkpWeight = reweighter.predict_weights(original_v)
  np.save(path.replace('.root',f'_{t}.npy'),kkpWeight)
  print(f" * GB-weighting {m}-{y}-{t} sample is done")
  KS_test(original_v, target_v, original_w*kkpWeight, target_w)


def kkp_weighting_bins(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter_bin.fit(original = original_v, target = target_v,
                     original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter_bin.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{t}.npy',kkpWeight)
  #print(f" * GB-weighting {m}-{y}-{t} sample is done")
  print(f" * GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")


def get_angular_acceptance(mc, kkpWeight=False):
  # cook weight for angular acceptance
  weight  = mc.df.eval(f'angWeight*polWeight*{weight_rd}').values
  i = len(mc.kkpWeight.keys())

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[i])
  weight = ristra.allocate(weight)

  # compute angular acceptance
  ans = badjanak.get_angular_acceptance_weights(mc.true, mc.reco, weight, **mc.params.valuesdict())

  # create ipanema.Parameters
  w, uw, cov, corr = ans
  mc.angaccs[i] = Parameters()
  for k in range(0,len(w)):
    correl = {f'w{j}': corr[k][j]
              for j in range(0, len(w)) if k > 0 and j > 0}
    mc.angaccs[i].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
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
      badjanak.delta_gamma5_data_Bd(dt.data, dt.lkhd, **pars_dict,
                   **dt.angacc.valuesdict(),
                   #**dt.csp.valuesdict(),
                   tLL=tLL, tUL=tUL, use_timeacc=0, set_tagging=1, use_timeres=0)
                   #set_tagging =2

      chi2.append( -2.0 * (ristra.log(dt.lkhd) * dt.weight).get() );
      #exit()

  return np.concatenate(chi2)



# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description='Check iterative procedure for angular acceptance.')
  p.add_argument('--sample-mc-std', help='Bd2JpsiKstar MC sample')
  p.add_argument('--params-mc-std', help='BdJpsiKstar MC sample')
  p.add_argument('--output-weights-biased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-weights-unbiased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-angular-weights-mc-std', help='BdJpsiKstar MC sample')
  p.add_argument('--output-tables-biased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-tables-unbiased', help='BdJpsiKstar MC sample')
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
  YEARS = args['year'].split(',')
  MODE = 'Bd2JpsiKstar'
  ANGACC = args['angacc']
  # Prepare the cuts -----------------------------------------------------------
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  # Load Monte Carlo samples

  CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')

  # List samples, params and tables --------------------------------------------
  #samples
  samples_std   = args['sample_mc_std'].split(',')
  #params
  input_std_params = args['params_mc_std'].split(',')
  params_biased      = args['output_weights_biased'].split(',')
  params_unbiased    = args['output_weights_unbiased'].split(',')

  tables_biased      = args['output_tables_biased'].split(',')
  tables_unbiased    = args['output_tables_unbiased'].split(',')

  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')


  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdconfig.values()):50}\n")



  # %% Load samples ------------------------------------------------------------
  print(f"\n{80*'='}\nLoading samples\n{80*'='}\n")

  reco  = ['cosK', 'cosL', 'hphi', 'time']
  true  = [f'gen{i}' for i in reco]
  reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
  true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']

  real  = ['cosK','cosL','hphi','time']
  real += ['mHH','0*sigmat', 'idB','idB', '0*time', '0*time']
  weight_rd = f'sw_{VAR}' if VAR else 'sw'
  weight_mc = f'(polWeight*{weight_rd})'

  # Load Monte Carlo samples ---------------------------------------------------
  mc = {}; data = {};
  mcmodes = ['MC_Bd2JpsiKstar']
  params_init = Parameters.load(input_std_params[0])
  for i, y in enumerate( YEARS ):
    print(f'\nLoading {y} MC samples')
    mc[y] = {}
    for m, v in zip(mcmodes, [samples_std]):
      mc[y][m] = {'biased':   Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      mc[y][m]['biased'].name = f"{y}-biased"
      mc[y][m]['unbiased'].name = f"{y}-unbiased"
    for m,v in zip(mcmodes, [input_std_params]):
      mc[y][m]['biased'].assoc_params(v[i])
      mc[y][m]['unbiased'].assoc_params(v[i])
      mc[y][m]['biased'].chop(cuts_and(trigger_scissors('biased'),'(evtN % 2) == 0', CUT))
      mc[y][m]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'), '(evtN % 2) == 0', CUT))
      for t in ['biased', 'unbiased']:
        mc[y][m][t].allocate(reco=reco, true=true, pdf='0*time', weight=weight_mc)
        mc[y][m][t].df['angWeight'] = 0.0
        mc[y][m][t].angaccs = {}
        mc[y][m][t].kkpWeight = {}
        mc[y][m][t].pdfWeight = {}
    for m, v in zip(mcmodes,[kkpWeight_std]):
      mc[y][m]['biased'].path_to_weights = v[i]
      mc[y][m]['unbiased'].path_to_weights = v[i]

  if MODE== 'Bd2JpsiKstar':
    badjanak.config['x_m'] =  [826, 861, 896, 931, 966]
  mass = badjanak.config['x_m']

  for i, y in enumerate( YEARS ):
    print(f'\nLoading {y} MC as data samples')
    data[y] = {}
    for v in [samples_std]:
      data[y] = {'biased':   Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      data[y]['biased'].name = f"data {y}-biased"
      data[y]['unbiased'].name = f"data {y}-unbiased"
      data[y]['biased'].chop(cuts_and(trigger_scissors('biased'),'(evtN % 2) != 0', CUT))
      data[y]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'), '(evtN % 2) != 0', CUT))
    for t, path in zip(['biased','unbiased'],[params_biased,params_unbiased]):
        data[y][t].params_path = path[i]
    for t, path in zip(['biased','unbiased'],[tables_biased,tables_unbiased]):
        data[y][t].tables_path = path[i]

  for i,y in enumerate (YEARS):
      for t in ['biased','unbiased']:
        print('Compute angWeights correcting MC sample in kinematics')
        print(f" * Computing kinematic GB-weighting in pTB, pB and mHH")
        print(mc[y][m][t].df.eval(weight_mc))
        print(data[y][t].df.eval(weight_rd))
        reweighter.fit(original        = mc[y][m][t].df[['mHH','pB','pTB']],
                       target          = data[y][t].df[['mHH','pB','pTB']],
                       original_weight = mc[y][m][t].df.eval(weight_mc),
                       target_weight   = data[y][t].df.eval(weight_rd));
        angWeight = reweighter.predict_weights(mc[y][m][t].df[['mHH', 'pB', 'pTB']])
        mc[y][m][t].df['angWeight'] = angWeight
        mc[y][m][t].olen = len(angWeight)
        angacc = badjanak.get_angular_acceptance_weights(mc[y][m][t].true, mc[y][m][t].reco,
                                     mc[y][m][t].weight*ristra.allocate(angWeight),
                                     **mc[y][m][t].params.valuesdict())
        w, uw, cov, corr = angacc
        pars = Parameters()
        for i in range(0,len(w)):
          correl = {f'w{j}{t}': corr[i][j]
                    for j in range(0, len(w)) if i > 0 and j > 0}
          pars.add({'name': f'w{i}{t}', 'value': w[i], 'stdev': uw[i],
                    'correl': correl, 'free': False, 'latex': f'w_{i}^{t}'})
        print(f" * Corrected angular weights for {MODE}{y}-{t} sample are:")
        print(f"{pars}")
        data[y][t].angacc = pars
        data[y][t].angaccs = {0:pars}

  for i,y in enumerate (YEARS):
    print(f' *  Allocating {y} arrays in device ')
    for d in [data[y]['biased'],data[y]['unbiased']]:
      sw = np.zeros_like(d.df[f'{weight_rd}'])
      for l,h in zip(mass[:-1],mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
      d.df['sWeight'] = sw
      d.allocate(data=real,weight='sWeight',lkhd='0*time')

  # %% Prepare dict of parameters ----------------------------------------------
  print(f"\n{80*'='}\nParameters and initial status\n{80*'='}\n")

  print(f"\nFitting parameters\n{80*'='}")
  pars = Parameters()

  # P wave fractions normalmente fPlon
  pars.add(dict(name="fPlon", value=0.5240, min=0.1, max=0.9,
            free=True, latex=r'f_0'))
  pars.add(dict(name="fPper", value=0.170, min=0.1, max=0.9,
            free=True, latex=r'f_{\perp}'))
  # Weak phases
  pars.add(dict(name="pSlon", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_S - \phi_0"))
  pars.add(dict(name="pPlon", value= 0.00, min=-1.0, max=1.0,
            free=False , latex=r"\phi_0" ))
  pars.add(dict(name="pPpar", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_{\parallel} - \phi_0"))
  pars.add(dict(name="pPper", value= 0.00, min=-1.0, max=1.0,
            free=False, latex=r"\phi_{\perp} - \phi_0"))
  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.000, min=-3.14, max=3.14,
            free=False, latex=r"\delta_0"))
  pars.add(dict(name="dPpar", value=2.501, min=-3.14, max=3.14,
            free=True, latex=r"\delta_{\parallel} - \delta_0"))
  pars.add(dict(name="dPper", value=-0.17, min=-3.14, max=3.14,
            free=True, latex=r"\delta_{\perp} - \delta_0"))
  # lambdas
  pars.add(dict(name="lPlon", value=1.0, min=-1.5, max=1.5,
            free=False,  latex="\lambda_0"))
  pars.add(dict(name="lPpar", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\parallel}/\lambda_0"))
  pars.add(dict(name="lPper", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\perp}/\lambda_0"))
  # life parameters
  pars.add(dict(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
            free=False, latex=r"\Gamma_d"))
  #pars.add(dict(name="DGs", value= 0.0917, min= 0.03, max= 0.15,
            #free=True, latex=r"\Delta\Gamma_s"))
  pars.add(dict(name="DGs", value= 0.0, min= 0.0, max= 0.15,
            free=False, latex=r"\Delta\Gamma_s"))
  pars.add(dict(name="DGsd", value= 0.0, min=-0.2, max= 0.2,
            free=False, latex=r"\Gamma_s - \Gamma_d"))
  pars.add(dict(name="DM", value=0.5, min=0.1, max=1.0,
            free=False, latex=r"\Delta m"))
  print(pars)

  # print angular acceptance
  lb = [ data[y]['biased'].angaccs[0].__str__(['value']).splitlines() for i,y in enumerate( YEARS ) ]
  lu = [ data[y]['unbiased'].angaccs[0].__str__(['value']).splitlines() for i,y in enumerate( YEARS ) ]
  print(f"\nBiased angular acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")
  print(f"\nUnbiased angular acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")
  print(f"\n")

  badjanak.get_kernels(True)

  # The iterative procedure starts ---------------------------------------------
  #     First print angular acceptance before iterative procedure
  CHECK_DICT = {}
  likelihoods = []
  for i in range(1,30):
    print(f"\n{80*'='}\nIteration {i} of the procedure\n{80*'='}\n")
    checker = []                # here we'll store if weights do converge or not
    for ci in range(0,i):
      CHECK_DICT[ci] = []
    itstr = f"[iteration #{i}]"

    # 1st step: fit data -------------------------------------------------------
    for v in pars.values():
      v.init = v.value # start where we left
    print(f'Simultaneous fit Bd2JpsiKstar {"&".join(list(mc.keys()))} {itstr}')
    print(pars)
    print(data.__str__())
    result = optimize(fcn_data,
                      method='minuit', params=pars, fcn_kwgs={'data':data},
                      verbose=True, timeit=True, tol=0.5, strategy=1)
    print(result)
    likelihoods.append(result.chi2)

    pars = Parameters.clone(result.params)

    if not '2018' in data.keys() and not '2017' in data.keys():
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DGs',
                'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
        print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
    else:
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'lPlon', 'DGsd',
                'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4']:
        try:
            print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
        except:
            0
    # 2nd step: pdf weights ----------------------------------------------------
    #   We need to change badjanak to handle MC samples and then we compute the
    #   desired pdf weights for a given set of fitted pars in step 1. This
    #   implies looping over years
    print(f'\nPDF weighting MC samples to match Bd2JpsiKstar data {itstr}')
    t0 = timer()
    for y, dy in mc.items(): # loop over years
      for m, dm in dy.items(): # loop over mc_std and mc_dg0
        for t, v in dm.items(): # loop over mc_std and mc_dg0
            print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
            v.pdfWeight[i] = pdf_reweighting(v,v.params,pars)#+data[y][t].csp)
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bd2JpsiKstar':<24}")
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')


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
          ov  = v.df[['pTHm','pTHp','pHm','pHp']]
          ow  = v.df.eval(f'angWeight*polWeight*{weight_rd}/gb_weights')
          ow *= v.pdfWeight[i]
          # target variables + weight (real data)
          tv = data[y][t].df[['pTHm','pTHp','pHm','pHp']]
          tw = data[y][t].df.eval(f'{weight_rd}')
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
      print(f"{'MC_Bs2JpsiPhi':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0,10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')



    # 5th step: merge MC std and dg0 results -----------------------------------
    #Step 5 comentado todo por ramon
    print(f'\Parameters Angacc MC_BdJsiKstar {itstr}')

    for y, dy in mc.items(): # loop over years
      for trigger in ['biased','unbiased']:
        # Get angular weights for each MC
        std = dy['MC_Bd2JpsiKstar'][trigger].angaccs[i]

        # Create w and cov arrays
        std_w = np.array([std[f'w{i}'].value for i in range(1,len(std))])
        std_cov = std.cov()[1:,1:];

        # Some matrixes
        std_covi = np.linalg.inv(std_cov)
        cov_comb_inv = np.linalg.inv( std_cov )
        cov_comb = np.linalg.inv( std_covi )
        dof = len(std_w)

        # Combine angular weights
        w = np.ones((dof+1))
        w[1:] = cov_comb.dot( std_covi.dot(std_w.T))

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
          correl = {f'w{j}{trigger[0]}': corr[k][j]
                    for j in range(0, len(w)) if k > 0 and j > 0}
          merged_w.add({'name': f'w{k}{trigger[0]}', 'value': w[k], 'stdev': uw[k],
                        'free': False, 'latex': f'w_{k}^{trigger[0]}', 'correl': correl})
        print(f"Current angular weights for BdJpsiKstar-{y}-{trigger} sample are:")
        print(f"{'MC':>8}")
        for _i in range(len(merged_w.keys())):
          print(f"{np.array(merged_w)[_i]:+1.5f}")

        if i>5:
          for wk in merged_w.keys():
            new = merged_w[wk].value
            old = data[y][trigger].angaccs[i-1][wk].value
            print(f"  = > {new}+{old} ")
            merged_w[wk].set(value = 0.5*(new+old))
          print(f"{'MC':>8}")
          for _i in range(len(merged_w.keys())):
            print(f"{np.array(merged_w)[_i]:+1.5f}")

        # Save current set of angular weights
        data[y][trigger].angacc = merged_w
        data[y][trigger].angaccs[i] = merged_w

        # Check for all iterations if existed convergence
        for ci in range(0,i):
            CHECK_DICT[ci].append(check_for_convergence(
            data[y][trigger].angaccs[ci], data[y][trigger].angaccs[i]))

        qwe = check_for_convergence( data[y][trigger].angaccs[i-1], data[y][trigger].angaccs[i] )
        checker.append( qwe )
        merged_w.dump(data[y][trigger].params_path.replace('.json',f'i{i}.json'))
        #print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}\n')

    # Check if they are the same as previous iteration
    lb = [ data[y]['biased'].angaccs[i].__str__(['value']).splitlines() for i__,y in enumerate( YEARS ) ]
    lu = [ data[y]['unbiased'].angaccs[i].__str__(['value']).splitlines() for i__,y in enumerate( YEARS ) ]
    print(f"\n{80*'-'}\nBiased angular acceptance")
    for l in zip(*lb):
      print(*l, sep="| ")
    print("\nUnbiased angular acceptance")
    for l in zip(*lu):
      print(*l, sep="| ")
    print(f"\n{80*'-'}\n")

    print("CHECK: ", checker)
    print("CHECK_DICT: ", CHECK_DICT)
    print("LIKELIHOODs: ", likelihoods)

    if all(checker):
      print(f"\nDone! Convergence was achieved within {i} iterations")
      names = ['fPlon', 'fPper', 'dPpar', 'dPper','DGsd']
      values, oldvalues, std = [], [], []
      for p in names:
        values.append(pars[p].value)
        oldvalues.append(params_init[p].value)
        std.append(pars[p].stdev)
      df = pd.DataFrame({"names": names, "values": values, "std": std, "oldvalues": oldvalues})
      df['PULL'] = df.eval('(values-oldvalues)/sqrt(std**2)')
      print('PULL con los valores generados en el MC')
      print(df)
#Comentado checkeo w's con primera interacción
#      for y, dy in data.items(): # loop over years
#        for trigger in ['biased','unbiased']:
#          values, oldvalues, std, oldstd, names = [], [],[], [], []
#          pars = data[y][trigger].angaccs[i]
#          pars0 = check0[y][trigger]
#          names = [check0[y][trigger][p].name for p in check0[y][trigger]]
#          for p in names[1:]:
#            values.append(pars[p].value)
#            oldvalues.append(pars0[p].value)
#            std.append(pars[p].stdev)
#            oldstd.append(pars0[p].stdev)
#          df = pd.DataFrame({"names": names[1:], "values": values, "stdev": std, "oldvalues": oldvalues, "oldstd": oldstd})
#          df['PULL'] = df.eval('(values-oldvalues)/sqrt(stdev**2+oldstd**2)')
#          print(f"PULLS {y}-{trigger}")
#          print(df)
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
  ld_x = [i+1 for i, j in enumerate(likelihoods)]
  ld_y = [j+0 for i, j in enumerate(likelihoods)]
  import termplotlib
  ld_p = termplotlib.figure()
  ld_p.plot(ld_x, ld_y, xlabel='iteration', label='likelihood',xlim=(0,30))
  ld_p.show()

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
      pool.update({f'angWeight': wb + wu})
      with uproot.recreate(v['biased'].path_to_weights) as f:
        f['DecayTree'] = uproot.newtree({var:np.float64 for var in pool.keys()})
        f['DecayTree'].extend(pool)
  print(f' * Succesfully writen')
