#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ['Ramon Ruiz']
__email__  = ['mromerol@cern.ch']





################################################################################
# %% Modules ###################################################################
# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot3 as uproot # warning - upgrade to uproot4 asap
import os
import sys
import hjson
import uncertainties as unc
from uncertainties import unumpy as unp
from scipy.stats import chi2
from timeit import default_timer as timer


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
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=1, use_angacc=0,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=0, use_angacc=0,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h /= mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=1, use_angacc=0,
                           **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
  target_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc_Bd(mcsample.true, mcsample.pdf, use_fk=0, use_angacc=0,
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


def kkp_weighting(original_v, original_w, target_v, target_w, path, year, mode,
                  trigger, iter, verbose=False):

  reweighter.fit(original = original_v, target = target_v,
                 original_weight = original_w, target_weight = target_w );
  kkpWeight = reweighter.predict_weights(original_v)
  np.save(path.replace('.root',f'_{trigger}.npy'),kkpWeight)
  if verbose:
    print(f" * GB-weighting {mode}-{year}-{trigger} sample is done")
    KS_test(original_v, target_v, original_w*kkpWeight, target_w)

"""
def kkp_weighting_bins(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter_bin.fit(original = original_v, target = target_v,
                     original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter_bin.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{t}.npy',kkpWeight)
  #print(f" * GB-weighting {m}-{y}-{t} sample is done")
  print(f" * GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")
"""

def get_angular_acceptance(mc, kkpWeight=False):
  # cook weight for angular acceptance
  weight  = mc.df.eval(f'angWeight*{weight_rd}').values
  i = len(mc.kkpWeight.keys())

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[i])
  weight = ristra.allocate(weight)

  # compute angular acceptance
  ans = badjanak.get_angular_acceptance_weights_Bd(mc.true, mc.reco, weight, **mc.params.valuesdict())

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

      chi2.append( -2.0 * np.log(ristra.get(dt.lkhd)) * dt.weight.get() );
      #exit()

  return np.concatenate(chi2)

###Iterative convergencia para cualquier BDT, ayuda Marcos.
# Multiple categories functions
#     They run over multiple categories

def do_fit(verbose=True):
  """
  Fit
  """
  # Get'em from the global scope
  global pars, data

  # start where we left
  for v in pars.values():
    v.init = v.value

    result = optimize(fcn_data, method='minuit', params=pars,
                      fcn_kwgs={'data':data}, verbose=True, timeit=True,
                      tol=0.1, strategy=1)
    #likelihoods.append(result.chi2)
    #print(result.chi2)

    #names = ['fPlon', 'fPper', 'dPpar', 'dPper', 'Gd']
    #corr_run1 = Parameters.build(result.params, names).corr()
    if verbose:
      if not '2018' in data.keys() and not '2017' in data.keys():
        for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'Gd']:
          try:
            print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
          except:
            0
      else:
        for p in ['fPlon', 'fPper', 'dPpar', 'dPper','Gd']:
          try:
            print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
          except:
            0
  # store parameters + add likelihood to list
  names = ['fPlon', 'fPper', 'dPpar', 'dPper']
  pars = Parameters.clone(result.params)
  values, oldvalues, std = [], [], []
  for p in names:
    values.append(pars[p].value)
    oldvalues.append(params_init[p].value)
    std.append(pars[p].stdev)
  df = pd.DataFrame({"names": names, "values": values, "std": std, "oldvalues": oldvalues})
  df['PULL'] = df.eval('(values-oldvalues)/sqrt(std**2)')
  print(df)
  #exit()
  return result.chi2

def do_pdf_weighting(verbose):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0)
  """
  global pars, data, mc

  for y, dy in mc.items(): # loop over years
    for m, dm in dy.items(): # loop over mc_std and mc_dg0
      for t, v in dm.items(): # loop over triggers
        if verbose:
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
        v.pdfWeight[i] = pdf_reweighting(v,v.params,pars)
  if verbose:
    for y, dy in mc.items(): # loop over years
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bd2JpsiKstar':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} ")
      for evt in range(0, 10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')

def do_kkp_weighting(verbose):
  # 3rd step: kinematic weights ----------------------------------------------
  #    We need to change badjanak to handle MC samples and then we compute the
  #    desired pdf weights for a given set of fitted pars in step 1. This
  #    implies looping over years and MC samples (std and dg0).
  #    As a matter of fact, it's important to have data[y][combined] sample,
  #    the GBweighter gives different results when having those 0s or having
  #    nothing after cutting the sample.
  global mc, data, weight_rd

  threads = list()
  for y, dy in mc.items(): # loop over years
    for m, dm in dy.items(): # loop over mc_std and mc_dg0
      for t, v in dm.items():
        # original variables + weight (mc)
        ov  = v.df[['pTHm','pTHp','pHm','pHp']]
        ow  = v.df.eval(f'angWeight*{weight_rd}')
        ow *= v.pdfWeight[i]
        # target variables + weight (real data)
        tv = data[y][t].df[['pTHm','pTHp','pHm','pHp']]
        tw = data[y][t].df.eval(f'{weight_rd}')
        # Run multicore (about 15 minutes per iteration)
        job = multiprocessing.Process(
          target=kkp_weighting,
          args=(ov.values, ow.values, tv.values, tw.values, v.path_to_weights,
                y, m, t, len(threads), verbose)
        )
        threads.append(job); job.start()

  # Wait all processes to finish
  if verbose:
    print(f' * There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]



def do_angular_weights(verbose):
  """
  dddd
  """
  global mc

  for y, dy in mc.items(): # loop over years
    for m, dm in dy.items(): # loop over mc_std and mc_dg0
      for t, v in dm.items(): # loop over biased and unbiased triggers
        i = len(v.kkpWeight.keys())+1
        path_to_weights = v.path_to_weights.replace('.root',f'_{t}.npy')
        v.kkpWeight[i] = np.load(path_to_weights)
        os.remove(path_to_weights)
        get_angular_acceptance(v, kkpWeight=True)
    if verbose:
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'MC_Bd2JpsiKstar':<24}")
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0,10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')



def do_mc_combination(verbose):
  """
  Combine
  """
  global mc, data
  checker = []
  for y, dy in mc.items(): # loop over years
    for trigger in ['biased','unbiased']:
      i = len(dy['MC_Bd2JpsiKstar'][trigger].angaccs)
      std = dy['MC_Bd2JpsiKstar'][trigger].angaccs[i]
      data[y][trigger].angacc = std
      data[y][trigger].angaccs[i] = std
      qwe = check_for_convergence(data[y][trigger].angaccs[i-1], data[y][trigger].angaccs[i])
      checker.append( qwe )

  check_dict = {}
  for ci in range(0,i):
    check_dict[ci] = []
    for y, dy in data.items(): # loop over years
      for t in ['biased','unbiased']:
        qwe = check_for_convergence(dy[t].angaccs[ci], dy[t].angaccs[i])
        check_dict[ci].append( qwe )

  return checker, check_dict



def angular_acceptance_iterative_procedure(verbose=False, iteration=0):
  global pars

  itstr = f"[iteration #{iteration}]"


  #1 fit RD sample obtaining pars
  print(f'{itstr} Simultaneous fit Bd2JpsiKstar {"&".join(list(mc.keys()))}')
  likelihood = do_fit(verbose=verbose)

  #2 pdfWeight MC to RD using pars
  print(f'\n{itstr} PDF weighting MC samples to match Bd2JpsiKstar RD')
  t0 = timer()
  do_pdf_weighting(verbose=verbose)
  tf = timer()-t0
  print(f'PDF weighting took {tf:.3f} seconds.')

  #3 kkpWeight MC to RD to match K+ and K- kinematic distributions
  print(f'\n{itstr} Kinematic reweighting MC samples in K momenta')
  t0 = timer()
  do_kkp_weighting(verbose)
  tf = timer()-t0
  print(f'Kinematic weighting took {tf:.3f} seconds.')

  # 4th step: angular weights
  print(f'\n{itstr} Extract angular normalisation weights')
  t0 = timer()
  do_angular_weights(verbose)
  tf = timer()-t0
  print(f'Extract angular normalisation weights took {tf:.3f} seconds.')

  # 5th step: merge MC std and dg0 results
  print(f'\n{itstr} Combining MC_Bd2JpsiKstar')
  t0 = timer()
  checker, checker_dict = do_mc_combination(verbose)
  tf = timer()-t0

  return likelihood, checker, checker_dict





def lipschitz_iteration(max_iter=30, verbose=True):
  global pars
  likelihoods = []

  for i in range(1,max_iter):

    ans = angular_acceptance_iterative_procedure(verbose, i)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # Check if they are the same as previous iteration
    if 1:
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
    print("checker_dict: ", checker_dict)
    print("LIKELIHOODs: ", likelihoods)

    if all(checker) or i > 25:
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items(): # loop over years
        for trigger in ['biased','unbiased']:
          pars_w = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars_w.dump(data[y][trigger].params_path)
          print('Saving table of params in tex')
          with open(data[y][trigger].tables_path, "w") as tex_file:
            tex_file.write(
              pars_w.dump_latex( caption="""
              Angular acceptance for \\textbf{%s} \\texttt{\\textbf{%s}}
              category.""" % (y,trigger)
              )
            )
          tex_file.close()
      break
  return all(checker), likelihoods



def aitken_iteration(max_iter=30, verbose=True):
  global pars
  likelihoods = []

  for i in range(1,max_iter):

    # x1 = angular_acceptance_iterative_procedure <- x0
    ans = angular_acceptance_iterative_procedure(verbose, 2*i-1)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # x2 = angular_acceptance_iterative_procedure <- x1
    ans = angular_acceptance_iterative_procedure(verbose, 2*i)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)


    # x2 <- aitken solution
    checker = []
    print(f"[aitken #{i}] Update solution")
    for y, dy in data.items():  #  loop over years
      for t, dt in dy.items():
        for p in dt.angacc.keys():
          x0 = dt.angaccs[2*i-2][p].uvalue
          x1 = dt.angaccs[2*i-1][p].uvalue
          x2 = dt.angaccs[2*i][p].uvalue
          # aitken magic happens here
          den = x2 -2*x1 - x0
          if den < 1e-6:
              #checker.append(True)
              aitken = x2
          else:
              #checker.append(False)
              aitken = x2 - ( (x2-x1)**2 ) / den # aitken
              #aitken = x0 - ( (x1-x0)**2 ) / den # steffensen
              #aitken = x1 - ( (x2-x1)**2 ) / den # romero

          # update angacc
          dt.angacc[p].set(value=aitken.n)
          dt.angacc[p].stdev=aitken.s
        # update dict of angaccs
        dt.angaccs[-1] = dt.angacc

        checker.append( check_for_convergence(dt.angaccs[2*(i-1)], dt.angaccs[2*i]) )
        #check_dict[ci].append( qwe )

    # Check if they are the same as previous iteration
    if 1:
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
    print("checker_dict: ", checker_dict)
    print("LIKELIHOODs: ", likelihoods)


    if all(checker) or i > 25:
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items(): # loop over years
        for trigger in ['biased','unbiased']:
          pars_w = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars_w.dump(data[y][trigger].params_path)
          print('Saving table of params in tex')
          with open(data[y][trigger].tables_path, "w") as tex_file:
            tex_file.write(
              pars_w.dump_latex( caption="""
              Angular acceptance for \\textbf{%s} \\texttt{\\textbf{%s}}
              category.""" % (y,trigger)
              )
            )
          tex_file.close()
      break
  return all(checker), likelihoods

# Parse arguments for this script
if __name__ == '__main__':

  #Parse arguments
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
  args = vars(p.parse_args())

  # Parse arguments ------------------------------------------------------------
  #args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',')
  #YEARS = ['2015', '2016']
  #YEARS = ['2016']
  MODE = 'Bd2JpsiKstar'
  ANGACC = args['angacc']
  # Prepare the cuts -----------------------------------------------------------
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')

  # List samples, params and tables --------------------------------------------
  #samples
  samples_std   = args['sample_mc_std'].split(',')
  #samples_std = [samples_std[1]]
  #samples_std = ['/scratch28/ramon.ruiz/sidecar/2016/MC_Bd2JpsiKstar/v0r0.root']
  #samples_std = ['/scratch28/ramon.ruiz/sidecar/2015/MC_Bd2JpsiKstar/v0r5.root','/scratch28/ramon.ruiz/sidecar/2016/MC_Bd2JpsiKstar/v0r5.root']
  #samples_std = ['/scratch28/ramon.ruiz/sidecar/2015/MC_Bd2JpsiKstar/v0r5.root','/scratch28/ramon.ruiz/sidecar/2016/MC_Bd2JpsiKstar/v0r5.root', '/scratch28/ramon.ruiz/sidecar/2017/MC_Bd2JpsiKstar/v0r5.root','/scratch28/ramon.ruiz/sidecar/2018/MC_Bd2JpsiKstar/v0r5.root']

  #params
  input_std_params = args['params_mc_std'].split(',')
  #input_std_params = [input_std_params[1]]
  #print(input_std_params)
  #exit()
  #print(input_std_params)
  #input_std_params = input_std_params[:2]
  print('tables params')
  params_biased      = args['output_weights_biased'].split(',')
  print(params_biased)
  #params_biased = [params_biased[1]]
  #params_biased = params_biased[:2]
  params_unbiased    = args['output_weights_unbiased'].split(',')
  print(params_unbiased)
  #params_unbiased = [params_unbiased[1]]
  #params_unbiased = params_unbiased[:2]
  print('paths tables')
  tables_biased      = args['output_tables_biased'].split(',')
  tables_unbiased    = args['output_tables_unbiased'].split(',')
  print(tables_biased)
  print(tables_unbiased)
  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')
  print('path to weights')
  print(kkpWeight_std)


  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdconfig.values()):50}\n")


  global mc, data, weight_rd
  # %% Load samples ------------------------------------------------------------
  print(f"\n{80*'='}\nLoading samples\n{80*'='}\n")

  reco  = ['cosK', 'cosL', 'hphi', 'time']
  true  = [f'gen{i}' for i in reco]
  #reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
  reco += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
  #true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
  true += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
  real  = ['cosK','cosL','hphi','time']
  real += ['mHH','0*sigmat', 'idB','idB', '0*time', '0*time']
  #weight_rd = f'sw_{VAR}' if VAR else 'sw'
  weight_rd = f'sw'
  weight_mc = f'({weight_rd})'
  # Load Monte Carlo samples ---------------------------------------------------
  mc = {}; data = {};
  mcmodes = ['MC_Bd2JpsiKstar']
  params_init = Parameters.load(input_std_params[0])
  print(params_init)
  for i, y in enumerate( YEARS ):
    print(f'\nLoading {y} MC samples')
    mc[y] = {}
    for m, v in zip(mcmodes, [samples_std]):
      print(v[i])
      mc[y][m] = {'biased':   Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      mc[y][m]['biased'].name = f"{y}-biased"
      mc[y][m]['unbiased'].name = f"{y}-unbiased"
    for m,v in zip(mcmodes, [input_std_params]):
      mc[y][m]['biased'].assoc_params(v[i])
      mc[y][m]['unbiased'].assoc_params(v[i])
      mc[y][m]['biased'].chop(cuts_and(trigger_scissors('biased'),'(evtN % 2) == 0', CUT))
      mc[y][m]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'), '(evtN % 2) == 0', CUT))
      print(mc[y][m]['biased'])
      print(mc[y][m]['unbiased'])
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
    badjanak.config['mHH'] =  [826, 861, 896, 931, 966]
  mass = badjanak.config['mHH']
  badjanak.get_kernels(True)

  for i, y in enumerate( YEARS ):
    print(f'\nLoading {y} MC as data samples')
    data[y] = {}
    for v in [samples_std]:
      data[y] = {'biased':   Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      data[y]['biased'].name = f"data {y}-biased"
      data[y]['unbiased'].name = f"data {y}-unbiased"
      data[y]['biased'].chop(cuts_and(trigger_scissors('biased'),'(evtN % 2) != 0', CUT))#, 'logIPchi2B >= 0', 'log(BDTFchi2) >=0'))
      data[y]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'), '(evtN % 2) != 0', CUT))#, 'logIPchi2B >= 0', 'log(BDTFchi2) >=0'))
      print(data[y]['biased'])
      print(data[y]['unbiased'])
      #exit()
    for t, path in zip(['biased','unbiased'],[params_biased,params_unbiased]):
        data[y][t].params_path = path[i]
    for t, path in zip(['biased','unbiased'],[tables_biased,tables_unbiased]):
        data[y][t].tables_path = path[i]
  for i,y in enumerate (YEARS):
      for t in ['biased','unbiased']:
        print('Compute angWeights correcting MC sample in kinematics')
        print(f" * Computing kinematic GB-weighting in pTB, pB and mHH")
        reweighter.fit(original        = mc[y][m][t].df[['mHH','pB','pTB']],
                        target          = data[y][t].df[['mHH','pB','pTB']],
                       original_weight = mc[y][m][t].df.eval(weight_mc),
                       target_weight   = data[y][t].df.eval(weight_rd));
        angWeight = reweighter.predict_weights(mc[y][m][t].df[['mHH', 'pB', 'pTB']])
        #print(f'angWeight {t}')
        #print(angWeight[0:20])
        #angWeight =  np.ones_like(mc[y][m][t].df['angWeight'])
        mc[y][m][t].df['angWeight'] = angWeight
        print(mc[y][m][t].df['angWeight'])
        mc[y][m][t].olen = len(angWeight)
        print(mc[y][m][t].params)
        angacc = badjanak.get_angular_acceptance_weights_Bd(mc[y][m][t].true, mc[y][m][t].reco,
                                     mc[y][m][t].weight*ristra.allocate(angWeight),
                             **mc[y][m][t].params.valuesdict())
        w, uw, cov, corr = angacc
        pars_ws = Parameters()
        for i in range(0,len(w)):
          correl = {f'w{j}': corr[i][j]
                    for j in range(0, len(w)) if i > 0 and j > 0}
          pars_ws.add({'name': f'w{i}', 'value': w[i], 'stdev': uw[i],
                    'correl': correl, 'free': False, 'latex': f'w_{i}'})
        print(f" * Corrected angular weights for {MODE}{y}-{t} sample are:")
        print(f"{pars_ws}")
        data[y][t].angacc = pars_ws
        data[y][t].angaccs = {0:pars_ws}

  #for i,y in enumerate (YEARS):
    #for d in [mc[y][m]['biased'],mc[y][m]['unbiased']]:
      #sw = np.zeros_like(d.df[f'sw'])
      #print(f'mc{y}{m}{d}')
      #for l,h in zip(mass[:-1],mass[1:]):
        #pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        #this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        #print(f'l=',l, 'h=',h, 'N=',np.sum(pos),'Neff=',np.sum(this_sw)*np.sum(this_sw)/(np.sum(this_sw*this_sw)))
        #sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)

  for i,y in enumerate (YEARS):
    print(f' *  Allocating {y} arrays in device ')
    for d in [data[y]['biased'],data[y]['unbiased']]:
      sw = np.zeros_like(d.df[f'{weight_rd}'])
      for l,h in zip(mass[:-1],mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        print(f'l=',l, 'h=',h, 'N=',np.sum(pos),'Neff=',np.sum(this_sw)*np.sum(this_sw)/(np.sum(this_sw*this_sw)))
        sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
      d.df['sWeight'] = sw
      d.allocate(data=real,weight='sWeight',lkhd='0*time')

  # %% Prepare dict of parameters ----------------------------------------------
  print(f"\n{80*'='}\nParameters and initial status\n{80*'='}\n")

  print(f"\nFitting parameters\n{80*'='}")
  global pars; pars = Parameters()

  # P wave fractions normalmente fPlon
  pars.add(dict(name="fPlon", value=0.5001, min=0.1, max=0.9,
            free=True, latex=r'f_0'))
  pars.add(dict(name="fPper", value=0.1601, min=0.1, max=0.9,
            free=True, latex=r'f_{\perp}')) #0.170)
  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.000, min=-3.14, max=3.14,
            free=False, latex=r"\delta_0"))
  pars.add(dict(name="dPpar", value=2.501, min=-2*3.14, max=2*3.14,
            free=True, latex=r"\delta_{\parallel} - \delta_0")) #2.501
  pars.add(dict(name="dPper", value=-0.17, min=-2*3.14, max=2*3.14,
            free=True, latex=r"\delta_{\perp} - \delta_0")) #-0.17
  # lambdas for Bd fix to 1
  # life parameters
  pars.add(dict(name="Gd", value= 0.65833, min= 0.0, max= 1.0,
                free=False, latex=r"\Gamma_d"))
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

  # run the procedure!


  ok, likelihoods = lipschitz_iteration(max_iter=5, verbose=False)

  if not ok:
    ok, likelihoods = aitken_iteration(max_iter=30, verbose=True)

  if not ok:
    print('WARNING: Convergence was not achieved!')

  if ok:
    names = ['fPlon', 'fPper', 'dPpar', 'dPper']
    values, oldvalues, std = [], [], []
    for p in names:
      values.append(pars[p].value)
      oldvalues.append(params_init[p].value)
      std.append(pars[p].stdev)
    df = pd.DataFrame({"names": names, "values": values, "std": std, "oldvalues": oldvalues})
    df['PULL'] = df.eval('(values-oldvalues)/sqrt(std**2)')
    print('PULL with gen values of MC')
    print(df)


  # plot likelihood evolution
  ld_x = [i+1 for i, j in enumerate(likelihoods)]
  ld_y = [j+0 for i, j in enumerate(likelihoods)]
  import termplotlib
  ld_p = termplotlib.figure()
  ld_p.plot(ld_x, ld_y, xlabel='iteration', label='likelihood',xlim=(0,30))
  ld_p.show()


  # Storing some weights in disk -----------------------------------------------
  #     For future use of computed weights created in this loop, these should be
  #     saved to the path where samples are stored.
  #     GBweighting is slow enough once!
  print('Storing weights in root file')
  for y, dy in mc.items(): # loop over years
    for m, v in dy.items(): # loop over mc_std and mc_dg0
      pool = {}
      for i in v['biased'].pdfWeight.keys(): # loop over iterations
        wb = np.zeros((v['biased'].olen))
        wu = np.zeros((v['biased'].olen))
        #wb[list(v['biased'].df.index)] = v['biased'].pdfWeight[i]
        #wu[list(v['unbiased'].df.index)] = v['unbiased'].pdfWeight[i]
        pool.update({f'pdfWeight{i}': wb + wu})
      for i in v['biased'].kkpWeight.keys():  # loop over iterations
        wb = np.zeros((v['biased'].olen))
        wu = np.zeros((v['biased'].olen))
        #wb[list(v['biased'].df.index)] = v['biased'].kkpWeight[i]
        #wu[list(v['unbiased'].df.index)] = v['unbiased'].kkpWeight[i]
        pool.update({f'kkpWeight{i}': wb + wu})
      wb = np.zeros((v['biased'].olen))
      wu = np.zeros((v['biased'].olen))
      #wb[list(v['biased'].df.index)] = v['biased'].df['angWeight'].values
      #wu[list(v['unbiased'].df.index)] = v['unbiased'].df['angWeight'].values
      pool.update({f'angWeight': wb + wu})
      with uproot.recreate(v['biased'].path_to_weights) as f:
        f['DecayTree'] = uproot.newtree({var:np.float64 for var in pool.keys()})
        f['DecayTree'].extend(pool)
  print(f' * Succesfully writen')
