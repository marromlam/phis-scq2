from hep_ml import reweight
import config
from utils.helpers import version_guesser, timeacc_guesser, trigger_scissors, parse_angacc
from utils.strings import cammel_case_split, cuts_and, printsec, printsubsec
from utils.plot import mode_tex
from analysis import badjanak
from ipanema import ristra, Sample, Parameters, Parameter, optimize
from ipanema import initialize
import multiprocessing
import time
import threading
import logging
from hep_ml.metrics_utils import ks_2samp_weighted
from timeit import default_timer as timer
from scipy.stats import chi2
from uncertainties import unumpy as unp
import uncertainties as unc
import hjson
import sys
import os
import uproot3 as uproot  # warning - upgrade to uproot4 asap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from warnings import simplefilter
from utils.helpers import parse_angacc
DESCRIPTION = """
    Angular acceptance iterative procedure for MC check
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']

__all__ = ['fcn_data', 'get_angular_acceptance', 'kkp_weighting', 'compute_pdf_ratio_weight', 'check_for_convergence']


################################################################################
# Modules ######################################################################

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# threading

# load ipanema
initialize(config.user['backend'], 1, real='double')

# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

# import some phis-scq utils

# binned variables
bdtconfig = config.angacc['bdtconfig']
#resolutions = config.timeacc['constants']
#all_knots = config.timeacc['knots']
#bdtconfig = config.timeacc['bdtconfig']
#Gdvalue = config.general['Gd']
#tLL = config.general['tLL']
#tUL = config.general['tUL']

# reweighting config
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
# reweighter = reweight.GBReweighter(**bdtconfig)
# 40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000


def acceptance_effect(p, a):
  return p**3 / (a + p**3)


def check_for_convergence(a, b):
  a_f = np.array([float(a[p].unc_round[0]) for p in a])
  b_f = np.array([float(b[p].unc_round[0]) for p in b])
  checker = np.abs(a_f - b_f).sum()
  if checker == 0:
    return True
  return False


def compute_pdf_ratio_weight(mcsample, mcparams, rdparams, tLL, tUL):
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                           **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
  original_pdf_h /= mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                           **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
  target_pdf_h = mcsample.pdf.get()
  badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                           **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
  target_pdf_h /= mcsample.pdf.get()
  return np.nan_to_num(target_pdf_h / original_pdf_h)

# core functions
#     They work for a given category only.


def pdf_reweighting(mcsample, mcparams, rdparams):
  if 'Bs2Jpsi' in MODE:
    badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                             **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
    original_pdf_h = mcsample.pdf.get()
    badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                             **mcparams.valuesdict(), tLL=tLL, tUL=tUL)
    original_pdf_h /= mcsample.pdf.get()
    badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=1,
                             **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
    target_pdf_h = mcsample.pdf.get()
    badjanak.delta_gamma5_mc(mcsample.true, mcsample.pdf, use_fk=0,
                             **rdparams.valuesdict(), tLL=tLL, tUL=tUL)
    target_pdf_h /= mcsample.pdf.get()
  else:
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
  return np.nan_to_num(target_pdf_h / original_pdf_h)


# Komogorov test

def KS_test(original, target, original_weight, target_weight):
  """
  Kolmogorov test
  """
  vars = ['pTHm', 'pTHp', 'pHm', 'pHp']
  for i in range(0, 4):
    xlim = np.percentile(np.hstack([target[:, i]]), [0.01, 99.99])
    print(f'   KS({vars[i]:>10}) =',
          ks_2samp_weighted(original[:, i], target[:, i],
                            weights1=original_weight, weights2=target_weight))


def kkp_weighting(original_v, original_w, target_v, target_w, path, year, mode,
                  trigger, iter, verbose=False):
  """
  Kinematic reweighting
  """
  reweighter = reweight.GBReweighter(**bdtconfig)
  # do reweighting
  reweighter.fit(original=original_v, target=target_v,
                 original_weight=original_w, target_weight=target_w)
  # predict weights
  kkpWeight = reweighter.predict_weights(original_v)
  # save them temp
  np.save(path.replace('.root', f'_{trigger}.npy'), kkpWeight)
  # some prints
  if verbose:
    print(f" * GB-weighting {mode}-{year}-{trigger} sample is done")
    KS_test(original_v, target_v, original_w * kkpWeight, target_w)


"""
def kkp_weighting_bins(original_v, original_w, target_v, target_w, path, y,m,t,i):
  reweighter_bin.fit(original = original_v, target = target_v,
                     original_weight = original_w, target_weight = target_w );
  kkpWeight = np.where(original_w!=0, reweighter_bin.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path)+f'/kkpWeight_{trigger}.npy',kkpWeight)
  #print(f" * GB-weighting {m}-{y}-{trigger} sample is done")
  print(f" * GB-weighting {m}-{y}-{trigger} sample\n  {kkpWeight[:10]}")
"""


def get_angular_acceptance(mc, tLL, tUL, kkpWeight=False):
  """
  Compute angular acceptance
  """
  # cook weight for angular acceptance
  weight = mc.df.eval(f'angWeight*polWeight*{weight_mc}').values  #  WARNING: put polWeight again
  i = len(mc.kkpWeight.keys())

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[i])
  weight = ristra.allocate(weight)
  # compute angular acceptance
  if 'Bs2Jpsi' in MODE:
    ans = badjanak.get_angular_acceptance_weights(mc.true, mc.reco, weight, **mc.params.valuesdict(), tLL=tLL, tUL=tUL)
  else:
    ans = badjanak.get_angular_acceptance_weights_Bd(mc.true, mc.reco, weight, **mc.params.valuesdict(), tLL=tLL, tUL=tUL)

  # create ipanema.Parameters
  w, uw, cov, corr = ans
  mc.angaccs[i] = Parameters()
  for k in range(0, len(w)):
    correl = {f'w{j}': corr[k][j]
              for j in range(0, len(w)) if k > 0 and j > 0}
    mc.angaccs[i].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                       'free': False, 'latex': f'w_{k}', 'correl': correl})
  #print(f"{  np.array(mc.angular_weights[t])}")


# this one should be moved
def fcn_data(parameters, data, tLL, tUL):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []

  for dy in data.values():
    for dt in dy.values():
      badjanak_gamma5(dt.data, dt.lkhd, pars_dict, **dt.timeacc.valuesdict(),
                      **dt.csp.valuesdict(), **dt.angacc.valuesdict(),
                      tLL=tLL, tUL=tUL, BLOCK_SIZE=128)
      chi2.append(-2.0 * (ristra.log(dt.lkhd) * dt.weight).get())

  return np.concatenate(chi2)


def merge_std_dg0(std, dg0, verbose=True, label=''):
  # Create w and cov arrays
  std_w = np.array([std[i].value for i in std])[1:]
  dg0_w = np.array([dg0[i].value for i in dg0])[1:]
  std_cov = std.cov()[1:, 1:]
  dg0_cov = dg0.cov()[1:, 1:]

  # Some matrixes
  std_covi = np.linalg.inv(std_cov)
  dg0_covi = np.linalg.inv(dg0_cov)
  cov_comb_inv = np.linalg.inv(std_cov + dg0_cov)
  cov_comb = np.linalg.inv(std_covi + dg0_covi)

  # Check p-value
  chi2_value = (std_w - dg0_w).dot(cov_comb_inv.dot(std_w - dg0_w))
  dof = len(std_w)
  prob = chi2.sf(chi2_value, dof)

  # Combine angular weights
  w = np.ones((dof + 1))
  w[1:] = cov_comb.dot(std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T))

  # Combine uncertainties
  uw = np.zeros_like(w)
  uw[1:] = np.sqrt(np.diagonal(cov_comb))

  # Build correlation matrix
  corr = np.zeros((dof + 1, dof + 1))
  for k in range(1, cov_comb.shape[0]):
    for j in range(1, cov_comb.shape[1]):
      corr[k, j] = cov_comb[k][j] / np.sqrt(cov_comb[k][k] * cov_comb[j][j])

  # Create parameters std_w
  out = Parameters()
  for k, wk in enumerate(std.keys()):
    correl = {f'{wk}{label}': corr[k][j] for j in range(0, len(w)) if k > 0 and j > 0}
    out.add({'name': f'{wk}{label}', 'value': w[k], 'stdev': uw[k],
             'free': False, 'latex': f'{std[wk].latex}^{label}', 'correl': correl})

  if verbose:
    print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8}")
    for k, wk in enumerate(std.keys()):
      print(f"{np.array(std)[k]:+1.5f}", end=' | ')
      print(f"{np.array(dg0)[k]:+1.5f}", end=' | ')
      print(f"{out[f'{wk}{label}'].uvalue:+1.2uP}")
  return out


# Multiple categories functions
#     They run over multiple categories

def do_fit(tLL, tUL, verbose=False):
  """
  Fit
  """
  # Get'em from the global scope
  global pars, data
  # start where we left
  for v in pars.values():
    v.init = v.value

  # do the fit
  result = optimize(fcn_data, method='minuit', params=pars,
                    fcn_kwgs=dict(data=data, tLL=tLL, tUL=tUL), verbose=True, timeit=True,
                    tol=0.05, strategy=2)

  # print fit results
  print(result)  # parameters are not blinded, so we dont print the result
  if verbose:
    if not '2018' in data.keys() and not '2017' in data.keys():
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd',
                'DGs', 'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5',
                'dSlon6', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5',
                'fSlon6']:
        try:
          print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
        except:
          0
    else:
      for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'lPlon', 'DGsd', 'DM',
                'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
        try:
          print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
        except:
          0
  # store parameters + add likelihood to list
  for p, par in result.params.items():
    if not par.casket:
      par.casket = {}
      i = 1
    else:
      i = len(par.casket.keys()) + 1
    par.casket.update({f"{i}": {"value": par.uvalue.n, "stdev": par.uvalue.s}})
  pars = Parameters.clone(result.params)
  return result.chi2


def do_pdf_weighting(tLL, tUL, verbose):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0)
  """
  global pars, data, mc

  for y, dy in mc.items():  #  loop over years
    for t, v in dy.items():  # loop over triggers
      if verbose:
        print(f' * Calculating pdfWeight for {MODE}-{y}-{t} sample')
      j = len(v.pdfWeight.keys()) + 1
      # v.pdfWeight[i] = pdf_reweighting(v, v.params, pars+data[y][t].csp)
      v.pdfWeight[j] = compute_pdf_ratio_weight(v, v.params, pars + data[y][t].csp, tLL=tLL, tUL=tUL)
  if verbose:
    for y, dy in mc.items():  #  loop over years
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['biased'].pdfWeight[j][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['unbiased'].pdfWeight[j][evt]:>+.8f}", end='\n')
        print(f" | ", end='')


def do_kkp_weighting(verbose):
  # 3rd step: kinematic weights ----------------------------------------------
  #    We need to change badjanak to handle MC samples and then we compute the
  #    desired pdf weights for a given set of fitted pars in step 1. This
  #    implies looping over years and MC samples (std and dg0).
  #    As a matter of fact, it's important to have data[y][combined] sample,
  #    the GBweighter gives different results when having those 0s or having
  #    nothing after cutting the sample.
  global mc, data, weight_rd, weight_mc

  threads = list()
  for y, dy in mc.items():  #  loop over years
    for t, v in dy.items():
        # original variables + weight (mc)
      j = len(v.pdfWeight.keys())
      ov = v.df[['pTHm', 'pTHp', 'pHm', 'pHp']]
      ow = v.df.eval(f'angWeight*polWeight*{weight_mc}')
      ow *= v.pdfWeight[j]
      # target variables + weight (real data)
      tv = data[y][t].df[['pTHm', 'pTHp', 'pHm', 'pHp']]
      tw = data[y][t].df.eval(f'{weight_rd}')
      # Run multicore (about 15 minutes per iteration)
      job = multiprocessing.Process(
          target=kkp_weighting,
          args=(ov.values, ow.values, tv.values, tw.values, v.path_to_weights,
                y, MODE, t, len(threads), verbose)
      )
      threads.append(job)
      job.start()

  # Wait all processes to finish
  if verbose:
    print(f' * There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]


def do_angular_weights(tLL, tUL, verbose):
  """
  dddd
  """
  global mc

  for y, dy in mc.items():  #  loop over years
    for t, v in dy.items():  # loop over biased and unbiased triggers
      i = len(v.kkpWeight.keys()) + 1
      path_to_weights = v.path_to_weights.replace('.root', f'_{t}.npy')
      v.kkpWeight[i] = np.load(path_to_weights)
      os.remove(path_to_weights)
      get_angular_acceptance(v, tLL, tUL, kkpWeight=True)
    if verbose:
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['unbiased'].kkpWeight[i][evt]:>+.8f}", end='\n')
        print(f" | ", end='')


def do_mc_combination(verbose):
  """
  Combine 
  """
  global mc, data
  checker = []
  for y, dy in mc.items():  #  loop over years
    for trigger in ['biased', 'unbiased']:
      i = len(dy[trigger].angaccs)
      w_mc = dy[trigger].angaccs[i]
      data[y][trigger].angacc = Parameters.clone(w_mc)
      data[y][trigger].angaccs[i] = Parameters.clone(w_mc)
      qwe = check_for_convergence(data[y][trigger].angaccs[i - 1], data[y][trigger].angaccs[i])
      checker.append(qwe)

  check_dict = {}
  for ci in range(0, i):
    check_dict[ci] = []
    for y, dy in data.items():  #  loop over years
      for t in ['biased', 'unbiased']:
        qwe = check_for_convergence(dy[t].angaccs[ci], dy[t].angaccs[i])
        check_dict[ci].append(qwe)

  return checker, check_dict


def angular_acceptance_iterative_procedure(tLL, tUL, verbose=False, iteration=0):
  global pars

  itstr = f"[iteration #{iteration}]"

  # 1 fit RD sample obtaining pars
  print(f'{itstr} Simultaneous fit Bs2JpsiPhi {"&".join(list(mc.keys()))}')
  likelihood = do_fit(tLL, tUL, verbose=verbose)

  # 2 pdfWeight MC to RD using pars
  print(f'\n{itstr} PDF weighting MC samples to match Bs2JpsiPhi RD')
  t0 = timer()
  do_pdf_weighting(tLL, tUL, verbose=verbose)
  tf = timer() - t0
  print(f'PDF weighting took {tf:.3f} seconds.')

  # 3 kkpWeight MC to RD to match K+ and K- kinematic distributions
  print(f'\n{itstr} Kinematic reweighting MC samples in K momenta')
  t0 = timer()
  do_kkp_weighting(verbose)
  tf = timer() - t0
  print(f'Kinematic weighting took {tf:.3f} seconds.')

  # 4th step: angular weights
  print(f'\n{itstr} Extract angular normalisation weights')
  t0 = timer()
  do_angular_weights(tLL, tUL, verbose)
  tf = timer() - t0
  print(f'Extract angular normalisation weights took {tf:.3f} seconds.')

  # 5th step: merge MC std and dg0 results
  print(f'\n{itstr} Combining MC_BsJpsiPhi and MC_BsJpsiPhi_dG0')
  t0 = timer()
  checker, checker_dict = do_mc_combination(verbose)
  tf = timer() - t0
  print(f'Combining MC_BsJpsiPhi and MC_BsJpsiPhi_dG0 {tf:.3f} seconds.')

  return likelihood, checker, checker_dict


def lipschitz_iteration(tLL, tUL, max_iter=30, verbose=True):
  global pars
  likelihoods = []

  for i in range(1, max_iter):

    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, i)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # Check if they are the same as previous iteration
    if 1:
      lb = [data[y]['biased'].angaccs[i].__str__(['value']).splitlines() for i__, y in enumerate(YEARS)]
      lu = [data[y]['unbiased'].angaccs[i].__str__(['value']).splitlines() for i__, y in enumerate(YEARS)]
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
      pars.dump(args["output_physics_params"])
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items():  #  loop over years
        for trigger in ['biased', 'unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars.dump(data[y][trigger].params_path)
      break
  return all(checker), likelihoods


def aitken_iteration(tLL, tUL, max_iter=30, verbose=True):
  global pars
  likelihoods = []

  for i in range(1, max_iter):

    # x1 = angular_acceptance_iterative_procedure <- x0
    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, 2 * i - 1)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # x2 = angular_acceptance_iterative_procedure <- x1
    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, 2 * i)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # x2 <- aitken solution
    checker = []
    print(f"[aitken #{i}] Update solution")
    for y, dy in data.items():  #  loop over years
      for t, dt in dy.items():
        for p in dt.angacc.keys():
          x0 = dt.angaccs[2 * i - 2][p].uvalue
          x1 = dt.angaccs[2 * i - 1][p].uvalue
          x2 = dt.angaccs[2 * i][p].uvalue
          # aitken magic happens here
          den = x2 - 2 * x1 - x0
          if den < 1e-6:
            # checker.append(True)
            aitken = x2
          else:
            # checker.append(False)
            # aitken = x2 - ( (x2-x1)**2 ) / den # aitken
            # aitken = x0 - ( (x1-x0)**2 ) / den # steffensen
            aitken = x1 - ((x2 - x1)**2) / den  # romero

          # update angacc
          dt.angacc[p].set(value=aitken.n)
          dt.angacc[p].stdev = aitken.s
        # update dict of angaccs
        dt.angaccs[-1] = dt.angacc

        checker.append(check_for_convergence(dt.angaccs[2 * (i - 1)], dt.angaccs[2 * i]))
        #check_dict[ci].append( qwe )

    # Check if they are the same as previous iteration
    if 1:
      lb = [data[y]['biased'].angaccs[i].__str__(['value']).splitlines() for i__, y in enumerate(YEARS)]
      lu = [data[y]['unbiased'].angaccs[i].__str__(['value']).splitlines() for i__, y in enumerate(YEARS)]
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
      pars.dump(args["output_physics_params"])
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items():  #  loop over years
        for trigger in ['biased', 'unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars.dump(data[y][trigger].params_path)
      break
  return all(checker), likelihoods


# Run and get the job done {{{
if __name__ == '__main__':

  # Parse arguments {{{
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--weights-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--csp', help='Bs2JpsiPhi MC sample')
  p.add_argument('--resolution', help='Bs2JpsiPhi MC sample')
  p.add_argument('--flavor', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angacc-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angacc-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-mc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-physics-params', help='physics params')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--timeacc', help='Year of data-taking')
  p.add_argument('--mode', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  args = vars(p.parse_args())

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',')
  global MODE
  MODE = args['mode']
  ANGACC = parse_angacc(args['angacc'])
  tLL = 0.3
  tUL = 15.0

  if 'Bs2Jpsi' in MODE:
    def badjanak_gamma5(data, lkhd, pars, **kwargs):
      return badjanak.delta_gamma5_data(data, lkhd, **pars, **kwargs,
                                        use_timeacc=2,
                                        use_timeres=0, set_tagging=0)
  else:
    def badjanak_gamma5(data, lkhd, pars, **kwargs):
      return badjanak.delta_gamma5_data_Bd(data, lkhd, **pars, **kwargs,
                                           use_timeacc=0,
                                           set_tagging=1, use_timeres=0)

  # Get badjanak model and configure it ----------------------------------------
  #initialize(os.environ['IPANEMA_BACKEND'], 1 if YEARS in (2015,2017) else -1)

  # Prepare the cuts -----------------------------------------------------------
  CUT = f'gentime>={tLL} & gentime<={tUL}'

  # Samples as lists, mc and data
  samples_mc = args['sample_mc'].split(',')
  samples_data = args['sample_data'].split(',')

  # generatol level set of parameters for MC
  params_mc = args['params_mc'].split(',')

  # previously computed disciplines
  timeacc_biased = args['timeacc_biased'].split(',')
  timeacc_unbiased = args['timeacc_unbiased'].split(',')
  csp_factors = args['csp'].split(',')
  time_resolution = args['resolution'].split(',')
  flavor_tagging = args['flavor'].split(',')

  # corrected angular acceptance as seed for the procedure
  weight0_mc = args['weights_mc'].split(',')
  angacc0_biased = args['angacc_biased'].split(',')
  angacc0_unbiased = args['angacc_unbiased'].split(',')

  # paths to output angular acceptance after iterative converges
  angacc_biased = args['output_angacc_biased'].split(',')
  angacc_unbiased = args['output_angacc_unbiased'].split(',')
  weights_mc = args['output_weights_mc'].split(',')

  reweighter = reweight.GBReweighter(**bdtconfig)
  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC['acc']:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdtconfig.values()):50}\n")

  # }}}

  # Load samples {{{
  printsec('Loading samples')
  global mc, data, weight_rd, weight_mc
  # MC reconstructed and generator level variable names
  reco = ['cosK', 'cosL', 'hphi', 'gentime']
  true = ['gencosK', 'gencosL', 'genhphi', 'gentime']
  real = ['cosK', 'cosL', 'hphi', 'gentime']
  if 'Bs2Jpsi' in MODE:
    reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
    true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
    real += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*mHH', '0*mHH']
  elif 'Bd2JpsiKstar' in MODE:
    reco += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
    true += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
    real += ['mHH', '0*sigmat', 'idB', 'idB', '0*mHH', '0*mHH']

  # sWeight variable
  weight_mc = "sWeight"
  weight_rd = "sWeight"
  # if "Bs2Jpsi" in MODE:
  # weight_mc += '/gb_weights'
  # if 'evt' in args['version']:
  # weight_rd = f'{weight_rd}/gb_weights'
  # if "bkgcat60" in args['version']:
  #   weight_mc = 'time/time'
  #   weight_rd = 'time/time'
  if ANGACC['use_oddWeight']:
    weight_rd = f'oddWeight*{weight_rd}'
  if ANGACC['use_pTWeight']:
    weight_rd = f'pTWeight*{weight_rd}'
  HAS_SWAVE = False
  if "Swave" in MODE:
    HAS_SWAVE = True

  print(f"Using weight = {weight_mc} for MC")
  print(f"Using weight = {weight_rd} for data")

  # Load Monte Carlo samples {{{
  printsubsec('Loading MC samples')
  mc = {}

  for i, y in enumerate(YEARS):
    mc[y] = {}
    for t in ['biased', 'unbiased']:
      # load sample in two categories
      mc[y][t] = Sample.from_root(samples_mc[i], share=SHARE)
      mc[y][t].name = f"{MODE}-{y}-{t}"

      # associate generator level params
      mc[y][t].assoc_params(params_mc[i])

      # add angWeight (from corrected procedure) to df
      angWeight = uproot.open(weight0_mc[i])['DecayTree'].array('angWeight')
      mc[y][t].df['angWeight'] = angWeight
      mc[y][t].olen = len(angWeight)
      mc[y][t].chop(trigger_scissors(t, CUT))

      # allocate variables in device
      mc[y][t].allocate(reco=reco, true=true, pdf='0*time')

      # add dicts for angaccs, kpp and pdf weights (for iterations)
      mc[y][t].angaccs = {}
      mc[y][t].kkpWeight = {}
      mc[y][t].pdfWeight = {}
      print(mc[y][t])

      # add path to store kkp and pdf weights when converged
      mc[y][t].path_to_weights = weights_mc[i]
  # }}}

  # Load corresponding data sample {{{
  data = {}

  printsubsec('Loading RD samples')
  for i, y in enumerate(YEARS):
    data[y] = {}
    # shared paramters between trigger categories
    csp = Parameters.load(csp_factors[i])
    resolution = Parameters.load(time_resolution[i])
    flavor = Parameters.load(flavor_tagging[i])
    mass = np.array(csp.build(csp, csp.find('mKK.*'))).tolist()
    badjanak.config['mhh'] = mass

    # loop in triggers
    for t, T in zip(['biased', 'unbiased'], [0, 1]):
      data[y][t] = Sample.from_root(samples_data[i], share=SHARE)
      data[y][t].name = f"{MODE}-{y}-{t}"
      data[y][t].csp = csp.build(csp, csp.find('CSP.*'))
      data[y][t].flavor = flavor
      data[y][t].resolution = resolution

    # add time acceptance
    for t, ta in zip(['biased', 'unbiased'], [timeacc_biased, timeacc_unbiased]):
      c = Parameters.load(ta[i])
      data[y][t].knots = Parameters.build(c, c.fetch('k.*'))
      badjanak.config['knots'] = np.array(data[y][t].knots).tolist()
      data[y][t].timeacc = Parameters.build(c, c.fetch('(a|b|c).*'))
      data[y][t].chop(trigger_scissors(t, CUT))
      if ANGACC['use_pTWeight']:
        pTp = np.array(data[y][t].df['pTHp'])
        pTm = np.array(data[y][t].df['pTHm'])
        pT_acc = np.ones_like(data[y][t].df['pTHp'])
        for k in range(len(pT_acc)):
          pT_acc[k] = acceptance_effect(pTp[k], 200**3)
          pT_acc[k] *= acceptance_effect(pTm[k], 200**3)
        data[y][t].df['pTWeight'] = pT_acc
        print(weight_rd)
      print(data[y][t])

    # add angular acceptance
    for t, aa in zip(['biased', 'unbiased'], [angacc0_biased, angacc0_unbiased]):
      w = Parameters.load(aa[i])
      data[y][t].angacc = Parameters.build(w, w.fetch('w.*'))
      data[y][t].angaccs = {0: data[y][t].angacc}

    # add angular acceptance result path
    for t, aa in zip(['biased', 'unbiased'], [angacc_biased, angacc_unbiased]):
      data[y][t].params_path = aa[i]

    # finally curate sWeights properly
    for d in [data[y]['biased'], data[y]['unbiased']]:
      sw = np.zeros_like(d.df.eval(f'{weight_rd}'))
      for l, h in zip(mass[:-1], mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        __sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        sw = np.where(pos, __sw * (sum(__sw) / sum(__sw * __sw)), sw)
      d.df['sWeight'] = sw
      d.allocate(data=real, weight='sWeight', lkhd='0*time')

    # }}}

  # }}}

  # Prepare dict of parameters {{{
  printsec('Parameters and initial status')

  print(f"\nFitting parameters\n{80*'='}")
  global pars
  pars = Parameters()

  if 'Bs2Jpsi' in MODE:
    # S wave fractions
    if HAS_SWAVE:
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
    pars.add(dict(name="pSlon", value=0.00, min=-1.0, max=1.0,
                  free=False, latex=r"\phi_S - \phi_0"))
    pars.add(dict(name="pPlon", value=0.05, min=-1.0, max=1.0,
                  free=True, latex=r"\phi_0"))
    pars.add(dict(name="pPpar", value=0.00, min=-1.0, max=1.0,
                  free=False, latex=r"\phi_{\parallel} - \phi_0"))
    pars.add(dict(name="pPper", value=0.00, min=-1.0, max=1.0,
                  free=False, latex=r"\phi_{\perp} - \phi_0"))

    # S wave strong phases
    if HAS_SWAVE:
      pars.add(dict(name='dSlon1', value=+2.34, min=-0.0, max=+4.0,
                    free=True, latex=r"\delta_S^{1} - \delta_{\perp}"))
      pars.add(dict(name='dSlon2', value=+1.64, min=-0.0, max=+4.0,
                    free=True, latex=r"\delta_S^{2} - \delta_{\perp}"))
      pars.add(dict(name='dSlon3', value=+1.09, min=-0.0, max=+4.0,
                    free=True, latex=r"\delta_S^{3} - \delta_{\perp}"))
      pars.add(dict(name='dSlon4', value=-0.25, min=-4.0, max=+0.0,
                    free=True, latex=r"\delta_S^{4} - \delta_{\perp}"))
      pars.add(dict(name='dSlon5', value=-0.48, min=-4.0, max=+0.0,
                    free=True, latex=r"\delta_S^{5} - \delta_{\perp}"))
      pars.add(dict(name='dSlon6', value=-1.18, min=-4.0, max=+0.0,
                    free=True, latex=r"\delta_S^{6} - \delta_{\perp}"))

    # P wave strong phases
    pars.add(dict(name="dPlon", value=0.000, min=-2 * 3.14, max=2 * 3.14,
                  free=False, latex=r"\delta_0"))
    pars.add(dict(name="dPpar", value=3.260, min=-2 * 3.14, max=2 * 3.14,
                  free=True, latex=r"\delta_{\parallel} - \delta_0"))
    pars.add(dict(name="dPper", value=3.026, min=-2 * 3.14, max=2 * 3.14,
                  free=True, latex=r"\delta_{\perp} - \delta_0"))

    # lambdas
    pars.add(dict(name="lSlon", value=1.0, min=0.7, max=1.6,
                  free=False, latex="\lambda_S/\lambda_0"))
    pars.add(dict(name="lPlon", value=1.0, min=0.7, max=1.6,
                  free=True, latex="\lambda_0"))
    pars.add(dict(name="lPpar", value=1.0, min=0.7, max=1.6,
                  free=False, latex="\lambda_{\parallel}/\lambda_0"))
    pars.add(dict(name="lPper", value=1.0, min=0.7, max=1.6,
                  free=False, latex="\lambda_{\perp}/\lambda_0"))

    # life parameters
    pars.add(dict(name="Gd", value=0.65789, min=0.0, max=1.0,
                  free=False, latex=r"\Gamma_d"))
    pars.add(dict(name="DGs", value=0.0917, min=-0.15, max=0.15,  #  WARNING THIS CAN BE FIXED TO ZERO
                  free=True, latex=r"\Delta\Gamma_s"))
    pars.add(dict(name="DGsd", value=0.03, min=-0.15, max=0.15,
                  free=True, latex=r"\Gamma_s - \Gamma_d"))
    pars.add(dict(name="DM", value=17.768, min=16.0, max=19.0,
                  free=True, latex=r"\Delta m"))

  else:
    pars.add(dict(name="fPlon", value=0.5001, min=0.1, max=0.9,
                  free=True, latex=r'f_0'))
    pars.add(dict(name="fPper", value=0.1601, min=0.1, max=0.9,
                  free=True, latex=r'f_{\perp}'))
    # P wave strong phases
    pars.add(dict(name="dPlon", value=0.000, min=-3.14, max=3.14,
                  free=False, latex=r"\delta_0"))
    pars.add(dict(name="dPpar", value=2.501, min=-2 * 3.14, max=2 * 3.14,
                  free=True, latex=r"\delta_{\parallel} - \delta_0"))
    pars.add(dict(name="dPper", value=-0.17, min=-2 * 3.14, max=2 * 3.14,
                  free=True, latex=r"\delta_{\perp} - \delta_0"))
    if ANGACC['use_pTWeight'] or ANGACC['use_oddWeight']:
      pars.add(dict(name="Gd", value=0.65833, min=0.0, max=1.0,
                    free=False, latex=r"\Gamma_d"))
    else:
      pars.add(dict(name="Gd", value=0.65833, min=0.0, max=1.0,
                    free=True, latex=r"\Gamma_d"))  # Warining a true
  print(pars)

  # }}}

  # print all input information {{{

  # print time acceptances
  lb = [data[y]['biased'].timeacc.__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  lu = [data[y]['unbiased'].timeacc.__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nBiased time acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  print(f"\nUnbiased time acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")

  # print csp factors
  lb = [data[y]['biased'].csp.__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nCSP factors\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print flavor tagging parameters
  lb = [data[y]['biased'].flavor.__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nFlavor tagging parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print time resolution
  lb = [data[y]['biased'].resolution.__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nResolution parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print angular acceptance
  lb = [data[y]['biased'].angaccs[0].__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  lu = [data[y]['unbiased'].angaccs[0].__str__(['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nBiased angular acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")
  print(f"\nUnbiased angular acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")
  print(f"\n")

  # }}}

  # The iterative procedure starts {{{

  # update kernels with the given modifications
  badjanak.get_kernels()

  # run the procedure!

  ok, likelihoods = lipschitz_iteration(max_iter=10, verbose=True, tLL=tLL, tUL=tUL)

  if not ok:
    ok, likelihoods = aitken_iteration(max_iter=30, verbose=True, tLL=tLL, tUL=tUL)

  if not ok:
    print('WARNING: Convergence was not achieved!')

  # plot likelihood evolution
  ld_x = [i + 1 for i, j in enumerate(likelihoods)]
  ld_y = [j + 0 for i, j in enumerate(likelihoods)]
  import termplotlib
  ld_p = termplotlib.figure()
  ld_p.plot(ld_x, ld_y, xlabel='iteration', label='likelihood', xlim=(0, 30))
  ld_p.show()

  # }}}

  # Storing some weights in disk {{{
  #     For future use of computed weights created in this loop, these should be
  #     saved to the path where samples are stored.
  #     GBweighting is slow enough once!
  print('Storing weights in root file')
  for y, v in mc.items():  #  loop over years
    pool = {}
    for i in v['biased'].pdfWeight.keys():  # loop over iterations
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
      pool.update({f'kkpWeight{i}': wb + wu})
    wb = np.zeros((v['biased'].olen))
    wu = np.zeros((v['unbiased'].olen))
    wb[list(v['biased'].df.index)] = v['biased'].df['angWeight'].values
    wu[list(v['unbiased'].df.index)] = v['unbiased'].df['angWeight'].values
    pool.update({f'angWeight': wb + wu})
    with uproot.recreate(v['biased'].path_to_weights) as f:
      f['DecayTree'] = uproot.newtree({var: np.float64 for var in pool.keys()})
      f['DecayTree'].extend(pool)
  print(f' * Succesfully writen')

  # }}}

# }}}


# vim:foldmethod=marker
# that's all folks!
