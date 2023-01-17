from hep_ml import reweight
import config
from analysis.angular_acceptance.bdtconf_tester import bdtmesh
from utils.helpers import version_guesser, parse_angacc, timeacc_guesser, trigger_scissors
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
    Angular acceptance iterative procedure
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

# reweighting config
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
# 40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000


def check_for_convergence(a, b):
  a_f = np.array([float(a[p].unc_round[0]) for p in a])
  b_f = np.array([float(b[p].unc_round[0]) for p in b])
  checker = np.abs(a_f - b_f).sum()
  if checker == 0:
    return True
  return False


# core functions
#     They work for a given category only.

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
    probs = []
    for i in range(0, 4):
      _prob = ks_2samp_weighted(original_v[:, i], target_v[:, i],
                                weights1=original_w * kkpWeight,
                                weights2=target_w)
      probs.append(_prob)
    print(f" * GB-weighting {mode}-{year}-{trigger} sample is done", "\n",
          f"  KS test : {probs}")


def get_angular_acceptance(mc: dict, tLL: float, tUL: float, kkpWeight=False):
  """
  Compute angular acceptance
  """
  # cook weight for angular acceptance
  weight = mc.df.eval(f'angWeight*polWeight*sWeight').values
  i = len(mc.kkpWeight.keys())

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[i])
  weight = ristra.allocate(weight)

  # compute angular acceptance
  ans = badjanak.get_angular_acceptance_weights(mc.true, mc.reco, weight,
                                                **mc.params.valuesdict(),
                                                tLL=tLL, tUL=tUL)

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

  for y, dy in data.items():
    # for dt in dy.values():
    for trig in ['biased', 'unbiased']:
      dt = dy[trig]
      badjanak.delta_gamma5_data(dt.data, dt.lkhd, **pars_dict,
                                 **dt.timeacc.valuesdict(),
                                 **dt.angacc.valuesdict(),
                                 **dt.resolution.valuesdict(),
                                 **dt.csp.valuesdict(),
                                 **dt.flavor.valuesdict(),
                                 tLL=tLL, tUL=tUL, use_timeacc=1,
                                 use_timeoffset=1, BLOCK_SIZE=128)
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
    out.add({'name': f'{wk}{label}', 'value': w[k], 'stdev': uw[k],
             'free': False, 'latex': f'{std[wk].latex}^{label}'})
  for k, wk in enumerate(std.keys()):
    correl = {f'w{j}{label}': corr[k][j] for j in range(0, len(w)) if k > 0 and j > 0}
    out[f'{wk}{label}'].correl = correl

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
                    fcn_kwgs=dict(data=data, tLL=tLL, tUL=tUL), verbose=True,
                    timeit=True, tol=0.05, strategy=2)
  # print(result.params)
  # print fit results
  # print(result) # parameters are not blinded, so we dont print the result
  if not '2018' in data.keys() and not '2017' in data.keys():
    for p in ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd',
              'DGs', 'DM',
              'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6',
              'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6']:
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
  pars = Parameters.clone(result.params)
  # pars['fPlon'].set(value=0.5123409167305097)
  # pars['fPper'].set(value=0.2478082459746342)
  # pars['dPpar'].set(value=3.0872017467210693)
  # pars['dPper'].set(value=2.6421773677470259)
  # pars['pPlon'].set(value=-0.1076586094443792)
  # pars['lPlon'].set(value=1.0164993275965450)
  # pars['DGsd'].set(value=-0.0045047145137569)
  # pars['DGs'].set(value=0.0798291503336716)
  # pars['DM'].set(value=17.7112349187736484)
  # pars['dSlon1'].set(value=2.3294147283804678)
  # pars['dSlon2'].set(value=1.6175131811176477)
  # pars['dSlon3'].set(value=1.0385341726506248)
  # pars['dSlon4'].set(value=-0.2310376097304339)
  # pars['dSlon5'].set(value=-0.4871378739706218)
  # pars['dSlon6'].set(value=-1.1430865193421349)
  # pars['fSlon1'].set(value=0.4739837863239510)
  # pars['fSlon2'].set(value=0.0370048508245155)
  # pars['fSlon3'].set(value=0.0047169340552120)
  # pars['fSlon4'].set(value=0.0091883450310787)
  # pars['fSlon5'].set(value=0.0682344201683382)
  # pars['fSlon6'].set(value=0.1431562192790002)

  return result.chi2


# pdf weighting {{{

def do_pdf_weighting(tLL: float, tUL: float, verbose: bool):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0)

  Parameters
  ----------
  tLL: float
  Time lower limit
  tLL: float
  Time lower limit
  """
  global pars, data, mc

  for y, dy in mc.items():  #  loop over years
    for m, dm in dy.items():  # loop over mc_std and mc_dg0
      for t, v in dm.items():  # loop over triggers
        if verbose:
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
        j = len(v.pdfWeight.keys()) + 1
        v.pdfWeight[j] = compute_pdf_ratio_weight(v, v.params,
                                                  pars + data[y][t].csp,
                                                  tLL=tLL, tUL=tUL)
  if verbose:
    for y, dy in mc.items():  #  loop over years
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_BsJpsiPhi']['biased'].pdfWeight[j][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi']['unbiased'].pdfWeight[j][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['biased'].pdfWeight[j][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['unbiased'].pdfWeight[j][evt]:>+.8f}")

# }}}


def do_kkp_weighting(verbose, kinematic_bmeson=False):
  # 3rd step: kinematic weights ----------------------------------------------
  #    We need to change badjanak to handle MC samples and then we compute the
  #    desired pdf weights for a given set of fitted pars in step 1. This
  #    implies looping over years and MC samples (std and dg0).
  #    As a matter of fact, it's important to have data[y][combined] sample,
  #    the GBweighter gives different results when having those 0s or having
  #    nothing after cutting the sample.
  global mc, data, weight_rd, weight_mc
  weighting_branches = ['pTHm', 'pTHp', 'pHm', 'pHp']
  if kinematic_bmeson:
    weighting_branches += ['pTB', 'pB', 'mHH']
    print("Adding kinematic B branches to reweighter.")

  threads = list()
  for y, dy in mc.items():  #  loop over years
    for m, dm in dy.items():  # loop over mc_std and mc_dg0
      for t, v in dm.items():  # loop in triggers
        # original variables + weight (mc)
        j = len(v.pdfWeight.keys())
        ov = v.df[weighting_branches]
        ow = v.df.eval(f'angWeight*polWeight*{weight_mc}')
        # WARNING: old school procedure:
        # ov  = v.df[['pTHm','pTHp','pHm','pHp']]
        # ow  = v.df.eval(f'angWeight*polWeight*{weight_mc}')
        ow *= v.pdfWeight[j]
        # target variables + weight (real data)
        tv = data[y][t].df[weighting_branches]
        tw = data[y][t].df.eval(weight_rd)
        # Run multicore (about 15 minutes per iteration)
        job = multiprocessing.Process(
            target=kkp_weighting,
            args=(ov.values, ow.values, tv.values, tw.values, v.path_to_weights,
                  y, m, t, len(threads), verbose)
        )
        threads.append(job)
        job.start()

  # Wait all processes to finish
  if verbose:
    print(f' * There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]


def do_kkp_weighting_old(verbose):
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
    for m, dm in dy.items():  # loop over mc_std and mc_dg0
      for t, v in dm.items():
        # original variables + weight (mc)
        j = len(v.pdfWeight.keys())
        ov = v.df[['pTHm', 'pTHp', 'pHm', 'pHp']]
        ow = v.df.eval(f'angWeight*polWeight*{weight_mc}')
        ow *= v.pdfWeight[j]
        # target variables + weight (real data)
        tv = data[y]['combined'].df[['pTHm', 'pTHp', 'pHm', 'pHp']]
        tw = data[y]['combined'].df.eval(weight_rd) * data[y]['combined'].df[f'{t}Weight']
        # Run multicore (about 15 minutes per iteration)
        job = multiprocessing.Process(
            target=kkp_weighting,
            args=(ov.values, ow.values, tv.values, tw.values, v.path_to_weights,
                  y, m, t, len(threads), verbose)
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
    for m, dm in dy.items():  # loop over mc_std and mc_dg0
      for t, v in dm.items():  # loop over biased and unbiased triggers
        i = len(v.kkpWeight.keys()) + 1
        path_to_weights = v.path_to_weights.replace('.root', f'_{t}.npy')
        v.kkpWeight[i] = np.load(path_to_weights)
        os.remove(path_to_weights)
        get_angular_acceptance(v, tLL, tUL, kkpWeight=True)
    if verbose:
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_BsJpsiPhi']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_BsJpsiPhi_dG0']['unbiased'].kkpWeight[i][evt]:>+.8f}")


def do_mc_combination(verbose=False, triggers=['biased', 'unbiased']):
  """
  Combine
  """
  global mc, data
  checker = []
  for y, dy in mc.items():  #  loop over years
    for trigger in triggers:
      i = len(dy['MC_BsJpsiPhi'][trigger].angaccs)
      std = dy['MC_BsJpsiPhi'][trigger].angaccs[i]
      dg0 = dy['MC_BsJpsiPhi_dG0'][trigger].angaccs[i]
      merged_w = merge_std_dg0(std, dg0, verbose=verbose, label=trigger[0])
      data[y][trigger].angacc = merged_w
      data[y][trigger].angaccs[i] = merged_w
      qwe = check_for_convergence(data[y][trigger].angaccs[i - 1],
                                  data[y][trigger].angaccs[i])
      checker.append(qwe)

  check_dict = {}
  for ci in range(0, i):
    check_dict[ci] = []
    for y, dy in data.items():  #  loop over years
      for t in triggers:
        qwe = check_for_convergence(dy[t].angaccs[ci], dy[t].angaccs[i])
        check_dict[ci].append(qwe)

  return checker, check_dict


def angular_acceptance_iterative_procedure(tLL, tUL, verbose=False, iteration=0, kinematic_bmeson=False):
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
  do_kkp_weighting(verbose, kinematic_bmeson=kinematic_bmeson)
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


def lipschitz_iteration(tLL, tUL, max_iter=30, verbose=True, kinematic_bmeson=False):
  global pars
  likelihoods = []

  for i in range(1, max_iter):

    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, i, kinematic_bmeson=kinematic_bmeson)
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
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items():  #  loop over years
        for trigger in ['biased', 'unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars.dump(data[y][trigger].params_path)
      break
  return all(checker), likelihoods


def aitken_iteration(tLL, tUL, max_iter=30, verbose=True, kinematic_bmeson=False):
  global pars
  likelihoods = []

  for i in range(1, max_iter):

    # x1 = angular_acceptance_iterative_procedure <- x0
    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, 2 * i - 1, kinematic_bmeson=kinematic_bmeson)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # x2 = angular_acceptance_iterative_procedure <- x1
    ans = angular_acceptance_iterative_procedure(tLL, tUL, verbose, 2 * i, kinematic_bmeson=kinematic_bmeson)
    likelihood, checker, checker_dict = ans
    likelihoods.append(likelihood)

    # x2 <- aitken solution
    checker = []
    print(f"[aitken #{i}] Update solution")
    for y, dy in data.items():  #  loop over years
      for t in ['biased', 'unbiased']:
        dt = dy[t]
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
            # aitken = x2 - ((x2 - x1)**2) / den  # aitken
            # aitken = x0 - ((x1 - x0)**2) / den  # steffensen
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
    print("checker_dict: ")
    for ck, cv in checker_dict.items():
      print(ck, cv)
    print("LIKELIHOODs: ", likelihoods)

    if all(checker) or i > 25:
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items():  #  loop over years
        for trigger in ['biased', 'unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in json')
          pars.dump(data[y][trigger].params_path)
      break
  return all(checker), likelihoods


################################################################################
# Run and get the job done #####################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--sample-data', help='Bs2JpsiPhi MC sample')
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
  p.add_argument('--output-angular-weights-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angular-weights-mc-dg0', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--timeacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  args = vars(p.parse_args())

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',')
  MODE = 'Bs2JpsiPhi'
  ANGACC = parse_angacc(args['angacc'])
  print(args['timeacc'])
  TIMEACC = timeacc_guesser(args['timeacc'])
  TIMEACC['use_upTime'] = TIMEACC['use_upTime'] | ('UT' in args['version'])
  TIMEACC['use_lowTime'] = TIMEACC['use_lowTime'] | ('LT' in args['version'])
  if "kinB" in args['angacc']:
    kinematic_bmeson = True
  else:
    kinematic_bmeson = False

  # Prepare the cuts
  if TIMEACC['use_transverse_time']:
    time = 'timeT'
  else:
    time = 'time'
  if TIMEACC['use_truetime']:
    time = f'gen{time}'

  if TIMEACC['use_upTime']:
    tLL = config.general['upper_time_lower_limit']
  else:
    tLL = config.general['time_lower_limit']
  if TIMEACC['use_lowTime']:
    tUL = config.general['lower_time_upper_limit']
  else:
    tUL = config.general['time_upper_limit']

  if 'T1' in args['version']:
    tLL, tUL = tLL, 0.92  # 47
    badjanak.config['final_extrap'] = False
  elif 'T2' in args['version']:
    tLL, tUL = 0.9247, 1.9725
    badjanak.config['final_extrap'] = False
  elif 'T3' in args['version']:
    tLL, tUL = 1.9725, tUL
    # tLL, tUL = 2, tUL
  else:
    print("SAFE CUT")

  print(TIMEACC['use_lowTime'], TIMEACC['use_upTime'])

  # Get badjanak model and configure it ----------------------------------------
  #initialize(os.environ['IPANEMA_BACKEND'], 1 if YEARS in (2015,2017) else -1)

  # Prepare the cuts -----------------------------------------------------------
  CUT = f'{time}>={tLL} & {time}<={tUL}'
  print(CUT)
  # List samples, params and tables --------------------------------------------
  samples_std = args['sample_mc_std'].split(',')
  samples_dg0 = args['sample_mc_dg0'].split(',')
  samples_data = args['sample_data'].split(',')

  input_std_params = args['params_mc_std'].split(',')
  input_dg0_params = args['params_mc_dg0'].split(',')

  if VERSION == 'v0r0':
    for i in range(len(input_std_params)):
      input_std_params[i] = input_std_params[i].replace('generator', 'generator_old')
      input_dg0_params[i] = input_dg0_params[i].replace('generator', 'generator_old')
  print(input_std_params, input_dg0_params)
  angWeight_std = args['angular_weights_mc_std'].split(',')
  angWeight_dg0 = args['angular_weights_mc_dg0'].split(',')

  w_biased = args['input_weights_biased'].split(',')
  w_unbiased = args['input_weights_unbiased'].split(',')

  coeffs_biased = args['input_coeffs_biased'].split(',')
  coeffs_unbiased = args['input_coeffs_unbiased'].split(',')

  csp_factors = args['input_csp'].split(',')
  time_resolution = args['input_time_resolution'].split(',')
  flavor_tagging = args['input_flavor_tagging'].split(',')

  params_biased = args['output_weights_biased'].split(',')
  params_unbiased = args['output_weights_unbiased'].split(',')

  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')
  kkpWeight_dg0 = args['output_angular_weights_mc_dg0'].split(',')

  # if version has bdt in name, then lets change it
  if 'bdt' in VERSION:
    bdtconfig = int(VERSION.split('bdt')[1])
    bdtconfig = bdtmesh(bdtconfig, config.general['bdt_tests'], False)
  reweighter = reweight.GBReweighter(**bdtconfig)

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC['acc']:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdtconfig.values()):50}\n")

  # Load samples ---------------------------------------------------------------
  printsec('Loading samples')

  global mc, data, weight_rd, weight_mc

  # only load the branches that are actually used
  mc_branches = ['cosK', 'cosL', 'hphi', time, 'mHH', 'sigmat', 'idB', 'etaB',
                 'gencosK', 'gencosL', 'genhphi', f'gen{time}', 'genidB', 'pTLm', 'pTLp', 'pLm', 'pLp',
                 'pHm', 'pHp', 'pTHp', 'pTHm', 'sWeight', 'hlt1b', 'polWeight', 'pB', 'pTB']
  rd_branches = ['cosK', 'cosL', 'hphi', time, 'mHH', 'sigmat', 'idB',
                 'tagOSdec', 'tagSSdec', 'tagOSeta', 'tagSSeta', 'pTLm', 'pTLp', 'pLm', 'pLp', 'etaB',
                 'pHm', 'pHp', 'pTHp', 'pTHm', 'sWeight', 'hlt1b', 'pB', 'pTB']

  # MC reconstructed and generator level variable names
  reco = ['cosK', 'cosL', 'hphi', time]
  true = [f'gen{i}' for i in reco]
  reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
  true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']

  # RD variable names
  real = ['cosK', 'cosL', 'hphi', time]
  real += ['mHH', 'sigmat', 'tagOSdec', 'tagSSdec', 'tagOSeta', 'tagSSeta']

  # sWeight variable
  weight_rd = 'sWeight'
  weight_mc = 'sWeight'
  if TIMEACC['use_veloWeight']:
    mc_branches += ['veloWeight']
    rd_branches += ['veloWeight']
    weight_rd = f'veloWeight*sWeight'
    weight_mc = f'veloWeight*sWeight'

  # Load Monte Carlo samples
  mc = {}
  mcmodes = ['MC_BsJpsiPhi', 'MC_BsJpsiPhi_dG0']

  printsubsec('Loading MC samples')
  for i, y in enumerate(YEARS):
    mc[y] = {}
    for m, v in zip(mcmodes, [samples_std, samples_dg0]):
      mc[y][m] = {'biased': Sample.from_root(v[i], branches=mc_branches, share=SHARE),
                  'unbiased': Sample.from_root(v[i], branches=mc_branches, share=SHARE)}
      # 'combined': Sample.from_root(v[i], branches=mc_branches, share=SHARE)}
      mc[y][m]['biased'].name = f"{m}-{y}-biased"
      mc[y][m]['unbiased'].name = f"{m}-{y}-unbiased"
      # mc[y][m]['combined'].name = f"{m}-{y}-unbiased"

    for m, v in zip(mcmodes, [input_std_params, input_dg0_params]):
      mc[y][m]['biased'].assoc_params(v[i])
      mc[y][m]['unbiased'].assoc_params(v[i])
      # mc[y][m]['combined'].assoc_params(v[i])

    for m, v in zip(mcmodes, [angWeight_std, angWeight_dg0]):
      angWeight = uproot.open(v[i])['DecayTree'].array('angWeight')
      if kinematic_bmeson:
        angWeight = np.ones_like(angWeight)
      # mc[y][m]['unbiased'].olen = len(angWeight)
      for t in ['biased', 'unbiased']:
        mc[y][m]['biased'].olen = len(angWeight)
        if 'v0r0' in args['version']:
          mc[y][m][t].olen = len(angWeight)
          # mc[y][m][t].df['angWeight'] = angWeight
          # mc[y][m][t].df['angWeight'] = angWeight
          trigWeight = mc[y][m][t].df.eval(trigger_scissors(t))
          # mc[y][m]['unbiased'].eval(trigger_scissors('unbiased', CUT))
          # mc[y][m]['biased'].chop(CUT)
          mc[y][m][t].df['angWeight'] = angWeight * trigWeight
          mc[y][m][t].chop(CUT)
        else:
          mc[y][m][t].df['angWeight'] = angWeight
          mc[y][m][t].olen = len(angWeight)
          mc[y][m][t].chop(trigger_scissors(t, CUT))
          # mc[y][m]['unbiased'].df['angWeight'] = angWeight
          # mc[y][m][t].chop(trigger_scissors(t, CUT))
        # mc[y][m]['combined'].df['angWeight'] = angWeight

      # for t in ['biased', 'unbiased', 'combined']:
      for t in ['biased', 'unbiased']:
        mc[y][m][t].allocate(reco=reco, true=true, pdf='0*time')
        mc[y][m][t].angaccs = {}
        mc[y][m][t].kkpWeight = {}
        mc[y][m][t].pdfWeight = {}

    for m, v in zip(mcmodes, [kkpWeight_std, kkpWeight_dg0]):
      mc[y][m]['biased'].path_to_weights = v[i]
      mc[y][m]['unbiased'].path_to_weights = v[i]

  # Load corresponding data sample
  data = {}

  printsubsec('Loading RD samples')
  for i, y in enumerate(YEARS):
    data[y] = {}
    csp = Parameters.load(csp_factors[i])
    resolution = Parameters.load(time_resolution[i])
    flavor = Parameters.load(flavor_tagging[i])
    mass = np.array(csp.build(csp, csp.find('mKK.*')))
    badjanak.config['mHH'] = mass.tolist()

    for t in ['biased', 'unbiased', 'combined']:
      data[y][t] = Sample.from_root(samples_data[i],
                                    branches=rd_branches, share=SHARE)
      data[y][t].name = f"{m}-{y}-{t}"
      data[y][t].csp = csp.build(csp, csp.find('CSP.*'))
      data[y][t].flavor = flavor
      data[y][t].resolution = resolution

    for t, coeffs in zip(['biased', 'unbiased'], [coeffs_biased, coeffs_unbiased]):
      c = Parameters.load(coeffs[i])
      data[y][t].knots = Parameters.build(c, c.fetch('k.*'))
      print(data[y][t].knots)
      badjanak.config['knots'] = np.array(data[y][t].knots).tolist()
      data[y][t].timeacc = Parameters.build(c, c.fetch('(a|b|c).*'))
      data[y][t].chop(trigger_scissors(t, CUT))
      trigWeight = data[y]['combined'].df.eval(trigger_scissors(t))
      data[y]['combined'].df[f'{t}Weight'] = trigWeight
      # data[y]['combined'].chop(CUT)
      print(data[y][t])

    for t, weights in zip(['biased', 'unbiased'], [w_biased, w_unbiased]):
      w = Parameters.load(weights[i])
      data[y][t].angacc = Parameters.build(w, w.fetch('w.*'))
      data[y][t].angaccs = {0: Parameters.build(w, w.fetch('w.*'))}

    for t, path in zip(['biased', 'unbiased'], [params_biased, params_unbiased]):
      data[y][t].params_path = path[i]

    for d in [data[y]['biased'], data[y]['unbiased']]:
      sw = np.zeros_like(d.df.eval(weight_rd))
      for l, h in zip(mass[:-1], mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        sw = np.where(pos, this_sw * (sum(this_sw) / sum(this_sw * this_sw)), sw)
      d.df['sWeight'] = sw
      d.allocate(data=real, weight='sWeight', lkhd='0*time')

  # Prepare dict of parameters -------------------------------------------------
  printsec('Parameters and initial status')

  print(f"\nFitting parameters\n{80*'='}")
  global pars
  SWAVE = True
  mass_knots = badjanak.config['mHH']
  pars = Parameters()

  # S wave fractions
  for i in range(len(mass_knots) - 1):
    pars.add(dict(
        name=f'fSlon{i+1}', value=SWAVE * 0.0, min=0.00, max=0.90,
        free=SWAVE, latex=rf'|A_S^{{{i+1}}}|^2'))

  # P wave fractions
  pars.add(dict(name="fPlon", value=0.5240, min=0.4, max=0.6,
                free=True, latex=r'f_0'))
  pars.add(dict(name="fPper", value=0.2500, min=0.1, max=0.3,
                free=True, latex=r'f_{\perp}'))

  # Weak phases
  pars.add(dict(name="pSlon", value=0.00, min=-1.0, max=1.0,
                free=False, latex=r"\phi_S - \phi_0"))
  pars.add(dict(name="pPlon", value=0.07, min=-1.0, max=1.0,
                free=True, latex=r"\phi_0"))
  pars.add(dict(name="pPpar", value=0.00, min=-1.0, max=1.0,
                free=False, latex=r"\phi_{\parallel} - \phi_0"))
  pars.add(dict(name="pPper", value=0.00, min=-1.0, max=1.0,
                free=False, latex=r"\phi_{\perp} - \phi_0"))

  # S wave strong phases
  for i in range(len(mass_knots) - 1):
    phase = np.linspace(2.3, -1.2, len(mass_knots) - 1)[i]
    pars.add(dict(
        name=f'dSlon{i+1}', value=SWAVE * phase,
        min=0 if 2 * i < (len(mass_knots) - 1) else -4,
        max=4 if 2 * i < (len(mass_knots) - 1) else 0,
        free=SWAVE,
        latex=rf"\delta_S^{{{i+1}}} - \delta_{{\perp}} \, \mathrm{{[rad]}}"))

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
  pars.add(dict(name="DGs", value=0.0917, min=-0.15, max=0.15,
                free=True, latex=r"\Delta\Gamma_s"))
  pars.add(dict(name="DGsd", value=0.03, min=-0.2, max=0.2,
                free=True, latex=r"\Gamma_s - \Gamma_d"))
  pars.add(dict(name="DM", value=17.768, min=16.0, max=20.0,
                free=True, latex=r"\Delta m"))
  print(pars)

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

  # The iterative procedure starts ---------------------------------------------

  # update kernels with the given modifications
  badjanak.get_kernels(True)

  # run the procedure!

  ok, likelihoods = lipschitz_iteration(max_iter=10, verbose=True, tLL=tLL, tUL=tUL, kinematic_bmeson=kinematic_bmeson)

  if not ok:
    ok, likelihoods = aitken_iteration(max_iter=7, verbose=True, tLL=tLL, tUL=tUL, kinematic_bmeson=kinematic_bmeson)

  if not ok:
    print('WARNING: Convergence was not achieved!')

  # plot likelihood evolution
  ld_x = [i + 1 for i, j in enumerate(likelihoods)]
  ld_y = [j + 0 for i, j in enumerate(likelihoods)]
  # import termplotlib
  # ld_p = termplotlib.figure()
  # ld_p.plot(ld_x, ld_y, xlabel='iteration', label='likelihood',xlim=(0,30))
  # ld_p.show()

  # Storing some weights in disk ----------------------------------------------
  #     For future use of computed weights created in this loop, these should
  #     be saved to the path where samples are stored.
  #     GBweighting is slow enough once!
  print('Storing weights in root file')
  for y, dy in mc.items():  #  loop over years
    for m, v in dy.items():  # loop over mc_std and mc_dg0
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


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
