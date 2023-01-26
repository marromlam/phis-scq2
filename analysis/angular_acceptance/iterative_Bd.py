#!/usr/bin/env python
# -*- coding: utf-8 -*-

from hep_ml import reweight
from utils.helpers import version_guesser, timeacc_guesser, trigger_scissors
from utils.strings import cammel_case_split, cuts_and
from utils.plot import mode_tex
import badjanak
from ipanema import ristra, Sample, Parameters, Parameter, optimize
from ipanema import initialize
import multiprocessing
import time
import threading
import logging
from hep_ml.metrics_utils import ks_2samp_weighted
from warnings import simplefilter
from timeit import default_timer as timer
from scipy.stats import chi2
from uncertainties import unumpy as unp
import uncertainties as unc
import hjson
import sys
import os
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
__all__ = []
__author__ = ['Ramon Ruiz']
__email__ = ['rruizfer@cern.ch']

################################################################################
# %% Modules ###################################################################


# reweighting config
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# threading

# load ipanema

initialize(config.user['backend'], 2)

# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

# import some phis-scq utils

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# reweighting config
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)

# Functions needed


def check_for_convergence(a, b):
  a_f = np.array([float(a[p].unc_round[0]) for p in a])
  b_f = np.array([float(b[p].unc_round[0]) for p in b])
  checker = np.abs(a_f - b_f).sum()
  if checker == 0:
    return True
  return False


def pdf_reweighting_Bd(mcsample, mcparams, rdparams):  # cambie true por reco
  print('empiezo pdf_reweighhting')
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
  return np.nan_to_num(target_pdf_h / original_pdf_h)


def KS_test(original, target, original_weight, target_weight):
  vars = ['pTHm', 'pTHp', 'pHm', 'pHp']
  for i in range(0, 4):
    xlim = np.percentile(np.hstack([target[:, i]]), [0.01, 99.99])
    print(f'   KS({vars[i]:>10}) =',
          ks_2samp_weighted(original[:, i], target[:, i],
                            weights1=original_weight, weights2=target_weight))


def kkp_weighting(original_v, original_w, target_v, target_w, path, y, m, t, i):
  reweighter.fit(original=original_v, target=target_v,
                 original_weight=original_w, target_weight=target_w)
  kkpWeight = reweighter.predict_weights(original_v)
  np.save(path.replace('.root', f'_{t}.npy'), kkpWeight)
  print(f" * GB-weighting {m}-{y}-{t} sample is done")
  KS_test(original_v, target_v, original_w * kkpWeight, target_w)


def kkp_weighting_bins(original_v, original_w, target_v, target_w, path, y, m, t, i):
  reweighter_bin.fit(original=original_v, target=target_v,
                     original_weight=original_w, target_weight=target_w)
  kkpWeight = np.where(original_w != 0, reweighter_bin.predict_weights(original_v), 0)
  # kkpWeight = np.where(original_w!=0, np.ones_like(original_w), 0)
  np.save(os.path.dirname(path) + f'/kkpWeight_{t}.npy', kkpWeight)
  #print(f" * GB-weighting {m}-{y}-{t} sample is done")
  print(f" * GB-weighting {m}-{y}-{t} sample\n  {kkpWeight[:10]}")


def get_angular_acceptance(mc, kkpWeight=False):
  # cook weight for angular acceptance
  weight = mc.df.eval(f'angWeight*polWeight*{weight_rd}').values
  i = len(mc.kkpWeight.keys())

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[i])
  weight = ristra.allocate(weight)

  # compute angular acceptance
  ans = badjanak.get_angular_acceptance_weights_Bd(mc.true, mc.reco, weight, **mc.params.valuesdict())

  # create ipanema.Parameters
  w, uw, cov, corr = ans
  mc.angaccs[i] = Parameters()
  for k in range(0, len(w)):
    correl = {f'w{j}': corr[k][j]
              for j in range(0, len(w)) if k > 0 and j > 0}
    mc.angaccs[i].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                       'free': False, 'latex': f'w_{k}', 'correl': correl})


def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []
  for y, dy in data.items():
    for dt in [dy['unbiased'], dy['biased']]:
      badjanak.delta_gamma5_data_Bd(dt.data, dt.lkhd, **pars_dict,
                                    **dt.angacc.valuesdict(),
                                    **dt.csp.valuesdict(),
                                    tLL=tLL, tUL=tUL, use_timeacc=0, set_tagging=1, use_timeres=0)
      # de momento el notas va a pasar de lo que le mandes a mayores (o eso debería)

      chi2.append(-2.0 * np.log(ristra.get(dt.lkhd)) * dt.weight.get())
  return np.concatenate(chi2)

# Parse arguments for this script


def argument_parser():
  p = argparse.ArgumentParser(description='Check iterative procedure for angular acceptance.')
  p.add_argument('--sample-mc-std', help='Bd2JpsiKstar MC sample')
  p.add_argument('--sample-data', help='Bd2JpsiKstar data sample')
  p.add_argument('--params-mc-std', help='BdJpsiKstar MC sample')
  p.add_argument('--input-csp', help='csp factors for Bd2JpsiKstar')
  p.add_argument('--output-weights-biased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-weights-unbiased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-angular-weights-mc-std', help='BdJpsiKstar MC sample')
  p.add_argument('--output-tables-biased', help='BdJpsiKstar MC sample')
  p.add_argument('--output-tables-unbiased', help='BdJpsiKstar MC sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  return p


if __name__ == '__main__':
  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',')
  #YEARS = ['2016']
  MODE = 'Bd2JpsiKstar'
  ANGACC = args['angacc']
  # Prepare the cuts -----------------------------------------------------------
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Paths with samples, params and output tables
  # samples
  samples_std = args['sample_mc_std'].split(',')

  samples_data = args['sample_data'].split(',')

  # params MC
  input_std_params = args['params_mc_std'].split(',')

  # csp_factors
  csp_factors = args['input_csp'].split(',')
  # outputs
  params_biased = args['output_weights_biased'].split(',')
  params_unbiased = args['output_weights_unbiased'].split(',')

  tables_biased = args['output_tables_biased'].split(',')
  tables_unbiased = args['output_tables_unbiased'].split(',')

  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')

  # Print settings
  print(f"\n{80*'='}\nSettings\n{80*'='}\n")
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdconfig.values()):50}\n")

  # Variables for the PDF
  print(f"\n{80*'='}\nLoading samples\n{80*'='}\n")

  reco = ['cosK', 'cosL', 'hphi', 'time']
  true = [f'gen{i}' for i in reco]
  reco += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
  true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']

  real = ['cosK', 'cosL', 'hphi', 'time']
  real += ['mHH', '0*sigmat', 'idB', 'idB', '0*time', '0*time']
  weight_rd = f'sw_{VAR}' if VAR else 'sw'
  weight_mc = f'(polWeight*{weight_rd})'

  # Load MC sample
  mc = {}
  mcmodes = ['MC_Bd2JpsiKstar']
  for i, y in enumerate(YEARS):
    print(f'\nLoading {y} MC samples')
    mc[y] = {}
    for m, v in zip(mcmodes, [samples_std]):
      print(v[i])
      mc[y][m] = {'biased': Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      mc[y][m]['biased'].name = f"{y}-biased"
      mc[y][m]['unbiased'].name = f"{y}-unbiased"
    for m, v in zip(mcmodes, [input_std_params]):
      print(v[i])
      mc[y][m]['biased'].assoc_params(v[i])
      mc[y][m]['unbiased'].assoc_params(v[i])
      mc[y][m]['biased'].chop(cuts_and(trigger_scissors('biased'), CUT))
      mc[y][m]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'), CUT))
      for t in ['biased', 'unbiased']:
        mc[y][m][t].allocate(reco=reco, true=true, pdf='0*time', weight=weight_mc)
        mc[y][m][t].angaccs = {}
        mc[y][m][t].kkpWeight = {}
        mc[y][m][t].pdfWeight = {}
    for m, v in zip(mcmodes, [kkpWeight_std]):
      mc[y][m]['biased'].path_to_weights = v[i]
      mc[y][m]['unbiased'].path_to_weights = v[i]
  if MODE == 'Bd2JpsiKstar':
    badjanak.config['mHH'] = [826, 861, 896, 931, 966]
  mass = badjanak.config['mHH']

  # Load data sample
  # Load corresponding data sample ---------------------------------------------
  data = {}
  for i, y in enumerate(YEARS):
    print(f'Fetching elements for {y}[{i}] data sample')
    data[y] = {}
    csp = Parameters.load(csp_factors[i])
    for t, T in zip(['biased', 'unbiased'], [0, 1]):
      data[y][t] = Sample.from_root(samples_data[i], share=SHARE)
      data[y][t].name = f"{m}-{y}-{t}"
      data[y][t].csp = csp.build(csp, csp.find('CSP.*'))
      data[y][t].chop(trigger_scissors(t, CUT))
    for t, path in zip(['biased', 'unbiased'], [params_biased, params_unbiased]):
      data[y][t].params_path = path[i]

    for t, path in zip(['biased', 'unbiased'], [tables_biased, tables_unbiased]):
      data[y][t].tables_path = path[i]

  # Obtain angweights (weights cinematics B) and angular acceptance for MC corrected
  # WARNING ES POSIBLE QUE ESTOS REWEIGHTINGS USEN LA PDF COMPROBAR -> MARCOS
  # WARNING get_angular_acceptance usa la pdf no entiendo dif fk=1 y fk=0
  for i, y in enumerate(YEARS):
    for t in ['biased', 'unbiased']:
      print('Compute angWeights correcting MC sample in kinematics')
      print(f" * Computing kinematic GB-weighting in pTB, pB and mHH")
      reweighter.fit(original=mc[y][m][t].df[['mHH', 'pB', 'pTB']],
                     target=data[y][t].df[['mHH', 'pB', 'pTB']],
                     original_weight=mc[y][m][t].df.eval(weight_mc),
                     target_weight=data[y][t].df.eval(weight_rd))
      angWeight = reweighter.predict_weights(mc[y][m][t].df[['mHH', 'pB', 'pTB']])
      print(f'angWeight {t}')
      print(angWeight[0:20])
      mc[y][m][t].df['angWeight'] = angWeight
      mc[y][m][t].olen = len(angWeight)
      print(mc[y][m][t].params)
      angacc = badjanak.get_angular_acceptance_weights_Bd(mc[y][m][t].true, mc[y][m][t].reco,
                                                          mc[y][m][t].weight * ristra.allocate(angWeight),
                                                          **mc[y][m][t].params.valuesdict())
      w, uw, cov, corr = angacc
      pars = Parameters()
      for i in range(0, len(w)):
        correl = {f'w{j}{t}': corr[i][j]
                  for j in range(0, len(w)) if i > 0 and j > 0}
        pars.add({'name': f'w{i}{t}', 'value': w[i], 'stdev': uw[i],
                  'correl': correl, 'free': False, 'latex': f'w_{i}^{t}'})
      print(f" * Corrected angular weights for {MODE}{y}-{t} sample are:")
      print(f"{pars}")
      data[y][t].angacc = pars
      data[y][t].angaccs = {0: pars}
  for i, y in enumerate(YEARS):
    print(f' *  Allocating {y} arrays in device ')
    for d in [data[y]['biased'], data[y]['unbiased']]:
      sw = np.zeros_like(d.df[f'{weight_rd}'])
      for l, h in zip(mass[:-1], mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        sw = np.where(pos, this_sw * (sum(this_sw) / sum(this_sw * this_sw)), sw)
      d.df['sWeight'] = sw
      d.allocate(data=real, weight='sWeight', lkhd='0*time')

  # %% Prepare dict of parameters ----------------------------------------------
  print(f"\n{80*'='}\nParameters and initial status\n{80*'='}\n")

  print(f"\nFitting parameters\n{80*'='}")
  pars = Parameters()
  # S wave fractions
  pars.add(dict(name='fSlon1', value=0.115, min=0.00, max=0.60,
                free=True, latex=r'f_S^{1}'))
  pars.add(dict(name='fSlon2', value=0.049, min=0.00, max=0.60,
                free=True, latex=r'f_S^{2}'))
  pars.add(dict(name='fSlon3', value=0.052, min=0.00, max=0.60,
                free=True, latex=r'f_S^{3}'))
  pars.add(dict(name='fSlon4', value=0.105, min=0.00, max=0.60,
                free=True, latex=r'f_S^{4}'))
  # P wave fractions normalmente fPlon
  pars.add(dict(name="fPlon", value=0.572, min=0.35, max=0.65,
                free=True, latex=r'f_0'))
  # pars.add(dict(name="fPpar", value=0.240, min=0.1, max=0.9,
  # free=True, latex=r'f_{\parallel}'))
  pars.add(dict(name="fPper", value=0.201, min=0.15, max=0.35,
                free=True, latex=r'f_{\perp}'))

  # S wave strong phases
  pars.add(dict(name='dSlon1', value=+0.150, min=-4.0, max=4.0,
                free=True, latex=r"\delta_S^{1} - \delta_{\perp}"))
  pars.add(dict(name='dSlon2', value=-0.280, min=-4.0, max=4.0,
                free=True, latex=r"\delta_S^{2} - \delta_{\perp}"))
  pars.add(dict(name='dSlon3', value=-1.00, min=-4.0, max=4.0,
                free=True, latex=r"\delta_S^{3} - \delta_{\perp}"))
  pars.add(dict(name='dSlon4', value=-1.43, min=-4.0, max=4.0,
                free=True, latex=r"\delta_S^{4} - \delta_{\perp}"))
  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.000, min=-3.14, max=3.14,
                free=False, latex=r"\delta_0"))
  pars.add(dict(name="dPpar", value=-2.94, min=-2 * 3.14, max=2 * 3.14,
                free=True, latex=r"\delta_{\parallel} - \delta_0"))
  pars.add(dict(name="dPper", value=2.94, min=-2 * 3.14, max=2 * 3.14,
                free=True, latex=r"\delta_{\perp} - \delta_0"))

  # life parameters
  pars.add(dict(name="Gd", value=0.65833, min=0.45, max=0.85,
                free=True, latex=r"\Gamma_d"))
  print(pars)
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

  badjanak.get_kernels()

  # The iterative procedure starts ---------------------------------------------
  #     First print angular acceptance before iterative procedure
  CHECK_DICT = {}
  likelihoods = []
  for i in range(1, 30):
    print(f"\n{80*'='}\nIteration {i} of the procedure\n{80*'='}\n")
    checker = []                # here we'll store if weights do converge or not
    for ci in range(0, i):
      CHECK_DICT[ci] = []
    itstr = f"[iteration #{i}]"
    for v in pars.values():
      v.init = v.value  # start where we left
    print(f'Simultaneous fit Bd2JpsiKstar {"&".join(list(mc.keys()))} {itstr}')
    result = optimize(fcn_data,
                      method='minuit', params=pars, fcn_kwgs={'data': data},
                      verbose=True, timeit=True, tol=0.001, strategy=1)
    likelihoods.append(result.chi2)

    pars = Parameters.clone(result.params)
    names = ['fPlon', 'fPper', 'dPpar', 'dPper',
             'fSlon1', 'dSlon1', 'fSlon2', 'dSlon2',
             'fSlon3', 'dSlon3', 'fSlon4', 'dSlon4']
    corr_run1 = Parameters.build(result.params, names).corr()
    for p in ['fPlon', 'fPper', 'dPpar', 'dPper',
              'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4',
              'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'Gd']:
      try:
        print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
      except:
        0
    # 2nd step: pdf reweighting with this new Parameters
    print(f'\nPDF weighting MC samples to match Bd2JpsiKstar data {itstr}')
    t0 = timer()
    for y, dy in mc.items():  #  loop over years
      print(y, dy)
      for m, dm in dy.items():  # loop over mc_std and mc_dg0
        for t, v in dm.items():  # loop over mc_std and mc_dg0
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
          print(t, v)
          print(v.df[['gencosK', 'gencosL', 'genhphi', 'gentime', 'genidB']])
          v.pdfWeight[i] = pdf_reweighting_Bd(v, v.params, pars + data[y][t].csp)
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bd2JpsiKstar':<24}")
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')
    tf = timer() - t0
    print(f'PDF weighting took {tf:.3f} seconds.')
    # 3rd step: kinematic weights
    print(f'\nKinematic reweighting MC samples in K momenta {itstr}')
    threads = list()
    t0 = timer()

    for y, dy in mc.items():  #  loop over years
      print(y, dy)
      for m, dm in dy.items():  # loop over mc_std and mc_dg0
        for t, v in dm.items():
          print(t, v)
          # original variables + weight (mc)
          ov = v.df[['pTHm', 'pTHp', 'pHm', 'pHp']]
          ow = v.df.eval(f'angWeight*polWeight*{weight_rd}')
          ow *= v.pdfWeight[i]
          # target variables + weight (real data)
          tv = data[y][t].df[['pTHm', 'pTHp', 'pHm', 'pHp']]
          tw = data[y][t].df.eval(f'{weight_rd}')
          job = multiprocessing.Process(target=kkp_weighting, args=(
              ov.values, ow.values, tv.values, tw.values,
              v.path_to_weights, y, m, t, len(threads)))
          threads.append(job)
          job.start()

      # Wait all processes to finish
    print(f' * There are {len(threads)} jobs running in parallel')
    [thread.join() for thread in threads]
    tf = timer() - t0
    print(f'Kinematic weighting took {tf:.3f} seconds.')

    # 4th step: angular weights ------------------------------------------------
    print(f'\nExtract angular weights {itstr}')
    for y, dy in mc.items():  #  loop over years
      for m, dm in dy.items():  # loop over mc_std and mc_dg0
        for t, v in dm.items():  # loop over biased and unbiased triggers
          path_to_weights = v.path_to_weights.replace('.root', f'_{t}.npy')
          v.kkpWeight[i] = np.load(path_to_weights)
          os.remove(path_to_weights)
          get_angular_acceptance(v, kkpWeight=True)
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_Bd2JpsiKstar']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bd2JpsiKstar']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')

    # 5th step: save current set of angular weights
    print(f'\Parameters Angacc MC_BdJsiKstar {itstr}')
    for y, dy in mc.items():  #  loop over years
      for trigger in ['biased', 'unbiased']:
        std = dy['MC_Bd2JpsiKstar'][trigger].angaccs[i]
        data[y][trigger].angacc = std
        data[y][trigger].angaccs[i] = std

        # Check for all iterations if existed convergence
        for ci in range(0, i):
          CHECK_DICT[ci].append(check_for_convergence(
              data[y][trigger].angaccs[ci], data[y][trigger].angaccs[i]))

        qwe = check_for_convergence(data[y][trigger].angaccs[i - 1], data[y][trigger].angaccs[i])
        checker.append(qwe)
        std.dump(data[y][trigger].params_path.replace('.json', f'i{i}.json'))
        #print(f'Value of chi2/dof = {chi2_value:.4}/{dof} corresponds to a p-value of {prob:.4}\n')
    # Check if they are the same as previous iteration
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
    print("CHECK_DICT: ", CHECK_DICT)
    print("LIKELIHOODs: ", likelihoods)
    if all(checker):
      print(f"\nDone! Convergence was achieved within {i} iterations")
      print(f"Comparación Run1")
      # names = ['fPpar', 'fPper', 'dPpar', 'dPper',
      #'dSlon1r1', 'dSlon2r1', 'dSlon3r1', 'dSlon4r1',
      # 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4']
      names = ['fPlon', 'fPper', 'dPpar', 'dPper',
               'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4',
               'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'Gd']
      valores = []
      std = []
      # Comparation with Run1 results
      old_values = [0.572, 0.201, -2.94, 2.94, 0.150, -0.280, -1.00, -1.410, 0.115, 0.049, 0.052, 0.105]
      old_std = [0.015, 0.009, 0.04, 0.03, 0.13, 0.09, 0.09, 0.11, 0.021, 0.008, 0.011, 0.016]
      for p in names:
        valores.append(pars[p].value)
        std.append(pars[p].stdev)
      df = pd.DataFrame({"names": names, "valores": valores, "stdev": std, "valores_viejos": old_values, "std_viejas": old_std})
      df['PULL'] = df.eval('(valores-valores_viejos)/sqrt(stdev**2+std_viejas**2)')
      print(df)
      print('Correlaciones para compararlas Run1')
      print(corr_run1)
      for y, dy in data.items():  #  loop over years
        for trigger in ['biased', 'unbiased']:
          pars = data[y][trigger].angaccs[i]
          print('Saving table of params in tex')
          pars.dump(data[y][trigger].params_path)
          print('Saving table of params in tex')
          with open(data[y][trigger].tables_path, "w") as tex_file:
            tex_file.write(
                pars.dump_latex(caption="""
              Angular acceptance for \\textbf{%s} \\texttt{\\textbf{%s}}
              category.""" % (y, trigger)
                )
            )
          tex_file.close()
      break

  # plot likelihood evolution
  ld_x = [i + 1 for i, j in enumerate(likelihoods)]
  ld_y = [j + 0 for i, j in enumerate(likelihoods)]
  import termplotlib
  ld_p = termplotlib.figure()
  ld_p.plot(ld_x, ld_y, xlabel='iteration', label='likelihood', xlim=(0, 30))
  ld_p.show()

  # Storing some weights in disk -------------------------------------------------
  #     For future use of computed weights created in this loop, these should be
  #     saved to the path where samples are stored.
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
        print(len(wb + wu))
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
