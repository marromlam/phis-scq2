__all__ = []
import os, sys
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

from analysis.angular_acceptance.iterative import fcn_data
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and, printsec, printsubsec
from utils.helpers import  version_guesser, parse_angacc, timeacc_guesser, trigger_scissors
from analysis.angular_acceptance.bdtconf_tester import bdtmesh

# threading
import logging
import threading
import time
import multiprocessing

from ipanema import initialize
initialize('cuda', 1)
from ipanema import ristra, Sample, Parameters, Parameter, optimize

# get badjanak and compile it with corresponding flags
import analysis.badjanak as badjanak
badjanak.config['fast_integral'] = 1
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels()

import config
bdtconfig = config.angacc['bdtconfig']

# reweighting config
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)   # ignore future warnings
from hep_ml import reweight
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000

reweighter = reweight.GBReweighter(**bdtconfig)

def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  if checker == 0:
    return True
  return False



def KS_test(original, target, original_weight, target_weight):
  """
  Kolmogorov test
  """
  vars = ['pTHm','pTHp','pHm','pHp']
  for i in range(0,4):
    xlim = np.percentile(np.hstack([target[:,i]]), [0.01, 99.99])
    print(f'   KS({vars[i]:>10}) =',
          ks_2samp_weighted(original[:,i], target[:,i],
                            weights1=original_weight, weights2=target_weight))


# this one should be moved
def fcn_data(parameters, data):
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []

  for y, dy in data.items():
    for dt in dy.values():
      badjanak.delta_gamma5_data(dt.data, dt.prob, **pars_dict,
                  **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
                  **dt.flavor.valuesdict(), tLL=tLL, tUL=tUL, use_timeacc = 1)
      chi2.append( -2.0 * (ristra.log(dt.prob) * dt.weight).get() );

  return np.concatenate(chi2)


def do_fit(ipars, data, verbose=False):
  """
  Fit
  """
  # start where we left
  # for v in pars.values():
  #   v.init = v.value 
  
  # do the fit
  result = optimize(fcn_data, method='minuit', params=ipars, 
                    fcn_kwgs={'data':data}, verbose=verbose, timeit=True, 
                    tol=0.05, strategy=2)
  if verbose:
    print(result.params) 
  rpars = Parameters.clone(result.params)
  return rpars, result.chi2


def pdf_reweighting(mcsample, mcparams, rdparams):
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
  return np.nan_to_num(target_pdf_h/original_pdf_h)


def kkp_weighting(original_v, original_w, target_v, target_w, path, year, mode,
                  trigger, iter, verbose=False):
  """
  Kinematic reweighting
  """
  # do reweighting
  reweighter.fit(original=original_v, target=target_v,
                 original_weight=original_w, target_weight=target_w);
  # predict weights
  kkpWeight = reweighter.predict_weights(original_v)
  # save them temp
  np.save(path.replace('.root',f'_{trigger}.npy'),kkpWeight)
  # some prints
  if verbose:
    print(f" * GB-weighting {mode}-{year}-{trigger} sample is done")
    KS_test(original_v, target_v, original_w*kkpWeight, target_w)


def get_angular_acceptance(mc, kkpWeight=False):
  """
  Compute angular acceptance
  """
  # cook weight for angular acceptance
  weight  = mc.df.eval(f'angWeight*polWeight').values
  weight *= ristra.get(mc.weight)

  if kkpWeight:
    weight *= ristra.get(mc.kkpWeight[-1])
  weight = ristra.allocate(weight)

  # compute angular acceptance
  ans = badjanak.get_angular_acceptance_weights(mc.true, mc.reco, weight,
                                                **mc.params.valuesdict(),
                                                tLL=mc.tLL, tUL=mc.tUL)

  # create ipanema.Parameters
  w, uw, cov, corr = ans
  mc.angacc_history.append(Parameters())
  for k in range(0,len(w)):
    correl = {f'w{j}': corr[k][j]
              for j in range(0, len(w)) if k > 0 and j > 0}
    mc.angacc_history[-1].add({'name': f'w{k}', 'value': w[k], 'stdev': uw[k],
                         'free': False, 'latex': f'w_{k}', 'correl': correl})
  #print(f"{  np.array(mc.angular_weights[t])}")


def do_pdf_weighting(pars, data, mc, verbose=False):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0)
  """
  
  for m, dm in mc.items(): # loop over mc_std and mc_dg0
    for y, dy in dm.items(): # loop over years
      for t, v in dy.items(): # loop over triggers
        j = len(v.pdfWeight) + 1
        if verbose:
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
          print(v.pdfWeight, j)
        v.pdfWeight.append(pdf_reweighting(v, v.params, pars+data[y][t].csp))

  if verbose:
    for y in mc[list(mc.keys())[0]].keys(): # loop over years
      print(f'Show 10 fist pdfWeight for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{mc['MC_Bs2JpsiPhi'][y]['biased'].pdfWeight[-1][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{mc['MC_Bs2JpsiPhi'][y]['unbiased'].pdfWeight[-1][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{mc['MC_Bs2JpsiPhi_dG0'][y]['biased'].pdfWeight[-1][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{mc['MC_Bs2JpsiPhi_dG0'][y]['unbiased'].pdfWeight[-1][evt]:>+.8f}")


def do_kkp_weighting(samples, mc_list=['MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0'],
                     verbose=False):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0).
  As a matter of fact, it's important to have data[y][combined] sample,
  the GBweighter gives different results when having those 0s or having
  nothing after cutting the sample.
  """

  threads = list()
  kM = mc_list[0].split('_')[1]
  for km in mc_list: # loop over mc_std and mc_dg0
    for ky, vy in samples[km].items(): # loop over years
      for kt, vt in vy.items():
        # original variables + weight (mc) 
        ov  = vt.df[['pTHm','pTHp','pHm','pHp']]
        ow  = vt.df.eval(f'angWeight*polWeight*sWeight')
        ow *= vt.pdfWeight[-1]
        # target variables + weight (real data)
        tv = samples[kM][ky][kt].df[['pTHm','pTHp','pHm','pHp']]
        tw = samples[kM][ky][kt].df.eval("sWeight")
        # Run multicore (about 15 minutes per iteration)
        job = multiprocessing.Process(
          target=kkp_weighting, 
          args=(ov.values, ow.values, tv.values, tw.values, vt.path_to_weights,
                ky, km, kt, len(threads), verbose)
        )
        threads.append(job); job.start()

  # Wait all processes to finish
  if verbose:
    print(f' * There are {len(threads)} jobs running in parallel')
  [thread.join() for thread in threads]


def do_angular_weights(samples, mc_list=['MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0'],
                       verbose=False):
  """
  This function loops over mc_list modes and computes angulat normalisation
  weights for all the years and triggers under samples[mode_list].
  """
  for km in mc_list: # loop over mc_std and mc_dg0
    for ky, vy in samples[km].items(): # loop over years
      for kt, vt in vy.items(): # loop over biased and unbiased triggers
        path_to_weights = vt.path_to_weights.replace('.root',f'_{kt}.npy')
        vt.kkpWeight.append(np.load(path_to_weights))
        os.remove(path_to_weights)
        get_angular_acceptance(vt, kkpWeight=True)

  if verbose:
    for y in samples[list(samples.keys())[0]].keys(): # loop over years
      print(f'Show 10 fist kkpWeight for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24} | {'MC_Bs2JpsiPhi_dG0':<24}")
      print(f"{'biased':<11}  {'unbiased':<11} | {'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{samples['MC_Bs2JpsiPhi'][y]['biased'].kkpWeight[-1][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{samples['MC_Bs2JpsiPhi'][y]['unbiased'].kkpWeight[-1][evt]:>+.8f}", end='')
        print(f" | ", end='')
        print(f"{samples['MC_Bs2JpsiPhi_dG0'][y]['biased'].kkpWeight[-1][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{samples['MC_Bs2JpsiPhi_dG0'][y]['unbiased'].kkpWeight[-1][evt]:>+.8f}")


def do_time_acceptance(samples, samples_list=['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar'],
                       verbose=False):
  """
  This function loops over mc_list modes and computes angulat normalisation
  weights for all the years and triggers under samples[mode_list].
  """
  years = list(samples['Bs2JpsiPhi'].keys())
  triggers = list(samples['Bs2JpsiPhi'][years[0]].keys())
  for ky in years: # loop over mc_std and mc_dg0
    for kt in triggers: # loop over biased and unbiased triggers
        # load previous time acceptance
        _pars = [samples['Bs2JpsiPhi'][ky][kt].knots]
        _data = []; _prob = []; _weight = []
        for time_mode in samples_list:
          _pars.append(samples[time_mode][ky][kt].timeacc_history[-1])
          _data.append(samples[time_mode][ky][kt].time)
          _prob.append(samples[time_mode][ky][kt].pdf)
          _weight.append(samples[time_mode][ky][kt].weight *
             samples[time_mode][ky][kt].preWeight[-1] *
             ristra.allocate(samples[time_mode][ky][kt].kkpWeight[-1]) )
        import analysis.time_acceptance.fcn_functions as fcns
        fcn_call = fcns.saxsbxscxerf
        fcn_pars = _pars[0] + _pars[1] + _pars[2] + _pars[3]
        fcn_kwgs={ 'data': _data, 'prob': _prob, 'weight': _weight,
                   'tLL': tLL, 'tUL': tUL
        }
        result = optimize(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs,
                          method='minuit', verbose=False, tol=0.1);
        _pars = Parameters.build(result.params, result.params.fetch('(c).*'))
        if verbose:
          print(f"Time acceptance for {ky}-{kt}")
          print(_pars)
        samples['Bs2JpsiPhi'][ky][kt].timeacc = _pars
        samples['Bs2JpsiPhi'][ky][kt].timeacc_history.append(_pars)


def merge_std_dg0(std, dg0, verbose=True, label=''):
  # Create w and cov arrays
  std_w = np.array( [ std[i].value for i in std ] )[1:]
  dg0_w = np.array( [ dg0[i].value for i in dg0 ] )[1:]
  std_cov = std.cov()[1:,1:];
  dg0_cov = dg0.cov()[1:,1:];

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

  # Create parameters std_w
  out = Parameters()
  for k, wk in enumerate(std.keys()):
    correl = {f'{wk}{label}':corr[k][j] for j in range(0,len(w)) if k>0 and j>0}
    out.add({'name': f'{wk}{label}', 'value': w[k], 'stdev': uw[k],
             'free': False, 'latex': f'{std[wk].latex}^{label}', 'correl': correl})

  if verbose:
    print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8}")
    for k, wk in enumerate(std.keys()):
      print(f"{np.array(std)[k]:+1.5f}", end=' | ')
      print(f"{np.array(dg0)[k]:+1.5f}", end=' | ')
      print(f"{out[f'{wk}{label}'].uvalue:+1.2uP}")
  return out


def do_mc_combination(samples, verbose):
  """
  Combine 
  """
  checker = []
  for ky, vy in samples['Bs2JpsiPhi'].items(): # loop over years
    for kt, vt in vy.items():
      std = samples['MC_Bs2JpsiPhi'][ky][kt].angacc_history[-1]
      dg0 = samples['MC_Bs2JpsiPhi_dG0'][ky][kt].angacc_history[-1]
      merged_w = merge_std_dg0(std, dg0, verbose=verbose, label=kt[0])
      vt.angacc = merged_w
      vt.angacc_history.append(merged_w)
      qwe = check_for_convergence(vt.angacc_history[-2], vt.angacc_history[-1])
      checker.append( qwe )
  
  return checker


if __name__ == '__main__':
    # parse arguments {{{
    p = argparse.ArgumentParser(description="mass fit")
    p.add_argument('--sample-BdRD')
    p.add_argument('--sample-BdMC')
    p.add_argument('--sample-BsRD')
    p.add_argument('--sample-BsMC')
    p.add_argument('--sample-BsMCdG0')
    p.add_argument('--weight-BsMC')
    p.add_argument('--weight-BsMCdG0')
    p.add_argument('--params-BsMC')
    p.add_argument('--params-BsMCdG0')
    p.add_argument('--angacc-biased-BsRD')
    p.add_argument('--angacc-unbiased-BsRD')
    p.add_argument('--timeacc-biased-BsMC')
    p.add_argument('--timeacc-unbiased-BsMC')
    p.add_argument('--timeacc-biased-BdMC')
    p.add_argument('--timeacc-unbiased-BdMC')
    p.add_argument('--timeacc-biased-BdRD')
    p.add_argument('--timeacc-unbiased-BdRD')
    p.add_argument('--timeres-BsRD')
    p.add_argument('--csp-BsRD')
    p.add_argument('--flavor-BsRD')
    p.add_argument('--params')
    p.add_argument('--fit')
    p.add_argument('--angacc')
    p.add_argument('--timeacc')
    p.add_argument('--trigger')
    p.add_argument('--version')
    args = vars(p.parse_args())
    for k, v in args.items():
        print(k, v)

    VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
    # YEARS = args['year'].split(',')
    MODE = 'Bs2JpsiPhi'
    ANGACC = parse_angacc(args['angacc'])
    print(args['timeacc'])
    TIMEACC = timeacc_guesser(args['timeacc'])
    TIMEACC['use_upTime'] = TIMEACC['use_upTime'] | ('UT' in args['version']) 
    TIMEACC['use_lowTime'] = TIMEACC['use_lowTime'] | ('LT' in args['version']) 

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

    print(TIMEACC['use_lowTime'], TIMEACC['use_upTime'])

    time_cut = f'{time}>={tLL} & {time}<={tUL}'

    timeacc_list = ['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar']
    angacc_list = ['MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi']

    # }}}

    # sample loader {{{

    def sample_loader(Bs2JpsiPhi=False, Bd2JpsiKstar=False,
            MC_Bs2JpsiPhi=False, MC_Bs2JpsiPhi_dG0=False,
            MC_Bs2JpsiKK_Swave=False, MC_Bd2JpsiKstar=False,
            trigger=['combined'], cut=None):
        samples = {}
        if Bs2JpsiPhi:
            samples['Bs2JpsiPhi'] = Bs2JpsiPhi
        if Bd2JpsiKstar:
            samples['Bd2JpsiKstar'] = Bd2JpsiKstar
        if MC_Bs2JpsiPhi:
            samples['MC_Bs2JpsiPhi'] = MC_Bs2JpsiPhi
        if MC_Bs2JpsiPhi_dG0:
            samples['MC_Bs2JpsiPhi_dG0'] = MC_Bs2JpsiPhi_dG0
        if MC_Bs2JpsiKK_Swave:
            samples['MC_Bs2JpsiKK_Swave'] = MC_Bs2JpsiKK_Swave
        if MC_Bd2JpsiKstar:
            samples['MC_Bd2JpsiKstar'] = MC_Bd2JpsiKstar

        s = {}
        for km, vm in samples.items():
          s[km] = {}
          for vy in vm:
            if '2015' in vy: ky = '2015'
            elif '2016' in vy: ky = '2016'
            elif '2017' in vy: ky = '2017'
            elif '2018' in vy: ky = '2018'
            else: ValueError("I dont get this year at all") 
            s[km][ky] = {}
            for kt in trigger:
              s[km][ky][kt] = Sample.from_root(vy, name=f"{km}-{ky}-{kt}")
              print(s[km][ky][kt])
              s[km][ky][kt].chop(cuts_and(trigger_scissors(kt), cut))
              print(s[km][ky][kt])

        return s

        # }}}}
    
    # MC reconstructed and generator level variable names
    reco  = ['cosK', 'cosL', 'hphi', time]
    true  = [f'gen{i}' for i in reco]
    reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
    true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
    
    # RD variable names
    real  = ['cosK','cosL','hphi', time]
    real += ['mHH','sigmat', 'tagOSdec','tagSSdec', 'tagOSeta', 'tagSSeta'] 

    # laod samples
    samples = sample_loader(
           Bs2JpsiPhi = args['sample_BsRD'].split(','),
           MC_Bs2JpsiPhi = args['sample_BsMC'].split(','),
           MC_Bs2JpsiPhi_dG0 = args['sample_BsMCdG0'].split(','),
           Bd2JpsiKstar = args['sample_BdRD'].split(','),
           MC_Bd2JpsiKstar = args['sample_BdMC'].split(','),
           trigger = ['biased', 'unbiased'],
           cut = time_cut
           )
    

    print('Attaching parameters to Bs2JpsiPhi')
    for i, y in enumerate(samples['Bs2JpsiPhi']):
      csp = Parameters.load(args['csp_BsRD'].split(',')[i]);
      resolution = Parameters.load(args['timeres_BsRD'].split(',')[i])
      flavor = Parameters.load(args['flavor_BsRD'].split(',')[i])
      mass = np.array( csp.build(csp, csp.find('mKK.*')) ).tolist()
      badjanak.config['mHH'] = mass
      #
      for kt, vt in samples['Bs2JpsiPhi'][y].items():
        vt.csp = csp.build(csp, csp.find('CSP.*'))
        vt.flavor = flavor
        vt.resolution = resolution
        # add time acceptace
        c = Parameters.load(args[f'timeacc_{kt}_BdRD'].split(',')[i])
        vt.knots = Parameters.build(c,c.fetch('k.*'))
        vt.timeacc = Parameters.build(c,c.fetch('(a|b|c).*'))
        vt.timeacc_history = [ vt.timeacc ]
        badjanak.config['knots'] = np.array( vt.knots ).tolist()
        print(vt.timeacc)
        # add angular acceptace
        w = Parameters.load(args[f'angacc_{kt}_BsRD'].split(',')[i])
        vt.angacc = Parameters.build(w,w.fetch('w.*'))
        vt.angacc_history = [ vt.angacc ]
        print(vt.angacc)


    print('Attaching time acceptance to other modes')
    for km in ['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar']:
      for i, ky in enumerate(samples[km]):
        for kt, vt in samples[km][ky].items():
          # add time acceptace
          mode = km.split('_')
          mode = mode[1][:2]+mode[0] if len(mode)>1 else mode[0][:2]+'RD'
          c = Parameters.load(args[f'timeacc_{kt}_{mode}'].split(',')[i])
          vt.knots = Parameters.build(c,c.fetch('k.*'))
          vt.timeacc = Parameters.build(c,c.fetch('((a|b|c).*|(mu.*|sigma.*|gamma.*))'))
          vt.timeacc_history = [ vt.timeacc ]
          print(f'Time acceptance for {km}-{ky}-{kt}')
          print(vt.timeacc)


    print('Attaching generator level parameters to MC samples')
    print('Merging angWeight (pB, pTB and mX) to MC samples')
    for km in samples:
      for i, ky in enumerate(samples[km]):
        for kt, vt in samples[km][ky].items():
          if km.startswith('MC'):
            if km == 'MC_Bs2JpsiPhi':
              vt.params = Parameters.load(args['params_BsMC'].split(',')[i]);
              angWeight = uproot.open(args['weight_BsMC'].split(',')[i])['DecayTree'].array('angWeight')
              print(km, ky, kt, vt.df.shape, len(angWeight))
              vt.df['angWeight'] = angWeight[vt.df.index]
            elif km == 'MC_Bs2JpsiPhi_dG0':
              vt.params = Parameters.load(args['params_BsMCdG0'].split(',')[i]);
              angWeight = uproot.open(args['weight_BsMCdG0'].split(',')[i])['DecayTree'].array('angWeight')
              print(km, ky, kt, vt.shape, len(angWeight))
              vt.df['angWeight'] = angWeight[vt.df.index]
            vt.path_to_weights = vt.path.replace('.root', '')
            vt.path_to_weights += f"_{args['fit']}_{args['angacc']}_vgc_amsrd_{args['timeacc']}_amsrd_{args['trigger']}.root"
            vt.angacc = None
            vt.angacc_history = [ vt.angacc ]
          

    print("Allocating arrays in device")
    for km, vm in samples.items():
      for ky, vy in vm.items():
        for kt, vt in vy.items():
          if km.startswith('MC'):
            vt.allocate(reco=reco, true=true)
            # vt.angWeight = [ np.ones_like(vt.df.eval("sWeight")) ]
            # vt.kkpWeight = [ np.ones_like(vt.df.eval("sWeight")) ]
          else:
            if km == 'Bs2JpsiPhi':
              sw = np.zeros_like(vt.df.eval("sWeight"))
              for l,h in zip(mass[:-1], mass[1:]):
                pos = vt.df.eval(f'mHH>={l} & mHH<{h}')
                _sw = vt.df.eval(f'sWeight*(mHH>={l} & mHH<{h})')
                sw = np.where(pos, _sw * ( sum(_sw)/sum(_sw*_sw) ), sw)
              vt.df['sWeight'] = sw
              vt.allocate(data=real, prob='0*time')
          vt.allocate(weight='sWeight', time=time, pdf='0*time')
          vt.tLL = tLL
          vt.tUL = tUL
          preWeight = []
          # TODO: found a bug. MC_Bs2JpsiPhi does not have kinWeight
          if 'pdfWeight' in vt.branches: preWeight.append('pdfWeight')
          if 'polWeight' in vt.branches: preWeight.append('pdfWeight')
          if 'angWeight' in vt.branches:
            # angWeight is a better version of kinWeight, so let's use it 
            # if it exists
            preWeight.append('angWeight')
          else: 
            # if we don't have angWeight we try to ge the kinWeight
            if 'kinWeight' in vt.branches: preWeight.append('kinWeight')

          if len(preWeight)>0:
            vt.preWeight = [ ristra.allocate(vt.df.eval("*".join(preWeight)).values) ]
          else:
            vt.preWeight = [ ristra.allocate(np.ones_like(vt.df.eval("sWeight"))) ]
          vt.pdfWeight = [ np.ones_like(vt.df.eval("sWeight")) ]
          vt.kkpWeight = [ np.ones_like(vt.df.eval("sWeight")) ]
    print(samples)


    badjanak.get_kernels()


    # shit {{{

    printsec('Parameters and initial status')

    print(f"\nFitting parameters\n{80*'='}")
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
    pars.add(dict(name="dPlon", value=0.000, min=-2*3.14, max=2*3.14,
              free=False, latex=r"\delta_0"))
    pars.add(dict(name="dPpar", value=3.260, min=-2*3.14, max=2*3.14,
              free=True, latex=r"\delta_{\parallel} - \delta_0"))
    pars.add(dict(name="dPper", value=3.026, min=-2*3.14, max=2*3.14,
              free=True, latex=r"\delta_{\perp} - \delta_0"))
    
    # lambdas
    pars.add(dict(name="lSlon", value=1.0, min=0.7, max=1.6,
              free=False, latex=r"\lambda_S/\lambda_0"))
    pars.add(dict(name="lPlon", value=1.0, min=0.7, max=1.6,
              free=True,  latex=r"\lambda_0"))
    pars.add(dict(name="lPpar", value=1.0, min=0.7, max=1.6,
              free=False, latex=r"\lambda_{\parallel}/\lambda_0"))
    pars.add(dict(name="lPper", value=1.0, min=0.7, max=1.6,
              free=False, latex=r"\lambda_{\perp}/\lambda_0"))
    
    # life parameters
    pars.add(dict(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
              free=False, latex=r"\Gamma_d"))
    pars.add(dict(name="DGs", value= 0.0917, min= 0.03, max= 0.15,
              free=True, latex=r"\Delta\Gamma_s"))
    pars.add(dict(name="DGsd", value= -0.1, min=-0.2, max= 0.2,
              free=True, latex=r"\Gamma_s - \Gamma_d"))
    pars.add(dict(name="DM", value=17.768, min=16.0, max=20.0,
              free=True, latex=r"\Delta m"))
    print(pars)

    # }}}



    for iteration in range(20):
      # print experimental effects
      for ky, vy in samples['Bs2JpsiPhi'].items():
        for kt, vt in vy.items():
          print(f"Exp. effects for {ky} {kt}")
          # for el in [vt.resolution, vt.flavor, vt.csp, vt.timeacc, vt.angacc]:
          for el in [vt.timeacc, vt.angacc]:
              print([f"{v:.2uP}" for v in el.uvaluesdict().values()])

      print(f"\n[#{iteration}] FIT DATA")
      pars, curret_chi2 = do_fit(pars, samples['Bs2JpsiPhi'], verbose=False)
      # print(pars)

      print(f"\n[#{iteration}] PDF WEIGHTING MC TO DATA")
      do_pdf_weighting(
          pars, samples['Bs2JpsiPhi'],
          {'MC_Bs2JpsiPhi': samples['MC_Bs2JpsiPhi'],
           'MC_Bs2JpsiPhi_dG0': samples['MC_Bs2JpsiPhi_dG0']},
          verbose=False
      )
      print(f"\n[#{iteration}] KINEMATIC WEIGHTING MC TO DATA")
      do_kkp_weighting(samples, verbose=False)

      print(f"\n[#{iteration}] ANGULAR ACCEPTANCE DETERMINATION")
      do_angular_weights(samples, mc_list=angacc_list, verbose=False)

      print(f"\n[#{iteration}] CHECK FOR CONVERGENCE")
      if len(angacc_list)>1:
        checker = do_mc_combination(samples, verbose=False)
        print(checker)
      if all(checker):
          print(checker)
          print("\n\n    This is a small step for the iterative procedure,")
          print("    but a huge step for future analyses.")
          print("    --- Marcosito.")
          pars.dump(args['params'])
          break

      print(f"\n[#{iteration}] TIME ACCEPTANCE DETERMINATION")
      do_time_acceptance(samples, samples_list=timeacc_list, verbose=False)
      print(pars)



# vim: fdm=marker
