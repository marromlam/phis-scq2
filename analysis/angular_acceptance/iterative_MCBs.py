DESCRIPTION = """
    Angular acceptance iterative procedure MC Bs Check iterative
"""

__author__ = ['Ramon Ruiz Fernandez']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

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
from scipy import stats
from scipy import special
from timeit import default_timer as timer
from hep_ml.metrics_utils import ks_2samp_weighted

# threading
import logging
import threading
import time
import multiprocessing
import uncertainties as unc
# load ipanema
from ipanema import initialize, plotting
initialize(os.environ['IPANEMA_BACKEND'],1)
from ipanema import ristra, Sample, Parameters, Parameter, optimize, plot_conf2d, Optimizer

# get badjanak and compile it with corresponding flags
import badjanak
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and, printsec, printsubsec
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
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000


def check_for_convergence(a,b):
  a_f = np.array( [float(a[p].unc_round[0]) for p in a] )
  b_f = np.array( [float(b[p].unc_round[0]) for p in b] )
  checker = np.abs(a_f-b_f).sum()
  #if checker<=10.0:
  if checker == 0:
    return True
  return False



# core functions
#     They work for a given category only.

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


# Komogorov test

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

def get_angular_acceptance(mc, kkpWeight=False):
  """
  Compute angular acceptance
  """
  # cook weight for angular acceptance
  weight  = mc.df.eval(f'angWeight*{weight_rd}').values
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



# this one should be moved
def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pars_dict = parameters.valuesdict(blind=False)
  chi2 = []

  for y, dy in data.items():
    for dt in dy.values():
      badjanak.delta_gamma5_data(dt.data, dt.lkhd, **pars_dict,
                  **dt.timeacc.valuesdict(),
                  **dt.angacc.valuesdict(),
                  **dt.resolution.valuesdict(), #**dt.csp.valuesdict(),
                  #**dt.flavor.valuesdict(),
                  tLL=tLL, tUL=tUL, use_timeacc=1, use_angacc=1,
                  use_timeres=1, set_tagging=0)  #set_tagging = 0
      chi2.append( -2.0 * (ristra.log(dt.lkhd) * dt.weight).get() );

  return np.concatenate(chi2)






















# Multiple categories functions
#     They run over multiple categories

def do_fit(verbose=True):
  """
  Fit
  """
  # Get'em from the global scope
  global pars, data, param_forPlotting

  # start where we left
  for v in pars.values():
    v.init = v.value

  # do the fit
  #mini = Optimizer(fcn_call = fcn_data, params = pars, fcn_kwgs={'data':data})
  #result = mini.optimize(method='minuit', verbose=True, tol=0.0001)
  import matplotlib.pyplot as plt
  result = optimize(fcn_data, method='minuit', params=pars,
                    fcn_kwgs={'data':data}, verbose=True, timeit=True,
                    tol=0.05, strategy=2)
  #result._minuit.draw_mncontour('dPper', 'lPlon', nsigma=3)
  print(result._minuit.fmin.fval)
  print(result)
  #Storing params for plotting
  
  i = len(mc[YEARS[0]]['MC_Bs2JpsiPhi']['biased'].angaccs)
  print('i do_fit', i)
  params_forPlotting[i] = Parameters.clone(result.params) 
  #names = ['fPlon','fPper', 'dPpar', 'dPper', 'lPlon', 'DGsd']
  #for p in names:
    #plt.close()
  #result._minuit.draw_mnprofile('lPlon', bins=1000, bound = 5, band=True, text=True, subtract_min=False)
  #plt.savefig('lPlon_profile_shit.pdf')
  #fig, ax = plot_conf2d(mini, result, ['lPlon', 'dPper'], size=(20,20) )
  #fig.savefig('LambdaVsdPper.pdf')
  #exit()
  #print(result._minuit.fmin.fval)
  #print fit results
  #print(result) # parameters are not blinded, so we dont print the result
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
      names = ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DM']
      for p in names:
        try:
          print(f"{p:>12} : {pars[p].value:+.8f} +/- {pars[p].stdev:+.8f}")
        except:
          0
  # store parameters + add likelihood to list
  names = ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGs', 'DGsd', 'DM']
  #names = ['fPlon','fPper', 'dPpar', 'dPper', 'lPlon', 'DM', 'DGsd']
  pars2 = Parameters.clone(result.params)
  values, oldvalues, std, latex = [], [], [], []
  shit1, shit2 = [], []
  dict = {'fPlon':'$|A_{0}|^{2}$', 'fPper':'$|A_{\perp}|^{2}$', 'dPpar': '$\delta_{\parallel}-\delta_{0}$','dPper':'$\delta_{\perp}-\delta_{0}$',
          'pPlon':'$\phi_0$', 'lPlon':'$\lambda_0$', 'DGs': '$\Delta\Gamma_s$', 'DGsd':'$\Gamma_s - \Gamma_d$', 'DM': '$\Delta m$'} 
  for p in names:
    latex.append(dict[pars2[p].name])
    values.append(pars2[p].value)
    oldvalues.append(params_init[p].value)
    std.append(pars2[p].stdev)
  df = pd.DataFrame({"names": latex, "values": values, "std": std, "oldvalues": oldvalues})
  df['PULL'] = df.eval('(values-oldvalues)/sqrt(std**2)')
  print(df)
  print(df.to_latex(index=False, escape=False))
  PULLS = df['PULL'].tolist()
  shit1 = "\\begin{table}[H]\n\centering\n\small\n\\begin{tabular}{l"+"|"+1*"c"+"}\n\\toprule\n"
  shit1 += 'Parameter & Values Obtained (stat. only)\\\ \n \midrule \n'
  for i in range(len(names)):
    shit1 += latex[i]+' & '+' $ '+'{:.2uL}'.format(unc.ufloat(values[i], std[i]))+' $ ('+str(np.round(PULLS[i],2))+")\\\ \n"
  shit1 += "\\bottomrule\n\end{tabular}\n"
  shit1 += "\caption{{{caption}}}\n"
  shit1 += "\end{table}\n"
  print('Table Latex:\n')
  print(shit1)


  ###Getting p-value: Ramon for Veronika
  gen_params = Parameters.build(params_init, names)
  params = Parameters.build(pars2, names)
  print('Correlation')
  print(params.corr(['dPper', 'lPlon', 'dPpar'])) 
  diff = np.array(list(gen_params.valuesdict().values()))-np.array(list(params.valuesdict().values())) 
  print('diff',diff) 
  cov = np.matrix(params.cov())
  print('cov', cov)
  chi2 = np.dot(np.dot(diff, cov.getI()), diff.T)
  dof = len(cov)
  p = stats.chi2.sf(chi2, dof)
  p2 = 1- stats.chi2.cdf(chi2, dof)
  print(chi2)
  print(p,p2)
  #p-values without dGsd for checks with no time acc.
  # names = ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGs', 'DM']
  # gen_params = Parameters.build(params_init, names)
  # params = Parameters.build(pars, names)
  # print('Correlation')
  # print(params.corr(['dPper', 'lPlon', 'dPpar'])) 
  # diff = np.array(list(gen_params.valuesdict().values()))-np.array(list(params.valuesdict().values())) 
  # print('diff',diff) 
  # cov = np.matrix(params.cov())
  # print('cov', cov)
  # chi2 = np.dot(np.dot(diff, cov.getI()), diff.T)
  # dof = len(cov)
  # p = stats.chi2.sf(chi2, dof)
  # p2 = 1- stats.chi2.cdf(chi2, dof)
  # print(chi2)
  # print(p,p2)
  #P-values por parámetro con DLL y overall p-value interacción.
  DLL = []
  PULL_DLL, p_value_parameter = [], []
  pars_dll = Parameters.clone(mc[YEARS[0]]['MC_Bs2JpsiPhi']['biased'].params)
  pars_dll.unlock(*names)
  print(pars_dll)
  L0 = optimize(fcn_data, method='minuit', params=pars_dll,
                   fcn_kwgs={'data':data}, verbose=True, timeit=True,
                   tol=0.05, strategy=2)
  pars_dll.lock()
  for i, key in enumerate(names):
    pars_dll.unlock(*names)
    pars_dll.lock(key)
    result_param = optimize(fcn_data, method='minuit', params=pars_dll,
                   fcn_kwgs={'data':data}, verbose=True, timeit=True,
                   tol=0.05, strategy=2)
    print(f'chi2 minimization per parameter {key}', result_param.chi2)
    DLL.append(result_param.chi2-L0.chi2)
    pars_dll.lock()
    p_value_parameter.append(stats.chi2.sf(DLL[i], 1))
    PULL_DLL.append(np.sqrt(2)*special.erfcinv(p_value_parameter[i]))
    print(f'sigma per parameter {key}', PULL_DLL[i])
  values, oldvalues, std, latex = [], [], [], []
  shit1, shit2 = [], []
  for p in names:
    latex.append(dict[pars2[p].name])
    values.append(L0.params[p].value)
    std.append(L0.params[p].stdev)
    oldvalues.append(params_init[p].value)
  df2 = pd.DataFrame({"names": latex, "values": values, "std": std, "oldvalues": oldvalues, "PULL with DLL": PULL_DLL})
  print(df2.to_latex(index=False, escape=False))
  shit1 = "\\begin{table}[H]\n\centering\n\small\n\\begin{tabular}{l"+"|"+1*"c"+"}\n\\toprule\n"
  shit1 += 'Parameter & Values Obtained (stat. only)\\\ \n \midrule \n'
  for i in range(len(names)):
    shit1 += latex[i]+' & '+' $ '+'{:.2uL}'.format(unc.ufloat(values[i], std[i]))+' $ ('+str(np.round(PULL_DLL[i],2))+")\\\ \n"
  shit1 += "\\bottomrule\n\end{tabular}\n"
  chi2_genparams = np.sum(fcn_data(pars_dll, data=data))
  overall_p_value = stats.chi2.sf(L0.chi2-chi2_genparams, len(names))
  overall_sigma = np.sqrt(2)*special.erfcinv(overall_p_value)
  shit1 += f"\caption{overall_p_value} ({overall_sigma})\n"
  shit1 += "\end{table}\n"
  print('Table Latex with DLLs:\n')
  print(shit1)
  pars = Parameters.clone(result.params)
  #exit()
  return result.chi2


def do_pdf_weighting(verbose):
  """
  We need to change badjanak to handle MC samples and then we compute the
  desired pdf weights for a given set of fitted pars in step 1. This
  implies looping over years and MC samples (std and dg0)
  """
  global pars, data, mc
  #i = len(v.kkpWeight.keys())
  for y, dy in mc.items(): # loop over years
    for m, dm in dy.items(): # loop over mc_std and mc_dg0
      for t, v in dm.items(): # loop over triggers
        if verbose:
          print(f' * Calculating pdfWeight for {m}-{y}-{t} sample')
          print('i pdf weighting', i)
          print('len pdfWeighting', len(v.pdfWeight))
        v.pdfWeight[i] = pdf_reweighting(v,v.params,pars)
  if verbose:
    for y, dy in mc.items(): # loop over years
      print(f'Show 10 fist pdfWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24}")
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0, 10):
        print(f"{dy['MC_Bs2JpsiPhi']['biased'].pdfWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bs2JpsiPhi']['unbiased'].pdfWeight[i][evt]:>+.8f}", end='')
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
        print('i pdf weighting', i)
        # target variables + weight (real data)
        tv = data[y][t].df[['pTHm','pTHp','pHm','pHp']]
        tw = data[y][t].df.eval(f'{weight_rd}')
        # Run multicore (about 15 minutes per iteration)
        #job = multiprocessing.Process(
          #target=kkp_weighting,
          #args=(ov.values, ow.values, tv.values, tw.values, v.path_to_weights,
                #y, m, t, len(threads), verbose)
        #)
        #threads.append(job); job.start()

  # Wait all processes to finish
  #if verbose:
    #print(f' * There are {len(threads)} jobs running in parallel')
  #[thread.join() for thread in threads]



def do_angular_weights(verbose):
  """
  dddd
  """
  global mc

  for y, dy in mc.items(): # loop over years
    for m, dm in dy.items(): # loop over mc_std and mc_dg0
      for t, v in dm.items(): # loop over biased and unbiased triggers
        i = len(v.kkpWeight.keys())+1
        print('i kkp weighting', i)
        path_to_weights = v.path_to_weights.replace('.root',f'_{t}.npy')
        v.kkpWeight[i] = v.pdfWeight[1]
        print(v.kkpWeight[i])
        #v.kkpWeight[i] = np.load(path_to_weights)
        #os.remove(path_to_weights)
        get_angular_acceptance(v, kkpWeight=True)
    if verbose:
      print(f'Show 10 fist kkpWeight[{i}] for {y}')
      print(f"{'MC_Bs2JpsiPhi':<24}")
      print(f"{'biased':<11}  {'unbiased':<11}")
      for evt in range(0,10):
        print(f"{dy['MC_Bs2JpsiPhi']['biased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f"  ", end='')
        print(f"{dy['MC_Bs2JpsiPhi']['unbiased'].kkpWeight[i][evt]:>+.8f}", end='')
        print(f" | ", end='')



def do_mc_combination(verbose):
  """
  Combine
  """
  global mc, data
  checker = []
  for y, dy in mc.items(): # loop over years
    for trigger in ['biased','unbiased']:
      i = len(dy['MC_Bs2JpsiPhi'][trigger].angaccs)
      print('i mc combination', i)
      std = dy['MC_Bs2JpsiPhi'][trigger].angaccs[i]
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
  print(f'{itstr} Simultaneous fit Bs2JpsiPhi {"&".join(list(mc.keys()))}')
  likelihood = do_fit(verbose=verbose)

  #2 pdfWeight MC to RD using pars
  print(f'\n{itstr} PDF weighting MC samples to match Bs2JpsiPhi RD')
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
  print(f'\n{itstr} Combining MC_Bs2JpsiPhi and MC_Bs2JpsiPhi_dG0')
  t0 = timer()
  checker, checker_dict = do_mc_combination(verbose)
  tf = timer()-t0
  print(f'Combining MC_Bs2JpsiPhi and MC_Bs2JpsiPhi_dG0 {tf:.3f} seconds.')

  return likelihood, checker, checker_dict





def lipschitz_iteration(max_iter=30, verbose=False):
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
    ###AÑADIDO para el Latex (borrar antes de gitpush)
    if all(checker) or i>25:
      cols = ['latex', 'value', 'stdev']
      col_offset = 3
      #table= "\\begin{table}[H]\n\centering\n\small\n\\begin{tabular}{l"+"|"+4*"c"+"}\n\\toprule\n"
      table= "\\begin{table}[H]\n\centering\n\small\n\\begin{tabular}{l"+"|"+2*"c"+"}\n\\toprule\n"
      #table += 'weight & 2015 & 2016 & 2017 & 2018\\\ \n \midrule \n'
      table += 'weight & 2017 & 2018\\ \n \midrule \n'
      #line = []
      for trigger in ['biased','unbiased']:
        line = []
        for y, dy in data.items():
            pars_latex = data[y][trigger].angaccs[i]
            par_dict, len_dict = pars_latex._params_to_string_(cols,col_offset)
            for name, par in zip(par_dict.keys(), par_dict.values()):
                line.append('$' + par['value'] + ' \pm ' +  par['stdev']+"$")
        for j in range(10):
            if j==9:
              #table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+' & '+line[20+j]+' & '+line[30+j]+'\\\ \hline \n'
              table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+'\\\ \hline \n'
            else:
              #table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+' & '+line[20+j]+' & '+line[30+j]+'\\\  \n'
              table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+'\\\  \n'
      table = table.replace('None', '0.0')
      table += "\\bottomrule\n\end{tabular}\n"
      table += "\caption{{{caption}}}\n"
      table += "\end{table}\n"
      print(table)

    if all(checker) or i > 25:
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items(): # loop over years
        for trigger in ['biased','unbiased']:
          pars_w = data[y][trigger].angaccs[i]
          print(pars_w)
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
              #aitken = x2 - ((x1-x0)**2 ) / den #ruiz

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
    ###AÑADIDO para el Latex
    if all(checker) or i>25:
      cols = ['latex', 'value', 'stdev']
      col_offset = 3
      table= "\\begin{table}[H]\n\centering\n\small\n\\begin{tabular}{l"+"|"+4*"c"+"}\n\\toprule\n"
      table += 'weight & 2015 & 2016 & 2017 & 2018\\\ \n \midrule \n'
      for trigger in ['biased','unbiased']:
        line = []
        for y, dy in data.items():
            pars_latex = data[y][trigger].angaccs[i]
            par_dict, len_dict = pars_latex._params_to_string_(cols,col_offset)
            for name, par in zip(par_dict.keys(), par_dict.values()):
                line.append('$' + par['value'] + ' \pm ' +  par['stdev']+"$")
        for j in range(10):
            if j==9:
              #table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+' & '+line[20+j]+' & '+line[30+j]+'\\\ \hline \n'
              table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+'\\\ \hline \n'
            else:
              #table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+' & '+line[20+j]+' & '+line[30+j]+'\\\  \n'
              table += f'$w^{trigger[0]}_{j}$ & ' +line[j]+' & '+line[10+j]+'\\\  \n'
      table = table.replace('None', '0.0')
      table += "\\bottomrule\n\end{tabular}\n"
      table += "\caption{{{caption}}}\n"
      table += "\end{table}\n"
      print(table)
    if all(checker) or i > 25:
      print(f"\nDone! Convergence was achieved within {i} iterations")
      for y, dy in data.items(): # loop over years
        for trigger in ['biased','unbiased']:
          pars_w = data[y][trigger].angaccs[i]
          print(pars_w)
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



























################################################################################
# Run and get the job done #####################################################

if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-coeffs-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-coeffs-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-time-resolution', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-weights-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angular-weights-mc-std', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--mode', help='Mode of the sample')
  args = vars(p.parse_args())

  VERSION, SHARE,EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',') 
  #YEARS = ['2016']
  YEARS = ['2015']#, '2016']
  MODE = args['mode']
  ANGACC = args['angacc']
  bkgcat = False
  # Get badjanak model and configure it ----------------------------------------
  #initialize(os.environ['IPANEMA_BACKEND'], 1 if YEARS in (2015,2017) else -1)

  # Prepare the cuts -----------------------------------------------------------
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = cuts_and(CUT,f'time>={tLL} & time<={tUL}')

  # List samples, params and tables --------------------------------------------
  samples_std   = args['sample_mc_std'].split(',')
  samples_std = [samples_std[0]]
  #samples_std = samples_std[0:1]
  input_std_params = args['params_mc_std'].split(',')
  input_std_params = [input_std_params[0]]
  #input_std_params = input_std_params[0:1]
  coeffs_biased      = args['input_coeffs_biased'].split(',')
  coeffs_biased = [coeffs_biased[0]]
  #coeffs_biased = coeffs_biased[0:1]
  coeffs_unbiased    = args['input_coeffs_unbiased'].split(',')
  coeffs_unbiased = [coeffs_unbiased[0]]
  #coeffs_unbiased = coeffs_unbiased[0:2]
  time_resolution = args['input_time_resolution'].split(',')
  #time_resolution = time_resolution[0:2]
  time_resolution = [time_resolution[0]]
  params_biased      = args['output_weights_biased'].split(',')
  params_unbiased    = args['output_weights_unbiased'].split(',')
  tables_biased      = args['output_tables_biased'].split(',')
  tables_unbiased    = args['output_tables_unbiased'].split(',')

  kkpWeight_std = args['output_angular_weights_mc_std'].split(',')

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'version':>15}: {VERSION:50}")
  print(f"{'year(s)':>15}: {args['year']:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC:50}")
  print(f"{'bdtconfig':>15}: {':'.join(str(x) for x in bdconfig.values()):50}\n")



  # Load samples ---------------------------------------------------------------
  printsec('Loading samples')

  global mc, data, weight_rd

  # MC reconstructed and generator level variable names
  reco  = ['cosK', 'cosL', 'hphi', 'time']
  true  = [f'gen{i}' for i in reco]
  #reco = ['cosK', 'cosL', 'hphi', 'gentime']
  reco += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']
  true += ['mHH', '0*sigmat', 'genidB', 'genidB', '0*time', '0*time']

  # RD variable names
  real  = ['cosK','cosL','hphi','time']
  real += ['mHH','sigmat', 'genidB','genidB', '0*time', '0*time']

  # sWeight variable
  weight_rd = f'sw/gb_weights'
  weight_mc = f'sw/gb_weights'
  if bkgcat == True:
    weight_rd = f'time/time'
    weight_mc = f'time/time'
  #weight_rd = f'sw/gb_weights'
  #weight_mc = f'sw/gb_weights'
  # Load Monte Carlo samples
  mc = {}
  mcmodes = ['MC_Bs2JpsiPhi']

  printsubsec('Loading MC samples')
  for i, y in enumerate(YEARS):
    mc[y] = {}
    for m, v in zip(mcmodes,[samples_std]):
      mc[y][m] = {'biased':   Sample.from_root(v[i], share=SHARE),
                  'unbiased': Sample.from_root(v[i], share=SHARE)}
      mc[y][m]['biased'].name = f"{m}-{y}-biased"
      mc[y][m]['unbiased'].name = f"{m}-{y}-unbiased"

    for m, v in zip(mcmodes,[input_std_params]):
      mc[y][m]['biased'].assoc_params(v[i])
      mc[y][m]['unbiased'].assoc_params(v[i])
      if bkgcat ==True:
        mc[y][m]['biased'].chop(cuts_and(trigger_scissors('biased'), CUT, 'bkgcatB != 60.0','(evtN % 2) == 0'))
        mc[y][m]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'),CUT, 'bkgcatB != 60.0','(evtN % 2) == 0'))
      else:
          mc[y][m]['biased'].chop(cuts_and(trigger_scissors('biased'), CUT,'(evtN % 2) == 0'))
          mc[y][m]['unbiased'].chop(cuts_and(trigger_scissors('unbiased'),CUT,'(evtN % 2) == 0'))
      for t in ['biased', 'unbiased']:
        mc[y][m][t].allocate(reco=reco, true=true, pdf='0*time', weight=weight_mc)
        mc[y][m][t].df['angWeight'] = 0.0
        mc[y][m][t].angaccs = {}
        mc[y][m][t].kkpWeight = {}
        mc[y][m][t].pdfWeight = {}
    for m, v in zip(mcmodes,[kkpWeight_std]):
      mc[y][m]['biased'].path_to_weights = v[i]
      mc[y][m]['unbiased'].path_to_weights = v[i]
  #exit()
  # Load corresponding data sample
  global params_init
  params_init = Parameters.load(input_std_params[0])
  mass = badjanak.config['mHH']
  data = {}

  printsubsec('Loading MC as data samples')
  for i, y in enumerate(YEARS):
    data[y] = {}
    resolution = Parameters.load(time_resolution[i])
    for key in resolution.keys():
        print(resolution[key].value)
    for t in ['biased','unbiased']:
      data[y][t] = Sample.from_root(samples_std[i], share=SHARE)
      print(data[y][t])
      data[y][t].name = f"{m}-{y}-{t}"
      data[y][t].resolution = resolution
      print(data[y][t].resolution)
    for t, coeffs in zip(['biased','unbiased'],[coeffs_biased,coeffs_unbiased]):
      c = Parameters.load(coeffs[i])
      data[y][t].knots = Parameters.build(c,c.fetch('k.*'))
      badjanak.config['knots'] = np.array( data[y][t].knots ).tolist()
      data[y][t].timeacc = Parameters.build(c,c.fetch('a.*'))
      if bkgcat==True:
        data[y][t].chop(cuts_and(trigger_scissors(t), CUT,'bkgcatB != 60.0', '(evtN % 2) != 0'))#, 'logIPchi2B >= 0', 'log(BDTFchi2) >=0'))
        print(data[y][t])
      else:
          data[y][t].chop(cuts_and(trigger_scissors(t), CUT, ('evtN % 2 != 0')))#,'logIPchi2B >= 0', 'log(BDTFchi2) >=0'))
          #, 'logIPchi2B >= 0'))#, 'log(BDTFchi2) >=0'))
    for t, path in zip(['biased','unbiased'],[params_biased,params_unbiased]):
      data[y][t].params_path = path[i]

    for t, path in zip(['biased','unbiased'],[tables_biased,tables_unbiased]):
      data[y][t].tables_path = path[i]
    #exit()


  ##Calculate the corrected that is obtained from a part of the MC
  for i,y in enumerate (YEARS):
      for t in ['biased','unbiased']:
        print('Compute angWeights correcting MC sample in kinematics')
        print(f" * Computing kinematic GB-weighting in pTB, pB and mHH")
        reweighter.fit(original        = mc[y][m][t].df[['mHH','pB','pTB']],
                        target          = data[y][t].df[['mHH','pB','pTB']],
                        original_weight = mc[y][m][t].df.eval(weight_mc),
                        target_weight   = data[y][t].df.eval(weight_rd));
        angWeight = reweighter.predict_weights(mc[y][m][t].df[['mHH', 'pB', 'pTB']])
        #angWeight = np.ones_like(mc[y][m][t].df['angWeight'])
        mc[y][m][t].df['angWeight'] = angWeight
        mc[y][m][t].olen = len(angWeight)
        print(mc[y][m][t].df[['sw', 'angWeight']])
        print('mc params')
        print(mc[y][m][t].params.valuesdict())
        angacc = badjanak.get_angular_acceptance_weights(mc[y][m][t].true, mc[y][m][t].reco,
                                     mc[y][m][t].weight*ristra.allocate(angWeight),
                             **mc[y][m][t].params.valuesdict())
        w, uw, cov, corr = angacc
        pars_w = Parameters()
        for i in range(0,len(w)):
          correl = {f'w{j}': corr[i][j]
                    for j in range(0, len(w)) if i > 0 and j > 0}
          pars_w.add({'name': f'w{i}', 'value': w[i], 'stdev': uw[i],
                    'correl': correl, 'free': False, 'latex': f'w_{i}'})
        print(f" * Corrected angular weights for {MODE}{y}-{t} sample are:")
        print(f"{pars_w}")
        data[y][t].angacc = pars_w
        data[y][t].angaccs = {0:pars_w}
  exit()

  for i, y in enumerate(YEARS):
    for d in [data[y]['biased'],data[y]['unbiased']]:
      #sw = np.zeros_like(d.df[f'{weight_rd}'])
      sw = np.zeros_like(d.df[f'sw'])
      for l,h in zip(mass[:-1],mass[1:]):
        pos = d.df.eval(f'mHH>={l} & mHH<{h}')
        this_sw = d.df.eval(f'{weight_rd}*(mHH>={l} & mHH<{h})')
        sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
      d.df['sWeight'] = sw
      d.allocate(data=real,weight='sWeight',lkhd='0*time')


  # Prepare dict of parameters -------------------------------------------------
  printsec('Parameters and initial status')

  print(f"\nFitting parameters\n{80*'='}")
  global pars; pars = Parameters()

  # P wave fractions
  pars.add(dict(name="fPlon", value=0.521284,
            free=True, latex=r'f_0'))
  pars.add(dict(name="fPper", value=0.249001,
            free=True, latex=r'f_{\perp}'))

  # Weak phases
  pars.add(dict(name="pSlon", value= 0.00, min=-1, max=+1, #min -1 max +1
            free=False, latex=r"\phi_S - \phi_0"))
  pars.add(dict(name="pPlon", value= 0.07, min=-1, max=+1,
            free=True , latex=r"\phi_0" ))
  pars.add(dict(name="pPpar", value= 0.00, min=-1, max=+1,
            free=False, latex=r"\phi_{\parallel} - \phi_0"))
  pars.add(dict(name="pPper", value= 0.00, min=-1, max=+1,
            free=False, latex=r"\phi_{\perp} - \phi_0"))



  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.000, min=-2*3.14, max=2*3.14,
            free=False, latex=r"\delta_0"))
  pars.add(dict(name="dPpar", value=3.30,  
            free=True, latex=r"\delta_{\parallel} - \delta_0"))
  pars.add(dict(name="dPper", value=3.07, 
            free=True, latex=r"\delta_{\perp} - \delta_0"))

  # lambdas
  pars.add(dict(name="lSlon", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_S/\lambda_0"))
  pars.add(dict(name="lPlon", value=1.0,#1.094, #min=0.7, max=2.0,  #0.8 for evtN ==2 1.0 for evtN != 2 tol=0.0001 strategy = 2 |si fijas dpper, min 1.0
            free=True,  latex="\lambda_0"))
  pars.add(dict(name="lPpar", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\parallel}/\lambda_0"))
  pars.add(dict(name="lPper", value=1.0, min=0.7, max=1.6,
            free=False, latex="\lambda_{\perp}/\lambda_0"))

  # life parameters
  pars.add(dict(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
            free=False, latex=r"\Gamma_d")) 
  pars.add(dict(name="DGs", value= 0.0, min= -0.15, max= 0.15,
                free=True, latex=r"\Delta\Gamma_s"))
  pars.add(dict(name="DGsd", value= 0.0034,
            free=True, latex=r"\Gamma_s - \Gamma_d"))
  pars.add(dict(name="DM", value=17.80, min=16.0, max=19.0,
            free=True, latex=r"\Delta m"))
  print(pars)
  #Esto unicamente es para tener unos parámetros fijos con los valores generados
   
  # print time acceptances
  lb = [ data[y]['biased'].timeacc.__str__(['value']).splitlines() for i,y in enumerate( YEARS ) ]
  lu = [ data[y]['unbiased'].timeacc.__str__(['value']).splitlines() for i,y in enumerate( YEARS ) ]
  print(f"\nBiased time acceptance\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  print(f"\nUnbiased time acceptance\n{80*'='}")
  for l in zip(*lu):
    print(*l, sep="| ")

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
  
  global params_forPlotting
  params_forPlotting = {}

  # The iterative procedure starts ---------------------------------------------

  # update kernels with the given modifications
  badjanak.get_kernels(True)

  # run the procedure!


  ok, likelihoods = lipschitz_iteration(max_iter=5, verbose=True)

  if not ok:
    ok, likelihoods = aitken_iteration(max_iter=30, verbose=True)

  if not ok:
    print('WARNING: Convergence was not achieved!')

  if ok:
    names = ['fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGs', 'DGsd', 'DM']
    # values, oldvalues, std = [], [], []
    # for p in names:
    #   values.append(pars[p].value)
    #   oldvalues.append(params_init[p].value)
    #   std.append(pars[p].stdev)
    # df = pd.DataFrame({"names": names, "values": values, "std": std, "oldvalues": oldvalues})
    # df['PULL'] = df.eval('(values-oldvalues)/sqrt(std**2)')
    # print('PULL with gen values of MC')
    # print(df)
    # print('DataFrame to Latex')
    # print(df.to_latex(index=False))
    #Plotting parameters in each iteration
    #print(params_forPlotting)
    for key in names:
      y, y_err = [],[]
      x = np.arange(0, len(params_forPlotting))
      z = [params_init[key].value]*len(x)
      plt.close()
      for i in range(0, len(params_forPlotting)):
        y.append(params_forPlotting[i][key].value)
        y_err.append(params_forPlotting[i][key].stdev)
        print('Values for Plotting', y)
        print('Stdev for Plotting', y_err)
      plt.xlim(x[0]-0.05, x[-1]+0.05)
      plt.ylim(min(z+y)-1.1*max(y_err), max(z+y)+1.1*max(y_err))
      plt.errorbar(x,y, yerr=y_err, fmt='*g')
      x = np.insert(x,0,-0.05)
      x = np.append(x, len(params_forPlotting)-0.95)
      z.extend([params_init[key].value]*2)
      print(x)
      print(z)
      plt.plot(x,z, '-r')
      plt.title(f"{key}")
      plt.savefig(f'/home3/ramon.ruiz/phis-scq/logs_diego_test/figures/{MODE}/bkgcat60/{key}.pdf')



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
