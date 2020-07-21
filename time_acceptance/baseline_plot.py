# -*- coding: utf-8 -*-

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import argparse
import numpy as np
import pandas as pd
import uproot
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import uncertainties as unc
from uncertainties import unumpy as unp

from ipanema import ristra, initialize
from ipanema import Parameters, fit_report, optimize
from ipanema import histogram
from ipanema import Sample
from ipanema import plotting
from ipanema import wrap_unc, get_confidence_bands

from utils.plot import watermark
from utils.strings import cammel_case_split
from utils.plot import mode_tex

initialize(os.environ['IPANEMA_BACKEND'],1)
from time_acceptance.single import bsjpsikk
from time_acceptance.single import splinexerf
from time_acceptance.simultaneous import saxsbxscxerf




def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--MC-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi_dG0/v0r5.root',
    help='MC sample path')
  parser.add_argument('--CMC-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/MC_Bd2JpsiKstar/v0r5.root',
    help='MC control sample path')
  parser.add_argument('--CRD-sample',
    default = '/scratch17/marcos.romero/phis_samples/2016/Bd2JpsiKstar/v0r5.root',
    help='Real data control sample path')
  # Output parameters
  parser.add_argument('--MC-params',
    default = 'output/params/time_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_Baseline_biased.json',
    help='MC time acceptance coefficients path')
  parser.add_argument('--CMC-params',
    default = 'output/params/time_acceptance/2016/MC_Bd2JpsiKstar/v0r5_Baseline_biased.json',
    help='MC control time acceptance coefficients path')
  parser.add_argument('--CRD-params',
    default = 'output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_Baseline_biased.json',
    help='Real data control time acceptance coefficients path')
  # Output parameters
  parser.add_argument('--figure',
    default = 'output/figures/time_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_Baseline_biased_Fit.json',
    help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
    default = 'MC_Bs2JpsiPhi_dG0',
    help='Configuration')
  parser.add_argument('--year',
    default = '2016',
    help='Year of data-taking')
  parser.add_argument('--version',
    default = 'v0r5',
    help='Year of data-taking')
  parser.add_argument('--trigger',
    default = 'biased',
    help='Trigger(s) to fit [comb/(biased)/unbiased]')
  # Report
  parser.add_argument('--script',
    default = 'Baseline',
    help='Save a fit report with the results')
  parser.add_argument('--figure',
    default = 'Fit',
    help='dsfsdf')

  return parser

################################################################################





def get_sample_info(sample, script):
  if 'MC_Bs2JpsiPhi' in sample:
    if 'MC_Bs2JpsiPhi_dG0' in sample:
      mode = 'MC_Bs2JpsiPhi_dG0'
      if script == 'Baseline':
        weight='(sw/gb_weights)*polWeight*pdfWeight*kinWeight'
      elif script == 'Nonkinweighted':
        weight='(sw/gb_weights)'
    else:
      mode = 'MC_Bs2JpsiPhi'
  elif 'MC_Bd2JpsiKstar' in sample:
    mode = 'MC_Bd2JpsiKstar'
    if script == 'Baseline':
      weight='sw*polWeight*pdfWeight*kinWeight'
    elif script == 'Nonkinweighted':
      weight='sw'
  elif 'Bd2JpsiKstar' in sample:
    mode = 'Bd2JpsiKstar'
    if script == 'Baseline':
      weight='sw*kinWeight'
    elif script == 'Nonkinweighted':
      weight='sw'
  else:
    print('ERROR')
  label = mode_tex(mode).split('\,')
  return mode, weight, label

print(f"\n{80*'='}\n", "Loading categories", f"\n{80*'='}\n")
samples = {}
for n in ['MC','CMC','CRD']:
  print( get_sample_info(args[f'{n}_sample'], SCRIPT) )
  tmp = Sample.from_root(args[f'{n}_sample'], cuts=cuts)
  # get some info
  mode, weight_str, label = get_sample_info(args[f'{n}_sample'], SCRIPT)
  tmp.mode = mode; tmp.label = label
  # allocate arrays
  tmp.allocate(time='time')
  tmp.allocate(time='time',lkhd='0*time',weight=weight_str)
  tmp.weight *= ristra.sum(tmp.weight)/ristra.sum(tmp.weight**2)
  # add parameters
  tmp.assoc_params(args[f'{n}_params'])
  knots = tmp.params.find('k.*') + ['tLL','tUL']
  tmp.knots = Parameters.build(tmp.params, knots)
  [tmp.params.pop(k, None) for k in knots]
  # for p in tmp.params:
  #   if p.startswith('a') or p.startswith('b') or p.startswith('c'):
  #     tmp.params[p].value = 1.0
  #     tmp.params[p].init = 1.0
  tmp.figure_path = args['figure']
  tmp.name = '\,'.join(tmp.label)
  tmp.kind = n
  samples[n] = tmp






#%% Hey ho!

def plot_splinexerf(sample, ylog=False, figname='test.pdf', version=None):
  # Histo
  ref = histogram.hist(ristra.get(sample.time), weights=ristra.get(sample.weight), bins=100)

  # Plot p.d.f.
  x = np.linspace(sample.knots['tLL'].value,sample.knots['tUL'].value,200)
  y = splinexerf(sample.params, ristra.allocate(x) )
  y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))

  # Actual plot
  fig, axplot, axpull = plotting.axes_plotpull();
  axplot.plot(x,y)
  axpull.fill_between(ref.cmbins,
                      histogram.pull_pdf(x,y,ref.cmbins,ref.counts,ref.errl,ref.errh),
                      0, facecolor="C0")
  axplot.errorbar(ref.cmbins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
  if ylog:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  if version:
    watermark(axplot, version=f'${version}$')
  return fig, axplot, axpull


plot_splinexerf(samples['MC'], ylog=False, figname='test.pdf', version='caca')



def plot_saxsbxscxerf(sample, ylog=False, figname='test.pdf', version=None, k=0):
  idx = {'MC':0, 'CMC':1, 'CRD':2}[k]
  # Histo
  ref = histogram.hist(ristra.get(sample[k].time), weights=ristra.get(sample[k].weight), bins=100)
  # Plot p.d.f.
  x = np.linspace(sample[k].knots['tLL'].value,sample[k].knots['tUL'].value,200)
  P = sample['MC'].params+sample['CMC'].params+sample['CRD'].params
  y = saxsbxscxerf(P, [x,x,x])[idx]
  y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
  # Actual plot
  fig, axplot, axpull = plotting.axes_plotpull();
  axplot.plot(x,y)
  axpull.fill_between(ref.cmbins,
                      histogram.pull_pdf(x,y,ref.cmbins,ref.counts,ref.errl,ref.errh),
                      0, facecolor="C0")
  axplot.errorbar(ref.cmbins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
  if ylog:
    axplot.set_yscale('log')
  axpull.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'Weighted candidates')
  if version:
    watermark(axplot, version=f'${version}$')
  return fig, axplot, axpull








plot_saxsbxscxerf(samples, ylog=False, figname='test.pdf', version='caca', k='MC')
plot_saxsbxscxerf(samples, ylog=False, figname='test.pdf', version='caca', k='CMC')
plot_saxsbxscxerf(samples, ylog=False, figname='test.pdf', version='caca', k='CRD')

samples



def plot_spline(sample, xlog=False, version=None, k='MC', bins = 30, cl=1,
                ax=None):
  if isinstance(sample,list) or isinstance(sample,tuple):
    lsample = sample
  elif isinstance(sample,dict):
    lsample = list(sample.values())
  else:
    print('no method for this sample')
    return 0
  #print(len(lsample))
  n_samples = len(lsample)
  idx = min({'MC':0, 'CMC':1, 'CRD':2}[k],n_samples)
  gamma = lsample[idx].params[ lsample[idx].params.find('gamma.*')[0] ].value
  time = lsample[idx].time; weight = lsample[idx].weight; coeffs = []
  tLL = lsample[idx].knots['tLL'].value; tUL = lsample[idx].knots['tUL'].value
  label = lsample[idx].label

  #print(f'idx= {idx}')
  for i in range(0,idx+1):
    arr = []
    #print(f'sample {i}')
    for p in lsample[i].params.find("(a|b|c).*"):
      #print(lsample[i].params[p].uvalue)
      arr.append(lsample[i].params[p].uvalue)
    coeffs.append( arr )
  """
    n_samples = len(sample.keys())
    idx = min({'MC':0, 'CMC':1, 'CRD':2}[k],n_samples)
    gamma = sample[k].params[ sample[k].params.find('gamma.*')[0] ].value
    time = sample[k].time; weight = sample[k].weight; coeffs = []
    tLL = sample[k].knots['tLL'].value; tUL = sample[k].knots['tUL'].value

    for i in range(0,idx+1):
      arr = []
      for p in sample[k].params.find("(a|b|c).*"):
        if sample[k].params[p].stdev:
          arr.append(unc.ufloat(sample[k].params[p].value,sample[k].params[p].stdev))
        else:
          arr.append(unc.ufloat(sample[k].params[p].value,0))
      coeffs.append( arr )
  """
  def distfunction(tLL, tUL, gamma, ti, nob):
    return np.log(-((np.exp(gamma*ti + gamma*tLL + gamma*tUL)*nob)/
    (-np.exp(gamma*ti + gamma*tLL) + np.exp(gamma*ti + gamma*tUL) -
      np.exp(gamma*tLL + gamma*tUL)*nob)))/gamma
  list_bins = [tLL]; ipdf = []; widths = []
  dummy = 0.4; # this is a general gamma to distribute the bins
  for k in range(0,bins):
    ti = list_bins[k]
    list_bins.append( distfunction(tLL, tUL, dummy, ti, bins)   )
    tf = list_bins[k+1]
    ipdf.append( 1.0/((-np.exp(-(tf*gamma)) + np.exp(-(ti*gamma)))/1.0) )
    widths.append(tf-ti)
  bins = np.array(list_bins); int_pdf = np.array(ipdf)

  # Manipulate the decay-time dependence of the efficiency
  x = np.linspace(tLL,tUL,200)
  y = wrap_unc(bsjpsikk.bspline, x, *coeffs[-1])
  coeffs.pop(-1)
  y_nom = unp.nominal_values(y)
  y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)
  #print(coeffs)

  # Manipulate data
  ref = histogram.hist(ristra.get(time), bins=bins, weights=ristra.get(weight))
  ref.counts *= int_pdf; ref.errl *= int_pdf; ref.errh *= int_pdf
  if coeffs:
    spline = bsjpsikk.bspline(ref.cmbins,*[c.n for c in coeffs[-1]])
    ref.counts /= spline; ref.errl /= spline; ref.errh /= spline
  counts_spline = interp1d(ref.bins, ref.counts, kind='cubic')
  int_5 = counts_spline(5)
  ref.counts /= int_5; ref.errl /= int_5; ref.errh /= int_5

  # Actual ploting
  axplot.set_ylim(0.45, 1.45)
  axplot.plot(x,y_nom/y_spl)
  axplot.errorbar(ref.cmbins,ref.counts,
                  yerr=[ref.errl,ref.errh],
                  xerr=[-ref.edges[:-1]+ref.cmbins,-ref.cmbins+ref.edges[1:]],
                  fmt='.', color='k')
  y_upp, y_low = get_confidence_bands(x,y, sigma=cl)
  axplot.fill_between(x, y_upp/y_spl, y_low/y_spl, alpha=0.2, edgecolor="none",
                      label='$'+str(cl)+'\sigma$ band')
  axplot.set_xlabel(r'$t$ [ps]')
  axplot.set_ylabel(r'$\epsilon^{%s}_{%s}$ [a.u.]' % tuple(label))
  axplot.legend()
  return axplot


fig, axplot = plotting.axes_plot()
plot_spline(samples, version='$v0r5$', k='CMC', bins = 25, cl=1, ax=axplot)





################################################################################

if __name__ == '__main__':
  # Parse arguments
  args = vars(argument_parser().parse_args(''))
  VERSION = args['version']
  YEAR = args['year']
  MODE = args['mode']
  TRIGGER = args['trigger']
  SCRIPT = args['script']
  FIGURE = args['figure']

  # Select trigger to fit
  if args['trigger'] == 'biased':
    trigger = 'biased'; cuts = "time>=0.3 & time<=15 & hlt1b==1"
  elif args['trigger'] == 'unbiased':
    trigger = 'unbiased'; cuts = "time>=0.3 & time<=15 & hlt1b==0"
  elif args['trigger'] == 'combined':
    trigger = 'combined'; cuts = "time>=0.3 & time<=15"






  # Plotting -------------------------------------------------------------------
  print(f"\n{80*'='}\n{'= Plotting':79}=\n{80*'='}\n")

  for name, cat in zip(cats.keys(),cats.values()):
    plot_fcn_spline(
      result.params,
      cat.time.get(),
      cat.weight.get(),
      name = os.path.join(cat.figs_path,f'{VERSION}_baseline_{trigger}_fit_log.pdf'),
      log=True
    )
    print(f"Plotted {os.path.join(cat.figs_path,f'{VERSION}_baseline_{trigger}_fit_log.pdf')}")

  for name, cat in zip(cats.keys(),cats.values()):
    plot_fcn_spline(
      result.params,
      cat.time.get(),
      cat.weight.get(),
      name = os.path.join(cat.figs_path,f'{VERSION}_baseline_{trigger}_fit.pdf'),
      log=False
    )
    print(f"Plotted {os.path.join(cat.figs_path,f'{VERSION}_baseline_{trigger}_fit.pdf')}")


  # BsMC
  plot_spline(# BsMC
    cats['BsMC'].params,
    cats['BsMC'].time.get(),
    cats['BsMC'].weight.get(),
    name = os.path.join(cats['BsMC'].figs_path,f'{VERSION}_baseline_{trigger}_spline.pdf'),
    label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
    conf_level=1,
    bins=25
  )

  # ratio
  plot_spline(# BsMC
    cats['BsMC'].params+cats['BdMC'].params,
    cats['BdMC'].time.get(),
    cats['BdMC'].weight.get(),
    name = os.path.join(cats['BdMC'].figs_path,f'{VERSION}_baseline_{trigger}_spline.pdf'),
    label=r'$\varepsilon_{\mathrm{MC}}^{B_d}/\varepsilon_{\mathrm{MC}}^{B_s^0}$',
    conf_level=1,
    bins=25
  )

  # BsDT
  plot_spline(# BsMC
    cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params,
    cats['BdDT'].time.get(),
    cats['BdDT'].weight.get(),
    name = os.path.join(cats['BdDT'].figs_path,f'{VERSION}_baseline_{trigger}_spline.pdf'),
    label=r'$\varepsilon_{\mathrm{data}}^{B_s^0}$',
    conf_level=1,
    bins=25
  )

  print(f"Splines were plotted!")
