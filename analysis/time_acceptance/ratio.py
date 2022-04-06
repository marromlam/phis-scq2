__all__ = []
# THIS FILE IS DEPRECATED # -*- coding: utf-8 -*-
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED #                    DECAY TIME ACCEPTANCE                                     #
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED #                                                                              #
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED __author__ = ['Marcos Romero']
# THIS FILE IS DEPRECATED __email__  = ['mromerol@cern.ch']
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED # %% Modules ###################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED import argparse
# THIS FILE IS DEPRECATED import numpy as np
# THIS FILE IS DEPRECATED import matplotlib.pyplot as plt
# THIS FILE IS DEPRECATED import pandas as pd
# THIS FILE IS DEPRECATED import uproot
# THIS FILE IS DEPRECATED import os, sys
# THIS FILE IS DEPRECATED import platform
# THIS FILE IS DEPRECATED import hjson
# THIS FILE IS DEPRECATED import pandas
# THIS FILE IS DEPRECATED import importlib
# THIS FILE IS DEPRECATED from scipy.interpolate import interp1d
# THIS FILE IS DEPRECATED import uncertainties as unc
# THIS FILE IS DEPRECATED from uncertainties import unumpy as unp
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED from ipanema import ristra
# THIS FILE IS DEPRECATED from ipanema import Parameters, fit_report, optimize
# THIS FILE IS DEPRECATED from ipanema import histogram
# THIS FILE IS DEPRECATED from ipanema import Sample
# THIS FILE IS DEPRECATED from ipanema import plotting
# THIS FILE IS DEPRECATED from ipanema import wrap_unc, get_confidence_bands
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def argument_parser():
# THIS FILE IS DEPRECATED   parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
# THIS FILE IS DEPRECATED   # Samples
# THIS FILE IS DEPRECATED   parser.add_argument('--BsMC-sample',
# THIS FILE IS DEPRECATED                       default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
# THIS FILE IS DEPRECATED                       help='Bs2JpsiPhi MC sample')
# THIS FILE IS DEPRECATED   parser.add_argument('--BdMC-sample',
# THIS FILE IS DEPRECATED                       default = 'samples/MC_Bd2JpsiKstar_2016_test.json',
# THIS FILE IS DEPRECATED                       help='Bd2JpsiKstar MC sample')
# THIS FILE IS DEPRECATED   parser.add_argument('--BdDT-sample',
# THIS FILE IS DEPRECATED                       default = 'samples/Bd2JpsiKstar_2016_test.json',
# THIS FILE IS DEPRECATED                       help='Bd2JpsiKstar data sample')
# THIS FILE IS DEPRECATED   # Output parameters
# THIS FILE IS DEPRECATED   parser.add_argument('--BsMC-params',
# THIS FILE IS DEPRECATED                       default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
# THIS FILE IS DEPRECATED                       help='Bs2JpsiPhi MC sample')
# THIS FILE IS DEPRECATED   parser.add_argument('--BdMC-params',
# THIS FILE IS DEPRECATED                       default = 'output/time_acceptance/parameters/2016/MC_Bs_Bd_ratio/test_biased.json',
# THIS FILE IS DEPRECATED                       help='Bd2JpsiKstar MC sample')
# THIS FILE IS DEPRECATED   parser.add_argument('--BdDT-params',
# THIS FILE IS DEPRECATED                       default = 'output/time_acceptance/parameters/2016/Bs2JpsiPhi/test_biased.json',
# THIS FILE IS DEPRECATED                       help='Bd2JpsiKstar data sample')
# THIS FILE IS DEPRECATED   # Configuration file ---------------------------------------------------------
# THIS FILE IS DEPRECATED   parser.add_argument('--mode',
# THIS FILE IS DEPRECATED                       default = 'baseline',
# THIS FILE IS DEPRECATED                       help='Configuration')
# THIS FILE IS DEPRECATED   parser.add_argument('--year',
# THIS FILE IS DEPRECATED                       default = '2016',
# THIS FILE IS DEPRECATED                       help='Year of data-taking')
# THIS FILE IS DEPRECATED   parser.add_argument('--flag',
# THIS FILE IS DEPRECATED                       default = 'test',
# THIS FILE IS DEPRECATED                       help='Year of data-taking')
# THIS FILE IS DEPRECATED   parser.add_argument('--trigger',
# THIS FILE IS DEPRECATED                       default = 'biased',
# THIS FILE IS DEPRECATED                       help='Trigger(s) to fit [comb/(biased)/unbiased]')
# THIS FILE IS DEPRECATED   # Report
# THIS FILE IS DEPRECATED   parser.add_argument('--pycode',
# THIS FILE IS DEPRECATED                       default = 'baseline',
# THIS FILE IS DEPRECATED                       help='Save a fit report with the results')
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   return parser
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #%% Configuration ##############################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Parse arguments
# THIS FILE IS DEPRECATED args = vars(argument_parser().parse_args())
# THIS FILE IS DEPRECATED PATH = os.path.abspath(os.path.dirname(args['pycode']))
# THIS FILE IS DEPRECATED NAME = os.path.splitext(os.path.basename('time/baseline.py'))[0]
# THIS FILE IS DEPRECATED FLAG = args['flag']
# THIS FILE IS DEPRECATED YEAR = args['year']
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Select trigger to fit
# THIS FILE IS DEPRECATED if args['trigger'] == 'biased':
# THIS FILE IS DEPRECATED   trigger = 'biased'; cuts = "time>=0.3 & time<=15 & hlt1b==1"
# THIS FILE IS DEPRECATED elif args['trigger'] == 'unbiased':
# THIS FILE IS DEPRECATED   trigger = 'unbiased'; cuts = "time>=0.3 & time<=15 & hlt1b==0"
# THIS FILE IS DEPRECATED elif args['trigger'] == 'comb':
# THIS FILE IS DEPRECATED   trigger = 'comb'; cuts = "time>=0.3 & time<=15"
# THIS FILE IS DEPRECATED   print('not implemented!')
# THIS FILE IS DEPRECATED   exit()
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED print(f"\n{80*'='}\n{'= Settings':79}=\n{80*'='}\n")
# THIS FILE IS DEPRECATED print(f"{'path':>15}: {PATH:50}")
# THIS FILE IS DEPRECATED print(f"{'script':>15}: {NAME:50}")
# THIS FILE IS DEPRECATED print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
# THIS FILE IS DEPRECATED print(f"{'trigger':>15}: {args['trigger']:50}")
# THIS FILE IS DEPRECATED print(f"{'cuts':>15}: {cuts:50}\n")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Initialize backend
# THIS FILE IS DEPRECATED from ipanema import initialize
# THIS FILE IS DEPRECATED initialize(os.environ['IPANEMA_BACKEND'],2)
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Get bsjpsikk model and configure it
# THIS FILE IS DEPRECATED import bsjpsikk
# THIS FILE IS DEPRECATED bsjpsikk.use_time_acc = 0,
# THIS FILE IS DEPRECATED bsjpsikk.use_time_offset = 0
# THIS FILE IS DEPRECATED bsjpsikk.use_time_res = 0
# THIS FILE IS DEPRECATED bsjpsikk.use_perftag = 0
# THIS FILE IS DEPRECATED bsjpsikk.use_truetag = 1
# THIS FILE IS DEPRECATED bsjpsikk.get_kernels()
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #%% Likelihood functions to minimize ###########################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def lkhd_single_spline(parameters, data, weight = None, prob = None):
# THIS FILE IS DEPRECATED   pars_dict = list(parameters.valuesdict().values())
# THIS FILE IS DEPRECATED   #print(pars_dict)
# THIS FILE IS DEPRECATED   if not prob: # for ploting, mainly
# THIS FILE IS DEPRECATED     data = ristra.allocate(data)
# THIS FILE IS DEPRECATED     prob = ristra.allocate(np.zeros_like(data.get()))
# THIS FILE IS DEPRECATED     bsjpsikk.single_spline_time_acceptance(data, prob, *pars_dict)
# THIS FILE IS DEPRECATED     return prob.get()
# THIS FILE IS DEPRECATED   else:
# THIS FILE IS DEPRECATED     bsjpsikk.single_spline_time_acceptance(data, prob, *pars_dict)
# THIS FILE IS DEPRECATED     if weight is not None:
# THIS FILE IS DEPRECATED       result = (ristra.log(prob)*weight).get()
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       result = (ristra.log(prob)).get()
# THIS FILE IS DEPRECATED     return -2*result + 2*weight.get()
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def lkhd_ratio_spline(parameters, data, weight = None, prob = None):
# THIS FILE IS DEPRECATED   pars_dict = parameters.valuesdict()
# THIS FILE IS DEPRECATED   if not prob:                                             # for ploting, mainly
# THIS FILE IS DEPRECATED     samples = []; prob = []
# THIS FILE IS DEPRECATED     for sample in range(0,2):
# THIS FILE IS DEPRECATED       samples.append(ristra.allocate(data[sample]))
# THIS FILE IS DEPRECATED       prob.append( ristra.allocate(np.zeros_like(data[sample])) )
# THIS FILE IS DEPRECATED     bsjpsikk.ratio_spline_time_acceptance(samples[0], samples[1], prob[0], prob[1], **pars_dict)
# THIS FILE IS DEPRECATED     return prob[1].get()
# THIS FILE IS DEPRECATED   else:                               # Optimizer.optimize ready-to-use function
# THIS FILE IS DEPRECATED     bsjpsikk.ratio_spline_time_acceptance(data[0], data[1], prob[0], prob[1], **pars_dict)
# THIS FILE IS DEPRECATED     if weight is not None:
# THIS FILE IS DEPRECATED       result  = np.concatenate(((ristra.log(prob[0])*weight[0]).get(),
# THIS FILE IS DEPRECATED                                 (ristra.log(prob[1])*weight[1]).get()
# THIS FILE IS DEPRECATED                               ))
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       result  = np.concatenate((ristra.log(prob[0]).get(),
# THIS FILE IS DEPRECATED                                 ristra.log(prob[1]).get()
# THIS FILE IS DEPRECATED                               ))
# THIS FILE IS DEPRECATED     return -2*result
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def lkhd_full_spline(parameters, data, weight = None, prob = None):
# THIS FILE IS DEPRECATED   pars_dict = parameters.valuesdict()
# THIS FILE IS DEPRECATED   if not prob:                                             # for ploting, mainly
# THIS FILE IS DEPRECATED     samples = []; prob = []
# THIS FILE IS DEPRECATED     for sample in range(0,3):
# THIS FILE IS DEPRECATED       samples.append(ristra.allocate(data[sample]))
# THIS FILE IS DEPRECATED       prob.append( ristra.allocate(np.zeros_like(data[sample])) )
# THIS FILE IS DEPRECATED     bsjpsikk.full_spline_time_acceptance(samples[0], samples[1], samples[2], prob[0], prob[1], prob[2], **pars_dict)
# THIS FILE IS DEPRECATED     return [ p.get() for p in prob ]
# THIS FILE IS DEPRECATED   else:                               # Optimizer.optimize ready-to-use function
# THIS FILE IS DEPRECATED     bsjpsikk.full_spline_time_acceptance(data[0], data[1], data[2], prob[0], prob[1], prob[2], **pars_dict)
# THIS FILE IS DEPRECATED     if weight is not None:
# THIS FILE IS DEPRECATED       result  = np.concatenate(((ristra.log(prob[0])*weight[0]).get(),
# THIS FILE IS DEPRECATED                                 (ristra.log(prob[1])*weight[1]).get(),
# THIS FILE IS DEPRECATED                                 (ristra.log(prob[2])*weight[2]).get()
# THIS FILE IS DEPRECATED                               ))
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       result  = np.concatenate((ristra.log(prob[0]).get(),
# THIS FILE IS DEPRECATED                                 ristra.log(prob[1]).get(),
# THIS FILE IS DEPRECATED                                 ristra.log(prob[2]).get()
# THIS FILE IS DEPRECATED                               ))
# THIS FILE IS DEPRECATED     return -2*result
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #%% Plotting functions #########################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def plot_fcn_spline(parameters, data, weight, log=False, name='test.pdf'):
# THIS FILE IS DEPRECATED   if name.split('/')[4].startswith('MC_Bs'): i = 0
# THIS FILE IS DEPRECATED   elif name.split('/')[4].startswith('MC_Bd'): i = 1
# THIS FILE IS DEPRECATED   else: i = 2
# THIS FILE IS DEPRECATED   ref = histogram.hist(data, weights=weight, bins = 100)
# THIS FILE IS DEPRECATED   fig, axplot, axpull = plotting.axes_plotpull();
# THIS FILE IS DEPRECATED   x = np.linspace(0.3,15,200)
# THIS FILE IS DEPRECATED   if len(parameters)>22:
# THIS FILE IS DEPRECATED     y = lkhd_full_spline(parameters, [x, x, x] )[i]
# THIS FILE IS DEPRECATED   elif len(parameters)>11:
# THIS FILE IS DEPRECATED     y = lkhd_ratio_spline(parameters, [x, x] )[i]
# THIS FILE IS DEPRECATED   else:
# THIS FILE IS DEPRECATED     y = lkhd_single_spline(parameters, x )[i]
# THIS FILE IS DEPRECATED   y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
# THIS FILE IS DEPRECATED   axplot.plot(x,y)
# THIS FILE IS DEPRECATED   axpull.fill_between(ref.cmbins,
# THIS FILE IS DEPRECATED                       histogram.pull_pdf(x,y,ref.cmbins,ref.counts,ref.errl,ref.errh),
# THIS FILE IS DEPRECATED                       0, facecolor="C0")
# THIS FILE IS DEPRECATED   axplot.errorbar(ref.cmbins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
# THIS FILE IS DEPRECATED   if log:
# THIS FILE IS DEPRECATED     axplot.set_yscale('log')
# THIS FILE IS DEPRECATED   axpull.set_xlabel(r'$t$ [ps]')
# THIS FILE IS DEPRECATED   axplot.set_ylabel(r'Weighted candidates')
# THIS FILE IS DEPRECATED   fig.savefig(name)
# THIS FILE IS DEPRECATED   plt.close()
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED def plot_spline(params, time, weights, conf_level=1, name='test.pdf', bins=30, label=None):
# THIS FILE IS DEPRECATED   """
# THIS FILE IS DEPRECATED   Hi Marcos,
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   Do you mean the points of the data?
# THIS FILE IS DEPRECATED   The binning is obtained using 30 bins that are exponentially distributed
# THIS FILE IS DEPRECATED   with a decay constant of 0.4 (this means that an exponential distribution
# THIS FILE IS DEPRECATED   with gamma=0.4 would result in equally populated bins).
# THIS FILE IS DEPRECATED   For every bin, the integral of an exponential with the respective decay
# THIS FILE IS DEPRECATED   width (0.66137,0.65833,..) is calculated and its inverse is used to scale
# THIS FILE IS DEPRECATED   the number of entries in this bin.
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   Cheers,
# THIS FILE IS DEPRECATED   Simon
# THIS FILE IS DEPRECATED   """
# THIS FILE IS DEPRECATED   list_coeffs = [key for key in params if key[0]=='c']
# THIS FILE IS DEPRECATED   if not list_coeffs:
# THIS FILE IS DEPRECATED     list_coeffs = [key for key in params if key[0]=='b']
# THIS FILE IS DEPRECATED     if not list_coeffs:
# THIS FILE IS DEPRECATED       list_coeffs = [key for key in params if key[0]=='a']
# THIS FILE IS DEPRECATED       if not list_coeffs:
# THIS FILE IS DEPRECATED         print('shit')
# THIS FILE IS DEPRECATED       else:
# THIS FILE IS DEPRECATED         gamma = params['gamma_a'].value
# THIS FILE IS DEPRECATED         kind = 'single'
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       gamma = params['gamma_b'].value
# THIS FILE IS DEPRECATED       if not [key for key in params if key[0]=='a']:
# THIS FILE IS DEPRECATED         kind = 'single'
# THIS FILE IS DEPRECATED       else:
# THIS FILE IS DEPRECATED         kind = 'ratio'
# THIS FILE IS DEPRECATED   else:
# THIS FILE IS DEPRECATED     gamma = params['gamma_c'].value
# THIS FILE IS DEPRECATED     if not [key for key in params if key[0]=='b']:
# THIS FILE IS DEPRECATED       kind = 'single'
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       kind = 'full'
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   # Prepare coeffs as ufloats
# THIS FILE IS DEPRECATED   coeffs = []
# THIS FILE IS DEPRECATED   for par in list_coeffs:
# THIS FILE IS DEPRECATED     if params[par].stdev:
# THIS FILE IS DEPRECATED       coeffs.append(unc.ufloat(params[par].value,params[par].stdev))
# THIS FILE IS DEPRECATED     else:
# THIS FILE IS DEPRECATED       coeffs.append(unc.ufloat(params[par].value,0))
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   # Cook where should I place the bins
# THIS FILE IS DEPRECATED   tLL = 0.3; tUL = 15
# THIS FILE IS DEPRECATED   def distfunction(tLL, tUL, gamma, ti, nob):
# THIS FILE IS DEPRECATED     return np.log(-((np.exp(gamma*ti + gamma*tLL + gamma*tUL)*nob)/
# THIS FILE IS DEPRECATED     (-np.exp(gamma*ti + gamma*tLL) + np.exp(gamma*ti + gamma*tUL) -
# THIS FILE IS DEPRECATED       np.exp(gamma*tLL + gamma*tUL)*nob)))/gamma
# THIS FILE IS DEPRECATED   list_bins = [tLL]; ipdf = []; widths = []
# THIS FILE IS DEPRECATED   dummy = 0.4; # this is a general gamma to distribute the bins
# THIS FILE IS DEPRECATED   for k in range(0,bins):
# THIS FILE IS DEPRECATED     ti = list_bins[k]
# THIS FILE IS DEPRECATED     list_bins.append( distfunction(tLL, tUL, dummy, ti, bins)   )
# THIS FILE IS DEPRECATED     tf = list_bins[k+1]
# THIS FILE IS DEPRECATED     ipdf.append( 1.0/((-np.exp(-(tf*gamma)) + np.exp(-(ti*gamma)))/1.0) )
# THIS FILE IS DEPRECATED     widths.append(tf-ti)
# THIS FILE IS DEPRECATED   bins = np.array(list_bins); int_pdf = np.array(ipdf)
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   # Manipulate the decay-time dependence of the efficiency
# THIS FILE IS DEPRECATED   x = np.linspace(0.3,15,200)
# THIS FILE IS DEPRECATED   y = wrap_unc(bsjpsikk.acceptance_spline, x, *coeffs)
# THIS FILE IS DEPRECATED   y_nom = unp.nominal_values(y)
# THIS FILE IS DEPRECATED   y_spl = interp1d(x, y_nom, kind='cubic', fill_value='extrapolate')(5)
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   # Manipulate data
# THIS FILE IS DEPRECATED   ref = histogram.hist(time, bins=bins, weights=weights)
# THIS FILE IS DEPRECATED   ref.counts *= int_pdf; ref.errl *= int_pdf; ref.errh *= int_pdf
# THIS FILE IS DEPRECATED   if kind == 'ratio':
# THIS FILE IS DEPRECATED     coeffs_a = [params[key].value for key in params if key[0]=='a']
# THIS FILE IS DEPRECATED     spline_a = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_a)
# THIS FILE IS DEPRECATED     ref.counts /= spline_a; ref.errl /= spline_a; ref.errh /= spline_a
# THIS FILE IS DEPRECATED   if kind == 'full':
# THIS FILE IS DEPRECATED     coeffs_a = [params[key].value for key in params if key[0]=='a']
# THIS FILE IS DEPRECATED     coeffs_b = [params[key].value for key in params if key[0]=='b']
# THIS FILE IS DEPRECATED     spline_a = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_a)
# THIS FILE IS DEPRECATED     spline_b = bsjpsikk.acceptance_spline(ref.cmbins,*coeffs_b)
# THIS FILE IS DEPRECATED     ref.counts /= spline_b; ref.errl /= spline_b; ref.errh /= spline_b
# THIS FILE IS DEPRECATED     #ref.counts *= spline_a; ref.errl *= spline_a; ref.errh *= spline_a
# THIS FILE IS DEPRECATED   counts_spline = interp1d(ref.bins, ref.counts, kind='cubic')
# THIS FILE IS DEPRECATED   int_5 = counts_spline(5)
# THIS FILE IS DEPRECATED   ref.counts /= int_5; ref.errl /= int_5; ref.errh /= int_5
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   # Actual ploting
# THIS FILE IS DEPRECATED   fig, axplot = plotting.axes_plot()
# THIS FILE IS DEPRECATED   axplot.set_ylim(0.4, 1.5)
# THIS FILE IS DEPRECATED   axplot.plot(x,y_nom/y_spl)
# THIS FILE IS DEPRECATED   axplot.errorbar(ref.cmbins,ref.counts,
# THIS FILE IS DEPRECATED                   yerr=[ref.errl,ref.errh],
# THIS FILE IS DEPRECATED                   xerr=[-ref.edges[:-1]+ref.cmbins,-ref.cmbins+ref.edges[1:]],
# THIS FILE IS DEPRECATED                   fmt='.', color='k')
# THIS FILE IS DEPRECATED   y_upp, y_low = get_confidence_bands(x,y, sigma=conf_level)
# THIS FILE IS DEPRECATED   axplot.fill_between(x, y_upp/y_spl, y_low/y_spl, alpha=0.2, edgecolor="none",
# THIS FILE IS DEPRECATED                       label='$'+str(conf_level)+'\sigma$ confidence band')
# THIS FILE IS DEPRECATED   axplot.set_xlabel(r'$t$ [ps]')
# THIS FILE IS DEPRECATED   axplot.set_ylabel(r'%s [a.u.]' % label)
# THIS FILE IS DEPRECATED   axplot.legend()
# THIS FILE IS DEPRECATED   fig.savefig(name)
# THIS FILE IS DEPRECATED   plt.close()
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #%% Get data into categories ###################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED print(f"\n{80*'='}\n{'= Loading categories':79}=\n{80*'='}\n")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Select samples
# THIS FILE IS DEPRECATED samples = {}
# THIS FILE IS DEPRECATED samples['BsMC'] = os.path.join(args['BsMC_sample'])
# THIS FILE IS DEPRECATED samples['BdMC'] = os.path.join(args['BdMC_sample'])
# THIS FILE IS DEPRECATED samples['BdDT'] = os.path.join(args['BdDT_sample'])
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED cats = {}
# THIS FILE IS DEPRECATED for name, sample in zip(samples.keys(),samples.values()):
# THIS FILE IS DEPRECATED   print(f'Loading {sample} as {name} category')
# THIS FILE IS DEPRECATED   name = name[:4] # remove _sample
# THIS FILE IS DEPRECATED   if name == 'BsMC':
# THIS FILE IS DEPRECATED     label = (r'\mathrm{MC}',r'B_s^0')
# THIS FILE IS DEPRECATED   elif name == 'BdMC':
# THIS FILE IS DEPRECATED     label = (r'\mathrm{MC}',r'B^0')
# THIS FILE IS DEPRECATED   elif name == 'BdDT':
# THIS FILE IS DEPRECATED     label = (r'\mathrm{data}',r'B_s^0')
# THIS FILE IS DEPRECATED   cats[name] = Sample.from_root(sample, cuts=cuts)
# THIS FILE IS DEPRECATED   cats[name].name = os.path.splitext(os.path.basename(sample))[0]+'_'+trigger
# THIS FILE IS DEPRECATED   cats[name].allocate(time='time',weight='sWeight',lkhd='0*time')
# THIS FILE IS DEPRECATED   param_path  = os.path.join(PATH,'init',NAME)
# THIS FILE IS DEPRECATED   param_path = os.path.join(param_path,f"{sample.split('/')[-2]}_{YEAR}.json")
# THIS FILE IS DEPRECATED   cats[name].assoc_params(Parameters.load(param_path))
# THIS FILE IS DEPRECATED   cats[name].label = label
# THIS FILE IS DEPRECATED   cats[name].pars_path = os.path.dirname(args[f'{name}_params'])
# THIS FILE IS DEPRECATED   cats[name].figs_path = cats[name].pars_path.replace('parameters','figures')
# THIS FILE IS DEPRECATED   os.makedirs(cats[name].pars_path, exist_ok=True)
# THIS FILE IS DEPRECATED   os.makedirs(cats[name].figs_path, exist_ok=True)
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
# THIS FILE IS DEPRECATED #%% Fit all categories #########################################################
# THIS FILE IS DEPRECATED """
# THIS FILE IS DEPRECATED fits = {}; FIT_EACH = 1; FIT_RATIO = 1; FIT_FULL = 1
# THIS FILE IS DEPRECATED # Fit each sample
# THIS FILE IS DEPRECATED if FIT_EACH:
# THIS FILE IS DEPRECATED   for name, cat in zip(cats.keys(),cats.values()):
# THIS FILE IS DEPRECATED     print('Fitting %s category...' % name)
# THIS FILE IS DEPRECATED     if cat.params:
# THIS FILE IS DEPRECATED       fits[cat.name] = optimize(lkhd_single_spline, method="hesse",
# THIS FILE IS DEPRECATED                             params=cat.params,
# THIS FILE IS DEPRECATED                             kwgs={'data': cat.time,
# THIS FILE IS DEPRECATED                                  'prob': cat.lkhd,
# THIS FILE IS DEPRECATED                                  'weight': cat.weight},
# THIS FILE IS DEPRECATED                             verbose=True);
# THIS FILE IS DEPRECATED       fits[cat.name].params.dump(TIMEACC_PATH+'/parameters/'+cat.name)
# THIS FILE IS DEPRECATED     fits[cat.name].label = r'$\varepsilon_{%s}^{%s}$' % cat.label
# THIS FILE IS DEPRECATED     print('\n')
# THIS FILE IS DEPRECATED   for name, cat in zip(cats.keys(),cats.values()):
# THIS FILE IS DEPRECATED     print('Plotting %s category...' % name)
# THIS FILE IS DEPRECATED     filename = TIMEACC_PATH+'/plots/'+cat.name+'_fit.pdf'
# THIS FILE IS DEPRECATED     plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
# THIS FILE IS DEPRECATED                     name = filename )
# THIS FILE IS DEPRECATED     filename = TIMEACC_PATH+'/plots/'+cat.name+'_fit_log.pdf'
# THIS FILE IS DEPRECATED     plot_fcn_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
# THIS FILE IS DEPRECATED                     name = filename, log= True )
# THIS FILE IS DEPRECATED     filename = TIMEACC_PATH+'/plots/'+cat.name+'_spline.pdf'
# THIS FILE IS DEPRECATED     plot_spline(fits[cat.name].params, cat.time.get(), cat.weight.get(),
# THIS FILE IS DEPRECATED                        name = filename, label=fits[cat.name].label,
# THIS FILE IS DEPRECATED                        conf_level=1, bins=30 )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Fit the ratio BsMC/BdMC ------------------------------------------------------
# THIS FILE IS DEPRECATED if FIT_RATIO:
# THIS FILE IS DEPRECATED   for trig in ['_biased']:
# THIS FILE IS DEPRECATED     name = cats['BdMC'+trig].name.replace('Bd2JpsiKstar','ratioBsBd')
# THIS FILE IS DEPRECATED     print('Fitting %s category...' % name)
# THIS FILE IS DEPRECATED     fits[name] = optimize(lkhd_ratio_spline, method="hesse",
# THIS FILE IS DEPRECATED                     params=cats['BsMC'+trig].params+cats['BdMC'+trig].params,
# THIS FILE IS DEPRECATED                     kwgs={'data':  [cats['BsMC'+trig].time,
# THIS FILE IS DEPRECATED                                    cats['BdMC'+trig].time],
# THIS FILE IS DEPRECATED                           'prob':  [cats['BsMC'+trig].lkhd,
# THIS FILE IS DEPRECATED                                    cats['BdMC'+trig].lkhd],
# THIS FILE IS DEPRECATED                          'weight': [cats['BsMC'+trig].weight,
# THIS FILE IS DEPRECATED                                    cats['BdMC'+trig].weight]},
# THIS FILE IS DEPRECATED                     verbose=True);
# THIS FILE IS DEPRECATED     fits[name].params.dump(TIMEACC_PATH+'/parameters/'+name)
# THIS FILE IS DEPRECATED     print('\n')
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED   for trig in ['_biased']:
# THIS FILE IS DEPRECATED     name = cats['BdMC'+trig].name.replace('Bd2JpsiKstar','ratioBsBd')
# THIS FILE IS DEPRECATED     plot_spline(
# THIS FILE IS DEPRECATED       {k:fits[name].params[k] for k in list(fits[name].params.keys())[0:12]},
# THIS FILE IS DEPRECATED       cats['BsMC'+trig].time.get(),
# THIS FILE IS DEPRECATED       cats['BsMC'+trig].weight.get(),
# THIS FILE IS DEPRECATED       name = TIMEACC_PATH+'/plots/'+cats['BsMC'+trig].name+'_spline.pdf',
# THIS FILE IS DEPRECATED       label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
# THIS FILE IS DEPRECATED       conf_level=1,
# THIS FILE IS DEPRECATED       bins=30
# THIS FILE IS DEPRECATED     )
# THIS FILE IS DEPRECATED     plot_spline(
# THIS FILE IS DEPRECATED       fits[name].params,
# THIS FILE IS DEPRECATED       cats['BdMC'+trig].time.get(),
# THIS FILE IS DEPRECATED       cats['BdMC'+trig].weight.get(),
# THIS FILE IS DEPRECATED       name = TIMEACC_PATH+'/plots/'+name+'_spline.pdf',
# THIS FILE IS DEPRECATED       label=r'$\varepsilon_{\mathrm{MC}}^{B^0/B_s^0}$',
# THIS FILE IS DEPRECATED       conf_level=1,
# THIS FILE IS DEPRECATED       bins=30
# THIS FILE IS DEPRECATED     )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED """
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Full fit to get decay-time acceptance ----------------------------------------
# THIS FILE IS DEPRECATED print(f"\n{80*'='}\n{'= Fitting three categories':79}=\n{80*'='}\n")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED result = optimize(lkhd_full_spline, method="bfgs",
# THIS FILE IS DEPRECATED             params=cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params,
# THIS FILE IS DEPRECATED             kwgs={'data': [cats['BsMC'].time,
# THIS FILE IS DEPRECATED                            cats['BdMC'].time,
# THIS FILE IS DEPRECATED                            cats['BdDT'].time],
# THIS FILE IS DEPRECATED                 'prob':   [cats['BsMC'].lkhd,
# THIS FILE IS DEPRECATED                            cats['BdMC'].lkhd,
# THIS FILE IS DEPRECATED                            cats['BdDT'].lkhd],
# THIS FILE IS DEPRECATED                 'weight': [cats['BsMC'].weight,
# THIS FILE IS DEPRECATED                            cats['BdMC'].weight,
# THIS FILE IS DEPRECATED                            cats['BdDT'].weight]},
# THIS FILE IS DEPRECATED             #ftol=1e2*np.finfo(float).eps
# THIS FILE IS DEPRECATED             #steps=2000, is_weighted=True, nan_policy='omit', progress=True
# THIS FILE IS DEPRECATED             );
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED print(result)
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Dumping fit parameters -------------------------------------------------------
# THIS FILE IS DEPRECATED print(f"\n{80*'='}\n{'= Dumping parameters':79}=\n{80*'='}\n")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED for name, cat in zip(cats.keys(),cats.values()):
# THIS FILE IS DEPRECATED   list_params = [par for par in cat.params if len(par) ==2]
# THIS FILE IS DEPRECATED   cat.params.add(*[result.params.get(par) for par in list_params])
# THIS FILE IS DEPRECATED   cat.params.dump(os.path.join(cat.pars_path,f'{FLAG}_{trigger}'))
# THIS FILE IS DEPRECATED   # latex export
# THIS FILE IS DEPRECATED   with open(os.path.join(cat.pars_path,f'{FLAG}_{trigger}.tex'), "w") as text:
# THIS FILE IS DEPRECATED     text.write(cat.params.dump_latex())
# THIS FILE IS DEPRECATED   print( os.path.join(cat.pars_path,f'{FLAG}_{trigger}') )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # Plotting ---------------------------------------------------------------------
# THIS FILE IS DEPRECATED print(f"\n{80*'='}\n{'= Plotting':79}=\n{80*'='}\n")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED for name, cat in zip(cats.keys(),cats.values()):
# THIS FILE IS DEPRECATED   plot_fcn_spline(
# THIS FILE IS DEPRECATED     result.params,
# THIS FILE IS DEPRECATED     cat.time.get(),
# THIS FILE IS DEPRECATED     cat.weight.get(),
# THIS FILE IS DEPRECATED     name = os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit_log.pdf'),
# THIS FILE IS DEPRECATED     log=True
# THIS FILE IS DEPRECATED   )
# THIS FILE IS DEPRECATED print(f"Plotted {os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit_log.pdf')}")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED for name, cat in zip(cats.keys(),cats.values()):
# THIS FILE IS DEPRECATED   plot_fcn_spline(
# THIS FILE IS DEPRECATED     result.params,
# THIS FILE IS DEPRECATED     cat.time.get(),
# THIS FILE IS DEPRECATED     cat.weight.get(),
# THIS FILE IS DEPRECATED     name = os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit.pdf'),
# THIS FILE IS DEPRECATED     log=False
# THIS FILE IS DEPRECATED   )
# THIS FILE IS DEPRECATED print(f"Plotted {os.path.join(cat.figs_path,f'{FLAG}_{trigger}_fit.pdf')}")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # BsMC
# THIS FILE IS DEPRECATED plot_spline(# BsMC
# THIS FILE IS DEPRECATED   cats['BsMC'].params,
# THIS FILE IS DEPRECATED   cats['BsMC'].time.get(),
# THIS FILE IS DEPRECATED   cats['BsMC'].weight.get(),
# THIS FILE IS DEPRECATED   name = os.path.join(cats['BsMC'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
# THIS FILE IS DEPRECATED   label=r'$\varepsilon_{\mathrm{MC}}^{B_s^0}$',
# THIS FILE IS DEPRECATED   conf_level=1,
# THIS FILE IS DEPRECATED   bins=25
# THIS FILE IS DEPRECATED )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # ratio
# THIS FILE IS DEPRECATED plot_spline(# BsMC
# THIS FILE IS DEPRECATED   cats['BsMC'].params+cats['BdMC'].params,
# THIS FILE IS DEPRECATED   cats['BdMC'].time.get(),
# THIS FILE IS DEPRECATED   cats['BdMC'].weight.get(),
# THIS FILE IS DEPRECATED   name = os.path.join(cats['BdMC'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
# THIS FILE IS DEPRECATED   label=r'$\varepsilon_{\mathrm{MC}}^{B_d}/\varepsilon_{\mathrm{MC}}^{B_s^0}$',
# THIS FILE IS DEPRECATED   conf_level=1,
# THIS FILE IS DEPRECATED   bins=25
# THIS FILE IS DEPRECATED )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED # BsDT
# THIS FILE IS DEPRECATED plot_spline(# BsMC
# THIS FILE IS DEPRECATED   cats['BsMC'].params+cats['BdMC'].params+cats['BdDT'].params,
# THIS FILE IS DEPRECATED   cats['BdDT'].time.get(),
# THIS FILE IS DEPRECATED   cats['BdDT'].weight.get(),
# THIS FILE IS DEPRECATED   name = os.path.join(cats['BdDT'].figs_path,f'{FLAG}_{trigger}_spline.pdf'),
# THIS FILE IS DEPRECATED   label=r'$\varepsilon_{\mathrm{data}}^{B_s^0}$',
# THIS FILE IS DEPRECATED   conf_level=1,
# THIS FILE IS DEPRECATED   bins=25
# THIS FILE IS DEPRECATED )
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED print(f"Splines were plotted!")
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED 
# THIS FILE IS DEPRECATED ################################################################################
