#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE
-- """


# %% Modules -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import uncertainties as unc
from uncertainties import unumpy as unp

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters

# get bsjpsikk and compile it with corresponding flags
import bsjpsikk
bsjpsikk.config['debug'] = 0
bsjpsikk.config['debug_evt'] = 1
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
# to get Simon
bsjpsikk.config['use_perftag'] = 1
bsjpsikk.config['use_truetag'] = 0
# to get Peilian
#bsjpsikk.config['use_perftag'] = 0
#bsjpsikk.config['use_truetag'] = 1
bsjpsikk.get_kernels()

# reweighting config
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from hep_ml import reweight
reweighter = reweight.GBReweighter(n_estimators     = 40,
                                   learning_rate    = 0.25,
                                   max_depth        = 5,
                                   min_samples_leaf = 500,
                                   gb_args          = {'subsample': 1})



################################################################################
################################################################################
#%% ############################################################################

# input parameters
VERSION = 'v0r4'
path = f'/scratch17/marcos.romero/phis_samples/{VERSION}'
YEAR = 2016
FLAG = 'test'
#FLAG = '200506a'

################################################################################
################################################################################
################################################################################
# os.listdir('output/v0r4/params/csp_factors/')
# print(Parameters.load('output/v0r4/params/csp_factors/2015/Bs2JpsiPhi/200506a.json'))
# print(Parameters.load('output/v0r4/params/csp_factors/2016/Bs2JpsiPhi/200506a.json'))
# print(Parameters.load('output/v0r4/params/csp_factors/2017/Bs2JpsiPhi/200506a.json'))
# print(Parameters.load('output/v0r4/params/csp_factors/2018/Bs2JpsiPhi/200506a.json'))


# %% Load samples --------------------------------------------------------------

# Load Monte Carlo samples
mc_std = Sample.from_root(f'{path}/{YEAR}/MC_Bs2JpsiPhi/{FLAG}.root')
mc_std.assoc_params(f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json')
mc_dg0 = Sample.from_root(f'{path}/{YEAR}/MC_Bs2JpsiPhi_dG0/{FLAG}.root')
mc_dg0.assoc_params(f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi_dG0.json')

# Load corresponding data sample
data = Sample.from_root(f'{path}/{YEAR}/Bs2JpsiPhi/{FLAG}.root')

# Build a dict
mc = {}
mc['MC_Bs2JpsiPhi'] = {}
mc['MC_Bs2JpsiPhi']['sample'] = mc_std
# mc['MC_Bs2JpsiPhi_dG0'] = {}
# mc['MC_Bs2JpsiPhi_dG0']['sample'] = mc_dg0

# Allocate some arrays with the needed branches
reco = ['helcosthetaK', 'helcosthetaL', 'helphi', 'time']
true = ['true'+i+'_GenLvl' for i in reco]
for k, v in mc.items():
  v['sample'].allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
  v['sample'].allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
  v['sample'].allocate(pdf='0*time', ones='time/time', zeros='0*time')
  v['sample'].allocate(weight='(polWeight*sw/gb_weights)')
  v['sample'].allocate(biased='(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)')
  v['sample'].allocate(unbiased='(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)')
  v['tables_path'] = f'output/{VERSION}/tables/angular_acceptance/{YEAR}/{k}/'
  v['params_path'] = f'output/{VERSION}/params/angular_acceptance/{YEAR}/{k}/'
  os.makedirs(v['tables_path'], exist_ok=True)
  os.makedirs(v['params_path'], exist_ok=True)
  v['biased']   = {'weight':v['sample'].biased*v['sample'].weight,
                   'trig_cut': '(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)*'}
  v['unbiased'] = {'weight':v['sample'].unbiased*v['sample'].weight,
                   'trig_cut': '(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*'}



################################################################################
################################################################################
################################################################################



#%% Compute angWeights without corrections ---------------------------------------
#     Let's start computing the angular weights in the most naive version, w/
#     any corrections
print(f"\n{80*'='}\n",
      "STEP 1: Compute angWeights without correcting MC sample",
      f"\n{80*'='}\n")

for k, v in mc.items():
  for t in ['biased', 'unbiased']:
    w, uw, cov, fcov = bsjpsikk.get_angular_cov(
                                 v['sample'].true,
                                 v['sample'].reco,
                                 v[t]['weight'],
                                 **v['sample'].params.valuesdict()
                                 )
    # Build arrays and parameters objects
    v[t]['w_uncorrected_params'] = Parameters()
    v[t]['cov'] = cov
    v[t]['fcov'] = fcov
    for i in range(0,len(w)):
      v[t]['w_uncorrected_params'].add(
        {'name': f'w{i}', 'value': w[i], 'stdev': uw[i], 'free': False, 'latex': f'w_{i}'})
    # Dump parameters in json files
    v[t]['w_uncorrected_params'].dump(v['params_path']+f'{FLAG}_naive_{t}')
    # Export parameters in tex tables
    with open(v['tables_path']+f'{FLAG}_naive_{t}.tex', "w") as tex_file:
      tex_file.write( v[t]['w_uncorrected_params'].dump_latex( caption="""
      Naive angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR,t,k.replace('_', ' ') ) ) )
    tex_file.close()
    print(f"\tCurrent angular weights for {k}-{t} sample are:")
    print(f"{v[t]['w_uncorrected_params']}")



#%% Compute standard kinematic weights -------------------------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
print(f"\n{80*'='}\n",
      "STEP 2: Weight the MC samples to match data ones",
      f"\n{80*'='}\n")
for k, v in mc.items():
  for t in ['biased', 'unbiased']:
    # reweighter.fit(original        = v['sample'].df[['X_M','B_P','B_PT']],
    #                target          = data.df[['X_M','B_P','B_PT']],
    #                original_weight = v['sample'].df.eval(v[t]['trig_cut']+'polWeight*sw/gb_weights'),
    #                target_weight   = data.df.eval(v[t]['trig_cut']+'sw')
    #               );
    # kinWeight = reweighter.predict_weights(v['sample'].df[['X_M','B_P','B_PT']])
    if t == 'biased':
      kinWeight = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Biased.root')[f'kinWeight_Biased_Step{0}'].array('kinWeight')
    v[t]['kinWeight'] = ristra.allocate(np.where( (v['sample'].df.eval(v[t]['trig_cut']+'polWeight*sw/gb_weights')) !=0, kinWeight, 0))
    if t == 'unbiased':
      kinWeight = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')[f'kinWeight_Unbiased_Step{0}'].array('kinWeight')
    v[t]['kinWeight'] = ristra.allocate(np.where( (v['sample'].df.eval(v[t]['trig_cut']+'polWeight*sw/gb_weights')) !=0, kinWeight, 0))
    print(f"The kinematic-weighting in B_PT, B_P and X_M is done for {k}-{t}")
    print(f"kinWeight:",v[t]['kinWeight'])


#%% Compute angWeights correcting with kinematic weights -------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
print(f"\n{80*'='}\n",
      "STEP 3: Compute angWeights correcting with kinematic weights",
      f"\n{80*'='}\n")
for k, v in mc.items():
  for t in ['biased', 'unbiased']:
    w, uw, cov, fcov = bsjpsikk.get_angular_cov(
                            v['sample'].true,
                            v['sample'].reco,
                            v[t]['weight']*v[t]['kinWeight'],
                            **v['sample'].params.valuesdict()
                         )
    w = w/w[0]
    v[t]['cov'] = cov
    v[t]['fcov'] = fcov
    v[t]['w_kinweighted'] = w
    v[t]['w_kinweighted_params'] = Parameters()
    for i in range(0,len(w)):
      v[t]['w_kinweighted_params'].add(
        {'name': f'w{i}', 'value': w[i], 'stdev': uw[i], 'free': False, 'latex': f'w_{i}'})
    # Dump parameters in json files
    v[t]['w_kinweighted_params'].dump(v['params_path']+f'corrected_{t}')
    # Export parameters in tex tables
    with open(v['tables_path']+f'corrected_{t}.tex', "w") as tex_file:
      tex_file.write( v[t]['w_kinweighted_params'].dump_latex( caption="""
      Naive angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
      category.""" % (YEAR,t,k ) ) )
    tex_file.close()
    print(f"Current angular weights for {k}-{t} sample are:")
    print(f"{v[t]['w_kinweighted_params']}")








#%% Iterative procedure computing angWeights with corrections --------------------
#     dfdfdf Weight MC to match data in the iterative variables namely
#              p and pT of K+ and K-



print('STEP 4: Launching iterative procedure, pdf and kinematic-weighting')
for i in range(1,5):
  for k, v in mc.items():
    # fit MC  ------------------------------------------------------------------
    print('\tFitting %s in %s iteration' % (k,str(i)))
    tparams_pdf = hjson.load(
                    open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_'+str(i-1)+'.json')
                  )
    for t in ['biased', 'unbiased']:
      # do the pdf-weighting -----------------------------------------------------
      print('\tCalculating pdfWeight in %s iteration' % str(i))
      bsjpsikk.diff_cross_rate_full(v['sample'].true, v['sample'].pdf, use_fk=1, **v['sample'].params.valuesdict())
      original_pdf_h = v['sample'].pdf.get()
      bsjpsikk.diff_cross_rate_full(v['sample'].true, v['sample'].pdf, use_fk=0, **v['sample'].params.valuesdict())
      original_pdf_h /= v['sample'].pdf.get()
      bsjpsikk.diff_cross_rate_full(v['sample'].true, v['sample'].pdf, use_fk=1, **tparams_pdf)
      target_pdf_h = v['sample'].pdf.get()
      bsjpsikk.diff_cross_rate_full(v['sample'].true, v['sample'].pdf, use_fk=0, **tparams_pdf)
      target_pdf_h /= v['sample'].pdf.get()
      v[f'pdfWeight{i}'] = np.nan_to_num(target_pdf_h/original_pdf_h)
      print(f"\tpdfWeight{i}:",v[f'pdfWeight{i}'])

      # kinematic-weighting over P and PT of K+ and K- ---------------------------
      print(f'\tCalculating p and pT of K+ and K- weight in {i} iteration')
      reweighter.fit(original        = v['sample'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     target          = data.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                     original_weight = v['sample'].df.eval(v[t]['trig_cut']+'polWeight*sw/gb_weights')*v[f'pdfWeight{i}']*v[t]['kinWeight'].get(),
                     target_weight   = data.df.eval(v[t]['trig_cut']+'sw')
                    );
      kkpWeight = reweighter.predict_weights(v['sample'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']])
      v[t][f'kkpWeight{i}'] = ristra.allocate(np.where((v['sample'].df.eval(v[t]['trig_cut']+'polWeight*sw/gb_weights')*v[f'pdfWeight{i}']*v[t]['kinWeight'].get())!=0, kkpWeight, 0))
      print(f"\tkkpWeight{i} = {v[t][f'kkpWeight{i}']}")

      # kinematic-weighting over P and PT of K+ and K- ---------------------------
      print(f"\tAngular weights for {k} category in {i} iteration")
      w, uw, cov, fcov = bsjpsikk.get_angular_cov(
                  v['sample'].true,
                  v['sample'].reco,
                  v[t]['weight']*v[t]['kinWeight']*v[t][f'kkpWeight{i}'],
                  **v['sample'].params.valuesdict()
                  )
      v[t]['cov'] = cov
      v[t]['fcov'] = fcov
      v[t][f'w_kkpweighted{i}_params'] = Parameters()
      for kk in range(0,len(w)):
        v[t][f'w_kkpweighted{i}_params'].add(
          {'name': f'w{kk}', 'value': w[kk], 'stdev': uw[kk], 'free': False, 'latex': f'w_{kk}'})
      # Dump parameters in json fkles
      v[t][f'w_kkpweighted{i}_params'].dump(v['params_path']+f'baseline_{t}')
      # Export parameters in tex tables
      with open(v['tables_path']+f'baseline_{t}.tex', "w") as tex_file:
        tex_file.write( v[t][f'w_kkpweighted{i}_params'].dump_latex( caption="""
        Naive angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
        category.""" % (YEAR,t,k ) ) )
      tex_file.close()
      print(f"Current angular weights for {k}-{t} sample are:")
      print(f"{v[t][f'w_kkpweighted{i}_params']}")

exit()

foo = uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')





"""
i = 1
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03782 +- 0.000949792 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03762 +- 0.000926573 (unbinned)
int f4 dcostheta dphi dcospsi = -0.00103561 +- 0.000740913 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000329971 +- 0.000447396 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000272403 +- 0.000457952 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01036 +- 0.000640444 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000336074 +- 0.000567024 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000553363 +- 0.000584319 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00322722 +- 0.00121307 (unbinned)

i = 2
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949104 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03758 +- 0.000926344 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966926 +- 0.00073987 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383365 +- 0.000446895 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456986 (unbinned)
int f7 dcostheta dphi dcospsi = 1.0103 +- 0.000640177 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000283837 +- 0.000566376 (unbinned)
int f9 dcostheta dphi dcospsi = 0.00058618 +- 0.000583348 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00280243 +- 0.00121186 (unbinned)

i = 3
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949098 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03757 +- 0.000926338 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966919 +- 0.000739864 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383704 +- 0.000446893 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456984 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01029 +- 0.000640172 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000284358 +- 0.000566372 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000586183 +- 0.000583344 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00278764 +- 0.00121186 (unbinned)








v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights') * v['kinWeight'].get() * v[f'kkpWeight{i}'].get()

v['sample'].weight*v[f'kkpWeight{1}']*v['kinWeight']
*

0.47/2


data = uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root')['DecayTree'].arrays(['sw','gb_weights'])

swg = data[b'sw']/data[b'gb_weights']
pol = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].array('PolWeight')

pdf_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step0'].array('pdfWeight')
pdf_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step1'].array('pdfWeight')
kin_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step0'].array('kinWeight')
kin_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step1'].array('kinWeight')


kin_0

kin_1

swg*pol*kin_0*kin_1

mc['MC_Bs2JpsiPhi'].df.eval(trig_cut+'polWeight*sw/gb_weights') *
mc['MC_Bs2JpsiPhi']['kinWeight'].get()
v[f'kkpWeight{1}'].get()




swg*pol
mc_std.df.eval('polWeight*(sw/gb_weights)')




























int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03714 +- 0.000948856 (unbinned)
int f3 dcostheta dphi dcospsi = 1.0369 +- 0.000926164 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000945802 +- 0.000738975 (unbinned)
int f5 dcostheta dphi dcospsi = 0.00035262 +- 0.000447158 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000285595 +- 0.000457371 (unbinned)
int f7 dcostheta dphi dcospsi = 1.00986 +- 0.00063955 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000366054 +- 0.000566205 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000567317 +- 0.000583275 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00220241 +- 0.00121173 (unbinned)

"""





v[f'kkpWeight{1}']


mc_std.df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{1}']*v['kinWeight'].get()


tweight


v['kkpWeight1']
v['kinWeight']

simon['MC_Bs2JpsiPhi']['w_kkpweighted1']



simon['MC_Bs2JpsiPhi']['kkpWeight1']



v['sample'].weight*v['kinWeight']*v[f'kkpWeight1']

os.listdir('/scratch08/marcos.romero/SideCar/')

a.keys()



################################################################################
################################################################################
################################################################################



################################################################################
# compare with Simon ###########################################################
################################################################################
simon = {}
for mode in ['MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0']:
  simon[mode] = {}
  for t, trig in zip(['biased','unbiased'],['Biased','Unbiased']):
    d = {}
    for i in range(-1,10):
      # Get angular weights
      f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_{trig}Trig_IncldG0_AngAcc_BaselineDef_15102018_Iteration{i}.root')['fi']
      if i==-1:
        d[f'w_uncorrected'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
      elif i==0:
        d[f'w_kinweighted'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
      else:
        d[f'w_kkpweighted{i}'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
      # Get kinWeight and kppWeights
      if i >=0:
        f = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_{trig}.root')[f'kinWeight_{trig}_Step{i}']
        if mode == 'MC_Bs2JpsiPhi_dG0':
          f = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_{trig}.root')[f'kinWeight_{trig}_Step{i}']
      if i==0:
        print(f.keys())
        d[f'kinWeight'] = f.array('kinWeight')
      elif i>0:
        d[f'kkpWeight{i}'] = f.array('kinWeight')
      # Get kinWeight and kppWeights
      if i >=0:
        f = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{i}"]
        if mode == 'MC_Bs2JpsiPhi_dG0':
          f = uproot.open(f'/scratch08/marcos.romero/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{i}"]
        d[f'pdfWeight{i+1}'] = f.array('pdfWeight')
    simon[mode][t] = d
s = simon#['MC_Bs2JpsiPhi']


s['MC_Bs2JpsiPhi']['unbiased']['pdfWeight2'] #OK
s['MC_Bs2JpsiPhi_dG0']['unbiased']['pdfWeight2'] #OK
s['MC_Bs2JpsiPhi']['unbiased']['kinWeight'] #OK
s['MC_Bs2JpsiPhi_dG0']['unbiased']['kinWeight'] #OK



s['MC_Bs2JpsiPhi']['unbiased']['kkpWeight1'][:10]
s['MC_Bs2JpsiPhi_dG0']['unbiased']['kkpWeight1'][:10]
s['MC_Bs2JpsiPhi']['biased']['kkpWeight1'][:10]
s['MC_Bs2JpsiPhi_dG0']['biased']['kkpWeight1'][:10]


uproot.open('/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/v0r0_angWeight.root')['2016'].arrays()


s['biased']['kkpWeight1']
s['unbiased']['kinWeight']












s['w_uncorrected']

f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_{trig}Trig_AngAcc_BaselineDef_15102018_Iteration{-1}.root')['fi']
mat = np.zeros((10,10))
for j1 in range(1,10):
  for j2 in range(1,10):
    mat[j1,j2] = f.array(f'cf{j1+1}{j2+1}')[0]

mat

f.arrays('cf*')
scale = mc['MC_Bs2JpsiPhi']['unbiased']['weight'].get().sum()
scale
0.8*mc['MC_Bs2JpsiPhi']['unbiased']['cov']/(scale*scale)
uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{0}"].array('pdfWeight')
