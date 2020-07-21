#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" --
ANGULAR ACCEPTANCE
-- """

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

# reweighting config
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters

# get bsjpsikk and compile it with corresponding flags
import bsjpsikk
bsjpsikk.config['debug'] = 0
bsjpsikk.config['debug_evt'] = 0
bsjpsikk.config['use_time_acc'] = 0
bsjpsikk.config['use_time_offset'] = 0
bsjpsikk.config['use_time_res'] = 0
bsjpsikk.config['use_perftag'] = 1
bsjpsikk.config['use_truetag'] = 0
bsjpsikk.get_kernels()

# reweighting config
from hep_ml import reweight
#reweighter = reweight.GBReweighter(n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 1})
reweighter = reweight.GBReweighter(n_estimators=40, learning_rate=0.25, max_depth=5, min_samples_leaf=500, gb_args={'subsample': 1})
#reweighter = reweight.GBReweighter(n_estimators=500, learning_rate=0.1, max_depth=2, min_samples_leaf=1000, gb_args={'subsample': 1})

#30:0.3:4:500
#20:0.3:3:1000

################################################################################



################################################################################

def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--sample-mc',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--sample-data',
                      default = 'samples/MC_Bs2JpsiPhi_DG0_2016_test.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--input-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-params',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--output-tables',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  parser.add_argument('--output-weights-file',
                      default = 'output/time_acceptance/parameters/2016/MC_Bs2JpsiPhi_DG0/test_biased.json',
                      help='Bs2JpsiPhi MC sample')
  # Configuration file ---------------------------------------------------------
  parser.add_argument('--mode',
                      default = 'baseline',
                      help='Configuration')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'test',
                      help='Year of data-taking')
  parser.add_argument('--trigger',
                      default = 'biased',
                      help='Trigger(s) to fit [comb/(biased)/unbiased]')

  return parser


# Parse arguments
try:
  args = vars(argument_parser().parse_args())
except:
  1+1

"""
YEAR = 2016
VERSION = 'v0r0'
MODE = 'MC_Bs2JpsiPhi'
TRIGGER = 'biased'
input_params_path = f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json'
sample_mc_path = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/{MODE}/{VERSION}.root'
sample_data_path = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/Bs2JpsiPhi/{VERSION}.root'
output_tables_path = f'output/v0r4/tables/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
output_params_path = f'output/v0r4/params/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
output_weight_file = f'/scratch17/marcos.romero/phis_samples/v0r4/{YEAR}/{MODE}/{VERSION}_angWeight.root'
"""


YEAR = args['year']
VERSION = args['version']
MODE = args['mode']
TRIGGER = args['trigger']
input_params_path = args['input_params']
sample_mc_path = args['sample_mc']
sample_data_path = args['sample_data']
output_tables_path = args['output_tables']
output_params_path = args['output_params']
output_weight_file = args['output_weights_file']

print(output_weight_file)

################################################################################
#%% ############################################################################



################################################################################
################################################################################
################################################################################


# %% Load samples --------------------------------------------------------------

# Load Monte Carlo samples
mc = Sample.from_root(sample_mc_path, cuts='truetime_GenLvl>0')
mc.assoc_params(input_params_path.replace('TOY','MC').replace('2021','2018'))

# Load corresponding data sample
data = Sample.from_root(sample_data_path)

# Variables and branches to be used
reco = ['cosK', 'cosL', 'hphi', 'time']
true = ['true'+i+'_GenLvl' for i in reco]
weight_mc='(polWeight*sw/gb_weights)'
weight_data='(sw)'


# Select trigger
if TRIGGER == 'biased':
  trigger = 'biased';
  weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
  weight_data += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==0)'
elif TRIGGER == 'unbiased':
  trigger = 'unbiased';
  weight_mc += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
  weight_data += '*(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)'
elif TRIGGER == 'comb':
  trigger = 'comb';

# Allocate some arrays with the needed branches
mc.allocate(reco=reco+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
mc.allocate(true=true+['X_M', '0*sigmat', 'B_ID_GenLvl', 'B_ID_GenLvl', '0*B_ID_GenLvl', '0*B_ID_GenLvl'])
mc.allocate(pdf='0*time', ones='time/time', zeros='0*time')
mc.allocate(weight=weight_mc)

mc.tables_path = output_tables_path
mc.params_path = output_params_path

################################################################################
################################################################################
################################################################################



#%% Compute standard kinematic weights -------------------------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
"""
Name        Value     Stdev    Min   Max  Free
w0            1.0      None   -inf   inf  False
w1         1.0204    0.0015   -inf   inf  False
w2         1.0205    0.0015   -inf   inf  False
w3         0.0026    0.0012   -inf   inf  False
w4        0.00313   0.00072   -inf   inf  False
w5       -0.00033   0.00071   -inf   inf  False
w6         1.0116    0.0010   -inf   inf  False
w7        0.00026   0.00091   -inf   inf  False
w8        0.00000   0.00092   -inf   inf  False
w9        -0.0013    0.0019   -inf   inf  False
"""




print(f"\n{80*'='}\n",
      "Weight the MC samples to match data ones",
      f"\n{80*'='}\n")

reweighter.fit(original        = mc.df[['X_M','B_P','B_PT']],
               target          = data.df[['X_M','B_P','B_PT']],
               original_weight = mc.df.eval(weight_mc),
               target_weight   = data.df.eval(weight_data)
                  );

kinWeight = reweighter.predict_weights(mc.df[['X_M','B_P','B_PT']])
kinWeight = np.where(mc.df.eval(weight_mc)!=0, kinWeight, 0)
print(f"The kinematic-weighting in B_PT, B_P and X_M is done for {MODE}-{TRIGGER}")
print(f"kinWeight: {kinWeight}")

if os.path.exists(output_weight_file):
  try:
    oldWeight = uproot.open(output_weight_file)['DecayTree'].array('kinWeight')
    kinWeight = np.where(kinWeight!=0,kinWeight,oldWeight).astype(np.float64)
  except:
    kinWeight = np.where(kinWeight!=0,kinWeight,0*kinWeight).astype(np.float64)
else:
  os.makedirs(os.path.dirname(output_weight_file), exist_ok=True)
  kinWeight = np.where(kinWeight!=0,kinWeight,0).astype(np.float64)

print(f"kinWeight: {kinWeight}")


print(f'Saving this kinWeight to {output_weight_file}')
with uproot.recreate(output_weight_file,compression=None) as out_file:
  out_file['DecayTree'] = uproot.newtree({'kinWeight':np.float64})
  out_file['DecayTree'].extend({'kinWeight':kinWeight})
  out_file.close()


#%% Compute angWeights correcting with kinematic weights -------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
print(f"\n{80*'='}\n",
      "Compute angWeights correcting MC sample in kinematics",
      f"\n{80*'='}\n")

print('Computing angular weights')
ang_acc = bsjpsikk.get_angular_cov(mc.true, mc.reco, mc.weight*ristra.allocate(kinWeight), **mc.params.valuesdict() )
w, uw, cov, corr = ang_acc
mc.w_corrected = Parameters()

for i in range(0,len(w)):
  correl = {f'w{j}':cov[i][j] for j in range(0,len(w)) if i>0 and j>0}
  mc.w_corrected.add({'name': f'w{i}',
                        'value': w[i],
                        'stdev': uw[i],
                        'free': False,
                        'latex': f'w_{i}',
                        'correl': correl
                      })

# Dump the parameters
print('Dumping parameters')
mc.w_corrected.dump(mc.params_path)
# Export parameters in tex tables
print('Saving table of params in tex')
with open(mc.tables_path, "w") as tex_file:
  tex_file.write(
    mc.w_corrected.dump_latex( caption="""
    Kinematically corrected angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
    category.""" % (YEAR,TRIGGER,MODE.replace('_', ' ') )
    )
  )
tex_file.close()
print(f"Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
print(f"{mc.w_corrected}")


exit()



BREAK





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

    # do the pdf-weighting -----------------------------------------------------
    print('\tCalculating pdfWeight in %s iteration' % str(i))
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **v['sample'].params.valuesdict())
    original_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **v['sample'].params.valuesdict())
    original_pdf_h /= v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **tparams_pdf)
    target_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **tparams_pdf)
    target_pdf_h /= v['sample'].pdf.get()
    v[f'pdfWeight{i}'] = np.nan_to_num(target_pdf_h/original_pdf_h)
    print(f"\tpdfWeight{i}:",v[f'pdfWeight{i}'])

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f'\tCalculating p and pT of K+ and K- weight in {i} iteration')
    reweighter.fit(original        = v['sample'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   target          = data.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   original_weight = v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{i}']*v['kinWeight'].get(),
                   target_weight   = data.df.eval(trig_cut+'sw')
                  );
    kkpWeight = reweighter.predict_weights(v['sample'].df[kin_vars])
    v[f'kkpWeight{i}'] = ristra.allocate(np.where(oweight!=0, kkpWeight, 0))
    print(f"\tkkpWeight{i} = {v[f'kkpWeight{i}']}")

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f"\tAngular weights for {k} category in {i} iteration")
    v[f'w_kkpweighted{i}'] = bsjpsikk.get_angular_weights(
                v['sample'].true,
                v['sample'].reco,
                v['sample'].weight*v['kinWeight']*v[f'kkpWeight{i}'],
                v['sample'].params.valuesdict()
                )
    v[f'w_kkpweighted{i}'] /= v[f'w_kkpweighted{i}'][0]
    print(10*"\t%+.8lf\n" % tuple(v[f'w_kkpweighted{i}']) )



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
for mode in ['MC_Bs2JpsiPhi']:
  d = {}
  for i in range(-1,10):
    # Get angular weights
    f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{i}.root')['fi']
    if i==-1:
      d[f'w_uncorrected'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    elif i==0:
      d[f'w_kinweighted'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    else:
      d[f'w_kkpweighted{i}'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')[f'kinWeight_Unbiased_Step{i}']
    if i==0:
      d[f'kinWeight'] = f.array('kinWeight')
    elif i>0:
      d[f'kkpWeight{i}'] = f.array('kinWeight')
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{i}"]
      d[f'pdfWeight{i+1}'] = f.array('pdfWeight')
  simon[mode] = d
s = simon['MC_Bs2JpsiPhi']

s['w_uncorrected']

f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{-1}.root')['fi']
mat = np.zeros((10,10))
for j1 in range(1,10):
  for j2 in range(1,10):
    mat[j1,j2] = f.array(f'cf{j1+1}{j2+1}')[0]

mat

f.arrays('cf*')
scale = mc['MC_Bs2JpsiPhi']['unbiased']['weight'].get().sum()
scale
0.8*mc['MC_Bs2JpsiPhi']['unbiased']['cov']/(scale*scale)
uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')
