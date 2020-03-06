import uproot
import importlib
import json
import numpy as np
import matplotlib.pyplot as plt

pdf = importlib.import_module('pdf_weighting')
polarity = importlib.import_module('polarity_weighting')
kinematic = importlib.import_module('kinematic_weighting')
path = '/scratch03/marcos.romero/phisRun2/SideCar/'



print('TEST: All weighting methods loaded!')



print('TEST: polarity-weighting starting...')
original_file     = path+'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root'#path+'MC_JpsiPhi_sample2016.root'
original_file     = "/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root"

target_file       = path+'BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel_comb_sw.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = path+'MC_JpsiPhi_sample2016_polWeight.root'

#polarity.polarity_weighting(original_file, original_treename,target_file, target_treename,output_file)
print('TEST: polarity-weighting is done.\n')



print('TEST: pdf-weighting starting...')
input_file      = path+'MC_JpsiPhi_sample2016_polWeight.root'
tree_name       = 'DecayTree'
output_file     = path+'MC_JpsiPhi_sample2016_pdfWeight.root'
original_params = '/home3/marcos.romero/phis-scq/backup/input/tad-2016-both-simon1.json'
target_params   = '/home3/marcos.romero/phis-scq/backup/input/tad-2016-both-simon2.json'

#pdf.pdf_weighting(input_file, tree_name, output_file, target_params, original_params, 'MC_BsJpsiPhi')
print('TEST: pdf-weighting is done.\n')





print('TEST: kinematic-weighting starting...')
original_file     = path+ 'MC_JpsiPhi_sample2016_pdfWeight.root'
target_file       = path+ 'BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel_comb_sw.root'
target_file       = '/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = path+ 'MC_JpsiPhi_sample2016_kinWeight.root'
original_vars     = 'B_PT X_M'
target_vars       = 'B_PT X_M'
original_weight   = '(sw/gb_weights)*polWeight*pdfWeight'
target_weight     = 'sw'
n_estimators      = 20
learning_rate     = 0.3
max_depth         = 3
min_samples_leaf  = 1000
trunc             = 0

kinematic.kinematic_weighting(original_file, original_treename, original_vars, original_weight, target_file, target_treename, target_vars, target_weight, output_file, n_estimators, learning_rate, max_depth, min_samples_leaf, trunc)
print('TEST: kinematic-weighting is done.\n')




# polWeight
foo = uproot.open(path+'MC_JpsiPhi_sample2016_polWeight.root')['DecayTree'].array('polWeight')
bar = uproot.open(path+'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].array('PolWeight')
plt.close()
print(foo-bar,np.amax(np.abs(foo-bar)))
plt.plot(foo-bar)
plt.title(np.amax(np.abs(foo-bar)))
plt.show()

# pdfWeight
foo = uproot.open(path+'MC_JpsiPhi_sample2016_pdfWeight.root')['DecayTree'].array('pdfWeight')
bar = uproot.open(path+'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights'].array('PDFWeight')
plt.close()
print(foo-bar,np.amax(np.abs(foo-bar)))
plt.plot(foo-bar)
plt.title(np.amax(np.abs(foo-bar)))
plt.show()

# kinWeight
foo = uproot.open(path+'MC_JpsiPhi_sample2016_kinWeight.root')['DecayTree'].array('kinWeight')
bar = uproot.open(path+'BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
print(foo,bar)
plt.close()
print(foo-bar,np.amax(np.abs(foo-bar)))
plt.plot(foo-bar)
plt.title(np.amax(np.abs(foo-bar)))
plt.show()
