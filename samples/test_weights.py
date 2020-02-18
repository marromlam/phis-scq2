import uproot
import importlib
import json
pdf = importlib.import_module('pdf-weighting')
polarity = importlib.import_module('polarity-weighting')
kinematic = importlib.import_module('kinematic-weighting')
path = '/scratch03/marcos.romero/phisRun2/time_acceptance/'

print('TEST: All weighting methods loaded!')



print('TEST: polarity-weighting starting...')
original_file     = path+'MC_JpsiPhi_sample2016.root'
target_file       = path+'JpsiPhi_sample2016.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = path+'MC_JpsiPhi_sample2016_polWeight.root'

polarity.polarity_weighting(original_file, original_treename,target_file, target_treename,output_file)
print('TEST: polarity-weighting is done.\n')



print('TEST: pdf-weighting starting...')
input_file      = path+'MC_JpsiPhi_sample2016_polWeight.root'
tree_name       = 'DecayTree'
output_file     = path+'MC_JpsiPhi_sample2016_pdfWeight.root'
original_params = json.load(open(path+'input/tad-MC_Bs2JpsiPhi-2016-both-baseline.json'))
target_params   = json.load(open(path+'input/tad-Bs2JpsiPhi-Run1-both-baseline.json'))

pdf.pdf_weighting(input_file, tree_name, output_file, target_params, original_params, 'MC_BsJpsiPhi')
print('TEST: pdf-weighting is done.\n')



print('TEST: kinematic-weighting starting...')
original_file     = path+ 'MC_JpsiPhi_sample2016_pdfWeight.root'
target_file       = path+ 'JpsiPhi_sample2016.root'
original_treename = 'DecayTree'
target_treename   = 'DecayTree'
output_file       = path+ 'MC_JpsiPhi_sample2016_kinWeight.root'
original_vars     = ['B_PT','X_M']
target_vars       = ['B_PT','X_M']
original_weight   = 'polWeight*pdfWeight/gb_weights'
target_weight     = 'sw'
n_estimators      = 20
learning_rate     = 0.3
max_depth         = 3
min_samples_leaf  = 1000
trunc             = 0

kinematic.kinematic_weighting(original_file, original_treename, original_vars, original_weight, target_file, target_treename, target_vars, target_weight, output_file, n_estimators, learning_rate, max_depth, min_samples_leaf, trunc)
print('TEST: kinematic-weighting is done.\n')
