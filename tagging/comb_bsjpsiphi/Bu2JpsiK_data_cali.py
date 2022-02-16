######################################
# Data, Latest tagger
######################################
import ./config/Config_tagger_new.py
#import ./config/Config_tagger_old.py

NumFiles  = 2
RootFile_1 = "/eos/lhcb/wg/B2CC/phis-run2/data/BuJpsiKplus/BuJpsiKplus_Data_2015_UpDown_20180821_tmva_cut-2_sel_sw_trigCat.root"
RootFile_2 = "/eos/lhcb/wg/B2CC/phis-run2/data/BuJpsiKplus/BuJpsiKplus_Data_2016_UpDown_20180821_tmva_cut-2_sel_sw_trigCat.root"
#RootFile  = "/eos/lhcb/wg/B2CC/phis-run2/data/BuJpsiKplus/BuJpsiKplus_Data_2015_UpDown_20180821_tmva_cut-2_sel_sw_trigCat.root"
TupleName = "DecayTree"
Selection = ""
Nmax      = -1

CalibrationMode  = "Bu"
CalibrationLink  = "MISTAG"
CalibrationModel = "POLY"
DoCalibrations    = 1
CalibrationDegree = 1
UseNewtonRaphson  = 0

BranchID  = "B_ID"
UseWeight = 1
WeightFormula = "sw"
