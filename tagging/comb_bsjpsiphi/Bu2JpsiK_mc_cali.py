########
#  MC  #
########
import ./config/Config_tagger_new.py
#import ./config/Config_tagger_old.py

# This is the file/directory that you want to run
# Multiple files can be specified by setting NumFiles = N
# and then setting RootFile_1, RootFile_2, ..., RootFile_N
NumFiles = 2
RootFile_1="/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2015_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root"
RootFile_2="/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2016_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root"

TupleName = "DecayTree"
Selection = ""
Nmax = -1  # Events to run, -1 means all

########################
# CALIBRATION SETTINGS #
########################

CalibrationMode   = "Bu"
DoCalibrations    = 1
CalibrationLink   = "MISTAG"
CalibrationDegree = 1
CalibrationModel  = "POLY"
UseNewtonRaphson  = 0

##########################
# BRANCH NAMES AND TYPES #
##########################

BranchID             = "B_TRUEID"
UseWeight            = 0
#WeightFormula        = "nsig_sw"
WeightFormula        = "sw"
