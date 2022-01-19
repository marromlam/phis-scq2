# OS combination in *B<sub>s</sub><sup>0</sup> &rightarrow; J/&psi;&Phi;* analysis
## Combination for *B<sup>+</sup>&rightarrow; J/&psi;K<sup>+</sup>, B<sub>s</sub><sup>0</sup> &rightarrow; J/&psi;&Phi;* data & MC samples

1. Setup EPM by [instructions](https://gitlab.cern.ch/lhcb-ft/EspressoPerformanceMonitor) where you 
can find [Wiki](https://gitlab.cern.ch/lhcb-ft/EspressoPerformanceMonitor/wikis/home) on left sidebar 
2. Calibrate OS taggers for B<sup>+</sup> Data & MC ( calibration paramters are saved in <code>‘EspressoCalibrations.py’</code> ) 
3. Use calibration parameters from last step to combine OS taggers for B<sup>+</sup> & B<sub>s</sub><sup>0</sup> samples
4. Add the output branches, ‘OS\_Combination\_DEC(ETA)’, into original file 

## Usage
1. Set taggers in <code>config/Config\_tagger\_new(old).py</code>
2. Calibrate taggers by <code>../../bin/SimpleEvaluator Bu2JpsiK\_data(mc)\_cali.py </code>
3. Combine taggers by <code>sh run\_oscomb\_data(mc).sh</code>
4. Get the tuples with combined os tagger by <code>get\_oscomb\_root.C</code>

## Instructions of the code setting
1. Set taggers you want to use
```python
############################
# Use OS_Kaon as example
############################

# 1, include it; 0, exclude it
OS_Kaon_Use            = 1

# Set brach name and type of tagger
OS_Kaon_BranchDec      = "B_OSKaonLatest_TAGDEC"
OS_Kaon_BranchProb     = "B_OSKaonLatest_TAGETA"
OS_Kaon_TypeDec        = "Int_t"
OS_Kaon_TypeProb       = "Double_t"
```
2. Set calibration code
```python
# Import the setting of taggers
import ./config/Config_tagger_new.py

# Multiple files can be specified by setting NumFiles = N
# and then setting RootFile_1, RootFile_2, ..., RootFile_N
NumFiles = 2
RootFile_1="/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2015_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root"
RootFile_2="/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2016_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root"

TupleName = "DecayTree"
Selection = ""
Nmax = -1  # Events to run, -1 means all

# Set the calibration mode
CalibrationMode   = "Bu"
DoCalibrations    = 1
CalibrationLink   = "MISTAG"
CalibrationDegree = 1
CalibrationModel  = "POLY"
UseNewtonRaphson  = 0

# Set the ID(TRUEID) for Data(MC) & weight
BranchID             = "B_TRUEID"
UseWeight            = 1
WeightFormula        = "sw"
```

3. Set the combination code
```python
# Import the setting of taggers and calibration parameters
import ./config/Config_tagger_new.py
import ./config/JpsiK_cali_mc_1516_os_new.py

# When you have more than 1 input file, EPM doesn't suppport outputing root file 
# So do not multiple input files 
RootFile = "/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root"
TupleName = "DecayTree"
Selection = ""
Nmax      = -1

# The mode should be same as that in calibration
# Set 'DoCalibrations' to be 0 for combination
CalibrationMode  = "Bu"
CalibrationLink  = "MISTAG"
CalibrationModel = "POLY"
DoCalibrations    = 0 
CalibrationDegree = 1
UseNewtonRaphson  = 0

# Perform combination and output root file
OS_Kaon_InOSComb     = 1
OS_Muon_InOSComb     = 1
OS_Electron_InOSComb = 1
OS_Charm_InOSComb    = 1
VtxCharge_InOSComb   = 1
PerformOfflineCombination_OS  = 1
WriteCalibratedMistagBranches = 1
OS_Combination_Write = 1
CalibratedOutputFile = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2016_sel_oscomb_new.root"
```
