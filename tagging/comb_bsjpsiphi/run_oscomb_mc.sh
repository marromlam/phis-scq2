#/bin/bash

PATH=/afs/cern.ch/user/w/whu/project_EPM/bin:$PATH; export PATH

export EPM_ROOT=/afs/cern.ch/user/w/whu/project_EPM/bin
echo "------------------------------------"
echo " EPM_ROOT points to ${EPM_ROOT}"

RunExe=${EPM_ROOT}/SimpleEvaluator
Script='oscomb_mc.py'

# Set taggers and calibration parameters
Tagger=( 
'./config/Config_tagger_new.py' 
'./config/Config_tagger_new.py' 
'./config/Config_tagger_new.py' 
'./config/Config_tagger_new.py' 
'./config/Config_tagger_old.py'  
'./config/Config_tagger_old.py'  
'./config/Config_tagger_old.py'  
'./config/Config_tagger_old.py'  
)

Parinf=( 
'./config/JpsiK_cali_mc_1516_os_new.py' 
'./config/JpsiK_cali_mc_1516_os_new.py'   
'./config/JpsiK_cali_mc_1516_os_new.py' 
'./config/JpsiK_cali_mc_1516_os_new.py'   
'./config/JpsiK_cali_mc_1516_os_old.py' 
'./config/JpsiK_cali_mc_1516_os_old.py'   
'./config/JpsiK_cali_mc_1516_os_old.py' 
'./config/JpsiK_cali_mc_1516_os_old.py'   
)

# Set input files and trees
Iffile=( 
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2015_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2016_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2015_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BuJpsiKplus_MC_2016_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root'
'/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root'
)
Iftree=( 
'DecayTree' 'DecayTree' 'DecayTree' 'DecayTree' 
'DecayTree' 'DecayTree' 'DecayTree' 'DecayTree' 
)

# Set output files
Offile=(
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BuJpsiK_MC_2015_sel_oscomb_new.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BuJpsiK_MC_2016_sel_oscomb_new.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2015_sel_oscomb_new.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2016_sel_oscomb_new.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BuJpsiK_MC_2015_sel_oscomb_old.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BuJpsiK_MC_2016_sel_oscomb_old.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2015_sel_oscomb_old.root'
'/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2016_sel_oscomb_old.root'
)

# Set the files you want run
IniI=3
FinI=4

# Create the code and run it
for((i=$IniI; i<$FinI; i++))
do
   tmpTagger="import ${Tagger[i]} "
   tmpParinf="import ${Parinf[i]} "
   tmpIffile="RootFile  = \"${Iffile[i]}\" "
   tmpIftree="TupleName = \"${Iftree[i]}\" "
   tmpOffile="CalibratedOutputFile = \"${Offile[i]}\" "
   echo ${tmpTagger} >  $Script 
   echo ${tmpParinf} >> $Script
   echo " ">> $Script
   echo ${tmpIffile} >> $Script
   echo ${tmpIftree} >> $Script
   echo "Selection = \"\"" >> $Script
   echo "Nmax      = -1"   >> $Script
   echo " ">> $Script
   echo "CalibrationMode  = \"Bu\""        >> $Script
   echo "CalibrationLink  = \"MISTAG\""    >> $Script
   echo "CalibrationModel = \"POLY\""      >> $Script
   echo "DoCalibrations    = 0" >> $Script
   echo "CalibrationDegree = 1" >> $Script
   echo "UseNewtonRaphson  = 0" >> $Script 
   echo " " >> $Script
   echo "OS_Kaon_InOSComb     = 1"  >> $Script
   echo "OS_Muon_InOSComb     = 1" >> $Script
   echo "OS_Electron_InOSComb = 1" >> $Script
   echo "OS_Charm_InOSComb    = 1" >> $Script
   echo "VtxCharge_InOSComb   = 1" >> $Script
   echo "PerformOfflineCombination_OS  = 1"  >> $Script
   echo "WriteCalibratedMistagBranches = 1" >> $Script
   echo "OS_Combination_Write = 1" >> $Script
   echo ${tmpOffile} >> $Script
   echo "------------------------------------"
   cat  $Script
   echo "------------------------------------"
   $RunExe $Script
   rm $Script
done

