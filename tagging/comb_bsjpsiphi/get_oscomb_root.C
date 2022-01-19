#include "get_oscomb_tagger.C"
void get_oscomb_root(){

   const int nNum = 9;
   string tree1[nNum]; 
   string tree2[nNum];
   string tree3[nNum];
   string offile[nNum];

   tree1[0] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2015_UpDown_20180821_tmva_cut58_sel.root/DecayTree";
   tree1[1] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel.root/DecayTree";
   tree1[2] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2017_UpDown_20180821_tmva_cut58_sel.root/DecayTree";
   tree1[3] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2015_UpDown_20180821_tmva_cut58_sel_noLbveto.root/DecayTree";
   tree1[4] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel_noLbveto.root/DecayTree";
   tree1[5] = "/eos/lhcb/wg/B2CC/phis-run2/data/BsJpsiPhi_Data_2017_UpDown_20180821_tmva_cut58_sel_noLbveto.root/DecayTree";
   tree1[6] = "/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root/DecayTree";
   tree1[7] = "/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2016_UpDown_DST_20181101_CombSim09bSim09c_tmva_cut58_sel.root/DecayTree";
   tree1[8] = "/eos/lhcb/wg/B2CC/phis-run2/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root/DecayTree";

   tree2[0] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2015_sel_oscomb_new.root/TaggingTree";
   tree2[1] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2016_sel_oscomb_new.root/TaggingTree";
   tree2[2] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2017_sel_oscomb_new.root/TaggingTree";
   tree2[3] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2015_sel_oscomb_new_noLbveto.root/TaggingTree";
   tree2[4] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2016_sel_oscomb_new_noLbveto.root/TaggingTree";
   tree2[5] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2017_sel_oscomb_new_noLbveto.root/TaggingTree";
   tree2[6] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2015_sel_oscomb_new.root/TaggingTree";
   tree2[7] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2016_sel_oscomb_new.root/TaggingTree";
   tree2[8] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2016_sel_oscomb_new_combdstmdst.root/TaggingTree";

   tree3[0] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2015_sel_oscomb_old.root/TaggingTree";
   tree3[1] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2016_sel_oscomb_old.root/TaggingTree";
   tree3[2] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2017_sel_oscomb_old.root/TaggingTree";
   tree3[3] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2015_sel_oscomb_old_noLbveto.root/TaggingTree";
   tree3[4] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2016_sel_oscomb_old_noLbveto.root/TaggingTree";
   tree3[5] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_Data_2017_sel_oscomb_old_noLbveto.root/TaggingTree";
   tree3[6] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2015_sel_oscomb_old.root/TaggingTree";
   tree3[7] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2016_sel_oscomb_old.root/TaggingTree";
   tree3[8] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/comb/BsJpsiPhi_MC_2016_sel_oscomb_old_combdstmdst.root/TaggingTree";

   offile[0] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2015_UpDown_20180821_tmva_cut58_sel_comb.root";
   offile[1] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel_comb.root";
   offile[2] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2017_UpDown_20180821_tmva_cut58_sel_comb.root";
   offile[3] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2015_UpDown_20180821_tmva_cut58_sel_comb_noLbveto.root";
   offile[4] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2016_UpDown_20180821_tmva_cut58_sel_comb_noLbveto.root";
   offile[5] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_Data_2017_UpDown_20180821_tmva_cut58_sel_comb_noLbveto.root";
   offile[6] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_comb_sw.root";
   offile[7] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2016_UpDown_DST_20181101_CombSim09bSim09c_tmva_cut58_sel_comb.root";
   offile[8] = "/eos/lhcb/wg/B2CC/phis-run2/FTCalib/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_comb_sw.root";

   int iniRun = 8;
   int finRun = 9;
   for( int i=iniRun; i<finRun; i++ ){
	get_oscomb_tagger( tree1[i].c_str(), tree2[i].c_str(), tree3[i].c_str(), offile[i].c_str() );
   }
   //get_oscomb_tagger( tree1[0].c_str(), tree2[0].c_str(), tree3[0].c_str(), offile[0].c_str() );

   exit(0);
}
