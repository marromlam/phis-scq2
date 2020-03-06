from collections import namedtuple

footprints = namedtuple('footprints', ['params', 'score'])
import uproot
a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_pdfWeight.root')['DecayTree']
b = uproot.open('/scratch03/marcos.romero/phisRun2/cooked_test_files/MC_Bs2JpsiPhi_dG0/test_kinWeight.root')['DecayTree']
c = uproot.open('/scratch03/marcos.romero/phisRun2/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root')['weights']
a.array('pdfWeight')
b.array('pdfWeight')
c.array('kinWeight')-b.array('kinWeight')
'MC_Bs'[:5]


d = uproot.open('/scratch03/marcos.romero/phisRun2/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights']
d.array('PDFWeight')



a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/Bd2JpsiKstar/test_kinWeight.root')['DecayTree']
b = uproot.open('/scratch03/marcos.romero/phisRun2/cooked_test_files/Bd2JpsiKstar/test.root')['DecayTree']
a.array('kinWeight')-b.array('kinWeight')



foo = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_selected_bdt_sw.root')['DecayTree']
bar = uproot.open('/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root')['DecayTree']
bar.keys()
foo.array('truehelcosthetaK_GenLvl')
bar.array('cosThetaKRef_GenLvl')
