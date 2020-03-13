from collections import namedtuple

footprints = namedtuple('footprints', ['params', 'score'])


import os
os.listdir('/scratch03/marcos.romero/phisRun2/original_test_files/2015')
%matplotlib inline
import uproot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2015/MC_Bd2JpsiKstar/test_pdfWeight.root')['DecayTree'].array('pdfWeight')
b = uproot.open('/home3/marcos.romero/BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsBdRun1.root')['PDFWeights'].array('PDFWeight')
c = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2015/BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root')['PDFWeights'].array('PDFWeight')
                                                                            BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd

plt.hist(b-c)
np.where(abs(a-b)>1e-5,1,0).sum()
np.where(abs(a-b)<1e-5,1,0).sum()
plt.hist(a-b)

os.

a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2015/MC_Bd2JpsiKstar/test_pdfWeight.root')['DecayTree'].array('pdfWeight')
b = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2015/BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root')['PDFWeights'].array('PDFWeight')
n_shit = np.where(abs(a-b)>1e-5,1,0).sum()
n_ok   = np.where(abs(a-b)>1e-5,0,1).sum()
print(n_shit, n_ok, n_shit/n_ok)
plt.plot(a-b)

a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bd2JpsiKstar/test_pdfWeight.root')['DecayTree'].array('pdfWeight')
b = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root')['PDFWeights'].array('PDFWeight')
plt.plot(a-b)
a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_pdfWeight.root')['DecayTree'].array('pdfWeight')
b = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2016/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights'].array('PDFWeight')
plt.plot(a-b)

n_shit = np.where(abs(a-b)>1e-5,1,0).sum()
n_ok   = np.where(abs(a-b)>1e-5,0,1).sum()
print(n_shit/n_ok)
n_shit
plt.hist(a-b)
for i in range(100):
  print( (a-b)[i] )





df_a = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2015/MC_Bd2JpsiKstar/test_kinWeight.root')['DecayTree'].pandas.df()
df_b = uproot.open('/scratch03/marcos.romero/phisRun2/original_test_files/2015/BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_BdMCToBdData_BaselineDef_15102018.root')['weights'].pandas.df()
df_a.rename(columns={"kinWeight": "me"}, inplace=True)
df_b.rename(columns={"kinWeight": "simon"}, inplace=True)


df_c = pd.concat([df_a, df_b], axis=1)
df_c
df_c.query("abs(me-simon)>1e-4")[['helcosthetaK','truehelcosthetaK_GenLvl']]
