from timeit import main
import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, plotting)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec

DOCAz_BINS = 8

__all__ = []


f = uproot.open("/scratch46/marcos.romero/Bu2JpsiKplus5r3.root")["DecayTree"]
df = f.pandas.df()

# eta
for b in ['Jpsi_LOKI_ETA', 'muplus_LOKI_ETA', 'muminus_LOKI_ETA', 'Kplus_LOKI_ETA']:
  plt.close()
  plt.hist(df[b].values, 100, range=(0,6))
  plt.hist(df.query(f"{b}>2 & {b}<4")[b], 100, range=(0,6))
  plt.xlabel("-".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")
# PID
for b in ['muplus_PIDmu', 'muminus_PIDmu', 'Kplus_PIDK']:
  plt.close()
  plt.hist(df[b].values, 100, range=(-6,6))
  plt.hist(df.query(f"{b}>0")[b], 100, range=(-6,6))
  plt.xlabel("\_".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")
# pT
for b in ['muplus_PT', 'muminus_PT', 'Kplus_PT']:
  plt.close()
  plt.hist(df[b].values, 100, range=(0,4000))
  plt.hist(df.query(f"{b}>550")[b].values, 100, range=(0,4000))
  plt.xlabel("\_".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")
# p
for b in ['Kplus_P']:
  plt.close()
  plt.hist(df[b].values, 100, range=(0,40000))
  plt.hist(df.query(f"{b}>1000")[b].values, 100, range=(0,40000))
  plt.xlabel("\_".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")
# chi2ndof
for b in ['Bu_LOKI_DTF_CHI2NDOF', 'muplus_TRACK_CHI2NDOF', 'muminus_TRACK_CHI2NDOF', 'Kplus_TRACK_CHI2NDOF']:
  plt.close()
  plt.hist(df[b].values, 100, range=(0,10))
  plt.hist(df.query(f"{b}<4")[b].values, 100, range=(0,10))
  plt.xlabel("\_".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")
# chi2ndof
for b in ['Bu_IPCHI2_OWNPV', 'Bu_MINIPCHI2', 'Kplus_TRACK_CHI2NDOF']:
  plt.close()
  plt.hist(df[b].values, 100, range=(0,10))
  plt.hist(df.query(f"{b}<4")[b].values, 100, range=(0,10))
  plt.xlabel("\_".join(b.split('_')))
  plt.savefig(f"mass_fit_cuts/{b}.pdf")

# others
plt.close()
plt.hist(df['Bu_PVConst_PV_Z'].values, 100, range=(0,200))
plt.hist(df.query(f"abs(Bu_PVConst_PV_Z)<100")['Bu_PVConst_PV_Z'].values, 100, range=(0,200))
plt.xlabel("\_".join("Bu_PVConst_PV_Z".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_PVConst_PV_Z.pdf")
plt.close()
plt.hist(df['Bu_MINIPCHI2NEXTBEST'].values, 100, range=(0,200))
plt.hist(df.query(f"Bu_MINIPCHI2NEXTBEST>50 | Bu_MINIPCHI2NEXTBEST==-1")['Bu_MINIPCHI2NEXTBEST'].values, 100, range=(0,200))
plt.xlabel("\_".join("Bu_MINIPCHI2NEXTBEST".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_MINIPCHI2NEXTBEST.pdf")
plt.close()
plt.hist(df['Jpsi_ENDVERTEX_CHI2'].values, 100, range=(0,20))
plt.hist(df.query(f"Jpsi_ENDVERTEX_CHI2<16")['Jpsi_ENDVERTEX_CHI2'].values, 100, range=(0,20))
plt.xlabel("\_".join("Jpsi_ENDVERTEX_CHI2".split('_')))
plt.savefig(f"mass_fit_cuts/Jpsi_ENDVERTEX_CHI2.pdf")
plt.close()
plt.hist(df['Bu_LOKI_FDS'].values, 100, range=(0,10))
plt.hist(df.query(f"Bu_LOKI_FDS<16")['Bu_LOKI_FDS'].values, 100, range=(0,10))
plt.xlabel("\_".join("Bu_LOKI_FDS".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_LOKI_FDS.pdf")
# Bu_M
plt.close()
plt.hist(df['Jpsi_M'].values, 100, range=(3000,3200))
plt.hist(df.query(f"Jpsi_M>3030 & Jpsi_M<3150")['Jpsi_M'].values, 100, range=(3000,3200))
plt.xlabel("\_".join("Jpsi_M".split('_')))
plt.savefig(f"mass_fit_cuts/Jpsi_M.pdf")
plt.close()
plt.hist(df['Bu_LOKI_MASS_JpsiConstr_NoPVConstr'].values, 100, range=(5000,5500))
plt.hist(df.query(f"Bu_LOKI_MASS_JpsiConstr_NoPVConstr>5170 & Bu_LOKI_MASS_JpsiConstr_NoPVConstr<5400")['Bu_LOKI_MASS_JpsiConstr_NoPVConstr'].values, 100, range=(5000,5500))
plt.xlabel("\_".join("Bu_LOKI_MASS_JpsiConstr_NoPVConstr".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_LOKI_MASS_JpsiConstr_NoPVConstr.pdf")
# time
plt.close()
plt.hist(df.eval('Bu_PVConst_ctau/0.299792458').values, 100, range=(0,15))
plt.hist(df.query(f"(Bu_PVConst_ctau/0.299792458)>0.3 & (Bu_PVConst_ctau/0.299792458)<14").eval('Bu_PVConst_ctau/0.299792458').values, 100, range=(0,15))
plt.xlabel("\_".join("Bu_PVConst_ctau".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_PVConst_ctau.pdf")
# Bu_PVConst_chi2 / Bu_PVConst_nDOF
plt.close()
plt.hist(df.eval('Bu_PVConst_chi2/Bu_PVConst_nDOF').values, 100, range=(0,10))
plt.hist(df.query(f"(Bu_PVConst_chi2/Bu_PVConst_nDOF)<5").eval('Bu_PVConst_chi2/Bu_PVConst_nDOF').values, 100, range=(0,10))
plt.xlabel("\_".join("(Bu_PVConst_chi2/Bu_PVConst_nDOF)".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_PVConst_chi2_Bu_PVConst_nDOF.pdf")
# Trigger
plt.close()
plt.hist(df.eval('Bu_L0MuonDecision_TOS').values, 10, range=(-1,1))
plt.hist(df.query(f"(Bu_L0MuonDecision_TOS==1 | Bu_L0DiMuonDecision_TOS==1)").eval('Bu_L0MuonDecision_TOS').values, 10, range=(-1,1))
plt.xlabel("\_".join("Bu_L0MuonDecision_TOS".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_L0MuonDecision_TOS.pdf")
plt.close()
plt.hist(df.eval('Bu_Hlt1DiMuonHighMassDecision_TOS').values, 10, range=(-1,1))
plt.hist(df.query(f"Bu_Hlt1DiMuonHighMassDecision_TOS==1").eval('Bu_Hlt1DiMuonHighMassDecision_TOS').values, 10, range=(-1,1))
plt.xlabel("\_".join("Bu_Hlt1DiMuonHighMassDecision_TOS".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_Hlt1DiMuonHighMassDecision_TOS.pdf")
plt.close()
plt.hist(df.eval('Bu_Hlt2DiMuonDetachedJPsiDecision_TOS').values, 10, range=(-1,1))
plt.hist(df.query(f"Bu_Hlt2DiMuonDetachedJPsiDecision_TOS==1").eval('Bu_Hlt2DiMuonDetachedJPsiDecision_TOS').values, 10, range=(-1,1))
plt.xlabel("\_".join("Bu_Hlt2DiMuonDetachedJPsiDecision_TOS".split('_')))
plt.savefig(f"mass_fit_cuts/Bu_Hlt2DiMuonDetachedJPsiDecision_TOS.pdf")
