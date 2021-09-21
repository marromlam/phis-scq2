import uproot3 as uproot
import argparse


p = argparse.ArgumentParser()
p.add_argument("--input-sample")
p.add_argument("--output-sample")
args = vars(p.parse_args())



# input_file = "/scratch46/marcos.romero/Bu2JpsiKplus5.root"
# input_file = "/scratch46/marcos.romero/MC_Bu2JpsiKplus.root"
# output_file = input_file.split(".root")[0] + "r3.root"

branches = [
  'Bu_M', 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr',
  'Bu_IPCHI2_OWNPV',
  'Jpsi_M',
  'Jpsi_ENDVERTEX_CHI2',
  'Bu_ENDVERTEX_CHI2',
  'Bu_TAU',
  'Jpsi_LOKI_ETA', 'muplus_LOKI_ETA', 'muminus_LOKI_ETA', 'Kplus_LOKI_ETA',
  'Bu_LOKI_DTF_CHI2NDOF',
  'muplus_TRACK_CHI2NDOF', 'muminus_TRACK_CHI2NDOF', 'Kplus_TRACK_CHI2NDOF',
  'Bu_IPCHI2_OWNPV',
  'Bu_MINIPCHI2', 'Bu_MINIPCHI2NEXTBEST',  # 'Bu_hasBestDTFCHI2',
  'Bu_LOKI_DTF_CHI2NDOF',
  'Kplus_PT', 'Kplus_P', 'muplus_PT', 'muminus_PT',
  'Jpsi_ENDVERTEX_CHI2',
  'Bu_LOKI_FDS',
  'Jpsi_M',
  'muplus_PIDmu', 'muminus_PIDmu', 'Kplus_PIDK',
  'Bu_L0MuonDecision_TOS', 'Bu_L0DiMuonDecision_TOS',
  'Bu_Hlt1DiMuonHighMassDecision_TOS',
  'Bu_Hlt2DiMuonDetachedJPsiDecision_TOS'
]

jagged_branches = [
  b'Bu_PVConst_veloMatch', b'Bu_PVConst_veloMatch_stdmethod',
  b'PVZ', b'Bu_PVConst_PV_Z',
  b'Bu_PVConst_J_psi_1S_muminus_0_DOCAz',
  # b'Bu_PVConst_J_psi_1S_muplus_0_DOCAz',
  b'Bu_PVConst_Kplus_DOCAz',
  # b'Bu_PVConstPVReReco_chi2',  # b'Bu_PVConstPVReReco_nDOF',
  # b'Bu_PVConstPVReReco_nDOF', 
  b'Bu_PVConst_nDOF',
  b'Bu_PVConst_ctau', b'Bu_PVConst_chi2', b'Bu_PVConst_nDOF'
]

# create dict of arrays
sample = uproot.open(args['input_sample'])
ttree = sample.keys()[0]
print(ttree)
sample = sample[ttree]
print(sample.keys())
#['Bu2JpsiKplus']['DecayTree']
arrs = sample.arrays(branches)

# transform jagged array of DOCAZ to array geting only first element
for b in jagged_branches:
  try:
    arrs[b] = sample[b].array()[:, 0]
  except:
    arrs[b] = sample[b].array()
arrs = {k.decode(): v for k, v in arrs.items()}

# write reduce tuple
with uproot.recreate(args['output_sample'], compression=None) as out_file:
 out_file["DecayTree"] = uproot.newtree({var: 'float64' for var in arrs})
 out_file["DecayTree"].extend(arrs)
out_file.close()
