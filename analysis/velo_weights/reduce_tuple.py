__all__ = []
import uproot3 as uproot
import argparse
import pandas as pd

from analysis.samples.sync_tuples import vsub_dict

p = argparse.ArgumentParser()
p.add_argument("--input-sample")
p.add_argument("--output-sample")
p.add_argument("--version")
p.add_argument("--mode")
p.add_argument("--year")
args = vars(p.parse_args())

MODE = args['mode']
YEAR = args['year']
VERSION = args['version']

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
    'Bu_Hlt2DiMuonDetachedJPsiDecision_TOS',
    # 'Bu_PT'
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
# ['Bu2JpsiKplus']['DecayTree']
arrs = sample.arrays(branches)

# transform jagged array of DOCAZ to array geting only first element
for b in jagged_branches:
    try:
        arrs[b] = sample[b].array()[:, 0]
    except:
        arrs[b] = sample[b].array()
arrs = {k.decode(): v for k, v in arrs.items()}

result = pd.DataFrame(arrs)
print(result)

# place cuts according to version substring
list_of_cuts = []
vsub_cut = None
for k, v in vsub_dict.items():
    if k in VERSION.split('@'):
        try:
            noe = len(result.query(v))
            if k in ("g210300", "l210300"):
                if "MC" in args['output']:
                    print("MCs are not cut in runNumber")
                elif "2018" not in args['output']:
                    print("Only 2018 is cut in runNumber")
                else:
                    list_of_cuts.append(v)
            elif (k in ("UcosK", "LcosK")) and 'Bd2JpsiKstar' not in MODE:
                print("Cut in cosK was only planned in Bd")
            else:
                list_of_cuts.append(v)
            if noe == 0:
                print(f"ERROR: This cut leaves df empty. {v}")
                print("       Query halted.")
        except:
            print(f"non hai variable para o corte {v}")

if list_of_cuts:
    vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"


# place the cut
print(f"{80*'-'}\nApplied cut: {vsub_cut}\n{80*'-'}")
if vsub_cut:
    result = result.query(vsub_cut)
print(result)

# write reduce tuple
with uproot.recreate(args['output_sample'], compression=None) as rf:
    rf['DecayTree'] = uproot.newtree({var: 'float64' for var in result})
    rf['DecayTree'].extend(result.to_dict(orient='list'))
rf.close()


# vim :set ts=4 sw=4 sts=4 et :
