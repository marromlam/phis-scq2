# create_v0r0
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


# tuples from v0r0 are ready!
#
#
import os
import time
import pandas as pd
import uproot3 as uproot
import yaml
from ipanema.tools.misc import get_vars_from_string


with open("v0r0_branches.yml") as yf:
    BRANCHES = yaml.load(yf, Loader=yaml.FullLoader)


input_path = "/scratch46/marcos.romero/oldtuples/original_test_files"
output_path = "/scratch49/marcos.romero/sidecar"


rfiles = {
    "2015": {
        "MC_Bd2JpsiKstar": {
            "sWeight": "BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat.root",
            "polWeight": "BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root",
            "pdfWeight": "BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root",
            "kinWeight": "BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_BdMCToBdData_BaselineDef_15102018.root",
        },
        "MC_Bs2JpsiPhi": {
            "sWeight": "BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root",
            "polWeight": "BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root",
        },
        "MC_Bs2JpsiPhi_dG0": {
            "sWeight": "BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root",
            "polWeight": "BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root",
            "pdfWeight": "BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root",
            "kinWeight": "BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root",
        },
        "MC_Bu2JpsiKplus": {
            "sWeight": "BuJpsiKplus_MC_2015_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root",
        },
        "Bd2JpsiKstar": {
            "sWeight": "BdJpsiKstar_Data_2015_UpDown_20180821_tmva_cut58_sel_sw_trigCat.root",
            "kinWeight": "BdJpsiKstar_Data_2015_UpDown_20180821_tmva_cut58_sel_sw_trigCat_BdDataToBsData_BaselineDef_15102018.root",
        },
        "Bs2JpsiPhi": {
            "sWeight": "BsJpsiPhi_Data_2015_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root",
        },
        "Bu2JpsiKplus": {
            "sWeight": "BuJpsiKplus_Data_2015_UpDown_20180821_tmva_cut-2_sel_sw_trigCat.root",
        },
    },
    "2016": {
        "MC_Bd2JpsiKstar": {
            "sWeight": "BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat.root",
            "polWeight": "BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root",
            "pdfWeight": "BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root",
            "kinWeight": "BdJpsiKstar_MC_2016_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_BdMCToBdData_BaselineDef_15102018.root",
        },
        "MC_Bs2JpsiPhi": {
            "sWeight": "BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root",
            "polWeight": "BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_PolWeight.root",
        },
        "MC_Bs2JpsiPhi_dG0": {
            "sWeight": "BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw.root",
            "polWeight": "BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root",
            "pdfWeight": "BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root",
            "kinWeight": "BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root",
        },
        "MC_Bu2JpsiKplus": {
            "sWeight": "BuJpsiKplus_MC_2016_UpDown_CombDSTLDST_20181101_Sim09b_tmva_cut-2_sel_sw_trigCat.root",
        },
        "Bd2JpsiKstar": {
            "sWeight": "BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat.root",
            "kinWeight": "BdJpsiKstar_Data_2016_UpDown_20180821_tmva_cut58_sel_sw_trigCat_BdDataToBsData_BaselineDef_15102018.root",
        },
        "Bs2JpsiPhi": {
            "sWeight": "BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root",
        },
        "Bu2JpsiKplus": {
            "sWeight": "BuJpsiKplus_Data_2016_UpDown_20180821_tmva_cut-2_sel_sw_trigCat.root",
        },
    },
}


for ky, vy in rfiles.items():
    for km, vm in vy.items():
        _dfs = []
        for kw, vw in vm.items():
            print(f"{ky} : {km:>20} : {kw:10}: {vw}")
            _rfile = uproot.open(os.path.join(input_path, ky, vw))
            _rtree = _rfile[list(_rfile.keys())[0]]
            # _rdf = _rtree.pandas.df(entrystop=10, flatten=None)
            _branches = list(BRANCHES[km].values())
            _branches = sum([get_vars_from_string(b) for b in _branches], [])
            _tree_branches = [b.decode() for b in _rtree.keys()]
            _branches = list(set(_branches).intersection(_tree_branches))
            print(_branches)
            _rdf = _rtree.pandas.df(branches=_branches, flatten=None)
            # if kw == "polWeight":
            #     _rdf.eval("polWeight = PolWeight", inplace=True)
            # if kw == "pdfWeight":
            #     _rdf.eval("pdfWeight = PDFWeight", inplace=True)
            _dfs.append(_rdf)
        # merge dataframe
        odf = pd.concat(_dfs, axis=1)
        del _dfs
        del _rdf
        _branches = []
        for kb, vb in BRANCHES[km].items():
            try:
                odf.eval(f"{kb} = {vb}", inplace=True)
                _branches.append(kb)
            except:
                print(f"{vb} was not found")
        odf = odf[_branches]
        print(odf)
        _ofile = os.path.join(output_path, ky, km, "v0r0_ready.root")
        print("Saving in:", _ofile)
        with uproot.recreate(_ofile) as of:
            of["DecayTree"] = uproot.newtree({var: "float64" for var in odf})
            of["DecayTree"].extend(odf.to_dict(orient="list"))
        time.sleep(2)
        print(f"xrdcp -f {_ofile} root://eoslhcb.cern.ch//eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/tuples/v0r0/{km}/{ky}/v0r0_{km}_{ky}_ready.root")
        os.system(f"xrdcp -f {_ofile} root://eoslhcb.cern.ch//eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/tuples/v0r0/{km}/{ky}/v0r0_{km}_{ky}_ready.root")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
