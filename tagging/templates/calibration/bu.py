# template for Bu2JpsiKplus tagging calibration
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


# OS tagger {{{

OS_Kaon_Use = 1
OS_Muon_Use = 1
OS_Electron_Use = 1
OS_Charm_Use = 1
VtxCharge_Use = 1

OS_Kaon_BranchDec = "B_OSKaonLatest_TAGDEC"
OS_Kaon_BranchProb = "B_OSKaonLatest_TAGETA"
OS_Kaon_TypeDec = "Int_t"
OS_Kaon_TypeProb = "Double_t"

OS_Muon_BranchDec = "B_OSMuonLatest_TAGDEC"
OS_Muon_BranchProb = "B_OSMuonLatest_TAGETA"
OS_Muon_TypeDec = "Int_t"
OS_Muon_TypeProb = "Double_t"

OS_Electron_BranchDec = "B_OSElectronLatest_TAGDEC"
OS_Electron_BranchProb = "B_OSElectronLatest_TAGETA"
OS_Electron_TypeDec = "Int_t"
OS_Electron_TypeProb = "Double_t"

OS_Charm_BranchDec = "B_OSCharm_TAGDEC"
OS_Charm_BranchProb = "B_OSCharm_TAGETA"
OS_Charm_TypeDec = "Int_t"
OS_Charm_TypeProb = "Double_t"

VtxCharge_BranchDec = "B_OSVtxCh_TAGDEC"
VtxCharge_BranchProb = "B_OSVtxCh_TAGETA"
VtxCharge_TypeDec = "Int_t"
VtxCharge_TypeProb = "Double_t"

# }}}

# Input configuration {{{
# This is the file/directory that you want to run
# Multiple files can be specified by setting NumFiles = N
# and then setting RootFile_1, RootFile_2, ..., RootFile_N

NumFiles = 1
RootFile_1 = "${input_tuple}"

TupleName = "DecayTree"
Selection = ""
Nmax = -1  # Events to run, -1 means all

# }}}

# calibration settings {{{

CalibrationMode = "Bu"
DoCalibrations = 1
CalibrationLink = "MISTAG"
CalibrationDegree = 1
CalibrationModel = "POLY"
UseNewtonRaphson = 0

# }}}

# branch names and types {{{

BranchID = "${idvar}"
UseWeight = 1
WeightFormula = "${sweight}"

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
