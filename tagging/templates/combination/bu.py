# combination
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


# import ./config/Config_tagger_new.py
# import ./config/JpsiK_cali_mc_1516_os_new.py

# config current tuple {{{

RootFile = "${input_tuple}"
TupleName = "DecayTree"
Selection = ""
Nmax = -1

CalibrationMode = "Bu"
CalibrationLink = "MISTAG"
CalibrationModel = "POLY"
DoCalibrations = 0
CalibrationDegree = 1
UseNewtonRaphson = 0

OS_Kaon_InOSComb = 1
OS_Muon_InOSComb = 1
OS_Electron_InOSComb = 1
OS_Charm_InOSComb = 1
VtxCharge_InOSComb = 1
PerformOfflineCombination_OS = 1
WriteCalibratedMistagBranches = 1
OS_Combination_Write = 1
CalibratedOutputFile = "${output_tuple}"

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
