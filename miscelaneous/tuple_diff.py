# tuple_diff
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import argparse

import numpy as np
import uproot3 as uproot

p = argparse.ArgumentParser()
p.add_argument("--versionA", default="v2r0p1_ready")
p.add_argument("--versionB", default="v2r3_lbWeight")
args = vars(p.parse_args())


versionA = args["versionA"]
versionB = args["versionB"]


tupleA = uproot.open(versionA)["DecayTree"]
tupleB = uproot.open(versionB)["DecayTree"]
_listA = list(tupleA.keys())
_listB = list(tupleB.keys())
_listA = [l.decode() for l in _listA]
_listB = [l.decode() for l in _listB]
_branches = [
    "time",
    "sigmat",
    "wLb",
    "sw",
    "bdtg3",
    "gb_weights",
    # 'B_PT',
    # 'B_ETA',
    # 'nLongTracks',
    # 'hplus_TRACK_CHI2NDOF',
    # 'hminus_TRACK_CHI2NDOF',
    # 'muplus_TRACK_CHI2NDOF',
    # 'muminus_TRACK_CHI2NDOF'
]
branches = []
for b in _branches:
    if b in _listA:
        if b in _listB:
            branches.append(b)

dfA = tupleA.pandas.df(branches)
dfB = tupleB.pandas.df(branches)


print(f"DATAFRAME {versionA}")
print(dfA)
print(f"\nDATAFRAME {versionB}")
print(dfB)
print(f"\n{80*'*'}")


for b in branches:
    print(f"Diff. in {b:>15} = {np.max(np.abs(dfA[b]-dfB[b]))}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
