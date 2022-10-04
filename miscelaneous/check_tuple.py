# check_tuple
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import argparse
import uproot3 as uproot
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--tuple1")
p.add_argument("--tuple2")

args = vars(p.parse_args())

branches = ['time', 'helcosthetaL', 'B_ConstJpsi_M_1']
df1 = uproot.open(args['tuple1'])['DecayTree'].pandas.df(branches=branches)
df2 = uproot.open(args['tuple2'])['DecayTree'].pandas.df(branches=branches)

for b in branches:
  _sum = df1[b].values - df2[b].values
  print(f"Diff {b} : {np.sum(_sum)}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
