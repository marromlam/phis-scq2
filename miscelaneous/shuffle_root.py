# shuffle_root
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import uproot3 as uproot
import argparse
import ROOT
import os
import time

p = argparse.ArgumentParser()
p.add_argument('--in-file')
# p.add_argument('--out-file')
args = vars(p.parse_args())

treename = list(uproot.open(args['in_file']).keys())[0].decode().split(';')[0]
# print(uproot.open(args['in_file'])['DecayTree'].pandas.df('time'))
# exit()

ROOT.EnableImplicitMT()
print('-- loading file')
df = ROOT.RDataFrame(treename, args['in_file'])
print('-- writting file')
df.Snapshot(treename, 'tmp.root')

time.sleep(1)
os.system(f"mv {args['in_file']} {args['in_file']}.bak")
os.system(f"mv tmp.root {args['in_file']}")

# time.sleep(2)


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
