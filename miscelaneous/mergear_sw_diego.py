import uproot3 as uproot
import pandas as pd
import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


import os
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir("/scratch24/diego/phis") if isfile(join("/scratch24/diego/phis", f))]


for year in [2015, 2016, 2017, 2018]:
    for mode in ['MC_Bs2JpsiPhi_dG0', 'MC_Bd2JpsiKstar', 'MC_Bs2JpsiPhi', 'MC_Bu2JpsiKplus', 'Bd2JpsiKstar', 'Bs2JpsiPhi', 'Bu2JpsiKplus']:
        print(f"Copy {year}-{mode}")
        os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/{mode}/{{v0r5,v0r6}}_sWeight.root")
        os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/{mode}/{{v0r5,v0r6@pTB1}}_sWeight.root")
        os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/{mode}/{{v0r5,v0r6@pTB2}}_sWeight.root")
        os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/{mode}/{{v0r5,v0r6@pTB3}}_sWeight.root")
        os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/{mode}/{{v0r5,v0r6@pTB4}}_sWeight.root")


tuples = {}
mode = "Bs"
for year in [2015, 2016, 2017, 2018]:
  tuples[year] = {}
  for trigger in ['Biased', 'Unbiased']:
    tuples[year][trigger] = {}
    for ptbin in ["1","2","3","4"]:
      tuples[year][trigger][ptbin] = {}
      for massbin in ["1","2","3","4","5","6"]:
        file_name = f"v0r5_sWeight{year}{trigger}pTB{ptbin}KKB{massbin}sW8s{mode}.root" 
        if file_name in onlyfiles:
          tuples[year][trigger][ptbin][massbin] = file_name
          print(f"Success for {file_name}")
        else:
          print(f"Error for {file_name}")


tuple_year = {}
for year in [2015, 2016, 2017,2018]:
    print(f"================ {year} =============================")
    tuple_year[year] = list(flatten(tuples[year]).values())
    if len(tuple_year[year]) > 0:
       _root = [uproot.open(f"/scratch24/diego/phis/{f}")['DecayTree'].pandas.df() for f in tuple_year[year]] 
       _pandas = pd.concat(_root)
       _pandas = _pandas.drop(["sw", "cor_sWeights_Bs"], axis=1)
       _pandas.eval(f"sw = sWeights_Bs", inplace=True)
       tuple_year[year] = _pandas
       with uproot.recreate(f"/scratch46/marcos.romero/sidecar/{year}/Bs2JpsiPhi/v0r6_sWeight.root") as rfile:
          rfile["DecayTree"] = uproot.newtree({var:'float64' for var in _pandas})
          rfile["DecayTree"].extend(_pandas.to_dict(orient='list'))
       os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/Bs2JpsiPhi/{{v0r6,v0r6@pTB1}}_sWeight.root")
       os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/Bs2JpsiPhi/{{v0r6,v0r6@pTB2}}_sWeight.root")
       os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/Bs2JpsiPhi/{{v0r6,v0r6@pTB3}}_sWeight.root")
       os.system(f"cp /scratch46/marcos.romero/sidecar/{year}/Bs2JpsiPhi/{{v0r6,v0r6@pTB4}}_sWeight.root")
    print(tuple_year[year])




