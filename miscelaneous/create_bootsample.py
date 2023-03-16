# create_bootsample
#
#

__all__ = []
__author__ = ["Ramón Ángel Ruiz Fernández"]
__email__ = ["rruizfer@CERN.CH"]



import uproot3 as uproot
import numpy as np
from typing import Tuple, Dict

def make_bootstraps(index: np.array, n_boot: int ) -> Dict[str, np.array]:
  """
    Function to generate bootstrapped samples
    
    Inputs:

        n_boot-> integer number of bootstraps to produce
        index         -> array of dataframe index
        
    Outputs:
        {'boot_n': np.array } -> dictionary of dictionaries containing the bootstrap index array
  """
  out = {}
  for i in range(0, n_boot):
	  out[f"{i}"] = np.random.choice(index, replace=True, size = len(index))
  return out




tuples = { 
	             "2015" : "/scratch48/ramon.ruiz/sidecar/2015/Bs2JpsiPhi/v5r4@LcosK.root",
	             "2016" : "/scratch48/ramon.ruiz/sidecar/2016/Bs2JpsiPhi/v5r4@LcosK.root",
	             "2017" : "/scratch48/ramon.ruiz/sidecar/2017/Bs2JpsiPhi/v5r4@LcosK.root",
	             "2018" : "/scratch48/ramon.ruiz/sidecar/2018/Bs2JpsiPhi/v5r4@LcosK.root"
           }

b_to_load  = ["cosK", "cosL", "hphi", "time", "mHH", "sigmat"]
b_to_load += ["tagOSdec", "tagSSdec", "tagOSeta", "tagSSeta"]
b_to_load += ["sWeight", "sw", "lbWeight", "mB"]
b_to_load += ["hlt1b"]

 #For what is seen on K* we will need more 
n_boots = 700

years = ["2015", "2016", "2017", "2018"]

for y in years:
  print(f"Doing {n_boots} Bootstrapping for year {y}")
  df = uproot.open(tuples[y])["DecayTree"].pandas.df(branches=b_to_load)
  boot = make_bootstraps(df.index, n_boots)
  for i in range(0, 700):
    # print(df)
    df_boot = df.loc[boot[f"{i}"].tolist()]
    # print(df_boot)
    output_path = tuples[y].replace("@LcosK", f"@LcosKboot{i}")
    print(output_path)
    with uproot.recreate(output_path) as f:
      _branches = {}
      for k, v in df_boot.items():
        if 'int' in v.dtype.name:
          _v = np.int32
        elif 'bool' in v.dtype.name:
          _v = np.int32
        else:
          _v = np.float64
        _branches[k] = _v
      mylist = list(dict.fromkeys(_branches.values()))
      f["DecayTree"] = uproot.newtree(_branches)
      f["DecayTree"].extend(df_boot.to_dict(orient='list'))






# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
