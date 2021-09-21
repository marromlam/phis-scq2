import uproot3 as uproot
import pandas as pd
import numpy as np


oname = "/home3/marcos.romero/Standalone/BsMuMuKK/20201008a/20201008a.root"
oname = "/scratch46/marcos.romero/old_evtgen_files/20201008a/20201008a.root"
oname = "/scratch46/marcos.romero/old_evtgen_files/20201007a/20201007a.root"
tname = "/scratch46/marcos.romero/sidecar/2021/MC_Bs2JpsiKK_Swave/20210914b.root"


translation_layer = {
  # 'RECO' variables
  'mHH': ['MKK','X_M'],
  'cosK': ['helcosthetaK', 'theta_K'],
  'cosL': ['helcosthetaL', 'theta_mu'],
  'hphi': ['helphi', 'phi'],
  'time': ['time'],
  'idB': ['B_ID', 'q'],
  # GenLvl variables
  'gencosK': ['truehelcosthetaK_GenLvl', 'theta_K'],
  'gencosL': ['truehelcosthetaL_GenLvl', 'theta_mu'],
  'genhphi': ['truehelphi_GenLvl', 'phi'],
  'gentime': ['truetime', 'time'],
  'genidB': ['B_ID_GenLvl', 'q'],
  # trigger
  'hlt1b': ['hlt1b']
}

# Load original tuple
ofile = uproot.open(oname)
odf = ofile["bsmumukk"].pandas.df()
all_branches = list(odf.keys())
print(odf)

# Just loop creating a new dataframe and adding variables
tdf = pd.DataFrame()
for k, v in translation_layer.items():
  for branch in v:
      if branch in all_branches:
          if branch == 'MKK':
            branch = '1000*MKK'
          print(f"Cloning {k} from {branch}")
          tdf[k] = np.float64(odf.eval(branch).values)
          break

# if there is no trigger variable, then we randomly generate one
if 'hlt1b' not in list(tdf.keys()):
  print('hlt1b was randomly generated')
  tdf['hlt1b'] = np.round(np.random.rand(len(tdf['time'])))
print(tdf)

# Write
with uproot.recreate(tname) as tfile:
  tfile["DecayTree"] = uproot.newtree({var:'float64' for var in tdf})
  tfile["DecayTree"].extend(tdf.to_dict(orient='list'))

