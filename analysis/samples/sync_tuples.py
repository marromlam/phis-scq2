DESCRIPTION = """
    This script downloads tuples from open('config.json')->['eos'] and places
    them, properly renamed within the convention of phis-scq, in the
    open('config.json')->['path'] (the so-called sidecar folder)
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []

import numpy as np
import hjson
import pandas as pd
import os
import argparse
import uproot3 as uproot
import shutil

ROOT_PANDAS = True
if ROOT_PANDAS:
  import root_pandas

binned_vars = {'etaB': 'B_ETA', 'pTB': 'B_PT', 'sigmat': 'sigmat'}
binned_files = {'etaB': 'eta', 'pTB': 'pt', 'sigmat': 'sigmat'}
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
EOSPATH = hjson.load(open('config.json'))['eos']



vsub_dict = {
  "evtOdd": "(eventNumber % 2) != 0",
  "evtEven": "(eventNumber % 2) == 0",
  "magUp": "Polarity == 1",
  "magDown": "Polarity == -1",
  "bkgCat60": "B_BKGCAT != 60",
  "pTB1": "B_PT >= 0 & B_PT < 3.8e3",
  "pTB2": "B_PT >= 3.8e3 & B_PT < 6e3",
  "pTB3": "B_PT >= 6e3 & B_PT <= 9e3",
  "pTB4": "B_PT >= 9e3",
  "etaB1": "B_ETA >= 0 & B_ETA <= 3.3",
  "etaB2": "B_ETA >= 3.3 & B_ETA <= 3.9",
  "etaB3": "B_ETA >= 3.9 & B_ETA <= 6",
  "sigmat1": "sigmat >= 0 & sigmat <= 0.031",
  "sigmat2": "sigmat >= 0.031 & sigmat <= 0.042",
  "sigmat3": "sigmat >= 0.042 & sigmat <= 0.15"
}


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  p.add_argument('--year', help='Full root file with huge amount of branches.')
  p.add_argument('--mode', help='Full root file with huge amount of branches.')
  p.add_argument('--version', help='Full root file with huge amount of branches.')
  p.add_argument('--tree', help='Input file tree name.')
  p.add_argument('--output', help='Input file tree name.')
  p.add_argument('--uproot-kwargs', help='Arguments to uproot.pandas.df')
  args = vars(p.parse_args())

  # Get the flags and that stuff
  v = args['version'].split("@")[0].split("bdt")[0]  # pipeline tuple version
  V = args['version'] # full version for phis-scq
  y = args['year']
  m = args['mode']
  tree = args['tree']

  path = os.path.dirname(os.path.abspath(args['output']))
  path = os.path.join(path, f"{abs(hash(f'{m}_{y}_selected_bdt_sw_{V}.root'))}")
  print(f"Required tuple : {m}_{y}_selected_bdt_sw_{V}.root", path)
  os.makedirs(path, exist_ok=False)  # create hashed temp dir
  all_files = []; all_dfs = []
  local_path = f'{path}/{v}.root'

  # some version substring imply using new sWeights (pTB, etaB and sigmat)
  if "pTB" in V:
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_pt_{v}.root'
    sw = 'sw_pt'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does not exist")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
      sw = 'sw'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
  elif "etaB" in V:
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_eta_{v}.root'
    sw = 'sw_eta'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given etaB bin does not exist")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
      sw = 'sw'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
  elif "sigmat" in V:
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_sigmat_{v}.root'
    sw = 'sw_sigmat'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given sigmat bin does not exist")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
      sw = 'sw'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
  else:
    eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    sw = 'sw'
  if status:
    print("These tuples are not yet avaliable at root://eoslhcb.cern.ch/*.",
    'You may need to create those tuples yourself or ask B2CC people to'
    'produce them')
    exit()

  # If we reached here, then all should be fine 
  print(f"Downloaded {m}_{y}_selected_bdt_sw_{v}.root")

  """
  # download main file
  if status == 0:
    all_files.append([f"{m}_{y}_selected_bdt_sw_{v}.root", None])
  else:
    print(f"- File {m}_{y}_selected_bdt_sw_{v}.root does not exist on server.")

  # donwload binned variable files
  for name, var in binned_files.items():
    print(f"Downloading {m}_{y}_selected_bdt_sw_{var}_{v}.root")
    _eos_path = eos_path.replace('selected_bdt_sw',f'selected_bdt_sw_{var}')
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{_eos_path} {path}")
    print(status)
    if status == 0:
      all_files.append([f"{m}_{y}_selected_bdt_sw_{var}_{v}.root", f"{var}"])
    else:
      print(f"- File {m}_{y}_selected_bdt_sw_{var}_{v}.root does not exist",
            "on server.")

  # Load and convert to pandas dfs
  for f, b in all_files:
    print(f, b)
    fp = f"{path}/{f}"
    if b:
      b = f"sw_{b}"
    all_dfs.append(uproot.open(fp)[tree].pandas.df(branches=b, flatten=None))
    print(uproot.open(fp)[tree].pandas.df(branches=b, flatten=None))

  # concatenate all columns
  result = pd.concat(all_dfs, axis=1)
  if "nsig_sw" in result.keys():
    result.eval("sw=nsig_sw", inplace=True)
  for var in binned_vars.keys():
    if not f'sw_{var}' in list(result.keys()):
      sw = np.zeros_like(result[f'sw'])
      for cut in bin_vars[var]:
        pos = result.eval(cut.replace(var, binned_vars[var]))
        this_sw = result.eval(f'sw*({cut.replace(var,binned_vars[var])})')
        sw = np.where(pos, this_sw * (sum(this_sw)/sum(this_sw*this_sw)), sw)
      result[f'sw_{var}'] = result[f'sw']  # sw
  """

  result = uproot.open(local_path)[tree].pandas.df(flatten=None)
  result.eval(f"sw = {sw}", inplace=True)  # overwrite sw variable

  # place cuts according to version substring
  list_of_cuts = []
  for k,v in vsub_dict.items():
    if k in V:
      list_of_cuts.append(v)
  vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"

  print(vsub_cut)

  #
  result.query(vsub_cut)

  # write
  print(f"\nStarting to write {os.path.basename(args['output'])} file.")
  if ROOT_PANDAS:
    root_pandas.to_root(result, args['output'], key=tree)
  else:
    f = uproot.recreate(args['output'])
    f[tree] = uproot.newtree({var: 'float64' for var in result})
    f[tree].extend(result.to_dict(orient='list'))
    f.close()
  print(f'    Succesfully written.')

  # delete donwloaded files
  shutil.rmtree(path, ignore_errors=True)
