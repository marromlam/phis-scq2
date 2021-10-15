DESCRIPTION = """
    This script downloads tuples from open('config.json')->['eos'] and places
    them, properly renamed within the convention of phis-scq, in the
    open('config.json')->['path'] (the so-called sidecar folder)
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{
import numpy as np
import hjson
# import pandas as pd
import os
import argparse
import uproot3 as uproot
import shutil

ROOT_PANDAS = False
if ROOT_PANDAS:
  import root_pandas

import config

# }}}


# Some config {{{

vsub_dict = {
  "evtOdd": "(eventNumber % 2) != 0",
  # "evtOdd": "(eventNumber % 2) != 0 & B_DTF_CHI2NDOF <= 1 & log_B_IPCHI2_mva <= 0",
  "evtEven": "(eventNumber % 2) == 0",
  "magUp": "Polarity == 1",
  "magDown": "Polarity == -1",
  "bkgcat60": "B_BKGCAT != 60",
  "LT": "time < 1.36",
  "UT": "time > 1.36",
  "g210300": "runNumber > 210300",
  "l210300": "runNumber < 210300",
  "pTB1": "B_PT >= 0 & B_PT < 3.8e3",
  "pTB2": "B_PT >= 3.8e3 & B_PT < 6e3",
  "pTB3": "B_PT >= 6e3 & B_PT <= 9e3",
  "pTB4": "B_PT >= 9e3",
  "etaB1": "B_ETA >= 0 & B_ETA <= 3.3",
  "etaB2": "B_ETA >= 3.3 & B_ETA <= 3.9",
  "etaB3": "B_ETA >= 3.9 & B_ETA <= 6",
  "sigmat1": "sigmat >= 0 & sigmat <= 0.031",
  "sigmat2": "sigmat >= 0.031 & sigmat <= 0.042",
  "sigmat3": "sigmat >= 0.042 & sigmat <= 0.15",
  "LcosK": "helcosthetaK<=0.0",
  "UcosK": "helcosthetaK>0.0",
}

# }}}


# CMDLINE interfrace {{{
if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  p.add_argument('--year', help='Full root file with huge amount of branches.')
  p.add_argument('--mode', help='Full root file with huge amount of branches.')
  p.add_argument('--version', help='Full root file with huge amount of branches.')
  p.add_argument('--tree', help='Input file tree name.')
  p.add_argument('--output', help='Input file tree name.')
  p.add_argument('--eos', help='Input file tree name.')
  p.add_argument('--uproot-kwargs', help='Arguments to uproot.pandas.df')
  args = vars(p.parse_args())

  # Get the flags and that stuff
  v = args['version'].split("@")[0].split("bdt")[0]  # pipeline tuple version
  V = args['version'].replace('bdt', '')  # full version for phis-scq
  y = args['year']
  m = args['mode']
  tree = args['tree']
  EOSPATH = args['eos']

  path = os.path.dirname(os.path.abspath(args['output']))
  path = os.path.join(path, f"{abs(hash(f'{m}_{y}_selected_bdt_sw_{V}.root'))}")
  print(f"Required tuple : {m}_{y}_selected_bdt_sw_{V}.root", path)
  os.makedirs(path, exist_ok=False)  # create hashed temp dir
  all_files = []; all_dfs = []
  local_path = f'{path}/{v}.root'

  # some version substring imply using new sWeights (pTB, etaB and sigmat)
  status = 1

  if "pTB" in V:
    # version@pTB {{{
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_pt_{v}.root'
    sw = 'sw_pt'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does not exist.")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_pt.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does not exist")
      print("         Downloading the standard tuple for this mode and year.")
    # }}}
  elif "etaB" in V:
    # version@etaB {{{
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_eta_{v}.root'
    sw = 'sw_eta'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given etaB bin does not exist.")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_eta.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given etaB bin does not exist")
      print("         Downloading the standard tuple for this mode and year.")
    # }}}
  elif "sigmat" in V: 
    # version@sigmat {{{
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_sigmat_{v}.root'
    sw = 'sw_sigmat'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given sigmat bin does not exist.")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_sigmat.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given sigmat bin does not exist")
      print("         Downloading the standard tuple for this mode and year.")
    # }}}

  # version (baseline tuples) {{{

  if status: 
    eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
    sw = 'sw'
    # eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt.root'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with sw does not exist")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with sw does not exist")
      print("         Could not found sw tuple. Downloading without sw.")
      # WARNING: eos tuples seem to do not have version anymore...
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
      if status:
        print("WARNING: Could not found v1r0 tuple. Downloading v0r5...")
        # WARNING: eos tuples seem to do not have version anymore...
        eos_path = f'{EOSPATH}/v0r5/{m}/{y}/{m}_{y}_selected_bdt_sw_v0r5.root'
        status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
  if status:
    print("These tuples are not yet avaliable at root://eoslhcb.cern.ch/*.",
    'You may need to create those tuples yourself or ask B2CC people to'
    'produce them')
    exit()

  # }}}

  # If we reached here, then all should be fine 
  print(f"Downloaded {eos_path}")

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
  try:
    print("There are sWeights variables")
    if 'sw_cosK_noGBw' in list(result.keys()):
      print('Adding Peilian sWeight')
      result.eval(f"sw = sw_cosK_noGBw", inplace=True)  # overwrite sw variable
    else:
      print("Adding standard sWeight")
      result.eval(f"sw = {sw}", inplace=True)  # overwrite sw variable
  except:
    print(result.keys())
    if 'B_BKGCAT' in list(result.keys()):
      print("sWeight is set to zero for B_BKGCAT==60")
      result['sw'] = np.where(result['B_BKGCAT'].values!=60,1,0)
    else:
      print("sWeight variable was not found. Set sw = 1")
      result['sw'] = np.ones_like(result[result.keys()[0]])


  # place cuts according to version substring
  list_of_cuts = []; vsub_cut = None
  for k,v in vsub_dict.items():
    if k in V:
      try:
        noe = len(result.query(v))
        if (k in ("g210300", "l210300")) and ("MC" in args['output']):
          print("MCs are not cut in runNumber")
        elif (k in ("g210300", "l210300")) and ("2018" not in args['output']):
          print("Only 2018 is cut in runNumber")
        elif (k in ("UcosK", "LcosK")) and 'Bd2JpsiKstar' not in m:
          print("Cut in cosK was only planned in Bd")
        else:
          list_of_cuts.append(v)
        if noe == 0:
          print(f"ERROR: This cut leaves df empty. {v}")
          print(f"       Query halted.")
      except:
        print(f"non hai variable para o corte {v}")
  if list_of_cuts:
    vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"


  # place the cut
  print(f"{80*'-'}\nApplied cut: {vsub_cut}\n{80*'-'}")
  if vsub_cut:
    result = result.query(vsub_cut)
  print(result)


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

# }}}


# vim: foldmethod=marker
