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
  # lower and upper time cuts
  "LT": f"time < {config.general['lower_time_upper_limit']}",
  "UT": f"time > {config.general['upper_time_lower_limit']}",
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
  # pXB and pYB cuts
  "pXB1": "B_PX >= 0 & B_PX < 2.7e3",
  "pXB2": "B_PX >= 2.7e3 & B_PX < 4.2e3",
  "pXB3": "B_PX >= 4.2e3 & B_PX <= 6.3e3",
  "pXB4": "B_PX >= 6.3e3",
  "pYB1": "B_PY >= 0 & B_PY < 2.7e3",
  "pYB2": "B_PY >= 2.7e3 & B_PY < 4.2e3",
  "pYB3": "B_PY >= 4.2e3 & B_PY <= 6.3e3",
  "pYB4": "B_PY >= 6.3e3",
}

# }}}


# CMDLINE interfrace {{{
if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  p.add_argument('--year', help='Full root file with huge amount of branches.')
  p.add_argument('--mode', help='Full root file with huge amount of branches.')
  p.add_argument('--version', help='Full root file with huge amount of branches.')
  p.add_argument('--weight', help='Full root file with huge amount of branches.')
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
    if m in ('Bd2JpsiKstar'):
      sw = 'sw_pt'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does ")
      print("         not exist. Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_pt.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does not exist.")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_pt_{v}.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with custon sw for given pTB bin does not exist.")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_pt.root'
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
    eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_{v}.root'
    sw = 'sw'
    # eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt.root'
    status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with sw does not exist")
      print("         Trying without the trailing version number.")
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
    if status:
      print("WARNING: Requested tuple with sw does not exist")
      print("         Could not found sw tuple. Downloading without sw.")
      # WARNING: eos tuples seem to do not have version anymore...
      eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt.root'
      status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
      if status:
        print(f"WARNING: Could not found {v} tuple. Downloading noveto...")
        # WARNING: eos tuples seem to do not have version anymore...
        #          Bs2JpsiPhi_Lb_2015_selected_bdt_noveto.root  
        eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_noveto.root'
        status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
      if status:
        print(f"WARNING: Could not found {v} tuple. Downloading v0r5...")
        # WARNING: eos tuples seem to do not have version anymore...
        eos_path = f'{EOSPATH}/v0r5/{m}/{y}/{m}_{y}_selected_bdt_sw_v0r5.root'
        status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
      if status:
        print(f"WARNING: Could not found {v} tuple. Downloading v0r5...")
        # WARNING: eos tuples seem to do not have version anymore...
        eos_path = f'{EOSPATH}/v0r5/{m}/{y}/{m}_{y}_selected_bdt_v0r5.root'
        status = os.system(f"xrdcp -f root://eoslhcb.cern.ch/{eos_path} {local_path}")
  if status:
    print("These tuples are not yet avaliable at root://eoslhcb.cern.ch/*.",
    'You may need to create those tuples yourself or ask B2CC people to'
    'produce them')
    shutil.rmtree(path, ignore_errors=True)
    exit()

  # }}}


  # If we reached here, then all should be fine 
  print(f"\n\n{80*'='}")
  print(f"Downloaded {eos_path}")
  print(f"{80*'='}\n\n")

  if args['weight'] == 'ready':
    try:
      result = uproot.open(local_path)[tree].pandas.df(flatten=None)
    except:
      result = uproot.open(local_path)['DVTuple'][tree].pandas.df(flatten=None)

    try:
      print("There are sWeights variables")
      if 'sw_cosK_noGBw' in list(result.keys()):
        print('Adding Peilian sWeight')
        result.eval(f"sw = sw_cosK_noGBw", inplace=True)  # overwrite sw variable
      else:
        print(f"Adding standard sWeight: {sw}")
        result.eval(f"sw = {sw}", inplace=True)  # overwrite sw variable
    except:
      # print(result.keys())
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
      _branches = {}
      for k, v in result.items():
          if 'int' in v.dtype.name:
              _v = np.int32
          elif 'bool' in v.dtype.name:
              _v = np.int32
          else:
              _v = np.float64
          _branches[k] = _v
      mylist = list(dict.fromkeys(_branches.values()))
      # print(mylist)
      # print(_branches)
      f[tree] = uproot.newtree(_branches)
      f[tree].extend(result.to_dict(orient='list'))
      f.close()
    print(f'    Succesfully written.')

    # delete donwloaded files
    shutil.rmtree(path, ignore_errors=True)
  else:
    shutil.copy(local_path, args['output'])
    shutil.rmtree(path, ignore_errors=True)

# }}}


# vim: ts=4 sw=4 sts=4 et
