DESCRIPTION = """
    This script downloads tuples from open('config.json')->['eos'] and places
    them, properly renamed within the convention of phis-scq, in the
    open('config.json')->['path'] (the so-called sidecar folder)
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []

import uproot
import argparse
import os
import pandas as pd
import hjson
import numpy as np

ROOT_PANDAS = True
if ROOT_PANDAS:
  import root_pandas

binned_vars = {'etaB':'B_ETA', 'pTB':'B_PT', 'sigmat':'sigmat'}
binned_files = {'etaB':'eta', 'pTB':'pt', 'sigmat':'sigmat'}
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
EOSPATH = hjson.load(open('config.json'))['eos']





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
  v = args['version']
  y = args['year']
  m = args['mode']
  tree = args['tree']
  scq_path = os.path.dirname(os.path.abspath(args['output']))
  all_files = []; all_dfs = []

  # Downloading everything xrdcp root://eoslhcb.cern.ch/
  print(f"Downloading {m}_{y}_selected_bdt_sw_{v}.root")
  eos_path = f'{EOSPATH}/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
  #status = os.system(f"""scp lxplus:{eos_path} {scq_path}""")
  status = os.system(f"""xrdcp -f root://eoslhcb.cern.ch/{eos_path} {scq_path}""")
  print(status)
  if status==0:
    all_files.append([f"{m}_{y}_selected_bdt_sw_{v}.root",None])
  else:
    print(f"    File {m}_{y}_selected_bdt_sw_{v}.root does not exist on server.")

  for name, var in binned_files.items():
    print(f"Downloading {m}_{y}_selected_bdt_sw_{var}_{v}.root")
    eos_path = f'{EOSPATH}/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_{var}_{v}.root'
    #status = os.system(f"""scp -r lxplus:{eos_path} {scq_path}""")
    status = os.system(f"""xrdcp -f root://eoslhcb.cern.ch/{eos_path} {scq_path}""")
    print(status)
    if status==0:
      all_files.append([f"{m}_{y}_selected_bdt_sw_{var}_{v}.root",f"{var}"])
    else:
      print(f"    File {m}_{y}_selected_bdt_sw_{var}_{v}.root does not exist",
             "on server.")


  # Load and convert to pandas dfs
  for f, b in all_files:
    print(f,b)
    fp = f"{scq_path}/{f}"
    if b: b = f"sw_{b}"
    all_dfs.append( uproot.open(fp)[tree].pandas.df(branches=b, flatten=None) )
    print(uproot.open(fp)[tree].pandas.df(branches=b, flatten=None))

  # concatenate all columns
  result = pd.concat(all_dfs, axis=1)
  for var in binned_vars.keys():
    if not f'sw_{var}' in list(result.keys()):
      sw = np.zeros_like(result[f'sw'])
      for cut in bin_vars[var]:
        pos = result.eval(cut.replace(var,binned_vars[var]))
        this_sw = result.eval(f'sw*({cut.replace(var,binned_vars[var])})')
        sw = np.where(pos, this_sw * ( sum(this_sw)/sum(this_sw*this_sw) ),sw)
      result[f'sw_{var}'] = result[f'sw']#sw

  # write
  print(f"\nStarting to write {os.path.basename(args['output'])} file.")
  if ROOT_PANDAS:
    root_pandas.to_root(result, args['output'], key=tree)
  else:
    f = uproot.recreate(args['output'])
    f[tree] = uproot.newtree({var: 'float64' for var in result})
    f[tree].extend(result.to_dict(orient='list'))
    f.close()
  print(f'    Succesfully writen.')

  # delete donwloaded files
  for file in all_files:
    os.remove(f"{scq_path}/{file[0]}")
