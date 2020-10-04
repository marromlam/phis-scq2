import uproot
import argparse
import os
import pandas as pd
import hjson
import numpy as np

binned_vars = {'eta':'B_ETA', 'pt':'B_PT', 'sigmat':'sigmat'}
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']

def argument_parser():
  parser = argparse.ArgumentParser(description='Sync tuples.')
  # Samples
  parser.add_argument('--year',
                      default = '2015',
                      help='Full root file with huge amount of branches.')
  parser.add_argument('--mode',
                      default = 'Bd2JpsiKstar',
                      help='Full root file with huge amount of branches.')
  parser.add_argument('--version',
                      default = 'v0r5',
                      help='Full root file with huge amount of branches.')
  parser.add_argument('--tree',
                      default = 'DecayTree',
                      help='Input file tree name.')
  parser.add_argument('--output',
                      default = '/scratch17/marcos.romero/phis_samples/2015/Bd2JpsiKstar/v0r5_sWeight.root',
                      help='Input file tree name.')
  parser.add_argument('--uproot-kwargs',
                      #default = '{"entrystart":0, "entrystop":100}',
                      help='Arguments to uproot.pandas.df')

  return parser







if __name__ == "__main__":
  args = vars(argument_parser().parse_args())

  # Get the flags and that stuff
  v = args['version']
  y = args['year']
  m = args['mode']
  tree = args['tree']
  scq_path = os.path.dirname(os.path.abspath(args['output']))
  all_files = []; all_dfs = []
  # Downloading everything xrdcp root://eoslhcb.cern.ch/
  print(f"Downloading {m}_{y}_selected_bdt_sw_{v}.root")
  eos_path  = f'/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2'
  eos_path += f'/{v}/{m}/{y}/{m}_{y}_selected_bdt_sw_{v}.root'
  status = os.system(f"""scp lxplus:{eos_path} {scq_path}""")
  print(status)
  if status==0:
    all_files.append([f"{m}_{y}_selected_bdt_sw_{v}.root",None])
  else:
    print(f"    File {m}_{y}_selected_bdt_sw_{v}.root does not exist on server.")

  for var in binned_vars.keys():
    print(f"Downloading {m}_{y}_selected_bdt_sw_{var}_{v}.root")
    eos_path  = f'/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2'
    eos_path += f'/{v}/fit_check/{m}/{y}/{m}_{y}_selected_bdt_sw_{var}_{v}.root'
    status = os.system(f"""scp -r lxplus:{eos_path} {scq_path}""")
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
  with uproot.recreate(args['output'],compression=None) as f:
    f[tree] = uproot.newtree({var:'float64' for var in result})
    f[tree].extend(result.to_dict(orient='list'))
  print(f'    Succesfully writen.')
  f.close()

  # delete donwloaded files
  for file in all_files:
    os.remove(f"{scq_path}/{file[0]}")
