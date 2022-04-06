__all__ = []
DESCRIPTION = """
    Sync CSP factor parameters from B2CC gitlab.
"""


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

import os
import argparse
import hjson
from ipanema import Parameters
from utils.strings import printsec

# }}}


# Command line runner {{{

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--year', help='Time resolution year')
  p.add_argument('--mode', help='Time resolution mode')
  p.add_argument('--version', help='Tuple version')
  p.add_argument('--csp', help='Time resolution flag')
  p.add_argument('--output', help='Place where to dump time resolution')
  p.add_argument('--repo', help='Repository')
  p.add_argument('--linker', help='Relationship between phis and repo flags')
  args = vars(p.parse_args())

  year = args['year']
  mode = args['mode']
  version = args['version'].split('@')[0]
  version = version.split('bdt')[0]
  csp = args['csp']
  out_path = args['output']
  linker = args['linker']
  repo = args['repo']

  if version in ('v0r0', 'v0r1', 'v0r2', 'v0r3', 'v0r4'):
    printsec("Plugging old CSP factors")
    old = f"analysis/params/csp_factors/{mode}/old.json"
    params = Parameters.load(old)
    print(params)
    params.dump(out_path)
    print(f"Dumping parameters to {out_path}")
  else:
    printsec("Get CSP factors from Bs2JpsiPhi-FullRun2 repository")

    # get name for time resolution
    _csp = hjson.load(open(linker, "r"))[csp]
    print(_csp)

    # cook local and remote paths
    tmp_path = out_path.replace('output', 'tmp')
    tmp_path = os.path.dirname(tmp_path)
    git_path = f"fitinputs/{version}/Csp/All/"
    os.makedirs(tmp_path, exist_ok=True)

    print(f'Downloading Bs2JpsiPhi-FullRun2: {git_path}')
    os.system(f"git archive --remote={repo} --prefix=./{tmp_path}/ HEAD:{git_path} {_csp} | tar -x")
    print(f"git archive --remote={repo} --prefix=./{tmp_path}/ HEAD:{git_path} {_csp} | tar -x")

    print(f"Loading CSP factors {year}")
    raw_json = hjson.load(open(f"{tmp_path}/{_csp}",'r'))
    raw_json = raw_json[f'All']['CspFactors']
  
    print(f'Parsing parameters to match phis-scq sctructure')
    cooked = {};
    for i, d in enumerate(raw_json):
      bin = i+1
      cooked[f'CSP{bin}'] = {'name':f'CSP{bin}'}
      cooked[f'CSP{bin}'].update({'value':d['Value'], 'stdev':d['Error'] })
      cooked[f'CSP{bin}'].update({'latex': f"C_{{SP}}^{{{bin}}}"})
      cooked[f'CSP{bin}'].update({'free': False})
      if not f'mKK{bin-1}' in cooked:
        cooked[f'mKK{bin-1}'] = {'name':f'mKK{bin-1}'}
        cooked[f'mKK{bin-1}'].update({'value':d['Bin_ll'], 'stdev':0 })
        cooked[f'mKK{bin-1}'].update({'latex':f'm_{{KK}}^{{{bin-1}}}'})
        cooked[f'mKK{bin-1}'].update({'free': False})
      if not f'mKK{bin}' in cooked:
        cooked[f'mKK{bin}'] = {'name':f'mKK{bin}'}
        cooked[f'mKK{bin}'].update({'value':d['Bin_ul'], 'stdev':0 })
        cooked[f'mKK{bin}'].update({'latex':f'm_{{KK}}^{{{bin}}}'})
        cooked[f'mKK{bin}'].update({'free': False})
  
    # Build the ipanema.Parameters object
    print(f"\nCSP parameters for {year} are:")
    list_params = list(cooked.keys())                 # list all parameter names
    list_params = sorted( list_params )               # sort them
    params = Parameters()
    [params.add(cooked[par]) for par in list_params]
    print(params)
    params.dump(out_path)
    print(f'Dumping parameters to {out_path}')
    print(f'Clean up {tmp_path}/{_csp}\n')
    os.remove(f"{tmp_path}/{_csp}")

# }}}


# vim:foldmethod=marker
# TODO: Do some cleaning 
