DESCRIPTION = """
    Sync flavor tagging parameters from B2CC gitlab.
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
  p.add_argument('--year', help='Tuple year')
  p.add_argument('--mode', help='Tuple mode')
  p.add_argument('--version', help='Tuple version')
  p.add_argument('--flavor', help='Flavor tagging flag')
  p.add_argument('--output', help='Place where to dump parameters')
  p.add_argument('--repo', help='Repository')
  p.add_argument('--linker', help='Relationship between phis and repo flags')
  args = vars(p.parse_args())

  year = args['year']
  mode = args['mode']
  version = args['version'].split('@')[0]
  version = version.split('bdt')[0]
  flavor = args['flavor']
  out_path = args['output']
  linker = args['linker']
  repo = args['repo']

  if version in ('v0r0', 'v0r1', 'v0r2', 'v0r3', 'v0r4'):
    printsec("Plugging old flavor factors")
    old = f"analysis/params/flavor_tagging/{mode}/old.json"
    params = Parameters.load(old)
    print(params)
    params.dump(out_path)
    print(f"Dumping parameters to {out_path}")
  else:
    printsec("Get flavor tagging parameters from Bs2JpsiPhi-FullRun2 repository")

    # get name for time resolution
    _flavor = hjson.load(open(linker, "r"))[flavor].format(year=year)
    print(_flavor)

    # cook local and remote paths
    tmp_path = out_path.replace('output', 'tmp')
    tmp_path = os.path.dirname(tmp_path)
    git_path = f"fitinputs/{version}/tagging_calibration/{year}/"
    if 'peilian' in flavor:
        git_path = f"fitinputs/{version}/tagging_calibration/"
    os.makedirs(tmp_path, exist_ok=True)

    print(f'Downloading Bs2JpsiPhi-FullRun2: {git_path}/{_flavor}')
    os.system(f"git archive --remote={repo} --prefix=./{tmp_path}/ HEAD:{git_path} {_flavor} | tar -x")

    print(f"Loading flavor factors {year}")
    raw_json = hjson.load(open(f"{tmp_path}/{_flavor}",'r'))
    rawd = raw_json['TaggingParameter']
    outd = Parameters.load("analysis/params/flavor_tagging/Bs2JpsiPhi/none.json")
  
    print(f'Parsing parameters to match phis-scq sctructure')
    for i, par in enumerate(list(outd.keys())):
        for _i in range(len(rawd)):
          if rawd[_i]['Name'][:5] == f'{par[:2]}_'+f'{par[3:]}'.upper():
            outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
            outd[par].correl = {}
            for j in range(0,3):
              pi = par[:2]; pj = f"p{j}"; tag = par[3:]
              for k in range(len(rawd)):
                if rawd[k]['Name'][:12].lower() == f'rho_{pi}_{pj}_{tag}':
                  outd[par].correl[f"{pj}_{tag}"] = rawd[k]['Value']
                elif rawd[k]['Name'][:12].lower() == f'rho_{pj}_{pi}_{tag}':
                  outd[par].correl[f"{pj}_{tag}"] = rawd[k]['Value']
          elif rawd[_i]['Name'][:6] == f'{par[:3]}_'+f'{par[4:]}'.upper():
            outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
          elif rawd[_i]['Name'][:10] == f'{par[:3]}_bar_'+f'{par[4:]}'.upper():
            outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
  
    # Build the ipanema.Parameters object
    print(f"\nflavor parameters for {year} are:")
    list_params = list(outd.keys())                 # list all parameter names
    list_params = sorted( list_params )               # sort them
    params = Parameters()
    [params.add(outd[par]) for par in list_params]
    print("\nParameter table")
    print(params)
    print("Correlation matrix")
    print(params.corr())
    params.dump(out_path)
    print(f'Dumping parameters to {out_path}')
    print(f'Clean up {tmp_path}/{_flavor}\n')
    os.remove(f"{tmp_path}/{_flavor}")

# }}}


# vim:foldmethod=marker
# TODO: Do some cleaning
