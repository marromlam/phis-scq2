from utils.strings import printsec
from ipanema import Parameters
import hjson
import argparse
import os
DESCRIPTION = """
    Sync time resolution parameters from B2CC gitlab.
"""


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{


# }}}


# Command line runner {{{

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--year', help='Time resolution year')
  p.add_argument('--mode', help='Time resolution mode')
  p.add_argument('--version', help='Tuple version')
  p.add_argument('--timeres', help='Time resolution flag')
  p.add_argument('--output', help='Place where to dump time resolution')
  p.add_argument('--repo', help='Repository')
  p.add_argument('--linker', help='Relationship between phis and repo flags')
  args = vars(p.parse_args())

  year = args['year']
  mode = args['mode']
  version = args['version'].split('@')[0]
  version = version.split('bdt')[0]
  timeres = args['timeres']
  out_path = args['output']
  linker = args['linker']
  repo = args['repo']

  if version in ('v0r0', 'v0r1', 'v0r2', 'v0r3', 'v0r4'):
    printsec("Plugging old time resolution")
    old = f"analysis/params/time_resolution/{mode}/old.json"
    params = Parameters.load(old)
    print(params)
    params.dump(out_path)
    print(f"Dumping parameters to {out_path}")
  else:
    printsec("Get time resolution from Bs2JpsiPhi-FullRun2 repository")

    # get name for time resolution
    if version == 'v0r5':
      _timeres = hjson.load(open(linker, "r"))[timeres][0].format(year=year)
    else:
      surname = 'nominal'
      if 'pT1' in args['version']:
        surname = 'pT0'
      elif 'pT2' in args['version']:
        surname = 'pT1'
      elif 'pT3' in args['version']:
        surname = 'pT2'
      elif 'pT4' in args['version']:
        surname = 'pT2'
      _timeres = hjson.load(open(linker, "r"))[timeres][1].format(surname=surname)

    # cook local and remote paths
    tmp_path = out_path.replace('output', 'tmp')
    tmp_path = os.path.dirname(tmp_path)
    git_path = f"fitinputs/{version}/time_resolution/{year}"
    os.makedirs(tmp_path, exist_ok=True)

    print(f'Downloading Bs2JpsiPhi-FullRun2: {git_path}')
    os.system(f"git archive --remote={repo} --prefix=./{tmp_path}/ HEAD:{git_path} {_timeres} | tar -x")

    print(f"Loading Time Resolution {year}")
    rawd = hjson.load(open(f"{tmp_path}/{_timeres}", 'r'))['TimeResParameters']
    outd = Parameters.load("analysis/params/time_resolution/Bs2JpsiPhi/none.json")

    # parse parameters
    # WARNING: This parser can be broken really easily
    print(f'Parsing parameters to match phis-scq sctructure')
    for i, par in enumerate(list(outd.keys())):
      what = par.split('_')[-1]
      outd[par].correl = {}
      for ii, _ in enumerate(list(rawd)):
        if rawd[ii]['Name'] == 'mu':
          outd[par].set(value=rawd[ii]['Value'], stdev=rawd[ii]['Error'])

        elif rawd[ii]['Name'] == f'p{i-1}' and f'sigma_{what}' == par:
          outd[par].set(value=rawd[ii]['Value'], stdev=rawd[ii]['Error'])
          for j, rap in enumerate(list(outd.keys())):
            for k in range(len(rawd)):
              if rawd[k]['Name'] == f'rho_p{i-1}_p{j-1}_time_res':
                outd[par].correl[rap] = rawd[k]['Value']
              elif rawd[k]['Name'] == f'rho_p{j-1}_p{i-1}_time_res':
                outd[par].correl[rap] = rawd[k]['Value']
              else:
                outd[par].correl[rap] = 1 if i == j else 0
        else:
          print(f"    - Parameter {par} does not exist")

    print("\nParameter table")
    print(outd)
    print("Correlation matrix")
    print(outd.corr(), '\n')
    print(f'Dumping parameters to {out_path}')
    outd.dump(out_path)
    print(f'Clean up {tmp_path}/{_timeres}\n')
    os.remove(f"{tmp_path}/{_timeres}")

# }}}


# vim:foldmethod=marker
