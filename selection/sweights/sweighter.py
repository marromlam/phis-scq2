import numpy as np
from ipanema import Sample
import argparse
import uproot3 as uproot


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="mass fit")
    p.add_argument('--input-sample')
    p.add_argument('--output-sample')
    p.add_argument('--biased-weights')
    p.add_argument('--unbiased-weights')
    p.add_argument('--mode')
    args = vars(p.parse_args())

    # Load full dataset and creae prxoy to store new sWeights
    sample = Sample.from_root(args['input_sample'], flatten=None)
    _proxy = np.float64(sample.df['time']) * 0.0

    # List all set of sWeights to be merged
    bpars = args["biased_weights"].split(",")
    upars = args["unbiased_weights"].split(",")

    list_of_weights = np.load(bpars[0], allow_pickle=True)
    list_of_weights = list_of_weights.item()
    list_of_weights = list(list_of_weights.keys())
    print(list_of_weights)
    for w in list_of_weights:
        __proxy = 0 * _proxy
        for sw in bpars+upars:
            _weight = np.load(sw, allow_pickle=True)
            _weight = _weight.item()
            __proxy += _weight[w]
        sample.df[f"{w[1:]}SW"] = __proxy
    print(sample.df)

    with uproot.recreate(args['output_sample']) as f:
        _branches = {}
        for k, v in sample.df.items():
            if 'int' in v.dtype.name:
                _v = np.int32
            elif 'bool' in v.dtype.name:
                _v = np.int32
            else:
                _v = np.float64
            _branches[k] = _v
        mylist = list(dict.fromkeys(_branches.values()))
        f["DecayTree"] = uproot.newtree(_branches)
        f["DecayTree"].extend(sample.df.to_dict(orient='list'))


"""
for i, b, u in zip(range(len(bpars)), bpars, upars):
  file_name = os.path.basename(b).split('.json')[0].split('_')
  print(i, b, u)
  # chech for mass_bin
  if 'Bd2JpsiKstar' in args['mode']:
    mass = [826, 861, 896, 931, 966]
  elif 'Bs2JpsiPhi' in args['mode']:
    mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  if file_name[2] == 'all':
    mLL = mass[0]
    mUL = mass[-1]
  else:
    bin = int(args['mass_bin'][-1])
    mLL = mass[bin-1]
    mUL = mass[bin]
  mX_cut = f"X_M>{mLL} & X_M<{mUL}"


  #
  # loop computing sweights for trig cats
  for ccut, cpars in zip([f"hlt1b==0 & ({mX_cut})", f"hlt1b==1 & ({mX_cut})" ], [b, u]):
    weights = np.load(cpars, allow_pickle=True)
    weights = weights.item()
    print(weights)
    cdf = isample.df.query(ccut)
    print(weights['nsigBd'])
    for y in weights.keys():
      print(weights[y].shape)
      print(cdf.shape)
      if not y in list(_proxies.keys()):
        _proxies[y] = _proxy
      _proxies[y][list(cdf.index)] = weights[y]
      print(_proxies)
"""


# vim: fdm=marker
