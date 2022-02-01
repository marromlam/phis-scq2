# Merge sweights
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]

import uproot3 as uproot
import argparse
from ipanema import Sample
import numpy as np


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


# vim: fdm=marker
