__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['split_tree_for_lambdab']


import argparse
import uproot3 as uproot
import pandas as pd
import numpy as np


def split_tree_for_lambdab(idf, output_type):
    """
    Cuts the dataframe according to the output type

    Parameters
    ----------
    idf : pandas.DataFrame
        Input dataframe
    output_type : str
        Particle string

    Returns
    -------
    pandas.DataFrame
        Chopped dataframe
    """
    if output_type == 'pk':
        _cut = 'hplus_TRUEID==2212 & hminus_TRUEID==-321'
    elif output_type == 'kp':
        _cut = 'hplus_TRUEID==321 & hminus_TRUEID==-2212'
    return idf.query(_cut)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-files', nargs='+',
                   help='Path to the input file')
    p.add_argument('--input-tree-name', default='DecayTree',
                   help='Name of the tree')
    p.add_argument('--output-file',
                   help='Output ROOT file')
    p.add_argument('--output-tree-name', default='DecayTree',
                   help='Name of the tree')
    p.add_argument('--output-type',
                   help='hplus=p,hminus=k or hplus=k,hminus=p')
    # p.add_argument('--output-file-pk',
    #                help='Output ROOT file hplus=p hminus=k')
    # p.add_argument('--output-tree-name-pk', default='DecayTree',
    #                help='Name of the tree hplus=p hminus=k')
    # p.add_argument('--output-file-kp',
    #                help='Output ROOT file hplus=k hminus=p')
    # p.add_argument('--output-tree-name-kp', default='DecayTree',
    #                help='Name of the tree hplus=k hminus=p')

    args = vars(p.parse_args())
    ifiles = args['input_files']
    ifiles = [ifiles] if type(ifiles) != type([]) else ifiles
    itree = args['input_tree_name']
    idf = pd.concat([uproot.open(f)[itree].pandas.df(flatten=None) for f in ifiles])
    odf = split_tree_for_lambdab(idf, args['output_type'])

    # write the file
    otree = args['output_tree_name']
    with uproot.recreate(args['output_file']) as ofile:
        _branches = {}
        for k, v in odf.items():
            if 'int' in v.dtype.name:
                _v = np.int32
            elif 'bool' in v.dtype.name:
                _v = np.int32
            else:
                _v = np.float64
            _branches[k] = _v
        mylist = list(dict.fromkeys(_branches.values()))
        ofile[otree] = uproot.newtree(_branches)
        ofile[otree].extend(odf.to_dict(orient='list'))
