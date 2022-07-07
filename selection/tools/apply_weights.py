import argparse
import pickle

import numpy as np
import pandas
import uproot as ur
import yaml
from hep_ml import reweight
from reweighting import read_variables_from_yaml
import awkward as ak


def dictarr2root(dictarr, wf, tree_name='DecayTree', verbose=True):
    """
    Writes  ...
    Rewritten for uproot4
    """
    create_tree = False
    list_of_trees = [t.decode().split(';')[0] for t in wf.keys()]
    if tree_name not in list_of_trees:
        create_tree = True
    varsdict = {}  # stores branches : types
    arrsdict = {}  # stores branches : arrs

    # for merda in dictarr.tolist():
    #     print(merda)

    for i, k in enumerate(dictarr.fields):
        # K = k.decode()  # uproot3
        v = dictarr[k]
        is_jagged = False
        if not isinstance(v, np.ndarray):
            is_jagged = True

        if is_jagged:
            # WARNING: assume it's a jagged array with the same type
            # _type = v[0][0].dtype.name
            # _type = v[0].dtype #.name
            _type = v.type #.name
        else:
            # _type = v[0].dtype #.name
            _type = v.type #.name
        print(_type)

        # uproot cant write some dtypes, we transform them here
        # if 'uint32' in str(_type):
        #     if verbose:
        #         print(f"{k} is type '{_type}', and was switched to 'int32'")
        #     _type = _type.replace('uint32', 'int32')
        # elif 'uint64' in str(_type):
        #     print(f"{k} is type '{_type}', and was switched to 'int64'")
        #     _type = _type.replace('uint64', 'int64')
        #     # _type = 'int64'

        if is_jagged:
            if create_tree:
                # varsdict.update({k: ur.newbranch(_type, size=f"__{k}")})
                # varsdict.update({k: f"var * {_type}"})
                varsdict.update({k: _type})
            # arrsdict.update({k: v, f"__{k}": v.counts})
            arrsdict.update({k: v})
        else:
            if create_tree:
                # varsdict.update({k: ur.newbranch(_type)})
                varsdict.update({k: _type})
            arrsdict.update({k: v})

    if create_tree:
        wf.mktree(tree_name, varsdict)

    wf[tree_name].extend(arrsdict)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='File to add weights to')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the tree in file')
    parser.add_argument('--input-weight', default='', help='Name of branch of sweights')
    parser.add_argument('--variables-files', nargs='+', help='Path to the file with variable lists')
    parser.add_argument('--weight-method', default='gb', choices=['gb', 'binned'], help='Specify method used to reweight')
    parser.add_argument('--weights-file', help='Pickle file containing weights')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--output-file', help='File to store the ntuple with weights')
    parser.add_argument('--output-weight-name', help='Name of the output weight')
    return parser


def apply_weights(input_file, input_tree_name, input_weight, variables_files,
                  weight_method, weights_file, mode, output_file,
                  output_weight_name):

    branches = read_variables_from_yaml(mode, variables_files)

    input_file = input_file[0] if type(input_file) == type([]) else input_file

    # dict of awkward arrays
    original = ur.open(f"{input_file}:{input_tree_name}").arrays()
    print(original)
    # original = pandas.DataFrame(original)

    reweighter = pickle.load(open(weights_file, 'rb'))
    
    
    if input_weight:
        # arr_weight = ur.open(f"{input_file}:{input_tree_name}")[input_weight].array(library="np")
        arr_weight = ak.to_pandas(original[input_weight])
    else:
        arr_weight = np.ones(len(original))
    # print(variables)
    # print(type(original[variables]))
    # print(np.array(original[variables].to_numpy()))
    # print(original[variables].values())
    arr_branches = ak.to_pandas(original[branches])  # [:, np.newaxis]
    print(arr_branches)
    # print(np.array(original[variables].values()))
    # weights = ak.from_numpy(
    #                 dict(output_weight_name,
    #                      reweighter.predict_weights(arr_branches, arr_weight))
    # )
    #
    # ans = ak.zip({**dict(ak.fields(original), ak.unzip(array)),
    #               **dict(ak.fields(original), ak.unzip(array))})

    weights = reweighter.predict_weights(arr_branches, arr_weight)
    original[output_weight_name] = ak.from_numpy(weights)

    print(original)

    # it seems uproot4 still has some problems writting jagged arrays
    with ur.recreate(output_file) as ofile:
        # ofile.mktree(input_tree_name, {k: v.dtype.name for k, v in original.items()})
        # ofile[input_tree_name].extend(original.to_dict(orient='list'))
        dictarr2root(original, ofile, tree_name=input_tree_name)
        # ofile[input_tree_name] = original


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    apply_weights(**vars(args))
