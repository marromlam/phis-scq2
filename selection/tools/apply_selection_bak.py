import argparse
import yaml
import cppyy

import pandas as pd
import uproot3 as uproot
import numpy as np

from ROOT import (
    vector,
    gInterpreter,
    ROOT,
    RDataFrame,
    TObject,
    std
)

from selection.tools.constants import C_LIGHT


def read_from_yaml(mode, selection_files):
    selection_dict = dict()
    for file in selection_files:
        with open(file, 'r') as stream:
            selection_dict.update(yaml.safe_load(stream)[mode])
    return selection_dict


def apply_cuts(cuts, dataframe, year):
    for key in cuts.keys():
        cut = cuts[key].format(C_LIGHT=C_LIGHT, year=year)
        if cut:
            dataframe = dataframe.Filter(cut, key)
    # check efficiencies
    report = dataframe.Report()
    report.Print()
    return dataframe


def apply_selection(input_files, input_tree_name, output_file, output_tree_name,
                    mode, cut_keys, cut_string, selection_files, branches_files,
                    keep_all_original_branches, year):
    # enable multithreading
    ROOT.EnableImplicitMT()
    input_files = [input_files] if type(
        input_files) != type([]) else input_files
    names = std.vector('string')()
    for n in input_files:
        names.push_back(n if n.endswith('.root') else n+'/*.root')
    print('Specified input files:\n', names)
    dataframe = RDataFrame(input_tree_name, names)


    # FUTURE print("START :: Loading files with uproot")

    # FUTURE rfiles = [uproot.open(f)[input_tree_name] for f in input_files]
    # FUTURE branches = sum([list(f.keys()) for f in rfiles], [])
    # FUTURE branches_in_df = [b.decode() for b in branches]
    # FUTURE # print(branches)
    # FUTURE # print(len(branches))
    # FUTURE # exit()
    # FUTURE # print(df)
    # FUTURE print("END :: Loading files with uproot")

    # read cuts from all input files
    cuts = read_from_yaml(mode, selection_files) if selection_files else {}
    # if cut keys are specified apply only desired cuts for given mode
    if 'all' not in cut_keys:
        cuts = {cut_key: cuts[cut_key] for cut_key in cut_keys}
    # if cut string is specified create corresponding cuts dictionary
    if cut_string:
        cuts = {'cut': cut_string}
    # read branches from all input files
    branches_to_add = read_from_yaml(
        mode, branches_files) if branches_files else {}

    # FUTURE WORK lambda_branches = ['helcosthetaK', 'helcosthetaL']
    # FUTURE WORK lambda_branches = []
    # FUTURE WORK load_branches = []
    # FUTURE WORK for b in branches_to_add:
    # FUTURE WORK     print(b)
    # FUTURE WORK     if b in branches_in_df+lambda_branches:
    # FUTURE WORK         load_branches.append(b)
    # FUTURE WORK     else:
    # FUTURE WORK         print(f'Branch {b} cannot be added')
    # FUTURE WORK         # branches_to_add.remove(b)
    # FUTURE WORK     # if b not in lambda_branches:
    # FUTURE WORK     #     print(f'Branch {b} cannot be added')
    # FUTURE WORK     #     branches_to_add.remove(b)
    # FUTURE WORK nevts = np.sum([f.numentries for f in rfiles])
    # FUTURE WORK print(nevts)
    # FUTURE WORK for
    # FUTURE WORK df = pd.concat([f.pandas.df(branches=load_branches, flatten=None) for f in rfiles])

    if branches_to_add:
        # get list of existing branches
        branches_in_df = dataframe.GetColumnNames()
        # define new branches and keep original branches if specified
        branches = vector('string')()
        if keep_all_original_branches:
            branches = branches_in_df

        # in case helicity angles and/or docaz are specified in branches
        gInterpreter.LoadMacro('selection/tools/calculate_helicity_angles.cpp')
        gInterpreter.LoadMacro('selection/tools/calculate_helicity_costheta.cpp')
        gInterpreter.LoadMacro('selection/tools/calculate_docaz.cpp')
        gInterpreter.LoadMacro('selection/tools/copy_array.cpp')
        # add new branches
        for branch in branches_to_add.keys():
            branch_value = branches_to_add[branch].format(
                C_LIGHT=C_LIGHT, year=year)
            if branch not in branches_in_df:
                if branch == branch_value:
                    print(
                        'WARNING: {} branch is not present in the original tree. Setting value to -99999.'.format(branch))
                    dataframe = dataframe.Define(branch, "-99999.0")
                elif not branch_value:
                    print('Skipping branch ', branch)
                    continue
                else:
                    dataframe = dataframe.Define(branch, branch_value)
            elif not branch_value:
                print('Skipping branch ', branch)
                continue
            branches.push_back(branch)
        # save new tree
        print('Branches kept in the pruned tree:', branches)
    else:
        print('All branches are kept in the tree')
    # apply all cuts
    if cuts:
        dataframe = apply_cuts(cuts, dataframe, year)
    # save new tree
    dataframe.Snapshot(output_tree_name, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', nargs='+',
                        help='Path to the input file')
    parser.add_argument('--input-tree-name',
                        default='DecayTree', help='Name of the tree')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--output-tree-name',
                        default='DecayTree', help='Name of the tree')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--cut-keys', default='all', nargs='+',
                        help='Specify which cuts for the mode should be applied, if not all')
    parser.add_argument('--cut-string', default=None,
                        help='Alternatively, specify cut string directly')
    parser.add_argument('--selection-files', nargs='+',
                        help='Yaml files with selection')
    parser.add_argument('--branches-files', nargs='+',
                        help='Yaml files with branches')
    parser.add_argument('--keep-all-original-branches', default=False,
                        help='Keeps all original branches if True, only adds specified branches if False')
    parser.add_argument('--year', required=True, help='Year of data taking')
    args = parser.parse_args()
    apply_selection(**vars(args))
