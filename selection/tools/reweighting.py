import argparse
import yaml
import uproot as ur
import numpy as np
import pandas
import pickle
from hep_ml import reweight


def read_variables_from_yaml(mode, variables_files):
    variables = []
    for file in variables_files:
        with open(file, 'r') as stream:
            variables += list(yaml.safe_load(stream)[mode].keys())
    return variables


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-file', nargs='+', help='Path to the original file, to be reweighted')
    parser.add_argument('--original-tree-name', default='DecayTree', help='Name of the tree in original file')
    parser.add_argument('--original-weight', default='', help='Name of branch of sweights')
    parser.add_argument('--target-file', help='Path to the target file to match')
    parser.add_argument('--target-tree-name', default='DecayTree', help='Name of the tree in target file')
    parser.add_argument('--target-weight', default='', help='Name of branch of sweights')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--weight-method', default='gb', choices=['gb', 'binned'], help='Specify method used to reweight')
    parser.add_argument('--variables-files', nargs='+', help='Path to the file with variable lists')
    parser.add_argument('--output-file', help='Output pickle file including new weights')
    return parser


def reweighting(original_file, original_tree_name, original_weight, target_file, target_tree_name,
    target_weight, mode, weight_method, variables_files, output_file):

    variables = read_variables_from_yaml(mode, variables_files)

    original_file = [original_file] if type(original_file)!=type([]) else original_file
    names = []
    for n in original_file:
      names.append(n if n.endswith('.root') else n+'*.root')
    print('Specified input files:\n', names)

    original = ur.concatenate([f"{fname}:{original_tree_name}" for fname in names], library="np", expressions=variables)
    original = pandas.DataFrame(original)
    if original_weight:
        original_weight = ur.concatenate([f"{fname}:{original_tree_name}" for fname in names], library="np", expressions=[original_weight])[original_weight].array()
    else:
        original_weight = np.ones(len(original))

    target = ur.open(f"{target_file}:{target_tree_name}").arrays(library="np", expressions=variables)
    target = pandas.DataFrame(target)
    if target_weight == "1":
        target_weight = False
    print("weight", target_weight)
    if target_weight:
        target_weight = ur.open(f"{target_file}:{target_tree_name}")[target_weight].array(library="np")
    else:
        target_weight = np.ones(len(target))

    if weight_method == 'gb':
        reweighter = reweight.GBReweighter(n_estimators=60,
                                           learning_rate=0.1,
                                           max_depth=6,
                                           min_samples_leaf=1000,
                                           gb_args={'subsample': 1.0})
        reweighter.fit(original, target,
                       original_weight=original_weight,
                       target_weight=target_weight)

    elif weight_method == 'binned':
        reweighter = reweight.BinsReweighter(n_bins=20,
                                             n_neighs=1.)
        reweighter.fit(original, target,
                       original_weight=original_weight,
                       target_weight=target_weight)
    else:
        print("ERROR: Invalid weighter type. Valid types are 'gb' and 'binned'. Exiting.")
        return

    with open(output_file, 'wb') as output:
        pickle.dump(reweighter, output, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    reweighting(**vars(args))


# vim: fdm=marker
