import argparse
import yaml
import root_numpy
import numpy as np
import pandas
import pickle
from hep_ml import reweight
from reweighting import read_variables_from_yaml

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


def apply_weights(input_file, input_tree_name, input_weight, variables_files, weight_method, weights_file, mode, output_file, output_weight_name):

    variables = read_variables_from_yaml(mode, variables_files)

    input_file = [input_file] if type(input_file)!=type([]) else input_file

    original = root_numpy.root2array(input_file, treename=input_tree_name)
    original = pandas.DataFrame(original)

    reweighter = pickle.load(open(weights_file, 'rb'))
    
    
    if input_weight:
        input_weight = root_numpy.root2array(input_file, treename=input_tree_name, branches=[input_weight])
    else:
        input_weight = np.ones(len(original))

    weights = reweighter.predict_weights(original[variables], input_weight)
        
        
    #original[weight_method+'_weights'] = weights
    original[output_weight_name] = weights

    original_array = original.to_records(False)
    weighted_tuple = root_numpy.array2root(original_array, output_file, mode='recreate', treename=input_tree_name)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    apply_weights(**vars(args))
