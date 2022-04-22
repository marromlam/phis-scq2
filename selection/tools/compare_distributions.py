import argparse
import yaml
import root_numpy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import rcParams, style
style.use('seaborn-muted')
rcParams.update({'font.size': 12})
rcParams['figure.figsize'] = 16, 8
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16


def read_from_yaml(file, mode):
    with open(file, 'r') as stream:
        vars_obj = yaml.safe_load(stream)[mode]
        if type(vars_obj)==type([]):
            return vars_obj
        elif type(vars_obj)==type({}):
            return list(vars_obj.keys())
        else:
            print('Provide a list or a dictionary with variables to compare')


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', action='append', help='Dataset to be compared. Args order: path_to_file, tree_name, weight_name(=none if N/A), label_name')
    parser.add_argument('--compare-vars', help='List of variables to be compared; if dict is given, then take list of the dictionary values')
    parser.add_argument('--mode', help='Name of the selection in yaml with variables')
    parser.add_argument('--plot-dir', help='Output path of pdfs')
    return parser


def draw_distributions(dataset, compare_vars, mode, plot_dir):

    compare_vars = read_from_yaml(compare_vars, mode)
    print('VARS:', compare_vars)
    for dset in dataset:
        dset.append(pandas.DataFrame(root_numpy.root2array(dset[0], treename=dset[1],
                                                                    branches=compare_vars)))
        if dset[2] == 'none':
            dset.append(np.ones(len(dset[4])))
        else:
            dset.append(root_numpy.root2array(dset[0], treename=dset[1],
                                                       branches=[dset[2]]))

    hist_settings = {'bins': 149, 'density': True, 'alpha': 0.8, 'histtype': 'step', 'lw': 2}
    for var in compare_vars:
        plt.figure()
        for dset in dataset:
            if dset[2] == 'none':
                plt.hist(dset[4][var], label=dset[3], **hist_settings)
            else:
                plt.hist(dset[4][var], weights=dset[5][dset[2]], label=dset[3], **hist_settings)
                print("var : ", var, " ", dset[2])

        plt.title(var, fontsize=20)
        plt.legend()
        plt.savefig(plot_dir + var + '.pdf')


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    draw_distributions(**vars(args))
