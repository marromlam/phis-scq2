# reweighter
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import argparse
# import yaml
from hep_ml import reweight
import uproot3 as uproot

# from analysis.reweightings import reweight


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-file',
                        help='Path to the original file, to be reweighted')
    parser.add_argument('--original-weight', default='',
                        help='Name of branch of sweights')
    parser.add_argument(
        '--target-file', help='Path to the target file to match')
    parser.add_argument('--target-weight', default='sw',
                        help='Name of branch of sweights')
    parser.add_argument('--mode', help='Mode')
    parser.add_argument(
        '--output-file', help='Output pickle file including new weights')
    return parser


def reweighting(original_file, original_weight, target_file, target_weight, mode, output_file):
    variables = ['B_PT', 'muminus_PT', 'muplus_PT', 'B_ETA', 'hminus_PT', 'hplus_PT']
    original = uproot.open(original_file)
    original = original[list(original.keys())[0]].pandas.df(flatten=None)
    target = uproot.open(original_file)
    target = target[list(target.keys())[0]].pandas.df(flatten=None)
    # create weights
    print(original)
    original_weight = 'time/time'
    original.eval(f"weight = {original_weight}", inplace=True)
    target.eval(f"weight = {target_weight}", inplace=True)
    reweighter = reweight.GBReweighter(n_estimators=60,
                                       learning_rate=0.1,
                                       max_depth=6,
                                       min_samples_leaf=1000,
                                       gb_args={'subsample': 0.6})
    reweighter.fit(original[variables].values, target[variables].values,
                   original_weight=original['weight'].values,
                   target_weight=target['weight'].values)
    gbw = reweighter.predict_weights(original[variables], original['weight'])
    original['gbWeight'] = gbw
    f = uproot.recreate(output_file)
    f['DecayTree'] = uproot.newtree({var: 'float64' for var in original})
    f['DecayTree'].extend(original.to_dict(orient='list'))
    f.close()


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    reweighting(**vars(args))


# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
