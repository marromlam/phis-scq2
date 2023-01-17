# merge_lambdab_weights
#
#

__all__ = ['add_tagging', 'add_lambdab_weight']
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import argparse
from selection.lb_prepare import tagging_fractions
import numpy as np
import ipanema
import uproot3 as uproot
import pandas as pd
import os

# np.random.seed(45353654)

# specify as values the actual name of taggers in the nTuples
TAGS = {
    'run2': {
        'tagdecision_os': 'OS_Combination_DEC',
        'tagomega_os': 'OS_Combination_ETA',
        'tagdecision_ss': 'B_SSKaonLatest_TAGDEC',
        'tagomega_ss': 'B_SSKaonLatest_TAGETA',
        'tagOS': 'OS_Combination_DEC',
        'etaOS': 'OS_Combination_ETA',
        'tagSS': 'B_SSKaonLatest_TAGDEC',
        'etaSS': 'B_SSKaonLatest_TAGETA'
    },
    'run1': {
        'tagdecision_os': 'tagos_dec_old',
        'tagomega_os': 'tagos_eta_old',
        'tagdecision_ss': 'B_SSKaon_TAGDEC',
        'tagomega_ss': 'B_SSKaon_TAGETA'
    },
}


def add_tagging(mcLb, rdLb, tag_type='run2'):
    """
    Add tagging branches to the Lb sample

    Notes
    -----
    TODO: This function is written in sucha a way there is a very direct
          equivalent in P2VV repo. For the future it requires a speed up. The
          hOS and hSS are also slow generations because they are ment to run
    """
    # get fraction of each of the taggers
    os, ss, ss_os, hOS, hSS = tagging_fractions(rdLb, tag_type)

    dummy = np.zeros((mcLb.shape[0]))

    def generate_tag_info(dummy):
        frac = np.random.rand()
        if (frac <= os):
            # we have information from te OS but not from SS
            _tagOS = -1 if np.random.rand() < 0.5 else 1
            _etaOS = hOS()
            _tagSS = 0
            _etaSS = 0.5
            # return 0
            return np.array([_tagOS, _etaOS, _tagSS, _etaSS])
        elif (frac > os) and (frac <= os+ss):
            # we have information from te SS but not from OS
            _tagOS = 0
            _etaOS = 0.5
            _tagSS = -1 if np.random.rand() < 0.5 else 1
            _etaSS = hSS()
            # return 0
            return np.array([_tagOS, _etaOS, _tagSS, _etaSS])
        elif (frac > os + ss) and (frac <= os + ss + ss_os):
            # we have information from BOTH taggers
            _tagOS = -1 if np.random.rand() < 0.5 else 1
            _etaOS = hOS()
            _tagSS = -1 if np.random.rand() < 0.5 else 1
            _etaSS = hSS()
            # return 0
            return np.array([_tagOS, _etaOS, _tagSS, _etaSS])
        elif frac > os + ss + ss_os:
            # untagged events
            _tagOS = 0
            _etaOS = 0.5
            _tagSS = 0
            _etaSS = 0.5
            # return 0
            return np.array([_tagOS, _etaOS, _tagSS, _etaSS])
        else:
            raise ValueError("que cona pasa")

    # generate the tagging information
    tag_info = np.vectorize(generate_tag_info, signature='()->(n)')(dummy)
    mcLb[TAGS[tag_type]['tagOS']] = tag_info[:, 0].astype(np.int32)
    mcLb[TAGS[tag_type]['etaOS']] = tag_info[:, 1].astype(np.float64)
    mcLb[TAGS[tag_type]['tagSS']] = tag_info[:, 2].astype(np.int32)
    mcLb[TAGS[tag_type]['etaSS']] = tag_info[:, 3].astype(np.float64)

    return mcLb


def add_lambdab_weight(df, nLb=1):
    """
    Add wLb weight to df. The number of events added is equal to nLb

    Parameters
    ----------
    df : pandas.DataFrame
      Original dataframe where we want to add wLb.
    nLB : int or float
      Number of Lb events to be added

    Returns
    -------
    pandas.DataFrame
      Original dataframe with the added weights.
    """

    sum_weight = np.sum(df.eval('wdp * wppt').values)

    # add wLb to a df copy
    dfOut = df.copy()
    dfOut.eval('wLb = -1 * wdp * wppt', inplace=True)
    dfOut['wLb'] = dfOut['wLb'] * (nLb / sum_weight)

    # check the amount of events added is correct
    sum_wLb = np.sum(dfOut['wLb'].values)
    print('New sum of weights: {}'.format(sum_wLb))
    return dfOut


def merge_lb(nLb, mcLbIn, rdLbIn, data_in):
    """
    mc are Lb mc
    rd are Bs2JpsiPhi rd
    """

    print('Resampling tagging variables in Lb tree...')
    mcLbOut = add_tagging(mcLbIn, rdLbIn, 'run2')

    print('Preparing Lb tree for merging...')
    mcLbOut = add_lambdab_weight(mcLbOut, nLb)

    print('Preparing JpsiPhi tree for merging...')
    # add_wlb_jpsiphi( data_in, data_out, tree_name, year, TAGS['run2'])
    rdOut = rdIn.copy()
    rdOut['wLb'] = np.ones((rdOut.shape[0]))

    br_phi = list(rdOut.keys())
    br_lb = list(mcLbOut.keys())

    not_in_lb = [name for name in br_phi if name not in br_lb]
    not_in_phi = [name for name in br_lb if name not in br_phi]

    print('Trees are being reduced...')
    mcLb_reduced = mcLbOut.drop(columns=not_in_phi)
    rd_reduced = rdOut.drop(columns=not_in_lb)

    print("MC Lb dataframe")
    print(mcLb_reduced)
    print("RD dataframe")
    print(rd_reduced)

    print('Merged RD dataframe')
    rdOut = pd.concat([rd_reduced, mcLb_reduced])
    print(rdOut)

    return mcLbOut, rdLbIn, rdOut


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Merge Lb mc into the data tuple")
    p.add_argument('--year', help='What data type?')
    p.add_argument('--number-of-lb', help='How much Lb MC events to inject')
    p.add_argument('--tree-name', default='DecayTree', help='Name of a tree')
    p.add_argument('--hist-loc', help='Where to save?')
    p.add_argument('--plot-loc', help='Where to save?')
    p.add_argument('--mc-lb-input', help='Lb tuple to be merged to data')
    p.add_argument('--rd-lb-input', help='To which data file merge the Lb')
    p.add_argument('--rd-input', help='Data to be add lbWeight')
    p.add_argument('--rd-output', help='Data with lbWeight added')
    p.add_argument('--ntuple-no-veto', help='Data tuple w/o Lb veto applied')
    p.add_argument('--version', help='Data tuple w/o Lb veto applied')

    # parse arguments
    args = vars(p.parse_args())
    nLb = ipanema.Parameters.load(args['number_of_lb'])['nLb'].value
    mcLbIn = uproot.open(args['mc_lb_input'])['DecayTree']
    mcLbIn = mcLbIn.pandas.df(flatten=None)
    rdLbIn = uproot.open(args['rd_lb_input'])['DecayTree']
    rdLbIn = rdLbIn.pandas.df(flatten=None)
    rdIn = uproot.open(args['rd_input'])['DecayTree']
    rdIn = rdIn.pandas.df(flatten=None)
    if 'wLb' in rdIn:
      # print(rdIn)
      print("WARNING: wLb already present. Just copying tuple")
      # rdIn = rdIn.query("wLb>0")
      print(rdIn)
      os.system(f"cp {args['rd_input']} {args['rd_output']}")
      exit(0)

    # update tuples with wLb
    mcLbOut, rdLbOut, rdOut = merge_lb(nLb, mcLbIn, rdLbIn, rdIn)

    # save merged tree if we want
    # with uproot.recreate(args['mc_lb_output']) as rf:
    #     rf['DecayTree'] = uproot.newtree({var: 'float64' for var in mcLbOut})
    #     rf['DecayTree'].extend(mcLbOut.to_dict(orient='list'))
    # with uproot.recreate(args['rd_lb_output']) as rf:
    #     rf['DecayTree'] = uproot.newtree({var: 'float64' for var in rdLbOut})
    #     rf['DecayTree'].extend(rdLbOut.to_dict(orient='list'))
    with uproot.recreate(args['rd_output']) as rf:
      _branches = {}
      for k, v in rdOut.items():
          if 'int' in v.dtype.name:
              _v = np.int32
          elif 'bool' in v.dtype.name:
              _v = np.int32
          else:
              _v = np.float64
          _branches[k] = _v
      # mylist = list(dict.fromkeys(_branches.values()))
      rf['DecayTree'] = uproot.newtree(_branches)
      rf['DecayTree'].extend(rdOut.to_dict(orient='list'))


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
