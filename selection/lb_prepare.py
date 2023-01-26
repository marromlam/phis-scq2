# create tagging branches from a RD sample to be sampled and then added to
# MC sample


__all__ = ['tagging_fractions']
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import numpy as np


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
    'run2Old': {
        'tagOS': 'tagdecision_os',
        'etaOS': 'tagomega_os',
        'tagSS': 'tagdecision_ss',
        'etaSS': 'tagomega_ss'
    },
    'run1': {
        'tagdecision_os': 'tagos_dec_old',
        'tagomega_os': 'tagos_eta_old',
        'tagdecision_ss': 'B_SSKaon_TAGDEC',
        'tagomega_ss': 'B_SSKaon_TAGETA'
    },
}


def tagging_fractions(df, tag_type='run2'):
    """
    Produces tagging fractions for the prepare_tree

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe with data
    tag_type : str
        String containing the family of run of LHCb: run1 or run2

    Returns
    -------
    pandas.Dataframe
        Dataframe with tagging fractions incorporated

    Notes
    -----
    TODO: clean this function
    """
    # TODO: automate creation of binning
    # binning is chosen to have similar stats in each bin
    # eOS_10 = np.array([0.00, 0.27, 0.33, 0.36, 0.39,
    #                   0.42, 0.44, 0.46, 0.48, 0.50])
    eOS_18 = np.array([0.0, 0.22, 0.27, 0.31, 0.33, 0.35, 0.36, 0.38, 0.39,
                       0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.5])
    # eSS_9 = np.array([0.00, 0.35, 0.41, 0.44, 0.45, 0.46, 0.47, 0.48, 0.50])
    eSS_12 = np.array([0.0, 0.30, 0.35, 0.38, 0.41, 0.42,
                      0.44, 0.45, 0.46, 0.47, 0.48, 0.5])

    # create OS histogram
    etaOS = df.query(f"{TAGS[tag_type]['tagOS']}!=0")[TAGS[tag_type]['etaOS']]
    swOS = df.query(f"{TAGS[tag_type]['tagOS']}!=0")['sigBsSW']
    hOS = np.histogram(etaOS, eOS_18, weights=swOS)
    # print(np.sum(hOS[0]))

    # create a generator for the OS tagger
    def generateOS():
        _x = 0.5*(hOS[1][1:]+hOS[1][:-1])
        _y = hOS[0] / np.sum(hOS[0])
        return np.random.choice(_x, 1, p=_y)[0]
    # test = [generateOS() for i in range(len(swOS))]

    # import matplotlib.pyplot as plt
    # plt.hist(etaOS, eOS_18, weights=swOS, density=True)
    # plt.hist(test, eOS_18, density=True)
    # plt.savefig('my_test_os.pdf')
    # breakpoint()
    # print(hOS)

    # create SS histogram
    etaSS = df.query(f"{TAGS[tag_type]['tagSS']}!=0")[TAGS[tag_type]['etaSS']]
    swSS = df.query(f"{TAGS[tag_type]['tagSS']}!=0")['sigBsSW']
    hSS = np.histogram(etaSS, eSS_12, weights=swSS)

    # create a generator for the OS tagger
    def generateSS():
        _x = 0.5*(hSS[1][1:]+hSS[1][:-1])
        _y = hSS[0] / np.sum(hSS[0])
        return np.random.choice(_x, 1, p=_y)[0]

    # import matplotlib.pyplot as plt
    # plt.hist(etaSS, eSS_12, weights=swSS)
    # plt.savefig('my_test_ss.pdf')
    # print(hSS)

    ######
    # get fractions OS/SS/OS&SS
    qOS = TAGS[tag_type]['tagOS']
    qSS = TAGS[tag_type]['tagSS']
    dssw_os = np.sum(df.query(f"{qOS}!=0 & {qSS}==0")['sigBsSW'])
    dssw_ss = np.sum(df.query(f"{qOS}==0 & {qSS}!=0")['sigBsSW'])
    dssw_ss_os = np.sum(df.query(f"{qOS}!=0 & {qSS}!=0")['sigBsSW'])
    dssw_ut = np.sum(df.query(f"{qOS}==0 & {qSS}==0")['sigBsSW'])
    tot = np.sum(df['sigBsSW'])

    os_tagged = dssw_os / tot
    ss_tagged = dssw_ss / tot
    both_tagged = dssw_ss_os / tot
    untagged = dssw_ut / tot

    print(f"{'Fraction of only OS tagged events':>50} : {os_tagged:.4f}")
    print(f"{'Fraction of only SS tagged events':>50} : {ss_tagged:.4f}")
    print(f"{'Fraction of only OS&SS tagged events':>50} : {both_tagged:.4f}")
    print(f"{'Fraction of only NOT tagged events':>50} : {untagged:.4f}")
    print(" ")

    dssw_os_p1 = np.sum(df.query(f"{qOS}==+1 & {qSS}==0")['sigBsSW'])
    dssw_os_m1 = np.sum(df.query(f"{qOS}==-1 & {qSS}==0")['sigBsSW'])
    dssw_ss_p1 = np.sum(df.query(f"{qOS}==0 & {qSS}==+1")['sigBsSW'])
    dssw_ss_m1 = np.sum(df.query(f"{qOS}==0 & {qSS}==-1")['sigBsSW'])

    os_p1 = dssw_os_p1 / dssw_os
    os_m1 = dssw_os_m1 / dssw_os
    ss_p1 = dssw_ss_p1 / dssw_ss
    ss_m1 = dssw_ss_m1 / dssw_ss

    print(f"{'Fraction of +1 OS tagged events':>50} : {os_p1:.4f}")
    print(f"{'Fraction of -1 OS tagged events':>50} : {os_m1:.4f}")
    print(f"{'Fraction of +1 SS tagged events':>50} : {ss_p1:.4f}")
    print(f"{'Fraction of -1 SS tagged events':>50} : {ss_m1:.4f}")

    return os_tagged, ss_tagged, both_tagged, generateOS, generateSS


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
