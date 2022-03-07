__all__ = ['create_mass_bins', 'epsmKK']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


import argparse
import os
import uproot3 as uproot
import numpy as np
# import matplotlib.pyplot as plt
import complot


def create_mass_bins(nob):
    """
    Creates a set of bins

    Parameters
    ----------
    nob: int
      Number of mass bins to be created.

    Returns
    -------
    mass_bins: list
      List with the edges for the required number of bins.
    """

    if int(nob) == 1:
        mass_bins = [990, 1050]
    elif int(nob) == 2:
        mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    elif int(nob) == 3:
        mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    elif int(nob) == 4:
        mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    elif int(nob) == 5:
        mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    elif int(nob) == 6:
        mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    else:
        raise ValueError("Number of bins cannot be higher than 6")
    return mass_bins


def epsmKK(df1, df2, mode, year, nbins=6, mass_branch='X_M', weight=False):
    r"""
    Get efficiency

    .. math::
      x_2


    Parameters
    ----------
    df1 : pandas.DataFrame
      Sample from the selection pipeline
    df2 : pandas.DataFrame
      Particle gun generated Sample
    mode : str
      Decay mode of the sample coming from the selection pipeline
    year : int
      Year of the sample coming from the selection pipeline
    nbins : int
      Number of bins to compute the CSP factors
    mass_branch : string
      Branch to be used as mass for the X meson
    weight : string or bool
      Weight to be used in the histograms. If it is set to false, then a weihgt
      of ones will be used.

    Returns
    -------
    masses: numpy.ndarray
      Mass bins
    ratios: numpy.ndarray
      Efficiency
    """

    has_swave = True if 'Swave' in mode else False

    if not weight:
        weight = f'{mass_branch}/{mass_branch}'

    mass_knots = create_mass_bins(6)
    mLL, mUL = mass_knots[0]-10, mass_knots[-1]+10+140*has_swave

    nwide = 100 + 150*has_swave
    nnarr = 200 + 300*has_swave

    # particle gun sample histogram {{{

    hwide = np.histogram(df2['mHH'].values, nwide, range=(mLL, mUL))[0]
    hnarr = np.histogram(df2['mHH'].values, nnarr, range=(mLL, mUL))[0]
    # just to have the same hitogram as the one from ROOT::Draw
    hwide = np.array([0.] + hwide.tolist())
    hnarr = np.array([0.] + hnarr.tolist())

    # }}}

    # histogram true mass of the MC {{{

    hb = []
    for i, ll, ul in zip(range(6), mass_knots[:-1], mass_knots[1:]):
        if ll == mass_knots[0] or ul == mass_knots[-1]:
            _nbins = nwide
        else:
            _nbins = nnarr
        mass_cut = f"{mass_branch} > {ll} & {mass_branch} < {ul}"
        true_mass_cut = f"truemHH > {mLL} & truemHH < {mUL}"
        _weight = (f"({mass_cut}) & ({true_mass_cut}) & (truthMatch)")
        _w = df1.eval(f"( {_weight} ) * {weight}")
        _v = df1['truemHH'].values
        _c, _b = np.histogram(_v, _nbins, weights=_w, range=(mLL, mUL))
        # print("bins", _b)
        hb.append([_b, [0] + _c.tolist() + [0]])
        # hb.append([0.5*(_b[1:]+_b[:-1]), [0] + _c.tolist() + [0]])

    # }}}

    # build afficiency histograms {{{

    masses = []
    ratios = []
    for j in range(len(hb)):
        _ratios = []
        _masses = []
        if(j == 0 or j == 5):
            NBINS = nwide
            # print("NBINS WIDE=",NBINS)
            for i in range(NBINS):
                ratio = hb[j][1][i] / max(hwide[i], 1)
                if j != 0 and hb[j][1][i] < mLL and has_swave:
                    ratio = 0.
                ratio = 0 if hwide[i] == 0 else ratio
                _ratios.append(ratio)
                _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i+1]))
        else:
            NBINS = nnarr
            # print("NBINS NARROW =",NBINS_NARROW)
            for i in range(NBINS):
                ratio = hb[j][1][i] / max(hnarr[i], 1)
                if j != 0 and hb[j][1][i] < mLL and has_swave:
                    ratio = 0.
                ratio = 0 if hnarr[i] == 0 else ratio
                _ratios.append(ratio)
                _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i+1]))
        masses.append(_masses)
        ratios.append(_ratios)

    # }}}

    # plot and dump {{{

    # ### To dump: NF with ratios and masses
    # functions = []
    # for i in range(len(mkk_bins)):
    #     functions.append(NF(masses[i],ratios[i]))
    #     # cPickle.dump(functions[i],open(),"w"))
    #     with open(dsafdsafadskfahds, "wb") as output_file:
    #          cPickle.dump(functions[i], output_file)

    # }}}

    return masses, ratios

# }}}


# command line {{{

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--simulated-sample',
                   help='Path to the preselected input file')
    p.add_argument('--pgun-sample', help='Path to the uncut input file')
    p.add_argument('--output-figure', help='Output directory')
    p.add_argument('--output-histos', help='Output directory')
    p.add_argument('--mode', help='Name of the selection in yaml')
    p.add_argument('--year', help='Year of the selection in yaml')
    p.add_argument('--nbins', help='Year of the selection in yaml')
    args = vars(p.parse_args())

    mass_branch = 'mHH'

    # selection branches
    list_branches = [
        mass_branch, 'gbWeights', 'truemHH', 'truthMatch'
    ]

    # load samples as dataframes
    sim = uproot.open(args['simulated_sample'])
    sim = sim[list(sim.keys())[0]].pandas.df(branches=list_branches)
    gun = uproot.open(args['pgun_sample'])
    gun = gun[list(gun.keys())[0]].pandas.df()

    # choose weights
    if args['mode'] == 'MC_Bs2JpsiKK_Swave':
        weight = 'gb_weights'
    else:
        weight = False

    masses, ratios = epsmKK(sim, gun, mode=args['mode'], year=args['year'],
                            nbins=args['nbins'], mass_branch=mass_branch,
                            weight=False)

    # create efficiency plot
    fig, axplot = complot.axes_providers.axes_plot()
    for i in range(6):
        axplot.fill_between(masses[i], ratios[i], 0, alpha=0.5)
    axplot.set_xlabel(r"$m(K^+K^-)$")
    axplot.set_ylabel(r"Efficiency, $\epsilon$")
    fig.savefig(args['output_figure'])
    print(args['output_histos'])
    # dump results
    np.save(os.path.join(args['output_histos']), [masses, ratios],
            allow_pickle=True)

    _masses, _ratios = np.load(os.path.join(args['output_histos']), allow_pickle=True)
    print(_masses, masses)
    print(_ratios, ratios)

# }}}


# vim: fdm=marker
