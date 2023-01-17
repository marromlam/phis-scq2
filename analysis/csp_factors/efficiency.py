__all__ = ['create_mass_bins', 'epsmKK']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


import argparse
import os
import uproot3 as uproot
import numpy as np
# import matplotlib.pyplot as plt
import complot
import pandas as pd

# from numericFunctionClass import NF  # taken from Urania
# import pickle as cPickle
from utils.plot import watermark
from utils.helpers import version_guesser


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
    mass_bins = [990, 1020, 1050]
  # elif int(nob) == 3:
  #     mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 4:
    # mass_bins = [990, 1017, 1020, 1027, 1050]  # equipoblado
    # mass_bins = [990, 1013, 1020, 1027, 1050]  # 1st -- best
    # mass_bins = [990, 1011, 1020, 1027, 1050]  # 2nd
    # mass_bins = [990, 1018, 1020, 1027, 1050]  # 3rd -- nada
    mass_bins = [990, 1012, 1020, 1027, 1050]  # 4nd -- definitive !!!!!
  elif int(nob) == 5:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 6:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    # mass_bins = [990, 1014.78, 1018.41, 1020, 10223.42, 1033, 1050]
  else:
    raise ValueError("Number of bins cannot be higher than 6")
  return mass_bins


def create_sigmam_bins(nob):
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
    mass_bins = [-200, 200]
  elif int(nob) == 2:
    mass_bins = [0, 5.66, 20]
  elif int(nob) == 3:
    mass_bins = [0, 5.22, 6.13, 20]
  elif int(nob) == 4:
    mass_bins = [0, 4.97, 5.66, 6.42, 20]
  elif int(nob) == 5:
    mass_bins = [0, 4.80, 5.40, 5.93, 6.62, 20]
  else:
    raise ValueError("Number of bins cannot be higher than 5")
  return mass_bins


def create_time_bins(nob):
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
    mass_bins = [-100, 100]
  elif int(nob) == 2:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 3:
    mass_bins = [0.3, 1.00, 2.05, 15]
  elif int(nob) == 4:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 5:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 6:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  else:
    raise ValueError("Number of bins cannot be higher than 6")
  return mass_bins


def create_cosK_bins(nob):
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
    mass_bins = [-1, 1]
  elif int(nob) == 2:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 3:
    mass_bins = [-1, -0.30365096, 0.30543088, 1]
  elif int(nob) == 4:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 5:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 6:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  else:
    raise ValueError("Number of bins cannot be higher than 6")
  return mass_bins


def create_cosL_bins(nob):
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
    mass_bins = [-1, 1]
  elif int(nob) == 2:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 3:
    mass_bins = [-1, -0.30443096, 0.30577088, 1]
  elif int(nob) == 4:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 5:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 6:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  else:
    raise ValueError("Number of bins cannot be higher than 6")
  return mass_bins


def create_hphi_bins(nob):
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
    mass_bins = [-3.15, 3.15]
  elif int(nob) == 2:
    mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  elif int(nob) == 3:
    mass_bins = [-3.14158648, -1.03376272, 1.0352542, 3.14158675]
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
  print("Weight:", weight)

  mass_knots = create_mass_bins(int(nbins))
  mLL, mUL = mass_knots[0] - 10, mass_knots[-1] + 10 + 140 * has_swave

  nwide = 100 + 150 * has_swave
  nnarr = 200 + 300 * has_swave

  # particle gun sample histogram {{{

  hwide = np.histogram(df2['mHH'].values, nwide, range=(mLL, mUL))[0]
  hnarr = np.histogram(df2['mHH'].values, nnarr, range=(mLL, mUL))[0]
  # just to have the same hitogram as the one from ROOT::Draw
  hwide = np.array(hwide.tolist())
  hnarr = np.array(hnarr.tolist())

  # }}}

  # histogram true mass of the MC {{{

  hb = []
  for i, ll, ul in zip(range(int(nbins)), mass_knots[:-1], mass_knots[1:]):
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
    hb.append([_b, _c.tolist()])

  # }}}

  # build afficiency histograms {{{

  masses = []
  ratios = []
  for j in range(len(hb)):
    _ratios = []
    _masses = []
    if(j == 0 or j == int(nbins) - 1):
      NBINS = nwide
      for i in range(NBINS):
        ratio = hb[j][1][i] / float(max(hwide[i], 1))
        if j != 0 and hwide[i] < mLL and has_swave:
          ratio = 0.
        ratio = 0 if hwide[i] == 0 else ratio
        _ratios.append(ratio)
        _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i + 1]))
    else:
      NBINS = nnarr
      # print("NBINS NARROW =",NBINS_NARROW)
      for i in range(NBINS):
        ratio = hb[j][1][i] / float(max(hnarr[i], 1))
        if j != 0 and hnarr[i] < mLL and has_swave:
          ratio = 0.
        ratio = 0 if hnarr[i] == 0 else ratio
        _ratios.append(ratio)
        _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i + 1]))
    masses.append(_masses)
    ratios.append(_ratios)

  # }}}

  # plot and dump {{{

  # ### To dump: NF with ratios and masses
  # functions = []
  # for i in range(len(mass_knots) - 1):
  #   functions.append(NF(masses[i], ratios[i]))
  #   # cPickle.dump(functions[i],open(),"w"))
  #   with open(f"/scratch46/forVeronika/histo{mass_knots[i]}_{mass_knots[i+1]}", "wb") as output_file:
  #     cPickle.dump(functions[i], output_file)

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
      mass_branch, 'gbWeight', 'truemHH', 'truthMatch'
  ]

  # load samples as dataframes
  # sim = uproot.open(args['simulated_sample'])
  sim = [uproot.open(f) for f in args['simulated_sample'].split(',')]
  sim = [f[list(f.keys())[0]].pandas.df(branches=list_branches) for f in sim]
  # sim = [f[list(f.keys())[0]].pandas.df() for f in sim]
  sim = pd.concat(sim)
  gun = [uproot.open(f) for f in args['pgun_sample'].split(',')]
  gun = [f[list(f.keys())[0]].pandas.df() for f in gun]
  gun = pd.concat(gun)
  # gun = gun[list(gun.keys())[0]].pandas.df()
  # print(gun.keys())
  # gun.eval("mHH = X_M", inplace=True)
  print(sim)
  print(gun)

  # choose weights
  if args['mode'] == 'MC_Bs2JpsiKK_Swave':
    weight = 'gbWeight'
  else:
    weight = False

  print(args['nbins'])
  masses, ratios = epsmKK(sim, gun, mode=args['mode'], year=args['year'],
                          nbins=args['nbins'], mass_branch=mass_branch,
                          weight=weight)

  # create efficiency plot
  fig, axplot = complot.axes_providers.axes_plot()
  for i in range(int(args['nbins'])):
    axplot.fill_between(masses[i], ratios[i], 0, alpha=0.5)
  axplot.set_xlabel(r"$m(K^+K^-)$")
  axplot.set_ylabel(r"Efficiency, $\epsilon$")
  # version_watermark = version_guesser(args['version'])[0]
  version_watermark = None
  watermark(axplot, version=f"final", scale=1.2)

  fig.savefig(args['output_figure'])
  print(args['output_histos'])
  # dump results
  np.save(os.path.join(args['output_histos']), [masses, ratios],
          allow_pickle=True)

  # _masses, _ratios = np.load(os.path.join(args['output_histos']), allow_pickle=True)
  # print(_masses, masses)
  # print(_ratios, ratios)

# }}}


# vim: fdm=marker
