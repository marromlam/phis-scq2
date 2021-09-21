version = 'v0r5'
import uproot3 as uproot
import pandas as pd
import numpy as np


def equibins1d(x, nbin):
  """
  This functions takes a random variable x and creates nbin bins with the same
  number of candidates in each of them (if possible).

  Parameters
  ----------
  x : ndarray
    Random variable to histogram.
  nbins : int
    Number of bins.

  Returns
  -------
  ndarray
    Set of edges to histogram x with.
  """
  n = len(x)
  return np.interp(np.linspace(0, n, nbin + 1), np.arange(n), np.sort(x))


all_data = pd.concat([
  uproot.open('/scratch46/marcos.romero/sidecar/2015/Bs2JpsiPhi/v0r5.root')['DecayTree'].pandas.df(branches='time'),
  uproot.open('/scratch46/marcos.romero/sidecar/2016/Bs2JpsiPhi/v0r5.root')['DecayTree'].pandas.df(branches='time'),
  uproot.open('/scratch46/marcos.romero/sidecar/2017/Bs2JpsiPhi/v0r5.root')['DecayTree'].pandas.df(branches='time'),
  uproot.open('/scratch46/marcos.romero/sidecar/2018/Bs2JpsiPhi/v0r5.root')['DecayTree'].pandas.df(branches='time')
])


print(equibins1d(np.array(all_data['time']), 2))
