__all__ = ['weighted_avg_and_std']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


import ipanema
import numpy as np
import complot
from analysis.angular_acceptance import bdtconf_tester
import matplotlib.pyplot as plt
from utils.plot import watermark, make_square_axes
import uncertainties as unc
import argparse


def weighted_avg_and_std(values, weights):
  """
  Return the weighted average and standard deviation.

  values, weights -- Numpy ndarrays with the same shape.
  """
  average = np.average(values, weights=weights)
  # Fast and numerically precise:
  variance = np.average((values-average)**2, weights=weights)
  return (average, np.sqrt(variance))


if __name__ == '__main__':

  p = argparse.ArgumentParser(description='Compute angular acceptance.')
  p.add_argument('--nominal-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--bdt-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--n-tests', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-table', help='Bs2JpsiPhi MC sample')
  args = vars(p.parse_args())

  nominal_params = args['nominal_params']
  bdt_params = args['bdt_params'].split(',')
  num_of_tests = int(args['n_tests'])

  # load nominal parameters
  pars = ipanema.Parameters.load(nominal_params)

  # load all bdt parameters
  bdtpars = []
  for i in range(1, num_of_tests+1):
    _item = list(bdtconf_tester.bdtmesh(i, num_of_tests, verbose=False).values())[:-1]
    _pars = ipanema.Parameters.load(bdt_params[i-1])
    _item += list(_pars.valuesdict().values())
    if abs(_item[31]) < 0.0078:
      print('yes!')
      print(_item[31])
      bdtpars.append(_pars)

  # create naming protocol
  _names = list(map(lambda x: pars[x].name, pars.valuesdict().keys()))
  all_pars = {}
  for _par in _names:
    all_pars[_par] = []
    for _set in bdtpars:
      all_pars[_par].append(_set[_par].uvalue)

  table = []
  table.append(r"\begin{tabular}{lccc|c}")
  _table = [
    f"{'Parameter':>50}",
    f"{'Nominal':>20}",
    f"{'Avg.':>10}",
    f"{'syst.':>10}",
    f"{'pull':>10}"
  ]
  table.append( " & ".join(_table) + r" \\")

  for _par in _names:
    if pars[_par].free:
      _vals = np.array([v.n for v in all_pars[_par]])
      _devs = np.array([v.s for v in all_pars[_par]])
      _nomi = pars[_par].uvalue
      # _mu = np.average(_vals, weights=1/(_devs**2))
      # _std = np.sqrt(np.sum((_vals-_mu)**2)/len(_vals))
      _mu, _std = weighted_avg_and_std(_vals, _devs)
      _avg = unc.ufloat(_mu, _std)
      _pull = np.abs(_nomi.n-_avg.n) / np.sqrt(_nomi.s**2+_avg.s**2)
      _syst = _nomi.n - _avg.n
      pars[_par].casket = {"gbconf_syst": _syst}
      # print(np.std(_vals), _std)
      # print(_par)
      __name = f"{pars[_par].latex}"
      __nomi = f"{pars[_par].uvalue:.2uP}"
      __mu = f"{_mu:.4f}"
      __pull = f"{_pull:.4f}"
      __syst = f"{_syst:.4f}"
      _table = [
        f"{__name:>50}",
        f"{__nomi:>20}",
        f"{__mu:>10}",
        f"{__syst:>10}",
        f"{__pull:>10}",
      ]
      table.append( " & ".join(_table) + r" \\")
      # print(f"{__name:>50} & {__nomi:>20} & {__mu:>10} & {__syst:>10} & {__pull:>10} \\")
  table.append(r"\end{tabular}")
  print("\n".join(table))
  with open(args['output_table'], 'w') as f:
    f.write("\n".join(table))
  pars.dump(args['output_params'])


