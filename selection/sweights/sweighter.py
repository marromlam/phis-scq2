# Merge sweights
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]

import uproot3 as uproot
import argparse
from ipanema import Sample
import numpy as np
import complot
import matplotlib.pyplot as plt
import os
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes


def shorten_mode(mode):
  if 'Bs' in mode:
    return 'Bs'
  elif 'Bd' in mode:
    return 'Bd'
  elif 'Bu' in mode:
    return 'Bu'
  else:
    print("adsfadsfds")
    exit()


if __name__ == '__main__':
  p = argparse.ArgumentParser(description="mass fit")
  p.add_argument('--input-sample')
  p.add_argument('--output-sample')
  p.add_argument('--output-plots')
  p.add_argument('--biased-weights')
  p.add_argument('--unbiased-weights')
  p.add_argument('--mode')
  p.add_argument('--version', default=None)
  args = vars(p.parse_args())

  mode = args['mode']
  version = args['version']
  short_mode = shorten_mode(args['mode'])
  ones = False
  if version:
    if 'bkgcat60' in version and 'MC' in mode:
      ones = True

  # Load full dataset and creae prxoy to store new sWeights
  sample = Sample.from_root(args['input_sample'], flatten=None)
  _proxy = np.float64(sample.df['time']) * 0.0

  # List all set of sWeights to be merged
  bpars = args["biased_weights"].split(",")
  upars = args["unbiased_weights"].split(",")

  list_of_weights = np.load(bpars[0], allow_pickle=True)
  list_of_weights = list_of_weights.item()
  list_of_weights = list(list_of_weights.keys())

  # first scan
  to_plot = {}
  to_df = {}
  create_plot = False
  for k in list_of_weights:
    for sw in bpars + upars:
      v = np.load(sw, allow_pickle=True)
      v = v.item()
      if k.startswith('mass'):
        create_plot = True
        to_plot['mass'] = np.zeros_like(v[k])
      elif k.startswith('bins'):
        to_plot['bins'] = np.zeros_like(v[k])
      elif k.startswith('counts'):
        to_plot['counts'] = np.zeros_like(v[k])
      elif k.startswith('pull'):
        to_plot['pull'] = np.zeros_like(v[k])
      elif k.startswith('yerr'):
        to_plot['yerr'] = np.zeros_like(v[k])
      elif k.startswith('xerr'):
        to_plot['xerr'] = np.zeros_like(v[k])
      elif k.startswith('pdf_'):
        to_plot[k[4:]] = {'arr': np.zeros_like(v[k]), 'label': k[4:]}
      else:
        to_df[k] = _proxy
  print("All weights:", list(to_df.keys()))
  print("All items to plot:", list(to_plot.keys()))

  # add sweights
  for w in to_df.keys():
    __proxy = 0 * _proxy
    for sw in bpars + upars:
      _weight = np.load(sw, allow_pickle=True)
      _weight = _weight.item()
      __proxy += _weight[w]
    sample.df[f"{w[1:]}SW"] = 1 + 0 * __proxy if ones else __proxy
  with uproot.recreate(args['output_sample']) as f:
    _branches = {}
    for k, v in sample.df.items():
      if 'int' in v.dtype.name:
        _v = np.int32
      elif 'bool' in v.dtype.name:
        _v = np.int32
      else:
        _v = np.float64
      _branches[k] = _v
    mylist = list(dict.fromkeys(_branches.values()))
    f["DecayTree"] = uproot.newtree(_branches)
    f["DecayTree"].extend(sample.df.to_dict(orient='list'))

  if create_plot:
    for k in list_of_weights:
      for i, sw in enumerate(bpars + upars):
        # print("file", i)
        v = np.load(sw, allow_pickle=True)
        v = v.item()
        if k.startswith('mass'):
          to_plot['mass'] = v[k]
          to_plot['xlabel'] = k.split('_')[-1]
          to_plot['xlabel'] = to_plot['xlabel'].replace('mu', r'\mu\!')
          to_plot['xlabel'] = to_plot['xlabel'].replace('pi', r'\pi\!')
          to_plot['xlabel'] = to_plot['xlabel'].replace('K', r'K\!')
        elif k.startswith('bins'):
          to_plot['bins'] = v[k]
        elif k.startswith('counts'):
          to_plot['counts'] += v[k]
        elif k.startswith('pull'):
          to_plot['pull'] += v[k]
        elif k.startswith('yerr'):
          to_plot['yerr'] += v[k]
        elif k.startswith('xerr'):
          to_plot['xerr'] = v[k]
        elif k.startswith('pdf_'):
          to_plot[k[4:]]['arr'] += v[k]
          if 'Bd' in k:
            label = r'$B_d^0$'
          elif 'Bs' in k:
            label = r'$B_s^0$'
          elif 'Bu' in k:
            label = r'$B_u^+$'
          elif 'Ds' in k:
            label = r'$D_s^+$'
          elif 'total' in k:
            label = "total fit"
          elif 'comb' in k:
            label = "combinatorial"
          elif 'exp' in k:
            label = "combinatorial"
          to_plot[k[4:]]['label'] = label
    # actual plotting
    mass = to_plot['mass']
    to_plot.pop('mass')
    hbins = to_plot['bins']
    to_plot.pop('bins')
    hcounts = to_plot['counts']
    to_plot.pop('counts')
    hyerr = to_plot['yerr']
    to_plot.pop('yerr')
    hxerr = to_plot['xerr']
    to_plot.pop('xerr')
    hpull = to_plot['pull']
    to_plot.pop('pull')
    xlabel = to_plot['xlabel']
    to_plot.pop('xlabel')

    fig, axplot, axpull = complot.axes_plotpull()
    # hdata = complot.hist(mass, weight, bins=100, density=False)
    mass = np.linspace(np.min(mass), np.max(mass), len(mass))
    # plot histogram
    axplot.errorbar(hbins, hcounts, yerr=hyerr, xerr=hxerr, fmt=".k")
    # plot pdf
    for i, k in enumerate(to_plot.keys()):
      print(to_plot[k]['label'])
      _color = f"C{i+1}" if not 'total' == k else 'C0'
      axplot.plot(mass, to_plot[k]['arr'], "-", color=_color,
                  label=to_plot[k]['label'])
    # plot pulls
    hpull = complot.compute_pdfpulls(mass, to_plot['total']['arr'], hbins, hcounts, *hyerr)
    axpull.fill_between(hbins, hpull, 0, facecolor="C0", alpha=0.5)
    # label and save the plot
    axpull.set_xlabel(rf"$m({xlabel})$ [MeV/$c^2$]")
    axpull.set_ylim(-6.5, 6.5 / np.inf)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel(rf"Candidates")
    axplot.legend(loc="upper right")
    v_mark = 'LHC$b$'  # watermark plots
    tag_mark = 'THIS THESIS'
    watermark(axplot, version=v_mark, tag=tag_mark, scale=1.3)
    os.makedirs(args['output_plots'], exist_ok=True)
    fig.savefig(os.path.join(args['output_plots'], f"fit.pdf"))
    axplot.set_yscale("log")
    try:
      axplot.set_ylim(1e0, 1.5 * np.max(to_plot['total']['arr']))
    except:
      print("axes not scaled")
    fig.savefig(os.path.join(args['output_plots'], f"logfit.pdf"))
    plt.close()
  else:
    os.makedirs(args['output_plots'], exist_ok=True)
    os.system(f"touch {os.path.join(args['output_plots'], 'placeholder.pdf')}")

# vim: fdm=marker
