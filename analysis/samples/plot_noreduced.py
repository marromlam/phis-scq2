__all__ = []
# Plot branches with noweight, sweight and rweight (all)
import ipanema
import uproot
import yaml
import matplotlib.pyplot as plt
import matplotlib
from ipanema import plotting, Sample, Parameter, Parameters
import numpy as np
import argparse
from utils.plot import watermark, mode_tex

with open(r"analysis/samples/branches_latex.yaml") as file:
    BRANCHES = yaml.load(file, Loader=yaml.FullLoader)

def argument_parser():
  p = argparse.ArgumentParser(description='Plot branches of samples')
  p.add_argument('--tuple', help='MC sample')
  p.add_argument('--control', help='RD sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--version', help='version of the tuple')
  p.add_argument('--mode', help='MODE')
  p.add_argument('--branch', help='branch which want to plot')
  p.add_argument('--output', help='output')
  return p

if __name__ == '__main__':
  args = vars(argument_parser().parse_args())
  YEARS = args['year'].split(',')
  input_tuples = args['tuple'].split(',')
  input_control = args['control'].split(',')
  output = args['output'].split(',')
  mode = args['mode']
  branch = args['branch']
  version = args['version']
  comp_version = False
  #w = weights for tuples, wc=weights for control channels
  w = ['sw']
  wc = ['sw/gb_weights']
  for j in range(len(output)):
    fig, axplot, axpull = plotting.axes_plotpull()
    for i, year in enumerate(YEARS):
      #Load tuple and control samples
      tuple = Sample.from_root(input_tuples[i], branches = ['sw', branch])
      control = Sample.from_root(input_control[i], branches= ['sw', 'gb_weights', branch])
      #histogram
      range = BRANCHES[mode][branch].get('range')
      ht = ipanema.hist(tuple.df[branch], bins=np.linspace(range[0],range[1],50),
                    weights=tuple.df.eval(w[j]), density=True)
      hc = ipanema.hist(control.df[branch], bins=ht.edges,
                    weights=control.df.eval(wc[j]), density=True)
      axplot.step(ht.bins, ht.counts, where='mid', label=f'{year} RD Bs v0r5',
                  linestyle='-', color=f'C{i}')
      if 'dG0' in mode:
        axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} MC Bs dG0 v0r5',
                linestyle='-.', color=f'C{i}')
      else:
        axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} MC Bs v0r5',
                linestyle='-.', color=f'C{i}')
      axpull.fill_between(ht.bins,hc.counts/ht.counts,1, facecolor=f'C{i}', alpha=0.4)
    axplot.set_ylabel(f"Candidates")
    axpull.set_xlabel(BRANCHES[mode][branch].get('latex_name'))
    axpull.set_ylabel(f"$\\frac{{N(MC)}}{{N(data)}}$")
    axpull.set_ylim(0.0,2.0)
    axpull.set_xlim(range[0], range[1])
    axpull.set_yticks([0.5, 1, 1.5])
    #if 'log' in branch:
        #axplot.set_yscale('log')
        #axplot.legend(loc='center left', fontsize='small')
        #watermark(axplot, version=f'$v0r5$', scale=100.0)
    #else:
    axplot.legend(fontsize='small')
    watermark(axplot, version=f'$v0r5$', scale=1.2)
    fig.savefig(f'{output[j]}')
