from ipanema import Parameters, Sample
import os
import hjson
import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
import numpy as np
import argparse

binned_vars = hjson.load(open(f'config.json'))['binned_variables']
vars_ranges = hjson.load(open(f'config.json'))['binned_variables_ranges']




def plot_comparison_binned_angular_weights(pars, variable, version):
    nterms = len(pars[0]) # trinagular number of the number of amplitudes T(4) = 10
    pcols = 3
    prows = (nterms-1)//pcols + (1 if (nterms-1)%pcols else 0)

    fig, ax = plt.subplots(prows, pcols, sharex=True, figsize=[prows*4.8,2*4.8])
    for i in range(1,nterms):
      ax[(i-1)//pcols ,(i-1)%pcols].fill_between(
        [vars_ranges[variable][0], vars_ranges[variable][-1]],
        2*[pars[0][f'w{i}'].value+pars[0][f'w{i}'].stdev],2*[pars[0][f'w{i}'].value-pars[0][f'w{i}'].stdev],
        alpha=0.8, label="Base")
      ax[(i-1)//pcols ,(i-1)%pcols].set_ylabel(f"$w_{i}$")
      for bin in range( len(vars_ranges[variable])-1 ):
        ax[(i-1)//pcols ,(i-1)%pcols].errorbar(
          (vars_ranges[variable][bin]+vars_ranges[variable][bin+1])*0.5,
          pars[bin+1][f'w{i}'].value,
          xerr=vars_ranges[variable][bin+1]-(vars_ranges[variable][bin]+vars_ranges[variable][bin+1])*0.5,
          yerr=pars[bin+1][f'w{i}'].stdev, fmt='.', color='k')
      #ax[(i-1)//pcols ,(i-1)%pcols].legend()
    watermark(ax[0,0],version=f"${version}$",scale=1.01)
    [make_square_axes(ax[ix,iy]) for ix,iy in np.ndindex(ax.shape)]
    [tax.set_xlabel(f"${get_var_in_latex(f'{variable}')}$") for tax in ax[prows-1,:]]
    return fig, ax


if __name__ == '__main__':
    def argument_parser():
      parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
      # Samples
      parser.add_argument('--params',
        help='Bs2JpsiPhi MC sample')
      parser.add_argument('--figure',
        help='Bd2JpsiKstar MC sample')
      # Configuration file ---------------------------------------------------------
      parser.add_argument('--mode',
        help='Mode to fit')
      parser.add_argument('--year',
        help='Year to fit')
      parser.add_argument('--version',
        help='Version of the tuples to use')
      parser.add_argument('--variable',
        help='Version of the tuples to use')
      parser.add_argument('--trigger',
        help='Trigger to fit, choose between comb, biased and unbiased')

      return parser

    # Get arguments
    args = vars(argument_parser().parse_args())
    VERSION = args['version']
    VAR = args['variable']

    # Get parameters
    params = [Parameters.load(p) for p in args['params'].split(',')]

    # Plot and save result
    fig, ax = plot_comparison_binned_angular_weights(params, VAR, VERSION)
    fig.savefig(args['figure'])
