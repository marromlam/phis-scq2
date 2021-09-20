import matplotlib.pyplot as plt
import hjson
import numpy as np
import ipanema
import argparse
import os
from ipanema import plotting, Sample, Parameter, Parameters
from utils.helpers import version_guesser
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  p = argparse.ArgumentParser(description='Plot parameters from ang acc in each iteration')
  p.add_argument('--input', help='Params from angular acceptance')
  p.add_argument('--naive', help='Params after iterative procedure wo acceptance effects')
  p.add_argument('--params', help='Generated parameters')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Mode fitted')
  p.add_argument('--version', help='Version and cuts fitted')
  p.add_argument('--angacc', help='version of ang acceptance')
  p.add_argument('--output', help='Directory where the plots will be storaged')
  args = vars(p.parse_args())
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  output = args['output']
  pars = hjson.load(open(args['input']))
  naive = hjson.load(open(args['naive']))
  gen_pars = Parameters.load(args['params'])
  os.mkdir(f'{output}')
  for key in pars.keys():
    if pars[key]['casket']['1']['stdev'] != 0.0:
        plt.close()
        y = [v['value'] for v in pars[key]['casket'].values()]
        y_err =[v['stdev'] for v in pars[key]['casket'].values()]
        n = [v['value'] for v in naive[key]['casket'].values()]
        n_err =[v['stdev'] for v in naive[key]['casket'].values()]
        n = n[-1]
        n_err = n_err[-1]
        n_max = np.array(n)+np.array(n_err)
        n_min= np.array(n)-np.array(n_err)
        n_max = [n_max]*len(y)
        n_min = [n_min]*len(y)
        z = [gen_pars[key].value]*(len(y))
        it = np.arange(0, len(y)).tolist()
        plt.errorbar(it, y, yerr=y_err, fmt='.g', label='iteration value')
        plt.xticks(np.arange(it[0], it[-1]+0.2))
        it[0] = it[0]-0.3; it[-1]=it[-1]+0.3
        plt.plot(it,z, '-r', label='gen value')
        plt.fill_between(it, n_min, n_max, facecolor='grey', alpha=0.5, label='naive values')
        plt.xlim(it[0], it[-1])
        plt.ylim(min(z+y)-1.5*max(y_err)-1.5*n_err, max(z+y)+1.5*max(y_err)+1.5*n_err)
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f"{key}")
        plt.legend()
        plt.savefig(f'{output}/{key}.pdf')
