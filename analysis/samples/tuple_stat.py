__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

import config
from utils.strings import printsec
from ipanema import Sample
import argparse
import numpy as np

# }}}


# Command line runner {{{

if __name__ == "__main__":
  DESCRIPTION = """
    Create tables with the number of events for each sample for a given set of
    years and splitting by trigger category.
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample', help='Sample or samples to compute stats from')
  p.add_argument('--year', help='Years')
  p.add_argument('--kind', help='Years')
  p.add_argument('--output', help='Path to write the table')
  args = vars(p.parse_args())

  printsec("Sample statistics")

  # Prepare data and totals
  data = {}
  totals = [0, 0, 0]
  kind = args['kind']
  for i, y in enumerate(config.years[args['year']]):
    this = Sample.from_root(args['sample'].split(',')[i])
    if kind == 'stat':
      data[y] = [
          this.df.query('hlt1b==1').shape[0],
          this.df.query('hlt1b==0').shape[0],
          this.df.shape[0],
      ]
    elif kind == 'effstat':
      data[y] = [
          int(np.round(this.df.query('hlt1b==1 & sWeight>0')['sWeight'].sum())),
          int(np.round(this.df.query('hlt1b==0 & sWeight>0')['sWeight'].sum())),
          int(np.round(this.df.query('sWeight>0')['sWeight'].sum())),
      ]
    totals[0] += data[y][0]
    totals[1] += data[y][1]
    totals[2] += data[y][2]

  # Create table and write it
  line = "{:>6} & {:>8} & {:>8} & {:>8} \\\\"
  table = []
  table.append(r'\begin{tabular}{lccc}')
  table.append(r'\toprule')
  table.append(
      rf"{'year':>6} & {'biased':>8} & {'unbiased':>8} & {'total':>8} \\")
  table.append(r"\midrule")
  for y, d in data.items():
    table.append(line.format(*[y, *d]))
  table.append(r"\midrule")
  table.append(line.format(*['total', *totals]))
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}")
  with open(args['output'], 'w') as fp:
    fp.write('\n'.join(table))
  print('\n'.join(table))

# }}}


# vim: fdm=marker
