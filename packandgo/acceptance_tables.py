__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['dump_joint_acceptance']


# Modules {{{

from utils.plot import mode_tex
from ipanema import Parameters
import argparse

# }}}


def dump_joint_acceptance(pu, pb, years):
  texcode = f"\\begin{{tabular}}{{{'c'*(len(years)+1)}}}\n"
  texcode += "\\toprule\n"
  texcode += " & ".join([f"  {'':>5}  "]+[f"$ {y:>20} $" for y in years])
  texcode += " \\\\ \n\\midrule\n"
  # biased table
  table = []
  for p in pb[0].find('(a|b|c|w).*'):
    print(pb[0][p].latex)
    line = [f'$ {pb[0][p].latex:>5} $']
    for i, y in enumerate(years):
      par = f"{pb[i][p].uvalue:.2uL}"
      line.append(f"$ {par:>20} $")
    table.append(' & '.join(line))
  texcode += ' \\\\ \n'.join(table)
  # unbiased table
  table = []
  for p in pu[0].find('(a|b|c|w).*'):
    line = [f'$ {pu[0][p].latex:>5} $']
    for i, y in enumerate(years):
      par = f"{pu[i][p].uvalue:.2uL}"
      line.append(f"$ {par:>20} $")
    table.append(' & '.join(line))
  texcode += ' \\\\\n\\midrule\n'+' \\\\ \n'.join(table)
  texcode += ' \\\\\n\\bottomrule\n'
  texcode += f"\\end{{tabular}}\n"
  return texcode


# CMDline run {{{

if __name__ == '__main__':
  DESCRIPTION = """
  hey there
  """

  # parse cmdline arguments
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--biased', help='Biased acceptance')
  p.add_argument('--unbiased', help='Unbiased acceptance')
  p.add_argument('--output', help='Path to the final table')
  p.add_argument('--mode', help='Mode')
  p.add_argument('--year', help='Year')
  p.add_argument('--version', help='Tuple version')
  p.add_argument('--timeacc', help='Time Acceptance Flag')
  p.add_argument('--angacc', default=False, help='Angular Acceptance Flag')
  args = vars(p.parse_args())

  # split inputs
  bpp = args['biased'].split(',')
  upp = args['unbiased'].split(',')
  years = args['year'].split(',')
  v = args['version']
  m = args['mode']
  acc = args['timeacc']

  # load parameters
  pu = [Parameters.load(ip) for ip in upp]
  pb = [Parameters.load(ip) for ip in bpp]

  # tabule
  table = dump_joint_acceptance(pu, pb, years)
  print(table)

  with open(args['output'], 'w') as fp:
    fp.write(table)

# }}}


# vim: fdm=marker
