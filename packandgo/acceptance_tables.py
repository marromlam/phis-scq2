DESCRIPTION = """
  hey there
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['dump_joint_acceptance']

from analysis.utils.plot import mode_tex
from ipanema import Parameters
import argparse

def dump_joint_acceptance(pu, pb, years, caption=""):
    texcode  = f"\\begin{{table}}[H]\n"
    texcode += f"\\centering\n"
    texcode += f"\\caption{{{caption}}}\n"
    texcode += f"\\begin{{tabular}}{{{'c'*(len(years)+1)}}}\n"
    texcode += "\\toprule\n"
    texcode += " & ".join([f"  {'':>5}  "]+[f"$ {y:>20} $" for y in years])
    texcode += " \\\\ \n\\midrule\n"
    table = []
    for p in pb[0].find('(a|b|c|w).*'):
      line = [f'$ {pb[0][p].latex:>5} $']
      for i, y in enumerate(years):
        par = f"{pb[i][p].uvalue:.2uL}"
        line.append(f"$ {par:>20} $")
      table.append(' & '.join(line))
    texcode += ' \\\\ \n'.join(table)

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
    texcode += f"\\end{{table}}\n"
    return texcode






if __name__ == '__main__':
  # parse cmdline arguments
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('--biased', help='Biased acceptance')
  parser.add_argument('--unbiased', help='Unbiased acceptance')
  parser.add_argument('--output', help='Path to the final table')
  parser.add_argument('--mode', help='Mode')
  parser.add_argument('--year', help='Year')
  parser.add_argument('--version', help='Tuple version')
  parser.add_argument('--timeacc', help='Time Acceptance Flag')
  parser.add_argument('--angacc', default=False, help='Angular Acceptance Flag')
  args = vars(parser.parse_args())

  # split inputs
  bpp = args['biased'].split(',')
  upp = args['unbiased'].split(',')
  years = args['year'].split(',')
  v = args['version']
  m = args['mode']
  acc = args['timeacc']
  if args['angacc']:
    acc = f"{args['angacc']}--{acc}"

  # load parameters
  pu = [Parameters.load(ip) for ip in upp]
  pb = [Parameters.load(ip) for ip in bpp]

  # create a caption
  if args['angacc']:
    caption = f"Angular acceptance [{acc}] for ${'+'.join(years)}$."
  else:
    caption = f"Time acceptance [{acc}] fitted against ${mode_tex(m)}$ for ${'+'.join(years)}$."

  # tabule
  table = dump_joint_acceptance(pu, pb, years, caption=caption)
  print(table)

  with open(args['output'], 'w') as fp:
    fp.write(table)
