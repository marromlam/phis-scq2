__all__ = []


import ipanema
from ipanema import Sample, Parameter, Parameters
import numpy as np
import argparse
from scipy import stats
import hjson
import uncertainties as unc
from utils.helpers import version_guesser

def pvalue(pars, genpars, names=None):
    if names != None:
      pars = Parameters.build(pars, names)
    names = [key for key in pars.keys() if pars[key].free==True]
    pars = Parameters.build(pars,names)
    genpars = Parameters.build(genpars, names)
    cov = np.matrix(pars.cov())
    dof = len(cov)
    diff = np.array(list(genpars.valuesdict().values()))-np.array(list(pars.valuesdict().values()))
    chi2 = np.dot(np.dot(diff, cov.getI()), diff.T)
    return stats.chi2.sf(chi2, dof)[0][0]


if __name__ == '__main__':
  p = argparse.ArgumentParser(description='overall p-value for Angular acceptance after convergence')
  p.add_argument('--target', help='Params from angular acceptance')
  p.add_argument('--original', help='Generated parameters')
  p.add_argument('--year', help='Year of the sample')
  p.add_argument('--mode', help='Mode fitted')
  p.add_argument('--version', help='Version and cuts fitted')
  p.add_argument('--angacc', help='Version of angular acceptance')
  p.add_argument('--output', help='Latex table with overall pvalue')
  args = vars(p.parse_args())
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  MODE = args['mode']
  output = args['output']
  pars = Parameters.load(args['target'])
  gen_pars = Parameters.load(args['original'])

  names = [key for key in pars.keys() if pars[key].free==True]
  if 'Bd2JpsiKstar' in MODE:
    names =  ['fPlon', 'fPper', 'dPpar', 'dPper']
  print(names)
  pars = Parameters.build(pars, names)
  latex = [pars[key].latex for key in pars.keys()]
  value = np.array([pars[key].value for key in pars.keys()])
  stdev = np.array([pars[key].stdev for key in pars.keys()])
  genvalue = np.array([gen_pars[key].value for key in pars.keys()])
  genstdev = np.array([gen_pars[key].stdev for key in pars.keys()])
  Comb = False
  if None in genstdev:
    comb = stdev
  else:
    Comb = True
    comb = np.sqrt(genstdev**2+stdev**2)
  PULL = (value-genvalue)/comb
  if 'Bd2JpsiKstar' in MODE:
    pvalue = pvalue(pars, gen_pars, names=names)
  else:
    pvalue = pvalue(pars, gen_pars)
  print(pvalue)

  #Latex table
  table = []
  table.append(r"\begin{tabular}{c|cc}")
  table.append(r"\toprule")
  col1 = 'Values (stats only)'
  col2 = 'Pulls (stats only)'
  if Comb:
    col3 = col2
    col2 = 'GenValues'
    table.append(f"{'Parameters':<40} & {col1:>25} & {col2:>25} & {col3:>17} \\\\")
  else:
    table.append(f"{'Parameters':<40} & {col1:>25} &  {col2:>17} \\\\")
  table.append(r"\midrule")
  for i in range(len(names)):
    line = []
    line.append(f"${latex[i]:<37}$  ")
    param = f"{unc.ufloat(value[i], stdev[i]):.2uL}"
    nsigma = PULL[i]
    line.append(f" ${param:>26}$ ")
    if Comb:
      genparam = f"{unc.ufloat(genvalue[i], genstdev[i]):.2uL}"
      line.append(f" ${genparam:>26}$ ")
    line.append(f"    $ {nsigma:>+.2f}\sigma $  ")
    table.append("&".join(line)+r"\\")
  table.append(r"\midrule")
  table.append(rf"\multicolumn{{3}}{{c}}{{overall p-value: {pvalue:>+.5f}}}")
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}") #Saving table in a .json
  with open(args['output'], "w") as tex_file:
    tex_file.write("\n".join(table))
  tex_file.close()
