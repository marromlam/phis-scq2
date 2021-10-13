DESCRIPTION = """
  hey there
"""

from analysis.utils.plot import mode_tex
from ipanema import Parameters
import argparse
import os
import numpy as np

from scipy.stats import chi2
from scipy import stats, special
import hjson


def blend_parameters(*p, p0=False, params=True, pvalue=False, chisqr=False, verbose=False):
  # Create arrays from ipanema.Parameters object
  vals = [np.array([i.value for i in r.values()]) for r in p]
  uncs = [np.array([i.stdev for i in r.values()]) for r in p]
  corr = [np.array(r.corr()) for r in p]
  if verbose:
    print("Values of different measurements:")
    for i,meas in enumerate(vals):
      print(f"    meas{i}: {meas}")
    print("Uncertainties of different measurements:")
    for i,meas in enumerate(uncs):
      print(f"    meas{i}: {meas}")

  # Get covariances and inverses
  covs = [uncs[i][:,np.newaxis]*corr[i]*uncs[i] for i in range(len(corr))]
  if verbose:
    print("Cov of different measurements:")
    for i,meas in enumerate(covs):
      print(f"    meas{i}: ", end='')
      for row in meas:
        for col in row:
          print(f"{col:+.8f}", end=' ')
        print('\n       ',end='')
      print('\n', end='')
  icovs = [np.linalg.inv(cov) for cov in covs]
  icovC = np.linalg.inv( sum(covs) )
  covC = np.linalg.inv( sum(icovs) )

  # Calculate mean value and uncertainties
  if p0:
    val = np.array([i.value for i in p0.values()])
    unc = np.array([i.stdev for i in p0.values()])
  else:
    val = covC.dot(sum( [icovs[i].dot(vals[i].T) for i in range(len(vals))] ))
    unc = np.sqrt(np.diagonal(covC))

  # Build correlation matrix
  corr = np.zeros((len(val),len(val)))
  for i in range(1,covC.shape[0]):
    for j in range(1,covC.shape[1]):
      corr[i,j] = covC[i][j]/np.sqrt(covC[i][i]*covC[j][j])

  # Build ipanema.Parameters object
  pars = Parameters()
  for i,n in enumerate(p[0].keys()):
    #print(f"{i}-{n}")
    correl = {f'{m}':corr[i][j] for j,m in enumerate(p[0].keys()) if i>0 and j>0}
    pars.add({'name': f'{n}', 'value': val[i], 'stdev': unc[i],
              'free': False, 'latex': p[0][n].latex, 'correl': correl})
  # Check p-value
  diffs = [(vals[i]-val) for i in range(len(vals))]
  chi2_value = sum([(diff).dot(icovC.dot(diff)) for diff in diffs]);
  dof = len(val)
  prob = chi2.sf(chi2_value,dof)

  # Build result list
  if params:
      if pvalue:
        return pars, prob
      if chisqr:
        return pars, prob, chi2_value
      return pars
  else:
      if pvalue:
        return prob
      if chisqr:
        return prob, chi2_value



# "output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB1_run2_run2_simul6.json"
#
#
pp = [
"output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul6.json",
#"output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul3.json",
"output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB1_run2_run2_simul6.json",
"output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB2_run2_run2_simul6.json",
"output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB3_run2_run2_simul6.json",
"output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB4_run2_run2_simul6.json"
]

lp = ['pPlon', 'lPlon', 'DGsd', "DGs", "DM", "fPper", "fPlon", "dPpar", "dPper"]

dlls = np.random.rand(50,50)*0

#titles = [(os.path.basename(p[:-5])).split('_')[:] for p in pp]




def dump_joint_physics(pars, titles, dlls=False):


  pval = blend_parameters(*pars[1:], p0=pars[0], params=False, pvalue=True)
  print(titles)
  texcode = ''
  if len(titles) > 2:
    texcode += "\\resizebox{\\textwidth}{!}{"
  texcode += f"\\begin{{tabular}}{{l{'r'*(len(titles))}}}\n"
  texcode += "\\toprule\n"
  # Header
  for i, w in enumerate(['tuple', 'fit', 'AA', 'CSP', 'FT', 'TA', 'TR', 'trigger']):
    texcode += "\n"+" & ".join([f"  {'':>60}  "]+[f"""  {f'{w}: {f"{y[0]}{y[i+1][6:]}" if "yearly" in y[i+1] else y[i+1]}':>20}  """ for y in titles])
    texcode += " \\\\"
  texcode += "\n\\midrule\n"
  # Parameters
  table = []
  for p in pars[0]:
    line = [f'$ {pars[0][p].latex:>60} $']
    for i, y in enumerate(titles):
      par = f"{pars[i][p].uvalue:.2uL}"
      line.append(f"$ {par:>20} $")
    table.append(' & '.join(line))
  texcode += ' \\\\ \n'.join(table)
  texcode += ' \\\\\n\\midrule\n\nAgreement [pull (pvalue)]\\\\\n'
  # Agreement
  try:
    for i, t in enumerate(titles):
      col0 = [f"{' '.join(t[1:]):<60}   "]
      if 'yearly' in ''.join(col0):
          print('yearly')
          col0 = [col0[0].replace('yearly', f"{t[0]:>6}")]
      print(i, len(titles))
      cols = [f"${f'{dlls[str(i)][str(n)][0]:.2f} ({dlls[str(i)][str(n)][1]:.2f})':>22}$" if n<i else f"${f'   ':>22}$" for n in range(len(titles))]
      #texcode += "\n"+" & ".join([])
      texcode += f"\n {' & '.join(col0+cols)} \\\\"
    texcode += '\n\\midrule\n'
  except:
    print("Most probably you have infinites in the agreement between measurements")
  texcode += rf"\multicolumn{{{len(titles)+1}}}{{c}}{{the averaged overall p-value is {pval:.4f}}}"
  texcode += '\\\\\n\\bottomrule\n'
  if len(titles) > 2:
    texcode += "\\end{tabular}}\n"
  else:
    texcode += "\\end{tabular}\n"
  return texcode











if __name__ == '__main__':
  # parse cmdline arguments
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('--params', help='Biased acceptance')
  parser.add_argument('--output', help='Path to the final table')
  parser.add_argument('--mode', help='Mode')
  parser.add_argument('--year', help='Year')
  parser.add_argument('--version', help='Tuple version')
  parser.add_argument('--timeacc', help='Time Acceptance Flag')
  parser.add_argument('--angacc', help='Angular Acceptance Flag')
  args = vars(parser.parse_args())

  # split inputs
  params = args['params'].split(',')
  years = args['year'].split(',')
  v = args['version']
  m = args['mode']
  acc = f"{args['angacc']}--{args['timeacc']}"

  # load parameters
  years_ = [0]+years
  print(years, params)
  print(len(years), len(params))
  titles = [[years[i]]+(os.path.basename(p[:-5])).split('_')[:] for i,p in enumerate(params)]
  # params = [
  # "output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul6.json",
  # #"output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul3.json",
  # "output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB1_run2_run2_simul6.json",
  # "output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB2_run2_run2_simul6.json",
  # "output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB3_run2_run2_simul6.json",
  # "output/params/physics_params/run2/Bs2JpsiPhi/v0r5@cutpTB4_run2_run2_simul6.json"
  # ]
  params = [Parameters.build(Parameters.load(p),lp) for p in params]
  print(hjson.load(open(args['output'].replace('.tex','.json'))))
  try:
    dlls = hjson.load(open(args['output'].replace('.tex','.json')))
  except:
    print('no dlls')
    dlls = False

  # tabule
  print(titles)
  table = dump_joint_physics(params, titles, dlls=dlls)
  print(table)

  print(  f"{args['output']}".replace('/packandgo/','/') )
  with open(args['output'].replace('/packandgo',''), 'w') as fp:
    fp.write(table)
