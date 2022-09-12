DESCRIPTION = """
  hey there
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['dump_joint_lifetimeBd', 'dump_joint_lifetimeBu']


from ipanema import Parameters
import uncertainties as unc
import numpy as np
import argparse


# PDG lifetimes {{{

tau = {}
tau['Bd'] = unc.ufloat(1.520, 0.004)  # PDG
tau['Bu'] = unc.ufloat(1.076, 0.004)  # PDG
tau['Bu'] = tau['Bu'] * tau['Bd']
tau['Bs'] = unc.ufloat(1.515, 0.004)  # PDG

# }}}


def dump_joint_lifetimeBd(pars, caption=""):
  # prepare table
  table = []
  table.append(r"\begin{tabular}{c|c|cccc}")
  table.append(r"\toprule")
  col1 = 'Event'
  col2 = r'$m(K^*)$'
  # col3 = r'$\sigma_t$'
  col4 = r'$\alpha$'
  # table.append(f"{'Year':>5} & {col1:<36} & {col2:>36} & {col3:>36} & {col4:>36}")
  table.append(f"{'Year':>5} & {col1:<36} & {col2:>36} & {col4:>36}")
  table.append(r"\\ \midrule")
  for year in pars:
    for flag in ['simul3', 'simul3Noncorr']:
      line = []
      _flag = 'C' if flag=='simul3' else 'U'
      line.append(f"{year} {_flag}")
      #for test in ['', 'mKstar', 'deltat', 'alpha']:
      for test in ['', 'mKstar', 'alpha']:
        print(pars[year])
        nsigma = abs(tau['Bd'].n - pars[year][flag][test].n)/pars[year][flag][test].s
        svalue = f"{pars[year][flag][test]:.2uL}"
        line.append(f"${svalue:>20}\,\,({nsigma:.1f}\sigma) $")
      table.append( "&".join(line) + r"\\" )
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}")

  return "\n".join(table)



def dump_joint_lifetimeBu(pars, caption=""):
  # prepare table
  table = []
  table.append(r"\begin{tabular}{c|cc}")
  table.append(r"\toprule")
  col1 = 'With corrections'
  col2 = 'Without corrections'
  table.append(f"{'Year':<4} & {col1:>36} & {col2:>36} \\\\")
  table.append(r"\midrule")
  for year in pars:
    line = []
    line.append(f"{year} ")
    for flag in pars[year]:
      nsigma = abs(tau['Bu'].n-pars[year][flag].n)/pars[year][flag].s
      svalue = f"{pars[year][flag]:.2uL}"
      line.append(f"${svalue:>20}\,\,({nsigma:.1f}\sigma) $")
    table.append( "&".join(line) + r"\\" )
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}")

  return "\n".join(table)

# Single-like lifetime table {{{

def dump_joint_lifetime_Bx(pars, mode):
  """
  Creates lifetime tables for different modes of Bx.

  Parameters
  ----------
  pars : dict,
    Dictionary with parameters.
  mode : string
    Bx mode to compara WA lifetime with.

  Returns
  -------
  string
    LaTeX-style table.
  """
  table = []
  table.append(r"\begin{tabular}{c|cc}")
  table.append(r"\toprule")
  col1 = 'With corrections'
  col2 = 'Without corrections'
  table.append(f"{'Year':<4} & {col1:>36} & {col2:>36} \\\\")
  table.append(r"\midrule")
  for year in pars:
    line = []
    line.append(f"{year} ")
    for flag in pars[year]:
      nsigma = tau[mode].n-pars[year][flag].n
      print(np.sqrt(pars[year][flag].s**2 + tau[mode].s**2) )
      nsigma /= np.sqrt(pars[year][flag].s**2 + tau[mode].s**2)
      svalue = f"{pars[year][flag]:.2uL}"
      line.append(f"${svalue:>20}\,\,({nsigma:+.2f}\sigma) $")
    table.append( "&".join(line) + r"\\" )
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}")
  return "\n".join(table)

# }}}


# CMDline interface {{{

if __name__ == '__main__':
  # parse cmdline arguments
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--corr', help='Biased acceptance')
  p.add_argument('--noncorr', help='Biased acceptance')
  p.add_argument('--output', help='Path to the final table')
  p.add_argument('--mode', help='Mode')
  p.add_argument('--year', help='Year')
  p.add_argument('--version', help='Tuple version')
  p.add_argument('--trigger', help='Tuple version')
  p.add_argument('--timeacc', help='Time Acceptance Flag')
  args = vars(p.parse_args())

  # loop over the parameters and load them
  FLAG = 'simul3'
  if 'Bs' in args['mode']:
    mode = 'Bs'
  elif 'Bd' in args['mode']:
    mode = 'Bd'
  elif 'Bu' in args['mode']:
    mode = 'Bu'
  else:
    print("Mode does not exist. Halted.")
    exit()

  print(f"PDG lifetime for {mode} is : {tau[mode]}")

  V = args['version']
  trigger = args['trigger']

  # tabule
  if 'single' in args['timeacc']:
    # when running single time acceptance and we try to get the lifetime for
    # some Bx RD sample.
    icorr = args['corr'].split(',')
    inoncorr = args['noncorr'].split(',')
    pars = {}
    for i, year in enumerate(args['year'].split(',')):
      pars[year] = {}
      flag = f'{args["timeacc"]}'
      pars[year][flag] = 1/Parameters.load(icorr[i])['gamma'].uvalue
      flag = f'{args["timeacc"]}Noncorr'
      pars[year][flag] = 1/Parameters.load(inoncorr[i])['gamma'].uvalue
    table = dump_joint_lifetime_Bx(pars, mode)
  elif args['mode'] == 'Bd2JpsiKstar':
    # this branch is used when we want to tabule all diffent tests with Bd
    # splitting in 
    pars = {}
    for year in args['year'].split(','):
      pars[year] = {}
      for flag in [f'{args["timeacc"]}', f'{args["timeacc"]}Noncorr']:
        pars[year][flag] = {}
        #for test in ['', 'mKstar', 'deltat', 'alpha']:
        for test in ['', 'mKstar', 'alpha']:
          try:
            pars[year][flag][test] = 1/Parameters.load(f'output/params/lifetime/{year}/Bd2JpsiKstar/{V}_{flag}{test}_{trigger}.json')['gamma'].uvalue
          except:
            pars[year][flag][test] = 1/Parameters.load(f'output/params/lifetime/{year}/Bd2JpsiKstar/{V}_{flag}{test}BdasBs_{trigger}.json')['gamma'].uvalue
    table = dump_joint_lifetimeBd(pars)
  else:
    icorr = args['corr'].split(',')
    inoncorr = args['noncorr'].split(',')
    pars = {}
    for i, year in enumerate(args['year'].split(',')):
      pars[year] = {}
      flag = f'{args["timeacc"]}'
      # pars[year][flag] = 1/(tau['Bd']*Parameters.load(icorr[i])['gamma'].uvalue)
      print(Parameters.load(icorr[i])['gamma'].uvalue)
      pars[year][flag] = 1/Parameters.load(icorr[i])['gamma'].uvalue #- 1/tau['Bd']
      flag = f'{args["timeacc"]}Noncorr'
      # pars[year][flag] = 1/(tau['Bd']*Parameters.load(inoncorr[i])['gamma'].uvalue)
      pars[year][flag] = 1/Parameters.load(inoncorr[i])['gamma'].uvalue #- 1/tau['Bd']
    table = dump_joint_lifetimeBu(pars)
  print(table)

  with open(args['output'], 'w') as fp:
    fp.write(table)

# }}}


# vim:foldmethod=marker
