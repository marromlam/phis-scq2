from ipanema import Parameters
import uncertainties as unc
import numpy as np
tauBd = unc.ufloat(1.520,0.004)
tauBu = unc.ufloat(1.076,0.004)

if __name__ == '__main__':
  # loop over the parameters and load them
  FLAG = 'simul'
  pars = {}
  for year in [2015, 2016, 2017, 2018]:
    pars[str(year)] = {}
    for flag in ['simul', 'simulNoncorr']:
      pars[str(year)][flag] = {}
      #for test in ['', 'mKstar', 'deltat', 'alpha']:
      for test in ['', 'mKstar', 'deltat']:
        pars[str(year)][flag][test] = 1/Parameters.load(f'output/params/time_acceptance/{year}/Bd2JpsiKstar/v0r5_lifetimeBd{flag}{test}.json')['gamma'].uvalue

  #%% prepare table
  table = []
  table.append(r"\begin{table}[H]")
  table.append(r"\begin{tabular}{c|c|cccc}")
  table.append(r"\toprule")
  col1 = 'Event'
  col2 = '$m(K^*)$'
  col3 = '$\sigma_t$'
  col4 = '$\alpha$'
  table.append(f"{'Year':>5} & {col1:<36} & {col2:>36} & {col3:>36} & {col4:>36}")
  table.append(r"\\ \midrule")
  for year in pars:
    for flag in ['simul', 'simulNoncorr']:
      line = []
      _flag = 'C' if flag=='simul' else 'U'
      line.append(f"{year} {_flag}")
      #for test in ['', 'mKstar', 'deltat', 'alpha']:
      for test in ['', 'mKstar', 'deltat']:
        nsigma = abs(tauBd.n - pars[year][flag][test].n)/pars[year][flag][test].s
        svalue = f"{pars[year][flag][test]:.2uL}"
        line.append(f"${svalue:>20}\,\,({nsigma:.1f}\sigma) $")
      table.append( "&".join(line) + r"\\" )
  table.append(r"\bottomrule")
  table.append(r"\end{tabular}")
  table.append(r"\caption{Values of $\tau(B_d^0)$ obtained for the validation of the time acceptance}")
  table.append(r"\end{table}")

  with open(f'output/tables/time_acceptance/run2/Bd2JpsiKstar/v0r5_lifetimeBd{FLAG}.tex', 'w') as fp:
    fp.write("\n".join(table))
