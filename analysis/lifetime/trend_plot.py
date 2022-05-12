DESCRIPTION = """
    plot lifetime trend
"""

__all__ = []

# Modules {{{

import argparse
import ipanema
import complot
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as unc
from scipy import stats

# }}}


# PDG lifetimes {{{

tau = {}
tau['Bd'] = unc.ufloat(1.520, 0.004)  # PDG
tau['Bu'] = unc.ufloat(1.076, 0.004)  # PDG
tau['Bu'] = tau['Bu'] * tau['Bd']
tau['Bs'] = unc.ufloat(1.515, 0.004)  # PDG
print(tau)
# }}}


# Run {{{

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument("--biased-params", help="Lifetime parameters")
  p.add_argument("--unbiased-params", help="Lifetime parameters")
  p.add_argument("--combined-params", help="Lifetime parameters")
  p.add_argument("--figure", help="Lifetime trend figure")
  p.add_argument("--year", help="Years to plot")
  p.add_argument("--mode", help="Yeprintars to plot")
  print(p.parse_args())
  args = vars(p.parse_args())
  print(args)

  MODE = args['mode']
  print(MODE)
  if 'Bs' in MODE:
    q = 's'
  elif 'Bd' in MODE:
    q = 'd'
  elif 'Bu' in MODE:
    q = 'u'
  else:
    print("Dont know that quark. Halt")
    exit()

  ipars = {}
  ipars['b'] = [ipanema.Parameters.load(p) for p in args['biased_params'].split(",")]
  ipars['u'] = [ipanema.Parameters.load(p) for p in args['unbiased_params'].split(",")]
  ipars['c'] = [ipanema.Parameters.load(p) for p in args['combined_params'].split(",")]
  years = [int(y) for y in args['year'].split(",")]
  print(years)
  figure = args['figure']

  fig, ax = complot.axes_plot()

  # generate some linspace to create plots
  x = np.linspace(2014,2019,10)

  # world average 1sigma band
  wa_x = [x[0], x[-1]]
  wa_yu = 2*[tau[f'B{q}'].n + tau[f'B{q}'].s]
  wa_yl = 2*[tau[f'B{q}'].n - tau[f'B{q}'].s]
  ax.fill_between(wa_x, wa_yl, wa_yu, color='C0', alpha=0.25, label='PDG')

  titles = []; pipas = []
  for k, trig in enumerate(['combined', 'biased', 'unbiased']):
    # prepare data
    data = np.zeros((len(years), 3))
    for i, y in enumerate(years):
      data[i,0] = np.float64(y)
      tau = 1/ipars[trig[0]][i]['gamma'].uvalue
      data[i,1] = np.float64(tau.n)
      data[i,2] = np.float64(tau.s)
    print("data = ", data)

    # build dict to fit parameters
    fpars = ipanema.Parameters()
    fpars.add(dict(name='a0', value=1, latex='a_0'))
    fpars.add(dict(name='a1', value=0, free=False, latex='a_1'))

    # create fcn to do a linear fit
    def fcn(pars, x, y=None, w=None):
      p = pars.valuesdict()
      ans = p['a0'] + p['a1']*x
      if y is not None:
        ans = (ans-y)/w
        return ans*ans
      return ans

    ax.errorbar(data[:,0], data[:,1], yerr=data[:,2], fmt='.', color=f'C{k+1}',
                label=rf"$\tau_{{{trig[0]}}}$")

    if trig in ('biased', 'unbiased'):
      chi2 = []
      for slope in [False, True]:
        if slope:
          fpars = ipanema.Parameters.clone(res.params)
          fpars.unlock('a1')
          fpars['a1'].set(value=1)

        # fit
        res = ipanema.optimize(fcn, fpars, method='minuit', tol=0.01,
                               fcn_kwgs=dict(x=data[:,0], y=data[:,1],
                                             w=data[:,2]))
        #print(res)
        chi2.append(res.chi2)

        # plot
        y = fcn(res.params, x)
        expr  = rf"{trig} & $({res.params['a0'].uvalue:+.2uL})"
        expr += rf" + ({res.params['a1'].uvalue:+.2uL}) y$ & "
        expr += rf"${res.chi2:.4f}$ & "
        expr += rf"${stats.chi2.sf(res.chi2, res.nvary):.4f}$"
        ax.plot(x, y, '-', color=f'C{k+1}')
        pipas.append(expr)
      pipas[-1] += f"& ${np.sqrt(chi2[0]-chi2[1]):.4f}$"
  # ax.set_title(" -- ".join(titles))
  shit = r"\begin{tabular}{r|l|c|c|c}"
  shit += r"trigger. & fit & $\chi^2$ & p-value & significance \\ "
  shit += r"\\ ".join(pipas)
  shit += r" \end{tabular}"
  print(shit)
  ax.set_title(shit, fontsize='x-small')

  # label and save
  ax.set_xlabel("year")
  ax.set_ylabel(rf"$\tau(B_{q})$", fontsize='x-small')
  # ax.set_ylim(1.45, 1.55)
  ax.set_xlim(2014, 2019)
  ax.legend(fontsize='x-small')
  fig.savefig(figure)

# }}}


# vim:foldmethod=marker
