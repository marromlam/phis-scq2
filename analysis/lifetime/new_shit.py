import os
import argparse
import numpy as np
import uproot3 as uproot
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt

from ipanema import Sample, optimize, ristra, Parameters

from selection.mass_fit.bd_mc import mass_fitter
from utils.strings import printsubsec
from trash_can.knot_generator import create_time_bins

tau = unc.ufloat(1.520, 0.004)

if __name__ == '__main__':
  p = argparse.ArgumentParser(description="adfasdf")
  p.add_argument("--rd-sample")
  p.add_argument("--mc-sample")
  p.add_argument("--output-params")
  p.add_argument("--output-figures")
  p.add_argument("--version")
  p.add_argument("--mode")
  p.add_argument("--trigger")
  p.add_argument("--year")
  p.add_argument("--timeacc")
  args = vars(p.parse_args())

  trigger = args['trigger']

  output_figures = args['output_figures']
  os.makedirs(output_figures, exist_ok=True)

  number_of_time_bins = 50
  mass_branch = 'B_ConstJpsi_M_1'
  time_bins = np.linspace(0.3, 15, number_of_time_bins+1)[:-1]
  print(time_bins)
  time_bins = create_time_bins(number_of_time_bins, 0.4, 10)
  time_bins[-2] = time_bins[-1]
  time_bins = time_bins[:-1]

  # Compute the efficiency
  printsubsec("Compute the efficiency")

  # histogram MC in bins of time
  mc = Sample.from_root(args['mc_sample'])
  rd = Sample.from_root(args['rd_sample'])
  # rd_hist = np.histogram(rd.df['time'], bins=time_bins, density=True)[0]
  mc_hist = np.histogram(mc.df['time'], bins=time_bins, density=True)[0]
  # mc_hist = np.histogram(mc.df['time'], bins=time_bins, density=False)[0]
  # mc_sum = np.sum(mc_hist)
  print("MC:", mc_hist)

  # compute the prediction for each bin
  pred = []
  for ll, ul in zip(time_bins[:-1], time_bins[1:]):
    integral_f = tau.n * np.exp(-ul/tau.n) 
    integral_s = tau.n * np.exp(-ll/tau.n)
    pred.append(integral_s - integral_f) 
  print("TOY:", pred)

  # eff = unp.uarray(mc_hist/pred, 0*np.sqrt(mc_hist)/pred)/mc_sum
  eff = unp.uarray(mc_hist, 0*np.sqrt(mc_hist)/pred)
  print("eff:", eff)


  printsubsec("Mass fit loop")
  nevts = []
  for ll, ul in zip(time_bins[:-1], time_bins[1:]):
    print(f"Bd mass fit in {ll}-{ul} ps time window")
    cdf = rd.df.query(f"time>{ll} & time<{ul}")
    # do mass fit here
    cpars = mass_fitter(cdf, figs=output_figures, trigger=trigger, verbose=False, label=f'{ll}-{ul}')
    nevts.append(cpars['nsig'].uvalue) 
    print(nevts)
    print(cpars)


  if number_of_time_bins>5:
    printsubsec("Lifetime fit")
    _x = 0.5*(time_bins[1:]+time_bins[:-1])
    __y = unp.uarray([v.n for v in nevts], [v.s for v in nevts]) * eff
    print(__y)
    _y = unp.nominal_values(__y)
    _uy = unp.std_devs(__y)
    _wy = 1/_uy**2

    print(_x, _y, _wy)
    plt.plot(_x, _y, 'o')
    plt.errorbar(_x, _y, yerr=_uy, fmt='.', color='k') 
    plt.savefig("meh.pdf")
    print("please cook fitter here")

    def fcn(pars, x, y=None, uy=False):
        _pars = pars.valuesdict()
        _model = _pars['N'] * np.exp( -x/_pars['tau'] )
        if y is not None:
            return ((y - _model) / uy)**2
        return _model

    
    pars = Parameters()
    pars.add(dict(name="tau", value=1.0))
    pars.add(dict(name="N", value=150))
    result = optimize(fcn, params=pars, method='minuit', fcn_kwgs={"x":_x, "y":_y, "uy":_uy}, verbose=True)

    __gamma = 1/result.params['tau'].uvalue
    gamma = Parameters()
    gamma.add(dict(name="gamma", value=__gamma.n, stdev=__gamma.s))
    gamma.dump(args['output_params'])

    # plot
    _proxy_x = np.linspace(0, 15, 100)
    _proxy_y = fcn(result.params, _proxy_x)
    plt.plot(_proxy_x, _proxy_y)
    # plt.xlim(-0.5, 16)
    plt.yscale('log')
    plt.savefig(os.path.join(output_figures, 'lifetime_fit.pdf'))
    print(result)

