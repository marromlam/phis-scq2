__all__ = []
import os
import argparse
import numpy as np
import uproot3 as uproot
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt

from ipanema import Sample, optimize, ristra, Parameters

from selection.mass_fit.bd_mc import mass_fitter
from utils.strings import printsec, printsubsec
from trash_can.knot_generator import create_time_bins

tau = unc.ufloat(1.520, 0.004)

if __name__ == '__main__':
  printsec("Lifetime crappy estimation")
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


  time_bins = create_time_bins(number_of_time_bins, 0.4, 12)
  print(time_bins)
  time_bins[-2] = time_bins[-1]
  time_bins = np.array(time_bins[:-1])

  # Compute the efficiency
  printsubsec("Compute the efficiency")

  # histogram MC in bins of time
  mc = Sample.from_root(args['mc_sample'])
  rd = Sample.from_root(args['rd_sample'])
  # rd_hist = np.histogram(rd.df['time'], bins=time_bins, density=True)[0]
  _mc_hist = np.histogram(mc.df['time'], bins=time_bins, density=True)[0]
  mc_hist = np.histogram(mc.df['time'], bins=time_bins, density=False)[0]
  mc_sum = mc_hist/_mc_hist
  print(mc_hist/mc_sum)
  print("MC:", mc_hist/mc_sum, _mc_hist)

  # compute the prediction for each bin
  pred = []
  for ll, ul in zip(time_bins[:-1], time_bins[1:]):
    integral_f = tau.n * np.exp(-ul/tau.n) 
    integral_s = tau.n * np.exp(-ll/tau.n)
    pred.append(integral_s - integral_f) 
  print("TOY:", pred)

  _eff = unp.uarray(mc_hist/pred, 0*np.sqrt(mc_hist)/pred)/mc_sum
  # eff = unp.uarray(mc_hist, 0*np.sqrt(mc_hist)/pred)
  print("eff:", _eff)


  printsubsec("Mass fit loop")
  nevts = []; tbins = []; eff = []; i = 0
  for ll, ul in zip(time_bins[:-1], time_bins[1:]):
    print(f"Bd mass fit in {ll}-{ul} ps time window")
    cdf = rd.df.query(f"time>{ll} & time<{ul}")
    # do mass fit here
    try:
      cpars = mass_fitter(cdf, figs=output_figures, trigger=trigger, verbose=False, label=f'{ll}-{ul}')
      nevts.append(cpars['nsig'].uvalue) 
      tbins.append( np.median(rd.df.query(f'time>{ll} & time>{ul}')['time']) )
      eff.append(_eff[i])
      # print(nevts[-1])
      print(cpars)
    except:
      print('  - fit failed')
    i += 1
  print(f"Succesfully fitted {len(nevts)} time bins")
  nevts = nevts[2:]
  tbins = tbins[2:]
  eff = eff[2:]

  nevts = nevts[:-2]
  tbins = tbins[:-2]
  eff = eff[:-2]

  if number_of_time_bins>5:
    printsubsec("Lifetime fit")
    _x = np.array(tbins)
    _y = unp.uarray([v.n for v in nevts], [v.s for v in nevts])
    _y *= unp.uarray([v.n for v in eff], [v.s for v in eff])
    _y, _uy = unp.nominal_values(_y), unp.std_devs(_y)
    _wy = 1/_uy**2

    print(_x, _y, _wy)
    # plt.plot(_xx, _y, 'o')
    plt.errorbar(_x, _y, yerr=_uy, fmt='.', color='k') 

    def fcn(pars, x, y=None, uy=False):
      _pars = pars.valuesdict()
      _model = _pars['N'] * np.exp( -x/_pars['tau'] )
      if y is not None:
          return ((y - _model) / uy)**2
      return _model

    # Create parameters and fit 
    pars = Parameters()
    pars.add(dict(name="tau", value=1.0, min=1.35, max=1.85))
    pars.add(dict(name="N", value=150))
    result = optimize(fcn, params=pars, method='minuit', fcn_kwgs={"x":_x, "y":_y, "uy":_uy}, verbose=True)
    print(result)
    
    # Dump Bx decay-width to json
    __gamma = 1/result.params['tau'].uvalue
    gamma = Parameters()
    gamma.add(dict(name="gamma", value=__gamma.n, stdev=__gamma.s ))
    gamma.dump(args['output_params'])

    # plot
    _proxy_x = np.linspace(0, 15, 100)
    _proxy_y = fcn(result.params, _proxy_x)
    plt.plot(_proxy_x, _proxy_y)
    # plt.xlim(-0.5, 16)
    plt.yscale('log')
    plt.savefig(os.path.join(output_figures, 'lifetime_fit.pdf'))
    print(result)

