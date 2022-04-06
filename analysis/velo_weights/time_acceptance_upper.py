__all__ = []
from ipanema import (Parameters, ristra, optimize, Sample, initialize)
import argparse
import numpy as np

initialize('opencl', 1)

if __name__ == '__main__':
  p = argparse.ArgumentParser(description='Get efficiency in DOCAz bin.')
  p.add_argument('--sample', help='Bu2JpsiKplus RD sample')
  p.add_argument('--eff', help='Mass fit parameters')
  p.add_argument('--timeacc', help='Mass fit parameters VELO matching')
  args = vars(p.parse_args())
  
  # we only want to load a few branches
  branches = ['time', 'docaz_hplus']
  sample = Sample.from_root(args["sample"], branches=branches)
  eff = Parameters.load(args['eff'])
  print(eff)

  # upper decay-time, let's say time > 2ps
  tLL = 4; tUL = 14

  # get efficiency
  eff_model = lambda x, p: p[0] * (1 + p[1] * x**2)
  # define model to fit time acceptance
  acc_model = lambda x, p: 1 + p[0]*x + p[1]*x*x


  # prepare efficiency weights
  eff_eval = lambda x: eff_model(x, list(eff.valuesdict().values()))

  sample.df.eval("veloWeight = @eff_eval(docaz_hplus)", inplace=True)
  sample.chop(f"time>{tLL} & time<{tUL}")
  sample.allocate(time="time", weight="1/veloWeight", prob="0*time")
  print(sample.df)

  def fcn(pars, x):
    p = pars.valuesdict()
    # numerator
    prob = acc_model(x.time, [p['a'], p['b']])
    prob *= ristra.exp(-p['gamma']*x.time)
    # denominator
    prob /= np.exp( -1. * p['gamma'] * ( tUL + tLL ) ) * ( p['gamma'] )**( -3 ) * ( -1 * np.exp( p['gamma'] * tLL ) * ( p['gamma'] * ( p['a'] + ( p['gamma'] + p['a'] * p['gamma'] * tUL ) ) + p['b'] * ( 2 + p['gamma'] * tUL * ( 2 + p['gamma'] * tUL ) ) ) + np.exp( p['gamma'] * tUL ) * ( p['gamma'] * ( p['a'] + ( p['gamma'] + p['a'] * p['gamma'] * tLL ) ) +p['b'] * ( 2 + p['gamma'] * tLL * ( 2 + p['gamma'] * tLL ) ) ) )
    #prob /= ristra.sum(prob)
    return ristra.get( -2 * ristra.log( prob ) * x.weight )

  pars = Parameters()
  pars.add(dict(name='a', value=0, min=-3, max=3, free=True))
  pars.add(dict(name='b', value=0, min=-3, max=3, free=True))
  pars.add(dict(name='gamma', value=0.61050, free=False))

  res = optimize(fcn, pars, method='minuit', fcn_kwgs={"x":sample},
                 verbose=True)
  print(res)
  res.params.dump(args['timeacc'])


# vim: ts=2 sw=2 sts=2 et
