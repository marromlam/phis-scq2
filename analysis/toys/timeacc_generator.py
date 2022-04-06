import ipanema
import numpy as np

__all__ = []

def randomize_timeacc(params, sigma=1):
  _p = ipanema.Parameters.clone(params)
  for k, v in _p.items():
    if 'c' in k and v.free:
      # gaussian
      #rng_val = np.random.normal(loc=v.value, scale=sigma*pv.stdev)
      # uniform
      rng_val = v.value + np.random.rand() * sigma * v.stdev
      v.set(value=rng_val)
  return _p


"""
import matplotlib.pyplot as plt
p0 = ipanema.Parameters.load(
     "output/params/time_acceptance/2015/Bd2JpsiKstar/v0r5_simul3_biased.json")

pars = [randomize_timeacc(p0) for i in range(0,10000)]

for i in range(0,6):
  fig, ax = ipanema.plotting.axes_plot()
  ax.hist([p[f'c{i}b'].value for p in pars]);
"""
