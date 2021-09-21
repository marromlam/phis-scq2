import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import uncertainties as unc
from uncertainties import unumpy as unp
import numba

# averages
# we measure DGsd = -0.0073 \pm 0.0014
run2_DGsd = unc.ufloat(-0.0073, 0.0014)
run2_lifetime = 1/ (1/unc.ufloat(1.520,0.004) + run2_DGsd)
print("Run2 Lifetime", run2_lifetime)

# for each year we measure
yearly_DGsd = {
  2015: unc.ufloat(-0.0035, 0.0066),
  2016: unc.ufloat(-0.0048, 0.0025),
  2017: unc.ufloat(-0.0065, 0.0025),
  2018: unc.ufloat(-0.0106, 0.0023)
}
yearly_lifetime = {k: 1/ (1/unc.ufloat(1.520,0.004) + v)
                   for k,v in yearly_DGsd.items()}
print("Yearly Lifetime", yearly_lifetime)




# years
years = list(yearly_lifetime.keys())
x = np.linspace(years[0], years[-1], 1000);

# we measure this trend slope
slope = unc.ufloat(0.009103, 0.000028)
intercept = unc.ufloat(-16.823,0.057)


# Create lines
yr = slope*x + intercept
y_upp = unp.nominal_values(yr) + unp.std_devs(yr)
y_low = unp.nominal_values(yr) - unp.std_devs(yr)


# Create random set of numbers following gaussian
# data = np.random.normal(tau.n, tau.s, (nevts,4))
# matched = nevts * [0]

@numba.jit
def polyfit(x, y, deg):
    '''Fit polynomial of order `deg` against x, y data points.'''
    mat = np.zeros(shape = (x.shape[0], deg + 1))
    mat[:, 0] = np.ones_like(x)
    for n in range(1, deg + 1):
        mat[:, n] = x**n

    p = np.linalg.lstsq(mat, y)[0]
    return p


# @numba.jit
def foo(years, mu, sigma, slope, nevts=1e6, data=False):
  matched = 0
  if data:
    _data = int(nevts) * [0]
  for i in range(0, int(nevts)):
    # generate four random numbers
    _r = [np.random.normal(mu[i], sigma[i]) for i in range(0,len(years))]
    _c = np.polyfit(years, _r, 1)  # linear fit with slope
    if _c[0] > slope:
      matched += 1  # if the slope is greater, then it's a match
    if data:
      _data[i] = _r
  if data:
    return matched, _data
  return matched


caca = [foo(years, len(years) * [run2_lifetime.n], [v.s for v in yearly_lifetime.values()], slope.n, 1e5, False)/1e5 for i in range(0,100)]
        
# plot
nevts = 1e5
matched, data = foo(years,
                 len(years) * [run2_lifetime.n],
                 [v.s for v in yearly_lifetime.values()],
                 slope.n,
                 nevts, True)
plt.close()
plt.fill_between(x, y_upp, y_low, alpha=0.2, label="1s conf. band. lifetime fit");
for i in range(1,1000):
  plt.plot(years, data[i], '.', color=f'C{i}')
  _c = np.polyfit(years, data[i], 1)
  plt.plot(x, _c[0]*x + _c[1], '-', color=f'C{i}')
plt.plot(years, [v.n for v in yearly_lifetime.values()], 'k*', label="real lifetimes")
plt.xlim(2014.7,2018.3)
plt.title(f"{matched} out of {nevts} : {matched/nevts}")
plt.legend()
plt.show()
