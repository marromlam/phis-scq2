# test_background_lifetime_3sigma
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import numpy as np
import uproot3 as uproot
import ipanema
import matplotlib.pyplot as plt

ipanema.initialize('python')

data = uproot.open("/scratch49/marcos.romero/sidecar/2016/Bs2JpsiPhi/v3r0@LcosK_sWeight.root")['DecayTree'].pandas.df()

high_bkg = data.query('B_ConstJpsi_M_1 > 5446 & hlt1b == 0 & time > 0.5')
low_bkg = data.query('B_ConstJpsi_M_1 < 5286 & hlt1b ==0 & time > 0.5')
t_proxy = np.linspace(0, 15, 50)


h = plt.hist(np.array(high_bkg['time']), bins=60, color='r', label="B_ConstJpsi_M_1 > 5446", density=True)
plt.hist(low_bkg['time'].array, bins=h[1], label="B_ConstJpsi_M_1 < 5286", color='g', alpha=0.5, density=True)
plt.yscale('log')


def model(b, x): return b * np.exp(-b * x) / (np.exp(-0.3) - np.exp(-15))


def fcn(p):
  b = p['b'].value
  # print(b, time)
  prob = model(b, time)
  return -2.0 * np.log(prob)


time = np.array(high_bkg['time'])
pars = ipanema.Parameters()
pars.add(dict(name='b', value=0.6, min=-10, max=10))
high_lifetime = ipanema.optimize(fcn, pars, method='minuit').params
print(1 / high_lifetime['b'].uvalue)
plt.plot(t_proxy, model(high_lifetime['b'].value, t_proxy), label=f"{high_lifetime['b'].uvalue:.2uP}", color='r')

time = np.array(low_bkg['time'])
pars = ipanema.Parameters()
pars.add(dict(name='b', value=0.6, min=-10, max=10))
low_lifetime = ipanema.optimize(fcn, pars, method='minuit').params
print(1 / low_lifetime['b'].uvalue)
plt.plot(t_proxy, model(low_lifetime['b'].value, t_proxy), label=f"{low_lifetime['b'].uvalue:.2uP}", color='g')

z = (high_lifetime['b'].value - low_lifetime['b'].value)
z /= np.sqrt(high_lifetime['b'].stdev**2 + low_lifetime['b'].stdev**2)
print(f"Agreement: {z}")


plt.legend()
plt.show()

# vim: fdm=marker ts=2 sw=2 sts=2 sr et
