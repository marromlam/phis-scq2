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
# import complot

ipanema.initialize('python')

data = uproot.open("/scratch49/marcos.romero/sidecar/2016/Bs2JpsiPhi/v3r0@LcosK_chopped.root")['DecayTree'].pandas.df()
#
# high_bkg = data.query('B_ConstJpsi_M_1 > 5446 & hlt1b == 0 & time > 0.5')
# low_bkg = data.query('B_ConstJpsi_M_1 < 5286 & hlt1b ==0 & time > 0.5')

# low and high
# cuts = {
#     "cut1": "B_ConstJpsi_M_1 > 5446",
#     "cut2": "B_ConstJpsi_M_1 < 5286"
# }
cuts = {
    "cut1": "B_ConstJpsi_M_1 > 5300 & B_ConstJpsi_M_1 < 5400",
    "cut2": "B_ConstJpsi_M_1 > 5300 & B_ConstJpsi_M_1 < 5400"
}

# split low in two
# cuts = {
#     "mycut1": "B_ConstJpsi_M_1 > 5255 & B_ConstJpsi_M_1 < 5310",
#     "mycut2": "B_ConstJpsi_M_1 < 5255 & B_ConstJpsi_M_1 > 5200"
# }

cuts = {
    "peak": "B_ConstJpsi_M_1 > 5340 & B_ConstJpsi_M_1 < 5400",
    "cut1": "B_ConstJpsi_M_1 > 5446",
    "cut2": "B_ConstJpsi_M_1 > 5255 & B_ConstJpsi_M_1 < 5310",
    "cut3": "B_ConstJpsi_M_1 > 5310 & B_ConstJpsi_M_1 < 5325",
    "cut4": "B_ConstJpsi_M_1 < 5255 & B_ConstJpsi_M_1 > 5200",
    "cut5": "B_ConstJpsi_M_1 > 5400 & B_ConstJpsi_M_1 < 5450",
}

weight = data['wLb']
print("ola", 0 * len(weight) - (len(weight) + weight.sum()))
print(data.query("wLb<0").shape)
print("shit", data.query("wLb<0")['wLb'].values.sum())
general_cut = "hlt1b == 0 & time > 0.5 & time < 7"
data = data.query(general_cut)
weight = data['wLb']
print(len(weight) - weight.sum())
print(data.shape)
fig, (ax_mass, ax_time) = plt.subplots(2, 1)
mass_bins = ax_mass.hist(data['B_ConstJpsi_M_1'].array, bins=60, weights=weight, color='k', alpha=0.2)[1]
time_bins = np.histogram(data['time'].array, weights=weight, bins=60)[1]


def model(b, x):
  gamma = b + 0.65789 * 1
  num = np.exp(-gamma * x)
  den = (np.exp(-gamma * 0.5) - np.exp(-gamma * 7)) / gamma
  return num / den


def fcn(p):
  b = p['b'].value
  # print(b, time)
  prob = model(b, _time)
  return -2.0 * _weight * np.log(prob)


fit = {}
t_proxy = np.linspace(0, 7, 50)
for i, k in enumerate(cuts.keys()):
  v = cuts[k]
  print(data.shape)
  _df = data.query(f'({v})')
  print(_df.shape)

  _time = np.array(_df['time'])
  _weight = np.array(_df['wLb'])
  print(_weight.sum())
  _mass = np.array(_df['B_ConstJpsi_M_1'])

  ax_mass.hist(_mass, bins=mass_bins, weights=_weight, color=f'C{i}',
               label=v, alpha=1, density=False)
  ax_mass.set_yscale('log')

  ax_time.hist(_time, bins=time_bins, ec=f'C{i}',
               density=True, weights=_weight, alpha=0.8, histtype='step', fc='none')
  ax_time.set_yscale('log')

  pars = ipanema.Parameters()
  pars.add(dict(name='b', value=0.6, min=-20, max=20))
  _fit = ipanema.optimize(fcn, pars, method='minuit').params
  # print(1 / high_lifetime['b'].uvalue)
  ax_time.plot(t_proxy, model(_fit['b'].value, t_proxy),
               label=f"Gs-Gd = {_fit['b'].uvalue:.2uP}", color=f'C{i}')

  # time = np.array(low_bkg['time'])
  # pars = ipanema.Parameters()
  # pars.add(dict(name='b', value=0.6, min=-10, max=10))
  # low_lifetime = ipanema.optimize(fcn, pars, method='minuit').params
  # print(1 / low_lifetime['b'].uvalue)
  # ax1.plot(t_proxy, model(low_lifetime['b'].value, t_proxy), label=f"Gs-Gd = {low_lifetime['b'].uvalue:.2uP}", color='g')

  # z = (high_lifetime['b'].value - low_lifetime['b'].value)
  # z /= np.sqrt(high_lifetime['b'].stdev**2 + low_lifetime['b'].stdev**2)
  # print(f"Agreement: {z}")

ax_mass.set_title(general_cut)
ax_mass.legend()
ax_time.legend()
fig.tight_layout()
fig.show()

# vim: fdm=marker ts=2 sw=2 sts=2 sr et
