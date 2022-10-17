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
import complot

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
    "cut4": "B_ConstJpsi_M_1 < 5255 & B_ConstJpsi_M_1 > 5200",
    "cut2": "B_ConstJpsi_M_1 > 5255 & B_ConstJpsi_M_1 < 5310",
    "cut3": "B_ConstJpsi_M_1 > 5310 & B_ConstJpsi_M_1 < 5325",
    "peak": "B_ConstJpsi_M_1 > 5340 & B_ConstJpsi_M_1 < 5400",
    "cut5": "B_ConstJpsi_M_1 > 5400 & B_ConstJpsi_M_1 < 5450",
    "cut1": "B_ConstJpsi_M_1 > 5446",
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
fig2, (ax2_mass, ax2_sigma) = plt.subplots(2, 1)
mass_bins = ax_mass.hist(data['B_ConstJpsi_M_1'].array, bins=60, weights=weight, color='k', alpha=0.2)[1]
ax2_mass.hist(data['B_ConstJpsi_M_1'].array, bins=60, weights=weight, color='k', alpha=0.2)[1]
time_bins = np.histogram(data['time'].array, weights=weight, bins=60)[1]
sigma_bins = np.histogram(data['B_ConstJpsi_MERR_1'].array, weights=weight, bins=60)[1]


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
  # print(data.shape)
  _df = data.query(f'({v})')
  # print(_df.shape)

  _time = np.array(_df['time'])
  _sigma = np.array(_df['B_ConstJpsi_MERR_1'])
  _weight = np.array(_df['wLb'])
  # print(_weight.sum())
  _mass = np.array(_df['B_ConstJpsi_M_1'])

  ax_mass.hist(_mass, bins=mass_bins, weights=_weight, color=f'C{i}',
               label=f"${v}$".replace('&', r'\&').replace('B_ConstJpsi_M_1', 'm'),
               alpha=1, density=False)
  ax_mass.set_yscale('log')
  ax2_mass.hist(_mass, bins=mass_bins, weights=_weight, color=f'C{i}',
                label=f"${v}$".replace('&', r'\&').replace('B_ConstJpsi_M_1', 'm'),
                alpha=1, density=False)
  ax2_mass.set_yscale('log')

  ax_time.hist(_time, bins=time_bins, ec=f'C{i}',
               density=True, weights=_weight, alpha=0.8, histtype='step', fc='none')
  ax_time.set_yscale('log')

  pars = ipanema.Parameters()
  pars.add(dict(name='b', value=0.6, min=-20, max=20))
  _fit = ipanema.optimize(fcn, pars, method='minuit').params
  # print(1 / high_lifetime['b'].uvalue)
  ax_time.plot(t_proxy, model(_fit['b'].value, t_proxy),
               label=f"$\Gamma_s-\Gamma_d = {_fit['b'].uvalue:.2uL}$", color=f'C{i}')

  sigma_mean = np.mean(_sigma)
  sigma_errr = np.std(_sigma)
  hsigma = np.histogram(_sigma, bins=sigma_bins, density=True, weights=_weight)

  ax2_sigma.plot(0.5 * (hsigma[1][1:] + hsigma[1][:-1]), hsigma[0], color=f'C{i}',
                 label=f"$mean={sigma_mean:.3f}, sigma={sigma_errr:.3f} $")
  ax2_sigma.set_yscale('log')

  # time = np.array(low_bkg['time'])
  # pars = ipanema.Parameters()
  # pars.add(dict(name='b', value=0.6, min=-10, max=10))
  # low_lifetime = ipanema.optimize(fcn, pars, method='minuit').params
  # print(1 / low_lifetime['b'].uvalue)
  # ax1.plot(t_proxy, model(low_lifetime['b'].value, t_proxy), label=f"Gs-Gd = {low_lifetime['b'].uvalue:.2uP}", color='g')

  # z = (high_lifetime['b'].value - low_lifetime['b'].value)
  # z /= np.sqrt(high_lifetime['b'].stdev**2 + low_lifetime['b'].stdev**2)
  # print(f"Agreement: {z}")

ax_mass.set_title(f"common cut: ${general_cut}$".replace("&", "\&").replace('_', r'\_'))
# ax_mass.legend()
# ax_time.legend()
ax_mass.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax_time.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax_mass.set_xlabel('$m=B\_ConstJpsi\_M\_1$ [MeV$/c^2$]')
ax_time.set_xlabel('$t$ [ps]')
ax_mass.set_ylabel('$\Lambda_b$-weighted candidates')
ax_time.set_ylabel('$\Lambda_b$-weighted candidates')
fig.tight_layout()
fig2.savefig("time.pdf")
fig.show()

ax2_mass.set_title(f"common cut: ${general_cut}$".replace("&", "\&").replace('_', r'\_'))
# ax_mass.legend()
# ax_time.legend()
ax2_mass.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2_sigma.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2_mass.set_xlabel('$m=B\_ConstJpsi\_M\_1$ [MeV$/c^2$]')
ax2_sigma.set_xlabel('$\sigma_m$ [ps]')
ax2_mass.set_ylabel('$\Lambda_b$-weighted candidates')
ax2_sigma.set_ylabel('$\Lambda_b$-weighted candidates')
fig2.tight_layout()
fig2.savefig("sigmam.pdf")
fig2.show()

# vim: fdm=marker ts=2 sw=2 sts=2 sr et
