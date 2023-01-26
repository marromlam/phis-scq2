# taggera
#
#

from uncertainties import unumpy as unp
__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import numpy as np
import ipanema
import complot
import typing as tp
ipanema.initialize('cuda', 1)


prog = ipanema.compile(open('ift_tagging/tagging.cu').read())
pdf = prog.tagging_calibration_pdf
# integraB = mod.get_function("omega_int")
# integraBbar = mod.get_function("omega_bar_int")

MODE = "MC_Bu2JpsiKplus"
# MODE = "MC_Bs2JpsiPhi"

GBWEIGHTED = 0

YEAR = "2018"

TAGGER = "OST"
# TAGGER = "SST"
# TAGGER = "IFT"

FIXDP = 0
FREE_DP = not FIXDP


# create parameters
pars = ipanema.Parameters()
pars.add(dict(name='p0', value=0.4, min=0.0, max=0.8, free=True))
pars.add(dict(name='p1', value=0.9, min=0.2, max=1.1, free=True))
pars.add(dict(name='dp0', value=0., min=-1, max=1, free=FREE_DP))
pars.add(dict(name='dp1', value=0., min=-1, max=1, free=FREE_DP))
pars.add(dict(name='eta_bar', value=0.5, min=-1, max=1, free=False))
pars.add(dict(name='p2', value=0., min=0.0, max=1.0, free=False))
pars.add(dict(name='dp2', value=0., min=-1, max=1., free=False))


def taggers_combination(qs: tp.List[np.array], etas: tp.List[np.array]):
  """
  Combines different OS tagges into one single tagging branch
  It returs this branch as numpy.ndarray

  Parameters
  ----------
  qs: List of arrays with all tagging decisions to combine
  etas: List of arrays with all mistagging decisions to combine

  Returns
  -------
  Tuple of tagging decision and mistagging decision combined
  """
  pB = np.ones_like(qs[0])
  pBbar = np.ones_like(qs[0])

  for i in range(len(qs)):
    pB *= 0.5 * (1. + qs[i]) - qs[i] * (1. - etas[i])
    pBbar *= 0.5 * (1. - qs[i]) + qs[i] * (1. - etas[i])

  q = np.ones_like(qs[0]) * 0
  eta = np.ones_like(qs[0]) * 0.5
  PB = pB / (pB + pBbar)
  PBbar = 1. - PB

  q = np.where(PB > PBbar, -1, q)
  eta = np.where(PB > PBbar, 1 - PB, eta)
  q = np.where(PB < PBbar, +1, q)
  eta = np.where(PB < PBbar, 1 - PBbar, eta)

  return q, eta


# load sample
rf = "/scratch49/marcos.romero/sidecar/2017/MC_Bu2JpsiKplus/v4r0@LcosK.root"
data = ipanema.Sample.from_root(rf)
weight_branch = 'sWeight'
weight_branch = 'sigBsSW'
weight_branch = 'time/time'
# weight_branch = 'gbw'


if TAGGER == 'IFT':
  q = data.df['B_IFT_InclusiveTagger_TAGDEC']
  eta = data.df['B_IFT_InclusiveTagger_TAGETA']
elif TAGGER == "SST":
  q = data.df['B_SSKaonLatest_TAGDEC']
  eta = data.df['B_SSKaonLatest_TAGETA']
elif TAGGER == "OST":
  q_list = ['B_OSKaonLatest_TAGDEC',
            'B_OSMuonLatest_TAGDEC',
            'B_OSElectronLatest_TAGDEC',
            'B_OSCharm_TAGDEC',
            'B_OSVtxCh_TAGDEC']
  eta_list = ['B_OSKaonLatest_TAGETA',
              'B_OSMuonLatest_TAGETA',
              'B_OSElectronLatest_TAGETA',
              'B_OSCharm_TAGETA',
              'B_OSVtxCh_TAGETA']
  q_list = [np.float64(data.df[b]) for b in q_list]
  eta_list = [np.float64(data.df[b]) for b in eta_list]
  q, eta = taggers_combination(q_list, eta_list)


data.df['xq'] = q
data.df['xeta'] = eta
data.df.eval(f"weight = {weight_branch}", inplace=True)


# TODO: move this to a YAML file {{{

if MODE == "MC_Bs2JpsiPhi":
  data.df['xid'] = data.df['B_TRUEID']
elif MODE == "MC_Bs2DsPi":
  data.df['xid'] = data.df['B_TRUEID']
elif MODE == "Bs2DsPi":
  data.df['xid'] = data.df['B_ID']
elif MODE == "MC_Bu2JpsiKplus" or MODE == "Bu2JpsiKplus":
  data.df['xid'] = data.df['B_ID']
else:
  exit()


tagger_cuts = {
    "IFT": "xq!=0",
    "SST": "xq!=0 & xeta!=0.5",
    "OST": "~(xeta < 0 | xeta >= 0.5 | xq == 0)"
}

if MODE == "MC_Bs2JpsiPhi":
  cuts = [
      tagger_cuts[TAGGER],
      "(B_BKGCAT==0 | B_BKGCAT==50)",
  ]
  cut = "(" + ") & (".join(cuts) + ")"
elif "Bs2DsPi" in MODE:
  cuts = [
      tagger_cuts[TAGGER],
  ]
elif MODE == "MC_Bu2JpsiKplus":
  cuts = [
      tagger_cuts[TAGGER],
      "(B_BKGCAT==0 | B_BKGCAT==50)",
  ]
elif MODE == "Bu2JpsiKplus":
  cuts = [
      tagger_cuts[TAGGER],
  ]

# }}}


cut = "(" + ") & (".join(cuts) + ")"
print("Applied cut:", cut)
data.chop(cut)
print(data)
print(data.branches)

# allocate in device
data.allocate(weight='weight')
data.allocate(x=['xq', 'xeta', 'xid'])
data.allocate(prob="0*weight")

print("Dataframe:")
print(data.df[['xq', 'xeta', 'xid', 'weight']])
print("eta_bar:", np.mean(data.df['xeta']))
pars['eta_bar'].set(value=np.mean(data.df['xeta']))
print(pars)


def model(x, prob, p0=0, dp0=0, p1=1, dp1=0, p2=0, dp2=0, eta_bar=0.5):
  pdf(x, prob, np.float64(p0), np.float64(dp0), np.float64(p1),
      np.float64(dp1), np.float64(p2), np.float64(dp2), np.float64(eta_bar),
      np.int32(len(prob)), global_size=(len(prob),))
  return ipanema.ristra.get(prob)


def omega(x, prob, p0=0, dp0=0, p1=1, dp1=0, p2=0, dp2=0, eta_bar=0.5):
  prog.calibrated_mistag(x, prob, np.float64(p0), np.float64(dp0), np.float64(p1),
                         np.float64(dp1), np.float64(p2), np.float64(dp2), np.float64(eta_bar),
                         np.int32(len(prob)), global_size=(len(prob),))
  return ipanema.ristra.get(prob)


def fcn(pars, data):
  p = pars.valuesdict()
  prob = model(data.x, data.prob, **p)
  return -2. * np.log(prob) * ipanema.ristra.get(data.weight)


res = ipanema.optimize(fcn, pars, fcn_args=(data,),
                       method='minuit', tol=0.05, verbose=False)

print(res)


rd = "/scratch49/marcos.romero/sidecar/2015/Bs2JpsiPhi/v3r0@LcosK_sWeight.root"
# rd = ipanema.Sample.from_root(rd)

# print(rd.branches)


# tagdec, tageta = 'B_SSKaonLatest_TAGDEC', 'B_SSKaonLatest_TAGETA'
# hrd = complot.hist(rd.df[tageta], bins=100)
fig, axplot = complot.axes_plot()

x = ipanema.ristra.linspace(0, 0.5, 100)
y = 0 * ipanema.ristra.linspace(0, 0.5, 100)
# y = model(data.x, data.prob, **res.params.valuesdict())
# var0 = omega(data.x, data.prob, **res.params.valuesdict())

# eta_arr = np.array(data.df['xeta'].array)
# counts, edges = np.histogram(eta_arr, 100, density=True)
# hrd = complot.hist(eta_arr, bins=edges, density=True)

# shit = []
# y = np.where(data.df['xid']/np.abs(data.df['xid']) == data.df['xq'], data.df['xq'], 0)
# y = np.where(data.df['xid']/np.abs(data.df['xid']) == data.df['xq'], data.df['xq'], 0)
# var1 = data.df['xid'] / np.abs(data.df['xid'])
# var2 = data.df['xeta']
# for i, p in enumerate(pos):
# hvar0 = []
# hvar1 = []
# hvar2 = []
# for el, er in zip(edges[:-1], edges[1:]):
#     hvar0.append(np.array(var0[(eta_arr >= el) & (eta_arr < er)]))
#     hvar1.append(np.array(var1[(eta_arr >= el) & (eta_arr < er)]))
#     hvar2.append(np.array(var2[(eta_arr >= el) & (eta_arr < er)]))
# hvar0 = np.array([s.sum() for s in hvar0])
# hvar1 = np.array([s.sum() for s in hvar1])
# hvar2 = np.array([s.sum() for s in hvar2])
# shit /= np.trapz(shit, hrd.bins)
# axplot.plot(ipanema.ristra.get(data.x), ipanema.ristra.get(y))


def calib(eta, p0, p1, p2, dp0, dp1, dp2, eta_bar, tag=1):
  result = 0
  result += (p0 + tag * 0.5 * dp0)
  result += (p1 + tag * 0.5 * dp1) * (eta - eta_bar)
  result += (p2 + tag * 0.5 * dp2) * (eta - eta_bar) * (eta - eta_bar)
  return result


# axplot.plot(hrd.bins, hrd.counts/np.sum(hrd.counts), color="k", label="gris")
# axplot.plot(hrd.bins, hvar1/hvar2, label="negro")

# axplot.plot(hrd.bins, hvar0, label="var0")
# axplot.plot(hrd.bins, hvar1, label="var1")
# axplot.plot(hrd.bins, hvar2, label="var2")
# axplot.plot(hrd.bins, hvar2/hvar0, label="var0-var2")


x = ipanema.ristra.linspace(0, 0.5, 100)
y = 0 * ipanema.ristra.linspace(0, 0.5, 100)
__x = x.get()
__y = calib(__x, **res.params.valuesdict())
norm = np.trapz(__y, __x)

_, edges = np.histogram(data.df['xeta'], 60, weights=data.df['xeta'])
hrd1 = complot.hist(data.df['xeta'], bins=edges,
                    weights=data.df['weight'])
hrd2 = complot.hist(data.df.query('xid*xq<0')['xeta'], bins=edges,
                    weights=data.df.query('xid*xq<0')['weight'])

num = unp.uarray(hrd2.counts, hrd2.yerr[0])
den = unp.uarray(hrd1.counts, hrd1.yerr[0])
ratio = num / den
ratio = norm * ratio / np.trapz(unp.nominal_values(ratio), hrd1.bins)
axplot.errorbar(hrd1.bins, unp.nominal_values(ratio),
                yerr=unp.std_devs(ratio), xerr=hrd1.xerr, color='k', fmt='.')

axplot.plot(__x, __y, label="calibration")
axplot.set_xlabel(rf"$\eta^{{{TAGGER}}}$")
axplot.set_ylabel(rf"$\omega(\eta^{{{TAGGER}}})$")
axplot.fill_between(hrd1.bins, 0.4 * norm * hrd1.counts / hrd1.norm,
                    0, color='k', alpha=0.2, step='mid')
# axplot.legend()
# axplot.set_xlim(0.15, 0.5)
axplot.set_ylim(0, 0.5)
fig.savefig('tagging.pdf')


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
