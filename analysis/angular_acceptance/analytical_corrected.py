import config
import hjson
import os
from scipy import stats, special
import uncertainties as unc
import uncertainties.unumpy as unp
import complot
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
from scipy.integrate import romb, simpson
from scipy.special import comb
from analysis import badjanak
import argparse
from scipy.interpolate import interp1d, interpn
from scipy.special import lpmv
import uproot3 as uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ipanema.core.python import ndmesh
from utils.strings import printsec, printsubsec
from utils.helpers import version_guesser, trigger_scissors, cuts_and
from ipanema import initialize, ristra, Parameters, Sample, optimize, IPANEMALIB
from ipanema import uncertainty_wrapper, get_confidence_bands
import ipanema
DESCRIPTION = """
    Computes the legendre-based angular acceptance with corrections in mHH, pB,
    pTB variables using an a reweight.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['transform_cijk', 'get_angular_prediction']


ipanema.initialize('cuda', 1)
badjanak.get_kernels()
# from ipanema import plotting


# tLL = config.general['tLL']
# tUL = config.general['tUL']
# all_knots = config.timeacc['knots']

order_cosK, order_cosL, order_hphi = config.angacc['legendre_order']
nob = config.angacc['analytic_bins'][0]
# nob = 20
# order_cosK, order_cosL, order_hphi = 2,4,0

# transforma cijk {{{


def transform_cijk(cijk, order_x, order_y, order_z, w=False):
  # TODO: I'm pretty sure this can be done much more easily and faster, but
  #       it seems work to the next slave working on phi_s at SCQ
  # order_x -> order of cosL poly
  # order_y -> order of hphi poly
  # order_z -> order of cosK poly
  noe = (order_x + 1) * (order_y + 1) * (order_z + 1)
  corr_x = np.zeros(noe)
  corr_y = np.zeros(noe)
  corr_z = np.zeros(noe)
  coeffs = np.zeros(noe)

  count = 0  # this should not be used now, but let's keep it just in case
  for i in range(0, order_x + 1):
    for j in range(0, order_y + 1):
      for k in range(0, order_z + 1):
        from_bin = i + (order_x + 1) * j + (order_x + 1) * (order_y + 1) * k
        for m in range(0, i // 2 + 1):
          to_bin = (i - 2 * m) + (order_x + 1) * j + (order_x + 1) * (order_y + 1) * k
          corr_x[to_bin] += (-1)**(m) * comb(i, m) * comb(2 * i - 2 * m, i) * cijk[from_bin] / (2.0**(i))
        count += 1

  for i in range(0, order_x + 1):
    for j in range(0, order_y + 1):
      for k in range(0, order_z + 1):
        from_bin = i + (order_x + 1) * j + (order_x + 1) * (order_y + 1) * k
        for m in range(0, j // 2 + 1):
          to_bin = i + (order_x + 1) * (j - 2 * m) + (order_x + 1) * (order_y + 1) * k
          corr_y[to_bin] += (1.0 if (m % 2 == 0) else -1.0) * comb(j, m) * comb(2 * j - 2 * m, j) * corr_x[from_bin] / (2.0**(j))

  for i in range(0, order_x + 1):
    for j in range(0, order_y + 1):
      for k in range(0, order_z + 1):
        from_bin = i + (order_x + 1) * j + (order_x + 1) * (order_y + 1) * k
        for m in range(0, k // 2 + 1):
          to_bin = i + (order_x + 1) * j + (order_x + 1) * (order_y + 1) * (k - 2 * m)
          corr_z[to_bin] += (1.0 if (m % 2 == 0) else -1.0) * comb(k, m) * comb(2 * k - 2 * m, k) * corr_y[from_bin] / (2.0**(k))

  # for i in range(0, noe):
  #    print(i, corr_z[i])
  # exit()
  # correct because hphi is not in (-1,1)
  coeffs = []
  crap = []
  for j in range(0, order_x + 1):
    for k in range(0, order_y + 1):
      for l in range(0, order_z + 1):
        lbin = j + (order_x + 1) * k + (order_x + 1) * (order_y + 1) * l
        crap.append(lbin)
        coeffs.append(corr_z[lbin] / np.pi**k)

  if w:
    ans = np.zeros(noe)
    for i in range(len(coeffs)):
      ans[crap[i]] = coeffs[i]
    return np.array(ans)

  return np.array(coeffs)

# }}}


def fcn_d(pars):
  pars_d = ristra.allocate(np.array(pars))
  chi2_d = ristra.allocate(0 * data_3d_d).astype(np.float64)
  cosK_l = len(cosK_d)
  cosL_l = len(cosL_d)
  hphi_l = len(hphi_d)
  badjanak.__KERNELS__.magia_borras(chi2_d, cosK_d, cosL_d, hphi_d, data_3d_d,
                                    prediction_3d_d, pars_d, np.int32(cosK_l),
                                    np.int32(cosL_l), np.int32(hphi_l),
                                    np.int32(order_cosK), np.int32(order_cosL),
                                    np.int32(order_hphi),
                                    global_size=(cosK_l, cosL_l, hphi_l))
  # exit()
  # breakpoint()
  return ristra.get(chi2_d).ravel()


# create margic function
@np.vectorize
def get_angular_prediction(params, cosKLL=-1, cosKUL=-0.9, cosLLL=-1,
                           cosLUL=-0.9, hphiLL=-np.pi, hphiUL=-0.9 * np.pi,
                           tLL=0.3, tUL=15):
  """
  Calculates the angular prediction in a given 3d bin in cosK, cosL and hphi.

  Parameters
  ----------
  params:

  """
  # TODO: This function is too slow. It was coded this way just to check
  #       good agreement with HD-fitter. Once this is done (it is), it should
  #       be rewritten in openCL.
  var = ristra.allocate(np.ascontiguousarray(np.float64([0.0] * 3 + [0.3] + [1020.] + [0.0] * 5)))
  pdf = ristra.allocate(np.float64([0.0]))
  badjanak.delta_gamma5_mc(var, pdf, **params, tLL=tLL, tUL=tUL)
  num = pdf.get()
  badjanak.delta_gamma5_mc(var, pdf, **params, cosKLL=cosKLL,
                           cosKUL=cosKUL, cosLLL=cosLLL, cosLUL=cosLUL,
                           hphiLL=hphiLL, hphiUL=hphiUL, tLL=tLL, tUL=tUL)
  den = pdf.get()
  return num / den


if __name__ == "__main__":
  p = argparse.ArgumentParser(description='Compute angular acceptance.')
  p.add_argument('--sample', help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tijk', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-angacc', help='Bs2JpsiPhi MC sample')
  p.add_argument('--mode', help='Configuration')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--figures', help='Year of data-taking')
  p.add_argument('--trigger', help='Trigger(s) to fit [comb/(biased)/unbiased]')
  args = vars(p.parse_args())

  tLL = 0.3
  tUL = 15
  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  ANGACC = args['angacc']
  MODE = args['mode']
  TRIGGER = args['trigger']
  print(ANGACC)
  # Prepare the cuts
  CUT = ''   # place cut attending to version

  from trash_can.knot_generator import create_time_bins
  if ANGACC != 'analytic':
    nknots, timebin = ANGACC.split('knots')
    timebin = int(timebin)
    knots = create_time_bins(int(nknots[8:]), tLL, tUL).tolist()
    tLL, tUL = knots[timebin - 1], knots[timebin]
    CUT = cuts_and(CUT, f'time>={tLL} & time<{tUL}')
  CUT = trigger_scissors(TRIGGER, CUT)

  # Print settings
  printsec('Settings')
  print(f"{'backend':>15}: {config.user['backend']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'angacc':>15}: {ANGACC} {order_cosK}{order_cosL}{order_hphi}")

  weight_str = 'polWeight*sWeight'
  if VERSION == 'v0r0':
    args['input_params'] = args['input_params'].replace('generator', 'generator_old')
    weight_str = 'polWeight*sWeight/gbWeight'
    weight_str = 'polWeight*sWeight'

  # load parameters
  gen = Parameters.load(args['input_params'])

  # load MC sample
  branches = ['cosK', 'cosL', 'hphi', 'time', 'gencosK', 'gencosL', 'genhphi',
              'gentime', 'polWeight', 'sWeight', 'gbWeight', 'hlt1b'
              ]
  mc = Sample.from_root(args['sample'], branches=branches)
  mc.chop(CUT)
  mc = mc.df
  print(mc)

  """
  mc = pd.concat((
  # uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root')['DecayTree'].pandas.df(['helcosthetaK','helcosthetaL','helphi', 'time', 'sw', 'gb_weights', 'hlt1b', 'cosThetaKRef_GenLvl', 'cosThetaMuRef_GenLvl', 'phiHelRef_GenLvl']),
  # uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].pandas.df(['PolWeight'])
  uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root')['DecayTree'].pandas.df(['helcosthetaK','helcosthetaL','helphi', 'time', 'sw', 'gb_weights', 'hlt1b', 'cosThetaKRef_GenLvl', 'cosThetaMuRef_GenLvl', 'phiHelRef_GenLvl']),
  uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].pandas.df(['PolWeight']),
  ), axis=1)

  mc.eval('cosK = helcosthetaK', inplace=True)
  mc.eval('cosL = helcosthetaL', inplace=True)
  mc.eval('hphi = helphi', inplace=True)
  mc.eval('gencosK = cosThetaKRef_GenLvl', inplace=True)
  mc.eval('gencosL = cosThetaMuRef_GenLvl', inplace=True)
  mc.eval('genhphi = phiHelRef_GenLvl', inplace=True)
  mc.eval('polWeight = PolWeight', inplace=True)
  mc.eval('sWeight = sw', inplace=True)
  mc.eval('gbWeight = gb_weights', inplace=True)

  # get only one trigger cat
  mc = mc.query('hlt1b == 0')
  print(mc)
  # weights = np.array(mc.eval(f'polWeight*sw/gb_weights'))
  # alpha = np.sum(weights)/np.sum(weights**2)
  """
  sum_weights = np.array(mc.eval(weight_str)).sum()

  # Prepare data and arrays {{{

  printsubsec("Prepare data")

  print(" * Prepare data")
  histdd = np.histogramdd(mc[['cosK', 'cosL', 'hphi']].values, bins=(nob, nob, nob),
                          weights=mc.eval(weight_str),
                          range=[(-1, 1), (-1, 1), (-np.pi, np.pi)])
  mccounts, (ecosK, ecosL, ehphi) = histdd
  noe = (order_hphi + 1) * (order_cosL + 1) * (order_cosK + 1)

  print(" * Prepare theory")
  # WARNING: Prepare mesh with get_angular_prediction, which is a very slow
  #          function
  m1 = ndmesh(ecosK[:-1], ecosL[:-1], ehphi[:-1])
  m2 = ndmesh(ecosK[1:], ecosL[1:], ehphi[1:])
  arry = np.ascontiguousarray(np.zeros((nob**3, 2 * 3)))
  arry[:, 0::2] = np.stack((m.ravel() for m in m1), axis=-1)
  arry[:, 1::2] = np.stack((m.ravel() for m in m2), axis=-1)
  pars_dict = np.ascontiguousarray([gen.valuesdict() for i in range(nob**3)])

  # generate predictions
  gencounts = get_angular_prediction(pars_dict, *arry.T, tLL=tLL, tUL=tUL)
  gencounts = gencounts.reshape(nob, nob, nob)

  # cook arrays with bin centers
  bcosK = 0.5 * (ecosK[1:] + ecosK[:-1])
  bcosL = np.copy(bcosK)
  bhphi = np.copy(bcosK)

  # TODO: These two functions sort elements in the HD-fitter way. This is
  #       clearly uneeded, but one should rewrite the whole code, and loses
  #       comparison capabilitis wrt. HD-fitter.
  def data_3d(i, N=20):
    c = i // (N * N)
    d = i % (N * N)
    a = d // N
    b = d % N
    return mccounts[c, b, a]

  def prediction_3d(i, N=20):
    c = i // (N * N)
    d = i % (N * N)
    a = d // N
    b = d % N
    return gencounts[a, b, c]

  # for i in range(nob**3):
  #   print(data_3d(i, nob), prediction_3d(i, nob), prediction_3d(i,nob)*sum_weights)
  # print(sum_weights)

  # }}}

  # Fit eff x prediction to data {{{

  printsubsec('Fitting legendre polynomials coefficients')

  # WARNING: I messed up. At some point I confused cosK with cosL, and therefore
  #          there are several functions treating these two as the other. So,
  #          be aware of this issue, and read the code properly.
  #          THIS NEEDS TO BE FIXED!
  # order_cosK, order_cosL, order_hphi = order_cosL, order_cosK, order_hphi

  print(' * Build parameters dict')
  pars = Parameters()
  for i in range(noe):
    cosL_bin = int(i % (order_cosL + 1))
    hphi_bin = int(((i - cosL_bin) / (order_cosL + 1)) % (order_hphi + 1))
    cosK_bin = int(((i - cosL_bin) / (order_cosL + 1) - hphi_bin) / (order_hphi + 1))
    pars.add({"name": f"b{cosL_bin}{hphi_bin}{cosK_bin}", "value": 0.0, "latex": f"b_{i+1}", "free": False})
    if ((cosK_bin % 2 == 0 or cosK_bin == 0) and
        (hphi_bin % 2 == 0 or hphi_bin == 0) and
            (cosL_bin % 2 == 0 or cosL_bin == 0)):
      pars[f'b{cosL_bin}{hphi_bin}{cosK_bin}'].free = True
      pars[f'b{cosL_bin}{hphi_bin}{cosK_bin}'].set(value=0.0)
  print(pars)
  pars['b000'].set(value=1.0)

  print(' * Allocating arrays')
  chi2_d = ristra.allocate(0 * bcosK).astype(np.float64)
  cosK_d = ristra.allocate(bcosK).astype(np.float64)
  cosL_d = ristra.allocate(bcosL).astype(np.float64)
  hphi_d = ristra.allocate(bhphi).astype(np.float64)
  data_3d_d = ristra.allocate(
      np.float64([data_3d(i, nob) for i in range(nob**3)])
  ).astype(np.float64)
  prediction_3d_d = sum_weights * ristra.allocate(
      np.float64([prediction_3d(i, nob) for i in range(nob**3)])
  ).astype(np.float64)

  print(' * Fitting...')
  result = optimize(fcn_d, pars, method='minuit', verbose=False, timeit=True)
  print(result)
  print(f"Max corr = {np.amax(result.params.corr()-np.eye(noe))*100:.4f}%")

  # WARNING: MESS UP END
  #    If you recall I messed up just before. Whatever I did is correct from
  #    the lines that follow up. So, basically, you do not need to touch from
  #    now on but before.
  # order_cosK, order_cosL, order_hphi = order_cosL, order_cosK, order_hphi
  peff = [p.uvalue for p in result.params.values()]
  upeff = uncertainty_wrapper(
      lambda p: transform_cijk(p, order_cosL, order_hphi, order_cosK),
      peff)

  cr = transform_cijk([p.uvalue.n for p in result.params.values()],
                      order_cosL, order_hphi, order_cosK, True)

  def tijk2weights_(tijk, order_cosK, order_cosL, order_hphi):
    tijk_h = transform_cijk(tijk, order_cosL, order_hphi, order_cosK, True)
    tijk_d = ristra.allocate(tijk_h)
    w = ristra.zeros(len(badjanak.config['tristan']))
    badjanak.__KERNELS__.py_tijk2weights(w, tijk_d, np.int32(order_cosK),
                                         np.int32(order_cosL),
                                         np.int32(order_hphi), global_size=(1,))
    return ristra.get(w)

  w = uncertainty_wrapper(
      lambda p: tijk2weights_(p, order_cosK, order_cosL, order_hphi),
      peff)
  print(w)
  w /= w[0]  # as usually, normalize to the first one

  # WARNING: Correlation matrix for upeff needs to be computed.
  #          This means, current output one is WRONG

  # create parameters object
  print(' * Dumping tijk parameters')

  for i, p in enumerate(result.params.values()):
    p.set(value=upeff[i].n)
    p.stdev = upeff[i].s

  result.params.dump(args['output_tijk'])

  print(f" * Naive tijk for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{result.params}")

  # create parameters object
  print(' * Dumping angular acceptance parameters')

  angacc = Parameters()
  for i, p in enumerate(w):
    angacc.add(dict(
        name=f"w{i}{TRIGGER[0]}", latex=f"w_{{{i}}}^{{{TRIGGER[0]}}}",
        value=p.n, stdev=p.s
    ))

  print(f" * Naive angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
  print(f"{angacc}")
  angacc.dump(args['output_angacc'])

  # exit()
  #%% THE PLOTS ##################################################################

  mc = data_3d_d.get().reshape(nob, nob, nob)
  th = prediction_3d_d.get().reshape(nob, nob, nob)

  cosK = 0.5 * (np.linspace(-1, 1, nob + 1)[1:] + np.linspace(-1, 1, nob + 1)[:-1])
  cosL = 0.5 * (np.linspace(-1, 1, nob + 1)[1:] + np.linspace(-1, 1, nob + 1)[:-1])
  hphi = 0.5 * (np.linspace(-1, 1, nob + 1)[1:] + np.linspace(-1, 1, nob + 1)[:-1])

  N = nob + 1
  cosKd = ristra.allocate(0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1]))
  cosLd = ristra.allocate(0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1]))
  hphid = ristra.allocate(0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1]))

  #upeff = np.array([p.uvalue for p in result.params.values()])

  for k, var in enumerate(['cosK', 'cosL', 'hphi']):
    if var == 'cosK':
      proj = (1, 2)
      bounds = 1
      tex = r'\mathrm{cos}\,\theta_K'
      dir = (0, 1)
      x = np.copy(cosK)
      xh = np.copy(ristra.get(cosKd))
    elif var == 'cosL':
      proj = (0, 1)
      bounds = 1
      tex = r'\mathrm{cos}\,\theta_{\mu}'
      dir = (1, 2)
      x = np.copy(cosL)
      xh = np.copy(ristra.get(cosLd))
    elif var == 'hphi':
      proj = (2, 0)
      bounds = np.pi
      tex = r'\phi_h\, \mathrm{[rad]}'
      dir = (2, 0)
      x = np.copy(hphi)
      xh = np.copy(ristra.get(hphid))

    # project data and theory into variable
    _mc = np.sum(mc, proj)
    _th = np.sum(th, proj)
    y = _mc / _th

    # get efficiency, project it and normalize
    #peff = transform_cijk(np.array(result.params), order_cosL, order_hphi, order_cosK)
    #peff = np.array(result.params)
    peff = np.array([p.n for p in upeff])
    print(peff)
    eff = badjanak.analytical_angular_efficiency(peff, cosKd, cosLd, hphid, None, order_cosK, order_cosL, order_hphi)
    eff = np.sum(eff * th, dir) / np.sum(th, dir)
    #norm = 1
    norm = 1 / (np.trapz(y, x) / np.trapz(eff, xh))
    #norm = 1/np.max(eff)

    # prepare errorbars
    _umc = np.sqrt(_mc)
    _uth = np.sqrt(_th)
    uy = np.sqrt((1 / _th)**2 * _umc**2 + (-_mc / _th**2)**2 * _uth**2)
    ux = 0.5 * (x[1] - x[0]) * np.ones_like(x)

    # compute confidence bands
    yunc = uncertainty_wrapper(
        lambda p: np.mean(th * badjanak.analytical_angular_efficiency(p, cosKd, cosLd, hphid, None, order_cosK, order_cosL, order_hphi), dir) / np.mean(th, dir),
        # uncertainty_wrapper(lambda p: transform_cijk(np.array(p), order_cosK, order_hphi, order_cosL), upeff)
        upeff
    )
    yl, yh = get_confidence_bands(yunc)  # /norm/np.max(y)

    # toy confidence bands (very slow, as you would expect)
    # pars = []; chi2 = []; accs = []
    # for rng in range(0,27*int(1e6)):
    #   rng_pars = np.ascontiguousarray([np.random.normal(p.n, 5*p.s) for p in upeff])
    #   rng_chi2 = bar(rng_pars)
    #   #print(rng_chi2)
    #   if chi2_probability(abs(rng_chi2-8351.725486964235), 27) > 0.317:
    #       print(rng_chi2)
    #       #pars.append(rng_pars)
    #       #chi2.append(rng_chi2)
    #       accs.append(foo(rng_pars, dir))
    # yh = np.max(np.array(accs),0)
    # yl = np.min(np.array(accs),0)
    # shit = np.mean(np.array(accs),0)

    # plot
    fig, ax = complot.axes_plot()
    ax.errorbar(bounds * x, y / np.max(y), yerr=uy, xerr=bounds * ux, fmt='.k')
    ax.plot(bounds * xh, eff / norm / np.max(y))
    #ax.plot(bounds*xh, shit/norm/np.max(y))
    ax.fill_between(bounds * x, yh / norm / np.max(y), yl / norm / np.max(y), alpha=0.2)
    ax.set_ylim(0.85, 1.05)
    ax.set_xlabel(f"${tex}$")
    ax.set_ylabel(rf"$\varepsilon({tex})$ [a.u.]")
    os.makedirs(args['figures'], exist_ok=True)
    _mark = args['version'].split('@')[0]  # watermark plots
    # watermark(ax, version=f"${_mark}$  {weight_str}", scale=1.02)
    watermark(ax, version=f"${_mark}$", scale=1.02)
    fig.savefig(os.path.join(args['figures'], f"{var}.pdf"))
