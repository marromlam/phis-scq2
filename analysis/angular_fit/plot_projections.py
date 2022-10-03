import config
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
import hjson
import os
from scipy import stats, special
import uncertainties as unc
import uncertainties.unumpy as unp
import complot
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
from ipanema import initialize, ristra, Parameters, Sample, optimize, IPANEMALIB, ristra
from ipanema import uncertainty_wrapper, get_confidence_bands
__all__ = []

import ipanema
ipanema.initialize('cuda', 1, real='double')


if __name__ == "__main__":

  def sample_loader(Bs2JpsiPhi=False, Bd2JpsiKstar=False,
                    MC_Bs2JpsiPhi=False, MC_Bs2JpsiPhi_dG0=False,
                    MC_Bs2JpsiKK_Swave=False, MC_Bd2JpsiKstar=False,
                    trigger=['combined'], cut=None):
    samples = {}
    if Bs2JpsiPhi:
      samples['Bs2JpsiPhi'] = Bs2JpsiPhi
    if Bd2JpsiKstar:
      samples['Bd2JpsiKstar'] = Bd2JpsiKstar
    if MC_Bs2JpsiPhi:
      samples['MC_Bs2JpsiPhi'] = MC_Bs2JpsiPhi
    if MC_Bs2JpsiPhi_dG0:
      samples['MC_Bs2JpsiPhi_dG0'] = MC_Bs2JpsiPhi_dG0
    if MC_Bs2JpsiKK_Swave:
      samples['MC_Bs2JpsiKK_Swave'] = MC_Bs2JpsiKK_Swave
    if MC_Bd2JpsiKstar:
      samples['MC_Bd2JpsiKstar'] = MC_Bd2JpsiKstar
    s = {}
    print(samples)
    for km, vm in samples.items():
      s[km] = {}
      for vy in vm:
        if '2015' in vy:
          ky = '2015'
        elif '2016' in vy:
          ky = '2016'
        elif '2017' in vy:
          ky = '2017'
        elif '2018' in vy:
          ky = '2018'
        else:
          raise ValueError("I dont get this year at all")
        s[km][ky] = {}
        for kt in trigger:
          s[km][ky][kt] = Sample.from_root(vy, cuts=cuts_and(trigger_scissors(kt), cut), name=f"{km}-{ky}-{kt}")
          # print(s[km][ky][kt])

    return s

  p = argparse.ArgumentParser(description="Plot projections 4D-fit")
  p.add_argument('--samples', help='Bs2JpsiPhi data sample')
  p.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  p.add_argument('--csp', help='Bs2JpsiPhi MC sample')
  p.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  p.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', default=False, help='Bs2JpsiPhi MC sample')
  p.add_argument('--figures', default=False, help='Bs2JpsiPhi MC sample')
  p.add_argument('--input-params', default=False, help='Bs2JpsiPhi MC sample')
  p.add_argument('--log-likelihood', default=False, help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--version', help='Year of data-taking')
  p.add_argument('--fit', help='Year of data-taking')
  p.add_argument('--angacc', help='Year of data-taking')
  p.add_argument('--timeacc', help='Year of data-taking')
  p.add_argument('--trigger', help='Year of data-taking')
  p.add_argument('--blind', default=1, help='Year of data-taking')
  p.add_argument('--plot-amplitudes', default=True, help='plot Ai')
  p.add_argument('--scan-likelihood', default=False, help='Year of data-taking')
  args = vars(p.parse_args())

  VERSION = args['version']
  TRIGGER = args['trigger']
  TIMEACC = args['timeacc']
  YEARS = args['year'].split(',')

  plot_subamps = args['plot_amplitudes']

  if TRIGGER == 'combined':
    triggers = ['biased', 'unbiased']
  else:
    triggers = [TRIGGER]

  original_pars = Parameters.load(args['params'])
  print(original_pars)

  # unblind them, if they are
  pars = original_pars.valuesdict(False)

  for k, v in original_pars.items():
    v.min = -np.inf
    v.max = +np.inf
    v.free = False
  # exit()
  all_pars = [[pars, 1, "full"]]
  if plot_subamps:
    _all_pars = list(pars.keys())
    _has_swave = 0
    _has_pwave = 0
    _has_dwave = 0
    for p in _all_pars:
      if 'fSlon' in p:
        _has_swave += 1
      if 'fP' in p:
        _has_pwave += 1
      if 'fD' in p:
        _has_dwave += 1
    _has_swave = True if _has_swave > 0 else False
    _has_pwave = True if _has_pwave > 0 else False
    _has_dwave = True if _has_dwave > 0 else False
    aprox_swave = 0
    aprox_pwave = 1
    aprox_dwave = 0
    if _has_swave:
      aprox_swave = np.mean(np.array([v for k, v in pars.items() if 'fS' in k]))
    if _has_dwave:
      aprox_dwave = np.mean(np.array([v for k, v in pars.items() if 'fD' in k]))
    aprox_pwave = 1 - aprox_swave - aprox_dwave

    if _has_dwave:
      raise ValueError("D wave was not correctly implemented!")

    # get lifetime measurement
    _Gs = pars['DGsd'] + pars['Gd']

    if _has_swave:
      _pars_swave = ipanema.Parameters.clone(original_pars)
      for k, v in pars.items():
        _pars_swave[k].value = 1 if 'fS' in k else v
        _pars_swave[k].value = 0 if 'fP' in k else _pars_swave[k].value
      all_pars.append([_pars_swave.valuesdict(), aprox_swave, "S-wave"])
      # _pars_swave['DGsd'].value = 0.5*(2*_Gs - 2*pars['DGs']) - pars['Gd']

    if _has_pwave:
      _pars_odd = ipanema.Parameters.clone(original_pars)
      _pars_even = ipanema.Parameters.clone(original_pars)
      for k, v in pars.items():
        if k == 'fPlon':
          _fper = pars['fPper']
          _flon = pars['fPlon']
          _fpar = 1 - _fper - _flon
          _sum = _fpar + _flon
          _pars_even['fPlon'].value /= _sum
          _pars_even['fPper'].value = 0
        elif k == 'fPper':
          _pars_odd['fPlon'].value = 0
          _pars_odd['fPper'].value = 1
          _Gs = pars['DGsd'] + pars['Gd']
        else:
          _pars_even[k].value = pars[k]
          _pars_odd[k].value = pars[k]
        if k.startswith('fS'):
          _pars_odd[k].value = 0
          _pars_even[k].value = 0
      # _pars_odd['DGsd'].value = 0.5*(2*_Gs - pars['DGs']) - pars['Gd']
      # _pars_even['DGsd'].value = 0.5*(2*_Gs + pars['DGs']) - pars['Gd']
      all_pars.append([_pars_even.valuesdict(), (aprox_pwave * 0.75), "CP-even"])
      all_pars.append([_pars_odd.valuesdict(), (aprox_pwave * 0.25), "CP-odd"])

  # all_samples = [
  #     f"/scratch49/marcos.romero/sidecar/{y}/Bs2JpsiPhi/{VERSION}.root" for y in YEARS]
  # all_csp = [
  #     f"output/params/csp_factors/{y}/Bs2JpsiPhi/{VERSION}_vgc.json" for y in YEARS]
  # all_flavor = [
  #     f"output/params/flavor_tagging/{y}/Bs2JpsiPhi/{VERSION}_amsrd.json" for y in YEARS]
  # all_timeacc = {
  #     'biased': [f"output/params/time_acceptance/{y}/Bd2JpsiKstar/{VERSION}_{TIMEACC}_biased.json" for y in YEARS],
  #     'unbiased': [f"output/params/time_acceptance/{y}/Bd2JpsiKstar/{VERSION}_{TIMEACC}_unbiased.json" for y in YEARS]
  # }
  # all_angacc = {
  #     'biased': [f"output/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_biased.json" for y in YEARS],
  #     'unbiased': [f"output/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_unbiased.json" for y in YEARS]
  # }
  all_samples = args['samples'].split(',')
  all_csp = args['csp'].split(',')
  all_timeacc = {
      'biased': args['timeacc_biased'].split(','),
      'unbiased': args['timeacc_unbiased'].split(',')
  }
  all_angacc = {
      'biased': args['angacc_biased'].split(','),
      'unbiased': args['angacc_unbiased'].split(',')
  }
  all_flavor = args['flavor_tagging'].split(',')

  # # p.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--csp', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  # # print(all_timeacc)

  samples = sample_loader(
      Bs2JpsiPhi=all_samples,
      trigger=triggers,
  )
  print(samples)

  # timeacc = Parameters.load("output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul3_biased.json")
  # angacc = Parameters.load("output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_correctedDual_none_unbiased.json").valuesdict()
  # pars = Parameters.load("analysis/params/generator/2016/MC_Bs2JpsiPhi_dG0.json").valuesdict()
  # all_samples = [
  #      "/scratch46/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi_dG0/v0r5.root",
  #      "/scratch46/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5.root",
  #      "/scratch46/marcos.romero/sidecar/2017/MC_Bs2JpsiPhi_dG0/v0r5.root",
  #      "/scratch46/marcos.romero/sidecar/2018/MC_Bs2JpsiPhi_dG0/v0r5.root"
  # ]
  # samples = sample_loader(
  #     MC_Bs2JpsiPhi_dG0=all_samples,
  #     trigger=triggers,
  # )

  # since now we have all samples in a good structure,
  # let's attach parameters to them
  average_resolution = []
  for km, vm in samples.items():
    iy = 0
    for ky, vy in vm.items():
      print(ky)
      _csp = Parameters.load(all_csp[iy])
      _csp = Parameters.build(_csp, _csp.find('CSP.*'))
      _tag = Parameters.load(all_flavor[iy])
      for kt, vt in vy.items():
        _ta = Parameters.load(all_timeacc[kt][iy])
        tLL = _ta['tLL'].value
        tUL = _ta['tUL'].value
        _knots = np.array(Parameters.build(_ta, _ta.find("k.*"))).tolist()
        _ta = Parameters.build(_ta, _ta.find('(a|b|c).*'))
        _aa = Parameters.load(all_angacc[kt][iy])
        _aa = Parameters.build(_aa, _aa.find('w.*'))
        # attaching
        vt.csp = _csp
        vt.timeacc = _ta
        vt.angacc = _aa
        vt.tagging = _tag
        vt.chop(f"time>={tLL} & time<{tUL}")
        print(f"Found csp     {km}::{ky}::{kt}  =  {np.array(vt.csp)}")
        print(f"Found timeacc {km}::{ky}::{kt}  =  {np.array(vt.timeacc)}")
        print(f"Found angacc  {km}::{ky}::{kt}  =  {np.array(vt.angacc)}")
        print(f"Found tagging {km}::{ky}::{kt}  =  {np.array(vt.tagging)}")
      iy += 1  # increase year number by one

  # TODO: It is very important to properly configure badjanak.
  #       I need to prepare some piece of code to do it automatically.
  # print(_knots)
  badjanak.config['knots'] = _knots
  print(_knots, tLL, tUL)
  # badjanak.config['knots'][0] = 0.5
  # badjanak.config['knots'] = [0.5, 0.91, 1.35, 1.96, 3.01, 7]
  badjanak.config['debug'] = 0
  badjanak.config['debug_evt'] = 0
  badjanak.config['fast_integral'] = 0
  badjanak.config['final_extrap'] = 1
  badjanak.get_kernels()
  # print(samples)

  # if TRIGGER == 'unbiased':
  ####     df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b==0 & time>0.3")
  # else:
  ####     df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b!=0 & time>0.3")
  #### dtime = np.array(df['time'])
  #### dcosL = np.array(df['cosL'])
  #### dcosK = np.array(df['cosK'])
  #### dhphi = np.array(df['hphi'])
  ####
  # timeres = Parameters.load("output/params/time_resolution/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
  #### csp = Parameters.load(f"output/params/csp_factors/{YEAR}/Bs2JpsiPhi/{VERSION}_vgc.json").valuesdict()
  # flavor = Parameters.load("output/params/flavor_tagging/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
  # angacc = Parameters.load("output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_correctedDual_none_unbiased.json").valuesdict()
  #### angacc = Parameters.load(f"output/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_simul3_amsrd_{TRIGGER}.json").valuesdict()
  # timeacc = Parameters.load("output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul3_biased.json")
  #### timeacc = Parameters.load(f"output/params/time_acceptance/{YEAR}/Bd2JpsiKstar/{VERSION}_simul3_{TRIGGER}.json")
  #### knots = np.array(Parameters.build(timeacc, timeacc.find("k.*"))).tolist()
  #### timeacc = Parameters.build(timeacc, timeacc.find("(a|c|b)(A|B)?.*")).valuesdict()

  def pdf_projector(params, edges, var='time', timeacc=False, angacc=False,
                    return_center=False, avgres=0.0042, csp=False):
    # create flags for acceptances
    use_timeacc = True if timeacc else False
    use_angacc = True if angacc else False
    # use_angacc = False
    # use_timeacc = False
    # print(use_angacc, use_timeacc)

    # defaults
    timeLL = 0.3
    timeUL = 15
    cosKLL = -1
    cosKUL = 1
    cosLLL = -1
    cosLUL = 1
    hphiLL = -np.pi
    hphiUL = +np.pi
    acc = 1.
    _x = 0.5 * (edges[:-1] + edges[1:])

    @np.vectorize
    def prob(pars, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL, timeLL, timeUL, avgres):
      var = np.float64([0.2] * 3 + [1] + [1020.] + [0.045] + [0.0] * 4)  # que el monchito se lo explique
      var = ristra.allocate(np.ascontiguousarray(var))
      pdf = ristra.allocate(np.float64([0.0]))
      __timeacc = timeacc if timeacc else {}
      __csp = csp if csp else {}
      badjanak.delta_gamma5_mc(var, pdf, **pars, tLL=tLL, tUL=tUL, **__csp)
      # exit()
      num = pdf.get()
      # __timeacc = {}
      badjanak.delta_gamma5_mc(var, pdf, **pars, cosKLL=cosKLL, cosKUL=cosKUL,
                               cosLLL=cosLLL, cosLUL=cosLUL, hphiLL=hphiLL,
                               hphiUL=hphiUL, tLL=timeLL, tUL=timeUL, **__csp, **__timeacc)
      den = pdf.get()
      return num / den

    @np.vectorize
    def vtimeacc(x):
      if use_timeacc:
        return badjanak.bspline(np.float64([x]), [v for v in timeacc.values()])
      else:
        return 1

    @np.vectorize
    def vangacc(x, proj):
      if use_angacc:
        _x = np.linspace(-1, 1, 50)
        __x, __y, __z = ristra.ndmesh(_x, _x, np.pi * _x)
        __x = ristra.allocate(__x.reshape(len(_x)**3))
        __y = ristra.allocate(__y.reshape(len(_x)**3))
        __z = ristra.allocate(__z.reshape(len(_x)**3))
        _arr = [__x, __y, __z]
        _arr[proj - 1] *= x / _arr[proj - 1]
        _ans = 1 / \
            badjanak.angular_efficiency_weights(
                [v for v in angacc.values()], *_arr, proj)
        return np.mean(_ans)
      else:
        return 1

    if var == 'time':
      timeLL, timeUL = edges[:-1], edges[1:]
      acc = vtimeacc(_x)
    elif var == 'cosL':
      cosLLL, cosLUL = edges[:-1], edges[1:]
      acc = vangacc(_x, 1)
    elif var == 'cosK':
      cosKLL, cosKUL = edges[:-1], edges[1:]
      acc = vangacc(_x, 2)
    elif var == 'hphi':
      hphiLL, hphiUL = edges[:-1], edges[1:]
      acc = vangacc(_x, 3)
    else:
      raise ValueError(f"The pdf is not {var} dependent")

    # print(acc, _x)
    # if angacc or timeacc:
    acc /= np.trapz(acc, _x)
    # acc = 1
    _pdf = prob(params, cosKLL, cosKUL, cosLLL, cosLUL,
                hphiLL, hphiUL, timeLL, timeUL, avgres=avgres)
    _pdf /= np.trapz(_pdf, _x)
    _pdf *= acc
    _pdf /= np.trapz(_pdf, _x)
    # exit()
    if return_center:
      return _pdf, _x
    return _pdf

  weight = 'sWeight'
  for branch in ['time', 'cosK', 'cosL', 'hphi']:
    # basic labels for {{{
    if branch == 'time':
      tLL = max(0.4, tLL)
      var = np.linspace(tLL, tUL, 50)
      edges = np.linspace(tLL, tUL, 51)
      xlabel = "$t$ [ps]"
    elif branch.startswith('cos'):
      var = np.linspace(-1, 1, 50)
      edges = np.linspace(-1, 1, 51)
      xlabel = r"$\cos \theta_K$" if 'K' in branch else r"$\cos \theta_{\mu}$"
    elif branch == 'hphi':
      var = np.linspace(-np.pi, np.pi, 50)
      edges = np.linspace(-np.pi, np.pi, 51)
      xlabel = r"$\phi_h$ [rad]"
    else:
      raise ValueError("not expedted branch")
    # }}}
    print(edges)

    # start plotting routine
    fig, axplot, axpull = complot.axes_plotpull()
    all_pdf_norm = 1

    for c, self_pars in enumerate(all_pars):
      hvar = []  # where to store counts
      hbin = []  # where to store the bining
      hyerr = []  # where to store the y errors
      hxerr = []  # where to store the x errors
      pdf_y = []
      pdf_x = []
      pars, _frac, _label = self_pars
      # print(f"pars {c}")
      # print(pars)
      for km, vm in samples.items():
        for ky, vy in vm.items():
          for kt, vt in vy.items():
            _weight = vt.df[weight].values
            # _weight *= np.sum(_weight) / np.sum(_weight**2)
            _hvar = complot.hist(vt.df[branch].values, bins=edges,
                                 weights=_weight)
            _pdfvar, _var = pdf_projector(
                pars, var, branch,  # timeacc=False,
                timeacc=vt.timeacc.valuesdict() if branch == 'time' else False,
                angacc=vt.angacc.valuesdict() if branch != 'time' else False,
                csp=vt.csp.valuesdict(),
                return_center=True, avgres=0.045)
            hvar.append(_hvar.counts)
            hbin.append(_hvar.bins)
            hyerr.append(np.nan_to_num(_hvar.yerr))
            hxerr.append(np.nan_to_num(_hvar.xerr))
            pdf_x.append(_var)
            _norm = np.trapz(_hvar.counts, _hvar.bins) / np.trapz(_pdfvar, _var)
            pdf_y.append(_pdfvar * _norm)

      # sum all contributions
      all_hvar = np.sum(hvar, 0)
      all_yerr = [np.sqrt(np.sum([r[0]**2 for r in hyerr], 0)),
                  np.sqrt(np.sum([r[1]**2 for r in hyerr], 0))]
      all_pdf_y = np.sum(pdf_y, 0)
      # all_yerr = [np.sqrt(all_hvar), np.sqrt(all_hvar)]
      # print(all_yerr)
      # print(pdf_y)
      # all_pdf_y *= np.trapz(all_hvar, hbin[0]) / np.trapz(all_pdf_y, pdf_x[0])
      # print(all_pdf_y)
      # print("x = ", pdf_x[0])
      # print("pdf norm = ", np.trapz(all_pdf_y, pdf_x[0]))
      if c == 0:
        all_pdf_norm = np.trapz(all_pdf_y, pdf_x[0])
      else:
        all_pdf_y *= all_pdf_norm / np.trapz(all_pdf_y, pdf_x[0])
        all_pdf_y *= _frac
        # _norm = np.trapz(all_pdf_y, pdf_x[0])
      axplot.plot(pdf_x[0], all_pdf_y, linestyle='-', color=f"C{c}",
                  label=_label)
      if c == 0:
        _pull = complot.compute_pdfpulls(pdf_x[0], all_pdf_y, hbin[0], all_hvar, *all_yerr)
        axpull.fill_between(hbin[0], _pull, 0, facecolor="C0", alpha=0.5)
        axplot.errorbar(hbin[0], all_hvar, yerr=all_yerr, xerr=hxerr[0], fmt='.k')
        axplot.set_ylim(0, np.max(all_pdf_y))

    if branch == "time":
      axplot.set_yscale('log')
      try:
        axplot.set_ylim(1e0, 1.5 * np.max(all_pdf_y))
      except:
        print("axes not scaled")

    axpull.set_xlabel(xlabel)
    axplot.set_ylabel("Weighted candidates")

    _watermark = version_guesser(VERSION)[0]  # watermark plots
    if 'time' in branch:  # this y-axes will be log
      watermark(axplot, version=rf"${_watermark}$", scale=10.01)
      axplot.legend()
    else:
      watermark(axplot, version=rf"${_watermark}$", scale=1.3)

    os.makedirs(args['figures'], exist_ok=True)
    plt.savefig(os.path.join(args['figures'], f"{branch}.pdf"))
    # plt.show()


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
