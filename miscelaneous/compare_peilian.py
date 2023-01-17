import pandas as pd
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

import config
from utils.helpers import version_guesser, timeacc_guesser, trigger_scissors
from utils.strings import cuts_and, printsec, printsubsec
# from utils.plot import mode_tex
from analysis import badjanak
from ipanema import initialize, Sample, Parameters, ristra, optimize
import complot

import argparse
import numpy as np
# import uproot
import os
import uncertainties as unc
import hjson
import json
import uproot3 as uproot

import matplotlib.pyplot as plt
initialize('cuda', 1)
# pd.options.mode.chained_assignment = None

# }}}


# some general configration for the badjanak kernel
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 0
badjanak.config['debug_evt'] = 774


if __name__ == "__main__":
  DESCRIPTION = """
  Fit data
  """
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  tLL = 0.3
  tUL = 15.0
  YEARS = ['2015', '2016', '2017', '2018']
  # YEARS = ['2016']
  TRIGGER = ['biased', 'unbiased']

  # Prepare the cuts
  CUT = cuts_and("", f'time>={tLL} & time<={tUL}')
  print(CUT)

  # print("INPUTS")
  # for k, v in args.items():
  #   print(f'{k}: {v}\n')

  # Load samples {{{

  printsubsec("Loading samples")

  branches_to_load = ['hlt1b']
  # Lists of data variables to load and build arrays
  real = ['helcosthetaK', 'helcosthetaL', 'helphi', 'time', 'X_M', 'sigmat']
  real += ['OS_Combination_DEC',
           'B_SSKaonLatest_TAGDEC',
           'OS_Combination_ETA',
           'B_SSKaonLatest_TAGETA']
  weight = 'sw'
  # weight = 'sw'
  branches_to_load += real
  branches_to_load += [weight]

  samples = {}

  inputs = {
      # str(y): hjson.load(open(f"fit_inputs_{y}.json", 'r')) for y in [2015, 2016, 2017, 2018]
      str(y): hjson.load(open(f"fit_inputs_{y}.json", 'r')) for y in YEARS
  }
  csp = {}
  flavor = {}
  timeres = {y: Parameters.load("analysis/params/time_resolution/Bs2JpsiPhi/none.json") for y in inputs}
  timeacc = {}
  angacc = {}

  for y in inputs.keys():
    # ntuple EOS paht {{{
    samples[y] = inputs[y]['ntuple'].replace("_selected_bdt", '')
    # }}}
    #
    # CSP factors {{{
    raw_json = inputs[y]['CspFactors']
    cooked = {}
    for i, d in enumerate(raw_json):
      bin = i + 1
      cooked[f'CSP{bin}'] = {'name': f'CSP{bin}'}
      cooked[f'CSP{bin}'].update({'value': d['Value'], 'stdev': d['Error']})
      cooked[f'CSP{bin}'].update({'latex': f"C_{{SP}}^{{{bin}}}"})
      cooked[f'CSP{bin}'].update({'free': False})
      if not f'mKK{bin-1}' in cooked:
        cooked[f'mKK{bin-1}'] = {'name': f'mKK{bin-1}'}
        cooked[f'mKK{bin-1}'].update({'value': d['Bin_ll'], 'stdev': 0})
        cooked[f'mKK{bin-1}'].update({'latex': f'm_{{KK}}^{{{bin-1}}}'})
        cooked[f'mKK{bin-1}'].update({'free': False})
      if not f'mKK{bin}' in cooked:
        cooked[f'mKK{bin}'] = {'name': f'mKK{bin}'}
        cooked[f'mKK{bin}'].update({'value': d['Bin_ul'], 'stdev': 0})
        cooked[f'mKK{bin}'].update({'latex': f'm_{{KK}}^{{{bin}}}'})
        cooked[f'mKK{bin}'].update({'free': False})
    list_params = list(cooked.keys())
    list_params = sorted(list_params)
    _csp = Parameters()
    [_csp.add(cooked[par]) for par in list_params]
    csp[y] = _csp
    # }}}
    #
    # Time resolution {{{
    rawd = inputs[y]['TimeResParameters']
    for i, par in enumerate(list(timeres[y].keys())):
      what = par.split('_')[-1]
      timeres[y][par].correl = {}
      for ii, _ in enumerate(list(rawd)):
        if rawd[ii]['Name'] == 'mu':
          timeres[y][par].set(value=rawd[ii]['Value'], stdev=rawd[ii]['Error'])
        elif rawd[ii]['Name'] == f'p{i-1}' and f'sigma_{what}' == par:
          timeres[y][par].set(value=rawd[ii]['Value'], stdev=rawd[ii]['Error'])
          for j, rap in enumerate(list(timeres[y].keys())):
            for k in range(len(rawd)):
              if rawd[k]['Name'] == f'rho_p{i-1}_p{j-1}_time_res':
                timeres[y][par].correl[rap] = rawd[k]['Value']
              elif rawd[k]['Name'] == f'rho_p{j-1}_p{i-1}_time_res':
                timeres[y][par].correl[rap] = rawd[k]['Value']
              else:
                timeres[y][par].correl[rap] = 1 if i == j else 0
        else:
          0
    timeres[y]['mu'].value = inputs[y]['TimeBias']['Value']
    timeres[y]['mu'].stdev = inputs[y]['TimeBias']['Error']
    # }}}
    #
    # Tagging {{{
    rawd = inputs[y]['TaggingParameter']
    outd = Parameters.load("analysis/params/flavor_tagging/Bs2JpsiPhi/none.json")

    for i, par in enumerate(list(outd.keys())):
      for _i in range(len(rawd)):
        if rawd[_i]['Name'][:5] == f'{par[:2]}_' + f'{par[3:]}'.upper():
          outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
          outd[par].correl = {}
          for j in range(0, 3):
            pi = par[:2]
            pj = f"p{j}"
            tag = par[3:]
            for k in range(len(rawd)):
              if rawd[k]['Name'][:12].lower() == f'rho_{pi}_{pj}_{tag}':
                outd[par].correl[f"{pj}_{tag}"] = rawd[k]['Value']
              elif rawd[k]['Name'][:12].lower() == f'rho_{pj}_{pi}_{tag}':
                outd[par].correl[f"{pj}_{tag}"] = rawd[k]['Value']
        elif rawd[_i]['Name'][:6] == f'{par[:3]}_' + f'{par[4:]}'.upper():
          outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
        elif rawd[_i]['Name'][:10] == f'{par[:3]}_bar_' + f'{par[4:]}'.upper():
          outd[par].set(value=rawd[_i]['Value'], stdev=rawd[_i]['Error'])
    list_params = list(outd.keys())                 # list all parameter names
    list_params = sorted(list_params)               # sort them
    _flavor = Parameters()
    [_flavor.add(outd[par]) for par in list_params]
    flavor[y] = Parameters()
    yt = y[2:] if y[2:] != '15' else '16'
    for k, v in _flavor.items():
      flavor[y].add(dict(name=f'{k}{yt}', value=v.value, stdev=v.stdev))
    for k, v in _flavor.items():
      # _correl = v.correl if v.correl else {}
      _correl = v.correl
      _correl = {f"{kk}{yt}": v for kk, v in _correl.items()} if _correl else {}
      flavor[y][f'{k}{yt}'].correl = _correl

    # Time acceptance
    timeacc[y] = {}
    _biased = []
    _unbiased = []
    for i in inputs[y]['TimeAccParameter']['SplineUnbiased']:
      _unbiased.append(i['Value'])
    for i in inputs[y]['TimeAccParameter']['SplineBiased']:
      _biased.append(i['Value'])
    _btimeacc = Parameters()
    _utimeacc = Parameters()
    for i in range(0, len(_biased)):
      _btimeacc.add(dict(name=f"c{i}b{y[2:]}", value=_biased[i]))
      _utimeacc.add(dict(name=f"c{i}u{y[2:]}", value=_unbiased[i]))
    timeacc[y]['biased'] = _btimeacc
    timeacc[y]['unbiased'] = _utimeacc

    # Angular acceptance
    angacc[y] = {}
    _biased = []
    _unbiased = []
    for i in inputs[y]['AngularParameterUnbiased']:
      _unbiased.append(i['Value'])
    for i in inputs[y]['AngularParameterBiased']:
      _biased.append(i['Value'])
    _bangacc = Parameters()
    _uangacc = Parameters()
    for i in range(0, len(_biased)):
      _bangacc.add(dict(name=f"w{i}b{y[2:]}", value=_biased[i]))
      _uangacc.add(dict(name=f"w{i}u{y[2:]}", value=_unbiased[i]))
    angacc[y]['biased'] = _bangacc
    angacc[y]['unbiased'] = _uangacc

  # print("Printing parsed information")
  # for year in samples.keys():
  #   print("\n\n")
  #   print(80 * "=")
  #   print(year)
  #   print(samples[year])
  #   print("Time acceptance")
  #   print(timeacc[year]['biased'])
  #   print(timeacc[year]['unbiased'])
  #   print("Angular acceptance")
  #   print(angacc[year]['biased'])
  #   print(angacc[year]['unbiased'])
  #   print("Time resolution")
  #   print(timeres[year])
  #   print(timeres[year].corr())
  #   print("Flavor tagging")
  #   print(flavor[year])
  #   print(flavor[year].corr())
  #   print("CSP factors")
  #   print(csp[year])

  mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  knots = np.array([0.3, 0.91, 1.96, 9])
  badjanak.config['knots'] = knots.tolist()
  data = {}
  for i, y in enumerate(list(samples.keys())):
    print(f'Fetching elements for {y}[{i}] data sample')
    data[y] = {}
    mass = np.array(csp[y].build(csp[y], csp[y].find('mKK.*')))
    badjanak.config['mHH'] = mass.tolist()
    for t in ['biased', 'unbiased']:
      print("\n\n")
      print(80 * "=")
      tc = trigger_scissors(t, CUT)
      print(f"{y}-{y} SAMPLE", tc)
      print(80 * "=")
      # pd_df = uproot.open(samples[y])['DecayTree'].pandas.df(flatten=None)
      # pd_df = uproot.open(f"{y}.root")['DecayTree'].pandas.df(flatten=None)
      # data[y][t] = Sample.from_pandas(pd_df, cuts=tc)
      data[y][t] = Sample.from_root(f"{y}.root", cuts=tc, flatten=None)
      data[y][t].name = f"Bs2JpsiPhi-{y}-{t}"
      print(data[y][t])
      # print("sum Lera:", np.sum(data[y][t].df['sw'].values))
      # print("sum me:", np.sum(data[y][t].df['sWeight'].values))
      data[y][t].csp = csp[y].build(csp[y], csp[y].find('CSP.*'))
      print(data[y][t].csp)
      data[y][t].flavor = flavor[y]
      print(data[y][t].flavor)
      data[y][t].resolution = timeres[y]
      print(data[y][t].resolution)
      # Time acceptance
      c = Parameters.clone(timeacc[y][t])
      data[y][t].timeacc = Parameters.build(c, c.fetch('(a|b|c).*'))
      print(data[y][t].timeacc)
      # Angular acceptance
      w = Parameters.clone(angacc[y][t])
      data[y][t].angacc = Parameters.build(w, w.fetch('w.*'))
      print(data[y][t].angacc)
      # Normalize sWeights per bin
      sw = np.zeros_like(data[y][t].df['time'])
      for ml, mh in zip(mass[:-1], mass[1:]):
        # for tl, th in zip([0.3, 0.92, 1.97], [0.92, 1.97, 15]):
        # sw_cut = f'mHH>={ml} & mHH<{mh} & time>={tl} &  time<{th}'
        sw_cut = f'X_M>={ml} & X_M<={mh}'
        pos = data[y][t].df.eval(sw_cut)
        _sw = data[y][t].df.eval(f'{weight}*({sw_cut})')
        print(np.sum(_sw))
        sw = np.where(pos, _sw * (sum(_sw) / sum(_sw * _sw)), sw)
      print(data[y][t].df.shape)
      data[y][t].df['sWeightCorr'] = sw
      print(data[y][t].df)
      data[y][t].allocate(data=real, weight='sWeightCorr', prob='0*time')

  # }}}
  # exit()

  # Compile the kernel
  #    so if knots change when importing parameters, the kernel is compiled
  # badjanak.config["precision"]='single'
  BLIND = True
  SWAVE = True
  POLDEP = False
  DGZERO = False
  CONSTR = True

  # Prepare parameters {{{

  mass_knots = badjanak.config['mHH']
  pars = Parameters()

  # S wave fractions
  for i in range(len(mass_knots) - 1):
    pars.add(dict(
        name=f'fSlon{i+1}', value=SWAVE * 0.2, min=0.00, max=0.90,
        free=SWAVE, latex=rf'|A_S^{{{i+1}}}|^2'))

  # P wave fractions
  pars.add(dict(name="fPlon", value=0.5241, min=0.4, max=0.6,
                free=True, latex=r'|A_0|^2'))
  pars.add(dict(name="fPper", value=0.25, min=0.1, max=0.3,
                free=True, latex=r'|A_{\perp}|^2'))

  # Weak phases
  if not POLDEP:
    pars.add(dict(name="pSlon", value=0.00, min=-5.0, max=5.0,
                  free=POLDEP, latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}",
                  blindstr="BsPhisSDelFullRun2",
                  blind=BLIND, blindscale=2., blindengine="root"))
    pars.add(dict(name="pPlon", value=0.3, min=-5.0, max=5.0,
                  free=True, latex=r"\phi_0 \, \mathrm{[rad]}",
                  blindstr="BsPhiszeroFullRun2" if POLDEP else "BsPhisFullRun2",
                  blind=BLIND, blindscale=2 if POLDEP else 1, blindengine="root"))
    pars.add(dict(name="pPpar", value=0.00, min=-5.0, max=5.0,
                  free=POLDEP, latex=r"\phi_{\parallel} - \phi_0 \, \mathrm{[rad]}",
                  blindstr="BsPhisparaDelFullRun2",
                  blind=BLIND, blindscale=2., blindengine="root"))
    pars.add(dict(name="pPper", value=0.00, min=-5.0, max=5.0,
                  free=POLDEP, blindstr="BsPhisperpDelFullRun2", blind=BLIND,
                  blindscale=2.0, blindengine="root",
                  latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))
  else:
    pars.add(dict(name="pAvg", value=0.00,  # min=-5.0, max=5.0,
             free=True,
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="pDiff", value=0.00,  # min=-5.0, max=5.0,
             free=True,
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    #
    pars.add(dict(name="pSlon", value=0.00,  # min=-5.0, max=5.0,
             free=True,
             blindstr="BsPhisSDelFullRun2",
             blind=BLIND, blindscale=2, blindengine="root",
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="pPpar", value=0.00,  # min=-5.0, max=5.0,
             free=True,
             blindstr="BsPhisparaDelFullRun2",
             blind=BLIND, blindscale=2, blindengine="root",
             latex=r"\phi_{\parallel} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="pPlon",
             formula="pAvg+pDiff-0.5*pSlon",
             blindstr="BsPhiszeroFullRun2",
             blind=BLIND, blindscale=2, blindengine="root",
             latex=r"\phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="pPper", value=0.00,  # min=-5.0, max=5.0,
             formula="2*pPlon+pPpar-2*pAvg",
             blindstr="BsPhisperpDelFullRun2",
             blind=BLIND, blindscale=2, blindengine="root",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))

  # S wave strong phases
  for i in range(len(mass_knots) - 1):
    phase = np.linspace(2.3, -1.2, len(mass_knots) - 1)[i]
    pars.add(dict(
        name=f'dSlon{i+1}', value=SWAVE * phase,
        min=0 if 2 * i < (len(mass_knots) - 1) else -4,
        max=4 if 2 * i < (len(mass_knots) - 1) else 0,
        free=SWAVE,
        latex=rf"\delta_S^{{{i+1}}} - \delta_{{\perp}} \, \mathrm{{[rad]}}"))

  # P wave strong phases
  pars.add(dict(name="dPlon", value=0.00, min=-2 * 3.14 * 0, max=2 * 3.14,
                free=False,
                latex=r"\delta_0 \, \mathrm{[rad]}"))
  pars.add(dict(name="dPpar", value=3.26, min=-2 * 3.14 * 0, max=1.5 * 3.14,
                free=True,
                latex=r"\delta_{\parallel} - \delta_0 \, \mathrm{[rad]}"))
  pars.add(dict(name="dPper", value=3.1, min=-2 * 3.14 * 0, max=1.5 * 3.14,
                free=True,
                latex=r"\delta_{\perp} - \delta_0 \, \mathrm{[rad]}"))

  # lambdas
  if not POLDEP:
    pars.add(dict(name="lSlon", value=1., min=0.4, max=1.6,
                  free=POLDEP,
                  latex=r"|\lambda_S|/|\lambda_0|"))
    pars.add(dict(name="lPlon", value=1., min=0.4, max=1.6,
                  free=True,
                  latex=r"|\lambda_0|"))
    pars.add(dict(name="lPpar", value=1., min=0.4, max=1.6,
                  free=POLDEP,
                  latex=r"|\lambda_{\parallel}|/|\lambda_0|"))
    pars.add(dict(name="lPper", value=1., min=0.4, max=1.6,
                  free=POLDEP,
                  latex=r"|\lambda_{\perp}|/|\lambda_0|"))
  else:
    pars.add(dict(name="CAvg", value=0.00,
             free=True,
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="CDiff", value=0.00,
             free=True,
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="CSlon", value=0.00,
             free=True,
             latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="CPpar", value=0.00,
             free=True,
             latex=r"\phi_{\parallel} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="CPlon",
             formula="CAvg+CDiff-0.5*CSlon",
             latex=r"\phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="CPper", value=0.00,
             formula="2*CPlon+CPpar-2*CAvg",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="lPlon", value=0.00,
             formula="sqrt((1-CPlon)/(1+CPlon))",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="lSlon", value=0.00,
             formula="sqrt((1-CSlon)/(1+CSlon))/lPlon",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="lPpar", value=0.00,
             formula="sqrt((1-CPpar)/(1+CPpar))/lPlon",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))
    pars.add(dict(name="lPper", value=0.00,
             formula="sqrt((1-CPper)/(1+CPper))/lPlon",
             latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}"))

  # lifetime parameters
  # pars.add(dict(name="Gd", value=0.65789, min=0.0, max=1.0,
  gamma_ref = 0.65789
  pars.add(dict(name="Gd",
                value=gamma_ref, min=0.0, max=1.0,
                free=False,
                latex=r"\Gamma_d \, \mathrm{[ps]}^{-1}"))
  pars.add(dict(name="DGs", value=(1 - DGZERO) * 0.3, min=0.0, max=1.7,
                free=1 - DGZERO,
                latex=r"\Delta\Gamma_s \, \mathrm{[ps^{-1}]}",
                blindstr="BsDGsFullRun2",
                blind=BLIND, blindscale=1.0, blindengine="root"))
  pars.add(dict(name="DGsd", value=0.03 * 0, min=-0.1, max=0.1,
                free=True,
                latex=r"\Gamma_s - \Gamma_d \, \mathrm{[ps^{-1}]}"))
  pars.add(dict(name="DM", value=17.757, min=15.0, max=20.0,
                free=True,
                latex=r"\Delta m_s \, \mathrm{[ps^{-1}]}"))

  # tagging
  # pars.add(dict(name="eta_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'eta_os{YEARS[0][2:]}'].value,
  #               free=False))
  # pars.add(dict(name="eta_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'eta_ss{YEARS[0][2:]}'].value,
  #               free=False))
  # pars.add(dict(name="p0_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'p0_os{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=0.0, max=1.0,
  #               latex=r"p^{\rm OS}_{0}"))
  # pars.add(dict(name="p1_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'p1_os{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=0.5, max=1.5,
  #               latex=r"p^{\rm OS}_{1}"))
  # pars.add(dict(name="p0_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'p0_ss{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=0.0, max=2.0,
  #               latex=r"p^{\rm SS}_{0}"))
  # pars.add(dict(name="p1_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'p1_ss{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=0.0, max=2.0,
  #               latex=r"p^{\rm SS}_{1}"))
  # pars.add(dict(name="dp0_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'dp0_os{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm OS}_{0}"))
  # pars.add(dict(name="dp1_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'dp1_os{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm OS}_{1}"))
  # pars.add(dict(name="dp0_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'dp0_ss{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm SS}_{0}"))
  # pars.add(dict(name="dp1_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor[f'dp1_ss{YEARS[0][2:]}'].value,
  #               free=CONSTR, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm SS}_{1}"))

  for year in list(samples.keys()):
    if CONSTR:
      pars += data[year][TRIGGER[0]].flavor
    else:
      data[year][TRIGGER[0]].flavor.lock()
      pars += data[year][TRIGGER[0]].flavor
  for p in pars:
    if p.startswith('eta'):
      print(p)
      pars.lock(p)
  print(pars)

  os_corr = np.array([[1.00, 0.03, 0.14, -0.00, 0.29, 0.00, -0.00, -0.00, 0.00, 0.00, 0.00, 0.00],
                      [0.03, 1.00, -0.00, 0.64, -0.00, 0.35, -0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                      [0.14, -0.00, 1.00, 0.05, 0.09, -0.00, -0.00, -0.00, 0.00, 0.00, 0.00, 0.00],
                      [-0.00, 0.64, 0.05, 1.00, -0.00, 0.28, -0.00, 0.00, -0.00, -0.00, 0.00, -0.00],
                      [0.29, -0.00, 0.09, -0.00, 1.00, 0.05, 0.00, -0.00, 0.00, 0.00, -0.00, 0.00],
                      [0.00, 0.35, -0.00, 0.28, 0.05, 1.00, 0.00, 0.00, -0.00, -0.00, -0.00, -0.00],
                      [-0.00, -0.00, -0.00, -0.00, 0.00, 0.00, 1.00, 0.07, 0.00, 0.00, -0.00, 0.00],
                      [-0.00, 0.00, -0.00, 0.00, -0.00, 0.00, 0.07, 1.00, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.00, -0.00, 0.00, -0.00, 0.00, 0.00, 1.00, 0.15, -0.00, -0.00],
                      [0.00, 0.00, 0.00, -0.00, 0.00, -0.00, 0.00, 0.00, 0.15, 1.00, -0.00, -0.00],
                      [0.00, 0.00, 0.00, 0.00, -0.00, -0.00, -0.00, 0.00, -0.00, -0.00, 1.00, 0.13],
                      [0.00, 0.00, 0.00, -0.00, 0.00, -0.00, 0.00, 0.00, -0.00, -0.00, 0.13, 1.00]])
  ss_corr = np.array([
      [1.00, -0.22, 0.05, -0.00, 0.06, -0.00, -0.00, -0.00, -0.00, 0.00, -0.00, -0.00],
      [-0.22, 1.00, 0.00, 0.08, -0.00, 0.07, -0.00, -0.00, 0.00, -0.00, -0.00, -0.00],
      [0.05, 0.00, 1.00, 0.01, 0.05, 0.00, 0.00, 0.00, -0.00, -0.00, 0.00, 0.00],
      [-0.00, 0.08, 0.01, 1.00, -0.00, 0.18, 0.00, 0.00, -0.00, -0.00, 0.00, 0.00],
      [0.06, -0.00, 0.05, -0.00, 1.00, 0.03, -0.00, -0.00, -0.00, 0.00, -0.00, -0.00],
      [-0.00, 0.07, 0.00, 0.18, 0.03, 1.00, -0.00, -0.00, -0.00, 0.00, -0.00, -0.00],
      [-0.00, -0.00, 0.00, 0.00, -0.00, -0.00, 1.00, 0.12, 0.00, -0.00, -0.00, -0.00],
      [-0.00, -0.00, 0.00, 0.00, -0.00, -0.00, 0.12, 1.00, 0.00, 0.00, -0.00, -0.00],
      [-0.00, 0.00, -0.00, -0.00, -0.00, -0.00, 0.00, 0.00, 1.00, 0.13, -0.00, 0.00],
      [0.00, -0.00, -0.00, -0.00, 0.00, 0.00, -0.00, 0.00, 0.13, 1.00, 0.00, 0.00],
      [-0.00, -0.00, 0.00, 0.00, -0.00, -0.00, -0.00, -0.00, -0.00, 0.00, 1.00, 0.11],
      [-0.00, -0.00, 0.00, 0.00, -0.00, -0.00, -0.00, -0.00, 0.00, 0.00, 0.11, 1.00]])
  pars.corr_from_matrix(os_corr,
                        ["p0_os16", "p1_os16", "p0_os17", "p1_os17", "p0_os18",
                         "p1_os18", "dp0_os16", "dp1_os16", "dp0_os17",
                         "dp1_os17", "dp0_os18", "dp1_os18", ])
  pars.corr_from_matrix(ss_corr,
                        ["p0_ss16", "p1_ss16", "p0_ss17", "p1_ss17", "p0_ss18",
                            "p1_ss18", "dp0_ss16", "dp1_ss16", "dp0_ss17",
                            "dp1_ss17", "dp0_ss18", "dp1_ss18", ])
  #
  # for year in list(samples.keys()):
  #   # if int(year) > 2015:
  #   str_year = str(year)
  #   syear = int(str_year[:-2])
  #   pars.add(dict(name=f"etaOS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['eta_os'].value,
  #                 free=False))
  #   pars.add(dict(name=f"etaSS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['eta_ss'].value,
  #                 free=False))
  #   pars.add(dict(name=f"p0OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p0_os'].value,
  #                 free=True, min=0.0, max=1.0,
  #                 latex=r"p^{\rm OS}_{0}"))
  #   pars.add(dict(name=f"p1OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p1_os'].value,
  #                 free=True, min=0.5, max=1.5,
  #                 latex=r"p^{\rm OS}_{1}"))
  #   pars.add(dict(name=f"p0SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p0_ss'].value,
  #                 free=True, min=0.0, max=2.0,
  #                 latex=r"p^{\rm SS}_{0}"))
  #   pars.add(dict(name=f"p1SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p1_ss'].value,
  #                 free=True, min=0.0, max=2.0,
  #                 latex=r"p^{\rm SS}_{1}"))
  #   pars.add(dict(name=f"dp0OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp0_os'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm OS}_{0}"))
  #   pars.add(dict(name=f"dp1OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp1_os'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm OS}_{1}"))
  #   pars.add(dict(name=f"dp0SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp0_ss'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm SS}_{0}"))
  #   pars.add(dict(name=f"dp1SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp1_ss'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm SS}_{1}"))

  # }}}

  print("The following set of parameters")
  print(pars)
  print("is going to be fitted to data with the following experimental")

  # print csp factors
  lb = [data[y][TRIGGER[0]].csp.__str__(
      ['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nCSP factors\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print flavor tagging parameters
  lb = [data[y][TRIGGER[0]].flavor.__str__(
      ['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nFlavor tagging parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print time resolution
  lb = [data[y][TRIGGER[0]].resolution.__str__(
      ['value']).splitlines() for i, y in enumerate(YEARS)]
  print(f"\nResolution parameters\n{80*'='}")
  for l in zip(*lb):
    print(*l, sep="| ")

  # print time acceptances
  for t in TRIGGER:
    lt = [data[y][t].timeacc.__str__(['value']).splitlines()
          for _, y in enumerate(YEARS)]
    print(f"\n{t.title()} time acceptance\n{80*'='}")
    for info in zip(*lt):
      print(*info, sep="| ")

  # print angular acceptance
  for t in TRIGGER:
    lt = [data[y][t].angacc.__str__(['value']).splitlines()
          for _, y in enumerate(YEARS)]
    print(f"\n{t.title()} angular acceptance\n{80*'='}")
    for info in zip(*lt):
      print(*info, sep="| ")
  print("\n")

  # define fcn function {{{

  def fcn_tag_constr_data(parameters: Parameters, data: dict) -> np.ndarray:
    """
    Cost function to fit real data. Data is a dictionary of years, and each
    year should be a dictionary of trigger categories. This function loops
    over years and trigger categories.  Here we are going to unblind the
    parameters to the p.d.f., thats why we call
    parameters.valuesdict(blind=False), by
    default `parameters.valuesdict()` has blind=True.

    Parameters
    ----------
    parameters : `ipanema.Parameters`
    Parameters object with paramaters to be fitted.
    data : dict

    Returns
    -------
    np.ndarray
    Array containing the weighted likelihoods

    """
    tag_names_all = parameters.find(r"\A((d)?p[0-2])(_os|OS|_ss|SS)(\d{2})?\Z")
    # print(tag_names_all)
    # _pars = Parameters.clone(parameters)
    # _pars.remove(*tag_names_all)
    pars_all = parameters.valuesdict(blind=False)
    pars_dict = {k: v for k, v in pars_all.items() if k not in tag_names_all}
    # print(pars_dict)
    # pars_dict = _pars.valuesdict(blind=False)
    # pars_dict = parameters.valuesdict(blind=False)

    chi2 = []
    for y, dy in data.items():
      for _, dt in dy.items():
        yt = y[2:] if y[2:] != "15" else "16"
        tag_names = parameters.find(rf"\A((d)?p[0-2])(_os|OS|_ss|SS)({yt})?\Z")
        _tag = {k: v for k, v in pars_all.items() if k in tag_names}
        # print(_tag)
        # _tag = dt.flavor.valuesdict()
        # if CONSTR:
        #   __tag = {}
        #   for k in tag_names:
        #     __tag[k] = _tag[k]
        #   _tag = __tag
        # print(_tag)
        badjanak.delta_gamma5_data(
            dt.data, dt.prob, **pars_dict,
            **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
            **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
            **_tag,
            tLL=tLL, tUL=tUL,
            use_fk=1, use_angacc=1, use_timeacc=1,
            use_timeoffset=1, set_tagging=1, use_timeres=1,
            BLOCK_SIZE=128
        )
        # exit()
        chi2.append(-2. * ((ristra.log(dt.prob)) * dt.weight).get())
    if CONSTR:
      c = parameters.valuesarray(tag_names)
      c0 = np.matrix(c - dt.flavor.valuesarray(tag_names))
      cov = dt.flavor.cov(tag_names)  # constraint covariance matrix
      cnstr = np.dot(np.dot(c0, np.linalg.inv(cov)), c0.T)
      cnstr += len(c0) * np.log(2 * np.pi) + np.log(np.linalg.det(cov))
      cnstr = np.float64(cnstr[0][0])
      # cnstr = np.float64(cnstr[0][0]) / len(dt.prob)
    else:
      cnstr = 0
      # print("NOT CONSTR")

    chi2conc = np.concatenate(chi2)
    # exit()
    return chi2conc + cnstr / len(chi2conc)

  cost_function = fcn_tag_constr_data

  # }}}

  # Minimization {{{

  print(pars)
  printsubsec("Simultaneous minimization procedure")
  badjanak.get_kernels(True)
  result = optimize(cost_function, method='minuit', params=pars,
                    fcn_kwgs={'data': data},
                    verbose=True, timeit=True, tol=0.05, strategy=1,
                    policy='filter')

  # full print of result
  print(result)
  print(result.params.dump_latex())

  # print some parameters
  for kp, vp in result.params.items():
    if vp.stdev:
      print(f"{kp:>12} : {vp.value:+.6f} ± {vp.stdev:+.6f}")

  lifeDiff = result.params['DGsd'].uvalue
  lifeBu = unc.ufloat(1.638, 0.004)
  lifeBd = unc.ufloat(1.520, 0.004)
  print(f"Bs lifetime: {1/(lifeDiff+1/lifeBd):.2uL}")
  # }}}

  # }}}


# vim: fdm=marker
