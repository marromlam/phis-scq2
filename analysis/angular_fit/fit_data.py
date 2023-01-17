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

import matplotlib.pyplot as plt
initialize(config.user['backend'], 1)

# }}}


# some general configration for the badjanak kernel
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 0
badjanak.config['debug_evt'] = 0

if __name__ == "__main__":
  DESCRIPTION = """
  Fit data
  """
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  # Samples
  parser.add_argument('--samples', help='Bs2JpsiPhi data sample')
  # Input parameters
  parser.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--csp', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  parser.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  # Output parameters
  parser.add_argument('--params', default=False, help='Bs2JpsiPhi MC sample')
  parser.add_argument('--input-params', default=False, help='Bs2JpsiPhi MC sample')
  parser.add_argument('--log-likelihood', default=False, help='Bs2JpsiPhi MC sample')
  # Configuration file
  parser.add_argument('--year', help='Year of data-taking')
  parser.add_argument('--version', help='Year of data-taking')
  parser.add_argument('--fit', help='Year of data-taking')
  parser.add_argument('--angacc', help='Year of data-taking')
  parser.add_argument('--timeacc', help='Year of data-taking')
  parser.add_argument('--trigger', help='Year of data-taking')
  parser.add_argument('--blind', default=1, help='Year of data-taking')
  parser.add_argument('--scan', default=False, help='Year of data-taking')
  args = vars(parser.parse_args())

  if not args['params'] and args['log_likelihood'] and args['input_params']:
    print("Just evaluate likelihood on data and input-params")

  VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEARS = args['year'].split(',')
  TRIGGER = args['trigger']
  MODE = 'Bs2JpsiPhi'
  FIT = args['fit']
  timeacc_config = timeacc_guesser(args['timeacc'])
  timeacc_config['use_upTime'] = timeacc_config['use_upTime'] | ('UT' in args['version'])
  timeacc_config['use_lowTime'] = timeacc_config['use_lowTime'] | ('LT' in args['version'])
  MINER = 'minuit'

  if timeacc_config['use_upTime']:
    tLL = config.general['upper_time_lower_limit']
  else:
    tLL = config.general['time_lower_limit']
  if timeacc_config['use_lowTime']:
    tUL = config.general['lower_time_upper_limit']
  else:
    tUL = config.general['time_upper_limit']
  print(timeacc_config['use_lowTime'], timeacc_config['use_upTime'])

  if 'T1' in args['version']:
    tLL, tUL = tLL, 0.92  # 47
    badjanak.config['final_extrap'] = False
  elif 'T2' in args['version']:
    tLL, tUL = 0.92, 1.97  # 25
    badjanak.config['final_extrap'] = False
  elif 'T3' in args['version']:
    tLL, tUL = 1.97, tUL
    # tLL, tUL = 2, tUL
  else:
    print("SAFE CUT")

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
  real = ['cosK', 'cosL', 'hphi', 'time', 'mHH', 'sigmat']
  real += ['tagOSdec', 'tagSSdec', 'tagOSeta', 'tagSSeta']
  weight = 'sWeight'
  # weight = 'sw'
  branches_to_load += real
  branches_to_load += ['sw', 'sWeight', 'lbWeight', 'mB']

  if timeacc_config['use_veloWeight']:
    weight = f'veloWeight*{weight}'
    branches_to_load += ["veloWeight"]

  if TRIGGER == 'combined':
    TRIGGER = ['biased', 'unbiased']
  else:
    TRIGGER = [TRIGGER]

  data = {}
  for i, y in enumerate(YEARS):
    print(f'Fetching elements for {y}[{i}] data sample')
    data[y] = {}
    csp = Parameters.load(args['csp'].split(',')[i])
    mass = np.array(csp.build(csp, csp.find('mKK.*')))
    csp = csp.build(csp, csp.find('CSP.*'))
    flavor = Parameters.load(args['flavor_tagging'].split(',')[i])
    resolution = Parameters.load(args['time_resolution'].split(',')[i])
    badjanak.config['mHH'] = mass.tolist()
    for t in TRIGGER:
      tc = trigger_scissors(t, CUT)
      data[y][t] = Sample.from_root(args['samples'].split(',')[i],
                                    branches=branches_to_load, cuts=tc)
      # print(data[y][t].df)
      # print("sum Lera:", np.sum(data[y][t].df['sw'].values))
      # print("sum me:", np.sum(data[y][t].df['sWeight'].values))
      data[y][t].name = f"Bs2JpsiPhi-{y}-{t}"
      data[y][t].csp = csp
      data[y][t].flavor = flavor
      data[y][t].resolution = resolution
      # Time acceptance
      c = Parameters.load(args[f'timeacc_{t}'].split(',')[i])
      tLL, tUL = c['tLL'].value, c['tUL'].value
      knots = np.array(Parameters.build(c, c.fetch('k.*')))
      badjanak.config['knots'] = knots.tolist()
      # Angular acceptance
      data[y][t].timeacc = Parameters.build(c, c.fetch('(a|b|c).*'))
      w = Parameters.load(args[f'angacc_{t}'].split(',')[i])
      data[y][t].angacc = Parameters.build(w, w.fetch('w.*'))
      # Normalize sWeights per bin
      sw = np.zeros_like(data[y][t].df['time'])
      for ml, mh in zip(mass[:-1], mass[1:]):
        # for tl, th in zip([0.3, 0.92, 1.97], [0.92, 1.97, 15]):
        # sw_cut = f'mHH>={ml} & mHH<{mh} & time>={tl} &  time<{th}'
        sw_cut = f'mHH>={ml} & mHH<{mh}'
        pos = data[y][t].df.eval(sw_cut)
        _sw = data[y][t].df.eval(f'{weight}*({sw_cut})')
        sw = np.where(pos, _sw * (sum(_sw) / sum(_sw * _sw)), sw)
      data[y][t].df['sWeightCorr'] = sw
      print(np.sum(sw))
      data[y][t].allocate(data=real, weight='sWeightCorr', prob='0*time')
      print(knots)

  # }}}

  # Compile the kernel
  #    so if knots change when importing parameters, the kernel is compiled
  # badjanak.config["precision"]='single'
  badjanak.get_kernels(True)

  # TODO: move this to a function which parses fit wildcard
  SWAVE = True
  if 'Pwave' in FIT:
    SWAVE = False
  if 'Dwave' in FIT:
    DWAVE = True
  DGZERO = False
  if 'DGzero' in FIT:
    DGZERO = True
  POLDEP = False
  if 'Poldep' in FIT:
    POLDEP = True
  BLIND = bool(int(args['blind']))
  TWO_GAMMA = False
  CONSTR = True
  # CONSTR = False
  # BLIND = False
  print("blind:", BLIND)
  print("polalization dependent:", POLDEP)

  # BLIND = False

  # Prepare parameters {{{

  mass_knots = badjanak.config['mHH']
  pars = Parameters()

  # S wave fractions
  for i in range(len(mass_knots) - 1):
    pars.add(dict(
        name=f'fSlon{i+1}', value=SWAVE * 0.0, min=0.00, max=0.90,
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
  if 'Bu' in args['timeacc']:
    gamma_ref = 1 / 1.638
  else:
    gamma_ref = 0.65789
  pars.add(dict(name="Gd",
                value=gamma_ref, min=0.0, max=1.0,
                free=False,
                latex=r"\Gamma_d \, \mathrm{[ps]}^{-1}"))
  if TWO_GAMMA:
    pars.add(dict(name="Gn",
                    value=0.6, min=0.0, max=5.5,
                    free=TWO_GAMMA,
                    latex=r"\Gamma_d \, \mathrm{[ps]}^{-1}"))
  pars.add(dict(name="DGs", value=(1 - DGZERO) * 0.3, min=0.0, max=1.7,
                free=1 - DGZERO,
                latex=r"\Delta\Gamma_s \, \mathrm{[ps^{-1}]}",
                blindstr="BsDGsFullRun2",
                blind=BLIND, blindscale=1.0, blindengine="root"))
  pars.add(dict(name="DGsd", value=0.03 * 0, min=-0.5, max=0.5,
                free=True,
                latex=r"\Gamma_s - \Gamma_d \, \mathrm{[ps^{-1}]}"))
  # pars.add(dict(name="Gs", value=0.03 * 0, min=-5.5, max=5.5,
  #               free=True,
  #               latex=r"\Gamma_s - \Gamma_d \, \mathrm{[ps^{-1}]}"))
  pars.add(dict(name="DM", value=17.757, min=15.0, max=20.0,
                free=True,
                latex=r"\Delta m_s \, \mathrm{[ps^{-1}]}"))

  # # tagging
  # pars.add(dict(name="eta_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['eta_os'].value,
  #               free=False))
  # pars.add(dict(name="eta_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['eta_ss'].value,
  #               free=False))
  # pars.add(dict(name="p0_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['p0_os'].value,
  #               free=True, min=0.0, max=1.0,
  #               latex=r"p^{\rm OS}_{0}"))
  # pars.add(dict(name="p1_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['p1_os'].value,
  #               free=True, min=0.5, max=1.5,
  #               latex=r"p^{\rm OS}_{1}"))
  # pars.add(dict(name="p0_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['p0_ss'].value,
  #               free=True, min=0.0, max=2.0,
  #               latex=r"p^{\rm SS}_{0}"))
  # pars.add(dict(name="p1_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['p1_ss'].value,
  #               free=True, min=0.0, max=2.0,
  #               latex=r"p^{\rm SS}_{1}"))
  # pars.add(dict(name="dp0_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].value,
  #               free=True, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm OS}_{0}"))
  # pars.add(dict(name="dp1_os",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].value,
  #               free=True, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm OS}_{1}"))
  # pars.add(dict(name="dp0_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].value,
  #               free=True, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm SS}_{0}"))
  # pars.add(dict(name="dp1_ss",
  #               value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].value,
  #               free=True, min=-0.1, max=0.1,
  #               latex=r"\Delta p^{\rm SS}_{1}"))

  # for year in YEARS:
  # # for year in list(samples.keys()):
  #   if CONSTR:
  #     pars += data[year][TRIGGER[0]].flavor
  #   else:
  #     data[year][TRIGGER[0]].flavor.lock()
  #     pars += data[year][TRIGGER[0]].flavor
  # for p in pars:
  #   if p.startswith('eta'):
  #     print(p)
  #     pars.lock(p)

  for year in YEARS:
    if int(year) > 2015:
      str_year = str(year)
      syear = int(str_year[2:])
      pars.add(dict(name=f"eta_os{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['eta_os'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['eta_os'].stdev,
                  free=False))
      pars.add(dict(name=f"eta_ss{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['eta_ss'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['eta_ss'].stdev,
                  free=False))
      pars.add(dict(name=f"p0_os{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['p0_os'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['p0_os'].stdev,
                  free=CONSTR, min=0.0, max=1.0,
                  latex=r"p^{\rm OS}_{0}"))
      pars.add(dict(name=f"p1_os{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['p1_os'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['p1_os'].stdev,
                  free=CONSTR, min=0.5, max=1.5,
                  latex=r"p^{\rm OS}_{1}"))
      pars.add(dict(name=f"p0_ss{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['p0_ss'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['p0_ss'].stdev,
                  free=CONSTR, min=0.0, max=2.0,
                  latex=r"p^{\rm SS}_{0}"))
      pars.add(dict(name=f"p1_ss{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['p1_ss'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['p1_ss'].stdev,
                  free=CONSTR, min=0.0, max=2.0,
                  latex=r"p^{\rm SS}_{1}"))
      pars.add(dict(name=f"dp0_os{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['dp0_os'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['dp0_os'].stdev,
                  free=CONSTR, min=-0.1, max=0.1,
                  latex=r"\Delta p^{\rm OS}_{0}"))
      pars.add(dict(name=f"dp1_os{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['dp1_os'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['dp1_os'].stdev,
                  free=CONSTR, min=-0.1, max=0.1,
                  latex=r"\Delta p^{\rm OS}_{1}"))
      pars.add(dict(name=f"dp0_ss{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['dp0_ss'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['dp0_ss'].stdev,
                  free=CONSTR, min=-0.1, max=0.1,
                  latex=r"\Delta p^{\rm SS}_{0}"))
      pars.add(dict(name=f"dp1_ss{syear}",
                  value=data[str(year)][TRIGGER[0]].flavor['dp1_ss'].value,
                  stdev=data[str(year)][TRIGGER[0]].flavor['dp1_ss'].stdev,
                  free=CONSTR, min=-0.1, max=0.1,
                  latex=r"\Delta p^{\rm SS}_{1}"))
  print(pars)

  os_corr = np.array([
	[    1.00, 0.03, 0.14,-0.00, 0.30,-0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ],
	[    0.03, 1.00,-0.00, 0.65,-0.00, 0.35, 0.00, 0.00, 0.00,-0.00,-0.00,-0.00 ],
	[    0.14,-0.00, 1.00, 0.05, 0.10,-0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ],
	[   -0.00, 0.65, 0.05, 1.00,-0.00, 0.29, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ],
	[    0.30,-0.00, 0.10,-0.00, 1.00, 0.04, 0.00, 0.00,-0.00,-0.00, 0.00,-0.00 ],
	[   -0.00, 0.35,-0.00, 0.29, 0.04, 1.00, 0.00,-0.00, 0.00, 0.00, 0.00,-0.00 ],
	[    0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.12, 0.00, 0.00, 0.00, 0.00 ],
	[    0.00, 0.00, 0.00, 0.00, 0.00,-0.00, 0.12, 1.00,-0.00,-0.00,-0.00,-0.00 ],
	[    0.00, 0.00, 0.00, 0.00,-0.00, 0.00, 0.00,-0.00, 1.00, 0.14,-0.00,-0.00 ],
	[    0.00,-0.00, 0.00, 0.00,-0.00, 0.00, 0.00,-0.00, 0.14, 1.00,-0.00,-0.00 ],
	[    0.00,-0.00, 0.00, 0.00, 0.00, 0.00, 0.00,-0.00,-0.00,-0.00, 1.00, 0.14 ],
	[    0.00,-0.00, 0.00, 0.00,-0.00,-0.00, 0.00,-0.00,-0.00,-0.00, 0.14, 1.00 ]
	])
  ss_corr = np.array([
	[    1.00,-0.03, 0.07,-0.00, 0.09,-0.00, 0.00, 0.00, 0.00, 0.00,-0.00,-0.00 ],
	[   -0.03, 1.00,-0.00, 0.36,-0.00, 0.32,-0.00,-0.00,-0.00,-0.00,-0.00, 0.00 ],
	[    0.07,-0.00, 1.00, 0.04, 0.05,-0.00,-0.00,-0.00,-0.00,-0.00, 0.00, 0.00 ],
	[   -0.00, 0.36, 0.04, 1.00,-0.00, 0.19,-0.00,-0.00,-0.00,-0.00,-0.00, 0.00 ],
	[    0.09,-0.00, 0.05,-0.00, 1.00, 0.05, 0.00, 0.00, 0.00, 0.00,-0.00,-0.00 ],
	[   -0.00, 0.32,-0.00, 0.19, 0.05, 1.00,-0.00,-0.00,-0.00,-0.00,-0.00, 0.00 ],
	[    0.00,-0.00,-0.00,-0.00, 0.00,-0.00, 1.00, 0.12, 0.00,-0.00,-0.00,-0.00 ],
	[    0.00,-0.00,-0.00,-0.00, 0.00,-0.00, 0.12, 1.00, 0.00, 0.00,-0.00,-0.00 ],
	[    0.00,-0.00,-0.00,-0.00, 0.00,-0.00, 0.00, 0.00, 1.00, 0.13,-0.00, 0.00 ],
	[    0.00,-0.00,-0.00,-0.00, 0.00,-0.00,-0.00, 0.00, 0.13, 1.00, 0.00, 0.00 ],
	[   -0.00,-0.00, 0.00,-0.00,-0.00,-0.00,-0.00,-0.00,-0.00, 0.00, 1.00, 0.11 ],
	[   -0.00, 0.00, 0.00, 0.00,-0.00, 0.00,-0.00,-0.00, 0.00, 0.00, 0.11, 1.00 ]
	])
  pars.corr_from_matrix(os_corr,
                        ["p0_os16", "p1_os16", "p0_os17", "p1_os17", "p0_os18",
                         "p1_os18", "dp0_os16", "dp1_os16", "dp0_os17",
                         "dp1_os17", "dp0_os18", "dp1_os18", ])
  pars.corr_from_matrix(ss_corr,
                        ["p0_ss16", "p1_ss16", "p0_ss17", "p1_ss17", "p0_ss18",
                         "p1_ss18", "dp0_ss16", "dp1_ss16", "dp0_ss17",
                         "dp1_ss17", "dp0_ss18", "dp1_ss18", ])
  # for year in YEARS:
  #   if int(year) > 2015:
  #     str_year = str(year)
  #     syear = int(str_year[:-2])
  #     pars.add(dict(name=f"etaOS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['eta_os'].value,
  #                 free=False))
  #     pars.add(dict(name=f"etaSS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['eta_ss'].value,
  #                 free=False))
  #     pars.add(dict(name=f"p0OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p0_os'].value,
  #                 free=True, min=0.0, max=1.0,
  #                 latex=r"p^{\rm OS}_{0}"))
  #     pars.add(dict(name=f"p1OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p1_os'].value,
  #                 free=True, min=0.5, max=1.5,
  #                 latex=r"p^{\rm OS}_{1}"))
  #     pars.add(dict(name=f"p0SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p0_ss'].value,
  #                 free=True, min=0.0, max=2.0,
  #                 latex=r"p^{\rm SS}_{0}"))
  #     pars.add(dict(name=f"p1SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['p1_ss'].value,
  #                 free=True, min=0.0, max=2.0,
  #                 latex=r"p^{\rm SS}_{1}"))
  #     pars.add(dict(name=f"dp0OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp0_os'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm OS}_{0}"))
  #     pars.add(dict(name=f"dp1OS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp1_os'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm OS}_{1}"))
  #     pars.add(dict(name=f"dp0SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp0_ss'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm SS}_{0}"))
  #     pars.add(dict(name=f"dp1SS{syear}",
  #                 value=data[str(year)][TRIGGER[0]].flavor['dp1_ss'].value,
  #                 free=True, min=-0.1, max=0.1,
  #                 latex=r"\Delta p^{\rm SS}_{1}"))

  # }}}

  # Print all ingredients of the pdf {{{

  # fit parameters
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

  # }}}

  # Calculate tagging constraints
  # currently using one value for all years only!!!
  # --- def --- {{{
  def taggingConstraints(data):
    corr = data[str(YEARS[0])][TRIGGER[0]].flavor.corr(['p0_os', 'p1_os'])
    print(corr)
    rhoOS = corr[1, 0]
    print(rhoOS)
    # print(Parameters.load('output/params/flavor_tagging/2015/Bs2JpsiPhi/v0r5.json')['rho01_os'].value)
    corr = data[str(YEARS[0])][TRIGGER[0]].flavor.corr(['p0_ss', 'p1_ss'])
    print(corr)
    # print(Parameters.load('output/params/flavor_tagging/2015/Bs2JpsiPhi/v0r5.json')['rho01_ss'].value)
    rhoSS = corr[1, 0]  # data[str(YEARS[0])][TRIGGER[0]].flavor['rho01_ss'].value

    pOS = np.matrix([
        data[str(YEARS[0])][TRIGGER[0]].flavor['p0_os'].value,
        data[str(YEARS[0])][TRIGGER[0]].flavor['p1_os'].value
    ])
    pSS = np.matrix([
        data[str(YEARS[0])][TRIGGER[0]].flavor['p0_ss'].value,
        data[str(YEARS[0])][TRIGGER[0]].flavor['p1_ss'].value
    ])
    print(f"pOS, pSS = {pOS}, {pSS}")

    p0OS_err = data[str(YEARS[0])][TRIGGER[0]].flavor['p0_os'].stdev
    p1OS_err = data[str(YEARS[0])][TRIGGER[0]].flavor['p1_os'].stdev
    p0SS_err = data[str(YEARS[0])][TRIGGER[0]].flavor['p0_ss'].stdev
    p1SS_err = data[str(YEARS[0])][TRIGGER[0]].flavor['p1_ss'].stdev
    print(p0OS_err, p0OS_err)

    covOS = np.matrix([[p0OS_err**2, p0OS_err * p1OS_err * rhoOS],
                       [p0OS_err * p1OS_err * rhoOS, p1OS_err**2]])
    covSS = np.matrix([[p0SS_err**2, p0SS_err * p1SS_err * rhoSS],
                       [p0SS_err * p1SS_err * rhoSS, p1SS_err**2]])
    print(f"covOS, covSS = {covOS}, {covSS}")
    print(f"covOS, covSS = {data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_os','p1_os'])}, {data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_ss','p1_ss'])}")

    print(np.linalg.inv(data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_os', 'p1_os'])))
    print(np.linalg.inv(data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_ss', 'p1_ss'])))
    covOSInv = covOS.I
    covSSInv = covSS.I

    print(covSSInv, covOSInv)
    dictOut = {'pOS': pOS, 'pSS': pSS, 'covOS': covOS, 'covSS': covSS, 'covOSInv': covOSInv, 'covSSInv': covSSInv}

    return dictOut

  tagConstr = taggingConstraints(data)
  for k, v in tagConstr.items():
    print(k)
    print(v)

  # }}}

  # define fcn function {{{

################## -old -  def fcn_tag_constr_data(parameters: Parameters, data: dict) -> np.ndarray:
################## -old -    """
################## -old -    Cost function to fit real data. Data is a dictionary of years, and each
################## -old -    year should be a dictionary of trigger categories. This function loops
################## -old -    over years and trigger categories.  Here we are going to unblind the
################## -old -    parameters to the p.d.f., thats why we call
################## -old -    parameters.valuesdict(blind=False), by
################## -old -    default `parameters.valuesdict()` has blind=True.
################## -old -
################## -old -    Parameters
################## -old -    ----------
################## -old -    parameters : `ipanema.Parameters`
################## -old -    Parameters object with paramaters to be fitted.
################## -old -    data : dict
################## -old -
################## -old -    Returns
################## -old -    -------
################## -old -    np.ndarray
################## -old -    Array containing the weighted likelihoods
################## -old -
################## -old -    """
################## -old -    # parameters['dp0_os'].value += 0.004
################## -old -    # parameters['dp1_os'].value += 0.004
################## -old -    # parameters['dp0_ss'].value -= 0.003
################## -old -
################## -old -    pars_dict = parameters.valuesdict(blind=False)
################## -old -    # os_names = parameters.find(rf"\A(p[0-2])(_os|OS)({y[2:]})?\Z")
################## -old -    # ss_names = parameters.find(rf"\A(p[0-2])(_ss|SS)({y[2:]})?\Z")
################## -old -    # dos_names = parameters.find(rf"\A(dp[0-2])(_os|OS)({y[2:]})?\Z")
################## -old -    # dss_names = parameters.find(rf"\A(dp[0-2])(_ss|SS)({y[2:]})?\Z")
################## -old -
################## -old -    chi2TagConstr = 0.
################## -old -
################## -old -    chi2TagConstr += (pars_dict['dp0_os'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].stdev**2
################## -old -    chi2TagConstr += (pars_dict['dp1_os'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].stdev**2
################## -old -    chi2TagConstr += (pars_dict['dp0_ss'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].stdev**2
################## -old -    chi2TagConstr += (pars_dict['dp1_ss'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].stdev**2
################## -old -
################## -old -    tagcvOS = np.matrix([pars_dict['p0_os'], pars_dict['p1_os']]) - tagConstr['pOS']
################## -old -    tagcvSS = np.matrix([pars_dict['p0_ss'], pars_dict['p1_ss']]) - tagConstr['pSS']
################## -old -    Y_OS = np.dot(tagcvOS, tagConstr['covOSInv'])
################## -old -    chi2TagConstr += np.dot(Y_OS, tagcvOS.T)
################## -old -    Y_SS = np.dot(tagcvSS, tagConstr['covSSInv'])
################## -old -    chi2TagConstr += np.dot(Y_SS, tagcvSS.T)
################## -old -
################## -old -    # c_dos = parameters.valuesarray(dos_names)
################## -old -    # c_dss = parameters.valuesarray(dss_names)
################## -old -    # c0_dos = np.matrix(c_dos - data[str(YEARS[0])][TRIGGER[0]].flavor.valuesarray(dos_names))
################## -old -    # c0_dss = np.matrix(c_dss - data[str(YEARS[0])][TRIGGER[0]].flavor.valuesarray(dss_names))
################## -old -    # cov_dos = np.matrix(data[str(YEARS[0])][TRIGGER[0]].flavor.cov(dos_names))  # constraint covariance matrix
################## -old -    # cov_dss = np.matrix(data[str(YEARS[0])][TRIGGER[0]].flavor.cov(dss_names))  # constraint covariance matrix
################## -old -    # cnstr_os = np.dot(np.dot(c0_dos, np.linalg.inv(cov_dos)), c0_dos.T)
################## -old -    # cnstr_ss = np.dot(np.dot(c0_dss, np.linalg.inv(cov_dss)), c0_dss.T)
################## -old -    # cnstr += len(c0_dos) * np.log(2 * np.pi) + np.log(np.linalg.det(cov_dos))
################## -old -    # cnstr = np.float64(cnstr[0][0]) / len(dt.prob)
################## -old -    # cnstr = cnstr_os + cnstr_ss
################## -old -
################## -old -    chi2 = []
################## -old -    chi2_2 = []
################## -old -    for y, dy in data.items():
################## -old -      for _, dt in dy.items():
################## -old -        os_names = parameters.find(rf"\A((d)?p[0-2])(_os|OS)({y[2:]})?\Z")
################## -old -        ss_names = parameters.find(rf"\A((d)?p[0-2])(_ss|SS)({y[2:]})?\Z")
################## -old -        dos_names = parameters.find(rf"\A(dp[0-2])(_os|OS)({y[2:]})?\Z")
################## -old -        dss_names = parameters.find(rf"\A(dp[0-2])(_ss|SS)({y[2:]})?\Z")
################## -old -        c_os = parameters.valuesarray(os_names)
################## -old -        c_ss = parameters.valuesarray(ss_names)
################## -old -        c0_os = np.matrix(c_os - dt.flavor.valuesarray(os_names))
################## -old -        c0_ss = np.matrix(c_ss - dt.flavor.valuesarray(ss_names))
################## -old -        # c_dos = parameters.valuesarray(dos_names)
################## -old -        # c_dss = parameters.valuesarray(dss_names)
################## -old -        # c0_dos = np.matrix(c_dos - dt.flavor.valuesarray(dos_names))
################## -old -        # c0_dss = np.matrix(c_dss - dt.flavor.valuesarray(dss_names))
################## -old -        cov_os = np.matrix(dt.flavor.cov(os_names))  # constraint covariance matrix
################## -old -        cov_ss = np.matrix(dt.flavor.cov(ss_names))  # constraint covariance matrix
################## -old -        # cov_dos = np.matrix(dt.flavor.cov(dos_names))  # constraint covariance matrix
################## -old -        # cov_dss = np.matrix(dt.flavor.cov(dss_names))  # constraint covariance matrix
################## -old -        cnstr_os = np.dot(np.dot(c0_os, np.linalg.inv(cov_os)), c0_os.T)
################## -old -        cnstr_ss = np.dot(np.dot(c0_ss, np.linalg.inv(cov_ss)), c0_ss.T)
################## -old -        # cnstr_os += np.dot(np.dot(c0_dos, np.linalg.inv(cov_dos)), c0_dos.T)
################## -old -        # cnstr_ss += np.dot(np.dot(c0_dss, np.linalg.inv(cov_dss)), c0_dss.T)
################## -old -        cnstr = cnstr_os + cnstr_ss
################## -old -        badjanak.delta_gamma5_data(
################## -old -        # badjanak.delta_gamma5_data_mod(
################## -old -            dt.data, dt.prob, **pars_dict,
################## -old -            **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
################## -old -            **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
################## -old -            # **dt.flavor.valuesdict(),
################## -old -            tLL=tLL, tUL=tUL,
################## -old -            use_fk=1, use_angacc=1, use_timeacc=1,
################## -old -            use_timeoffset=0, set_tagging=1, use_timeres=1,
################## -old -            BLOCK_SIZE=128
################## -old -        )
################## -old -        # exit()
################## -old -        _cnstr = np.float64(cnstr[0][0]) / len(dt.prob)
################## -old -        chi2_2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get() + _cnstr)
################## -old -        chi2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get())
################## -old -    # print('my', cnstr, "   vero", chi2TagConstr)
################## -old -
################## -old -    # chi2conc = np.concatenate(chi2)
################## -old -    chi2conc = np.concatenate(chi2_2)
################## -old -    # chi2conc = chi2conc + np.array(len(chi2conc)*[chi2TagConstr[0][0]/float(len(chi2conc))])
################## -old -
################## -old -    # chi2TagConstr = float(chi2TagConstr[0][0] / len(chi2conc))
################## -old -    # print(np.sum(chi2conc + chi2TagConstr))
################## -old -    # print(np.sum(np.concatenate(chi2_2)))
################## -old -    # exit()
################## -old -    # for i in range(len(chi2conc)): chi2conc[i] += chi2TagConstr
################## -old -
################## -old -    # print(chi2TagConstr)
################## -old -    # print( np.nan_to_num(chi2conc + chi2TagConstr, 0, 100, 100).sum() )
################## -old -    return chi2conc  # + chi2TagConstr  # np.concatenate(chi2)


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
    tag_names_all = parameters.find(r"\A((d)?p[0-2])(_os|OS|_ss|SS|IFT|ift)(\d{2})?\Z")
    tag_names_all2 = [shit.replace('15', '') for shit in tag_names_all]
    tag_names_all2 = [shit.replace('16', '') for shit in tag_names_all2]
    tag_names_all2 = [shit.replace('17', '') for shit in tag_names_all2]
    tag_names_all2 = [shit.replace('18', '') for shit in tag_names_all2]
    pars_all = parameters.valuesdict(blind=False)
    pars_dict = {k: v for k, v in pars_all.items() if k not in tag_names_all}
    # print(pars_all)
    # exit()

    chi2 = []
    for y, dy in data.items():
      for _, dt in dy.items():
        yt = y[2:] if y[2:] != "15" else "16"
        tag_names = parameters.find(rf"\A((d)?p[0-2])(_os|OS|_ss|SS|IFT|ift)({yt})?\Z")
        _tag = {k: v for k, v in pars_all.items() if k in tag_names}
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
      c = parameters.valuesarray(tag_names_all)
      c0 = np.matrix(c - dt.flavor.valuesarray(tag_names_all2))
      # cov = dt.flavor.cov(tag_names)  # constraint covariance matrix
      cov = pars.cov(tag_names_all)  # constraint covariance matrix
      cnstr = np.dot(np.dot(c0, np.linalg.inv(cov)), c0.T)
      cnstr += len(c0) * np.log(2 * np.pi) + np.log(np.linalg.det(cov))
      cnstr = np.float64(cnstr[0][0])
    else:
      cnstr = 0

    chi2conc = np.concatenate(chi2)
    # exit()
    return chi2conc + cnstr / len(chi2conc)


  # function without constraining on tagging parameters
  def fcn_data(parameters: Parameters, data: dict) -> np.ndarray:
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
    pars_dict = parameters.valuesdict(blind=False)
    chi2 = []

    for _, dy in data.items():
      for dt in dy.values():
        badjanak.delta_gamma5_data(
            dt.data, dt.prob, **pars_dict,
            **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
            **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
            **dt.flavor.valuesdict(),
            use_fk=1, use_angacc=1, use_timeacc=1,
            use_timeoffset=0, set_tagging=1, use_timeres=1,
            tLL=tLL, tUL=tUL
        )
        chi2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get())

    return np.concatenate(chi2)

  # cost_function = fcn_data
  cost_function = fcn_tag_constr_data

  # }}}

  # Minimization {{{

  if args['log_likelihood']:
    pars = Parameters.load(args['input_params'])
    ll = cost_function(pars, data).sum()
    print(f"Found sum(LL) to be {ll}")
    np.save(args['log_likelihood'], ll)
    exit(0)

  printsubsec("Simultaneous minimization procedure")

  result = optimize(cost_function, method=MINER, params=pars,
                    fcn_kwgs={'data': data},
                    verbose=True, timeit=True,  # tol=0.05, strategy=2,
                    policy='filter')

  # full print of result
  print(result)

  # print some parameters
  for kp, vp in result.params.items():
    if vp.stdev:
      if args['year'] == '2015,2016':
        print(f"{kp:>12} : {vp._getval(False):+.4f} ± {vp.stdev:+.4f}")
      else:
        print(f"{kp:>12} : {vp.value:+.4f} ({vp._getval(False):+.4f}) ± {vp.stdev:+.4f}")

  lifeDiff = result.params['DGsd'].uvalue
  lifeBu = unc.ufloat(1.638, 0.004)
  lifeBd = unc.ufloat(1.520, 0.004)
  print("Lifetime for Bs mesos is:")
  if 'Bu' in args['timeacc']:
    print(f"{1/(lifeDiff+1/lifeBu):.2uL}")
  else:
    print(f"{1/(lifeDiff+1/lifeBd):.2uL}")
  # }}}

  # Save results {{{

  print("Dump parameters")
  result.params.dump(args['params'])

  # if scan_likelihood, then we need to create some contours
  scan_likelihood = args['scan']
  # scan_likelihood = False

  if scan_likelihood != "0":
    print("scanning", scan_likelihood)
    if "+" in scan_likelihood:
      result._minuit.draw_mncontour(*scan_likelihood.split('+'),
                                    numpoints=100, nsigma=5)
    else:

      v = result.params[scan_likelihood].uvalue
      print(v)
      x, y = result._minuit.draw_mnprofile(scan_likelihood, bins=20, bound=3,
                                           subtract_min=True, band=True,
                                           text=True)
      fig, axplot = complot.axes_plot()
      axplot.plot(x, y, color="C0")  # scan
      axplot.axvline(v.n,
                     color="C1", linestyle="-")

      vmin = None
      vmax = None
      if (scan_likelihood, 1) in result._minuit.merrors:
        vmin = v.n + result._minuit.merrors[(scan_likelihood, -1)]
        vmax = v.n + result._minuit.merrors[(scan_likelihood, 1)]
      if scan_likelihood in result._minuit.errors:
        vmin = v.n - result._minuit.errors[scan_likelihood]
        vmax = v.n + result._minuit.errors[scan_likelihood]

      plt.axvspan(vmin, vmax, facecolor="C1", alpha=0.4)

      if vmin is None:
        str_value = f"{v:.2uL}"
      else:
        str_value = f"{v:.2uL}".split('\pm')[0]
        str_value = f"{str_value}_{{{v.n-vmax:.5f}}}^{{+{v.n-vmin:.5f}}}"
      str_latex = f"{result.params[scan_likelihood].latex}"
      str_units = str_latex.split("\,")[-1].replace("[", "")
      str_units = str_units.replace("]", "")
      str_latex = str_latex.split("\,")[0]
      print(f"${str_latex} = {str_value}$ ${str_units}$")
      axplot.set_xlabel(f"${str_latex} = {str_value}$ ${str_units}$")
      axplot.set_ylabel("$\Delta \log L$")
      print(result._minuit.merrors)
    _figure = args['params'].replace('/params', '/figures')
    _figure = _figure.replace('.json', f'/scans/{scan_likelihood}.pdf')
    print("Scan saved to:", _figure)
    os.makedirs(os.path.dirname(_figure), exist_ok=True)
    # plt.xlim(1.8, 3.5)
    # plt.ylim(2.7, 3.5)
    plt.savefig(_figure)

  # }}}


# vim: fdm=marker
# TODO: update tagging constraints to work with cov/corr methods from ipanema
#       parameters
#       use tagging per year and handle IFT
