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
badjanak.config['fast_integral'] = 1
badjanak.config['debug_evt'] = 774


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
  branches_to_load += ['sw', 'sWeight', 'lbWeight']

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
  badjanak.get_kernels()

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
  pars.add(dict(name="eta_os",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['eta_os'].value,
                free=False))
  pars.add(dict(name="eta_ss",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['eta_ss'].value,
                free=False))
  pars.add(dict(name="p0_os",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['p0_os'].value,
                free=True, min=0.0, max=1.0,
                latex=r"p^{\rm OS}_{0}"))
  pars.add(dict(name="p1_os",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['p1_os'].value,
                free=True, min=0.5, max=1.5,
                latex=r"p^{\rm OS}_{1}"))
  pars.add(dict(name="p0_ss",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['p0_ss'].value,
                free=True, min=0.0, max=2.0,
                latex=r"p^{\rm SS}_{0}"))
  pars.add(dict(name="p1_ss",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['p1_ss'].value,
                free=True, min=0.0, max=2.0,
                latex=r"p^{\rm SS}_{1}"))
  pars.add(dict(name="dp0_os",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].value,
                free=True, min=-0.1, max=0.1,
                latex=r"\Delta p^{\rm OS}_{0}"))
  pars.add(dict(name="dp1_os",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].value,
                free=True, min=-0.1, max=0.1,
                latex=r"\Delta p^{\rm OS}_{1}"))
  pars.add(dict(name="dp0_ss",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].value,
                free=True, min=-0.1, max=0.1,
                latex=r"\Delta p^{\rm SS}_{0}"))
  pars.add(dict(name="dp1_ss",
                value=data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].value,
                free=True, min=-0.1, max=0.1,
                latex=r"\Delta p^{\rm SS}_{1}"))

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

  # }}}

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
    pars_dict = parameters.valuesdict(blind=False)
    chi2TagConstr = 0.

    chi2TagConstr += (pars_dict['dp0_os'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_os'].stdev**2
    chi2TagConstr += (pars_dict['dp1_os'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_os'].stdev**2
    chi2TagConstr += (pars_dict['dp0_ss'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp0_ss'].stdev**2
    chi2TagConstr += (pars_dict['dp1_ss'] - data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].value)**2 / data[str(YEARS[0])][TRIGGER[0]].flavor['dp1_ss'].stdev**2

    tagcvOS = np.matrix([pars_dict['p0_os'], pars_dict['p1_os']]) - tagConstr['pOS']
    tagcvSS = np.matrix([pars_dict['p0_ss'], pars_dict['p1_ss']]) - tagConstr['pSS']

    Y_OS = np.dot(tagcvOS, tagConstr['covOSInv'])
    chi2TagConstr += np.dot(Y_OS, tagcvOS.T)
    Y_SS = np.dot(tagcvSS, tagConstr['covSSInv'])
    chi2TagConstr += np.dot(Y_SS, tagcvSS.T)

    chi2 = []
    for _, dy in data.items():
      for _, dt in dy.items():
        badjanak.delta_gamma5_data(
            dt.data, dt.prob, **pars_dict,
            **dt.timeacc.valuesdict(), **dt.angacc.valuesdict(),
            **dt.resolution.valuesdict(), **dt.csp.valuesdict(),
            # **dt.flavor.valuesdict(),
            tLL=tLL, tUL=tUL,
            use_fk=1, use_angacc=1, use_timeacc=1,
            use_timeoffset=0, set_tagging=1, use_timeres=1,
            BLOCK_SIZE=256
        )
        chi2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get())

    chi2conc = np.concatenate(chi2)
    # chi2conc = chi2conc + np.array(len(chi2conc)*[chi2TagConstr[0][0]/float(len(chi2conc))])

    chi2TagConstr = float(chi2TagConstr[0][0] / len(chi2conc))
    # for i in range(len(chi2conc)): chi2conc[i] += chi2TagConstr

    # print(chi2TagConstr)
    # print( np.nan_to_num(chi2conc + chi2TagConstr, 0, 100, 100).sum() )
    return chi2conc + chi2TagConstr  # np.concatenate(chi2)

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
                    verbose=False, timeit=True,  # tol=0.1, strategy=1,
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

      x, y = result._minuit.draw_mnprofile(scan_likelihood, bins=20, bound=3,
                                           subtract_min=True, band=True,
                                           text=True)
      fig, axplot = complot.axes_plot()
      axplot.plot(x, y)
      axplot.set_xlabel(f"${result.params[scan_likelihood].latex}$")
      axplot.set_ylabel("$\Delta \log L$")
      print(result._minuit.merrors)
    _figure = args['params'].replace('/params', '/figures')
    _figure = _figure.replace('.json', f'/scans/{scan_likelihood}.pdf')
    print(_figure)
    os.makedirs(os.path.dirname(_figure), exist_ok=True)
    # plt.xlim(1.8, 3.5)
    # plt.ylim(2.7, 3.5)
    plt.savefig(_figure)

  # }}}


# vim: fdm=marker
# TODO: update tagging constraints to work with cov/corr methods from ipanema
#       parameters
#       use tagging per year and handle IFT
