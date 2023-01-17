"""
Lb weight
"""


__all__ = ['ipatia_chebyshev']
__author__ = ["name"]
__email__ = ["email"]


# Modules {{{

import os
import ipanema
import argparse
import numpy as np
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
# from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and
import config
import complot

from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes


TIME_CUT = "(B_LOKI_DTF_CTAU/0.29979245)>0.3"
CUT_KMINUS = '((hminus_ProbNNp>0.7) & (hminus_ProbNNp>hplus_ProbNNp))'
CUT_KPLUS = '( (hplus_ProbNNp>0.7) & (hplus_ProbNNp>hminus_ProbNNp))'
CUT_KMINUS = f"({CUT_KMINUS}) & ({TIME_CUT})"
CUT_KPLUS = f"({CUT_KPLUS}) & ({TIME_CUT})"


# initialize ipanema3 and compile lineshapes
ipanema.initialize(config.user['backend'], 1)
prog = ipanema.compile("""
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])

# }}}


# ipatia as signal and chebyshev for background {{{

def ipatia_chebyshev(mass, signal, fsigBs=0, fcomb=0, mu=0, sigma=10, lambd=0,
                     zeta=0, beta=0, aL=0, nL=0, aR=0, nR=0, b=0, t0=1, t1=1,
                     t2=0, t3=0, t4=0, t5=0, t6=0, t7=0, mLL=False, mUL=False,
                     norm=1):
  if not mLL:
    mLL = ipanema.ristra.min(mass)
  if not mUL:
    mUL = ipanema.ristra.min(mass)
  t = np.float64([t0, t1, t2, t3, t4, t5, t6, t7])
  deg = len(t) - 1
  t = ipanema.ristra.allocate(t)

  # ipatia
  # prog.kernel_ipatia(signal, mass, np.float64(mu), np.float64(sigma),
  #                    np.float64(lambd), np.float64(zeta), np.float64(beta),
  #                    np.float64(aL), np.float64(nL), np.float64(aR),
  #                    np.float64(nR), np.float64(mLL), np.float64(mUL),
  #                    global_size=(len(mass)))
  prog.py_ipatia(signal, mass, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(
                     nL), np.float64(aR), np.float64(nR),
                 global_size=(len(mass)))
  pPeak = 1.0 * signal.get()
  signal = 0 * signal
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x * 0
  prog.py_ipatia(_y, _x, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(_x)))
  pPeak /= np.trapz(ristra.get(_y), ristra.get(_x))
  # background
  prog.kernel_chebyshev(signal, mass, t, np.int32(deg), np.float64(mLL),
                        np.float64(mUL), global_size=(len(mass)))
  pComb = signal.get()
  return norm * (fsigBs * pPeak + fcomb * pComb)

# }}}


# mass fitters {{{

def mass_fitter(odf, mass_range=False, mass_branch='B_ConstJpsi_M_1',
                mass_weight='B_ConstJpsi_M_1/B_ConstJpsi_M_1',
                extra_cut=False, figs=False, model=False, shift_peak=False,
                trigger='combined', input_pars=False, sweights=False,
                verbose=False):

  # mass range cut
  if not mass_range:
    mass_range = (min(odf[mass_branch]), max(odf[mass_branch]))
  mLL, mUL = mass_range
  mass_cut = f'{mass_branch} > {mLL} & {mass_branch} < {mUL}'

  # mass cut and trigger cut
  current_cut = trigger_scissors(trigger, cuts_and(mass_cut, extra_cut))
  print(current_cut)

  # Select model and set parameters {{{
  #    Select model from command-line arguments and create corresponding set
  #    of paramters

  pars = ipanema.Parameters()
  # Create common set of Bs parameters (all models must have and use)
  pars.add(dict(name='fsigBs', value=0.8, min=0.05, max=1, free=True,
           latex=r'N_{B_s}'))
  pars.add(dict(name='mu', value=5620, min=5500, max=5700,
           latex=r'\mu_{B_s}'))
  pars.add(dict(name='sigma', value=10, min=5, max=100, free=True,
           latex=r'\sigma_{B_s}'))

  if input_pars:
    _pars = ipanema.Parameters.clone(input_pars)
    _pars.lock()
    _pars.remove('fsigBs', 'mu', 'sigma')
    _pars.unlock('b')
    pars = pars + _pars
  else:
    if 'ipatia' in model:
      # Hypatia tails {{{
      pars.add(dict(name='lambd', value=-1.63529,
               free=False, latex=r'\lambda'))
      pars.add(dict(name='zeta', value=0.0, free=False, latex=r'\zeta'))
      pars.add(dict(name='beta', value=0.0, free=False, latex=r'\beta'))
      pars.add(dict(name='aL', value=2.0958, free=False, latex=r'a_l'))
      pars.add(dict(name='nL', value=0.80680, free=False, latex=r'n_l'))
      pars.add(dict(name='aR', value=2.6543, free=False, latex=r'a_r'))
      pars.add(dict(name='nR', value=0.56408, free=False, latex=r'n_r'))
      # }}}
    elif "crystalball" in model:
      # Crystal Ball tails {{{
      pars.add(dict(name='aL', value=1.4, min=0.5,
               max=3.5, free=True, latex=r'a_l'))
      pars.add(dict(name='nL', value=1, min=1,
               max=500, free=True, latex=r'n_l'))
      pars.add(dict(name='aR', value=1.4, min=0.5,
               max=3.5, free=True, latex=r'a_r'))
      pars.add(dict(name='nR', value=1, min=1,
               max=500, free=True, latex=r'n_r'))
      # }}}
    # Combinatorial background
    pars.add(dict(name='t1', value=0.14, min=-1, max=1, free=True,
                  latex=r't_1'))
    pars.add(dict(name='t2', value=-0.03, min=-1, max=1, free=True,
                  latex=r't_2'))
    pars.add(dict(name='fcomb', formula="1-fsigBs", latex=r'f_{comb}'))
  print(pars)

  # }}}

  # Chose model {{{

  if model == 'ipatia':
    pdf = ipatia_chebyshev
  elif model == 'ipatiaChebyshev':
    pdf = ipatia_chebyshev
  elif model == 'crystalball':
    pdf = ipatia_chebyshev

  def fcn(params, data):
    p = params.valuesdict()
    prob = pdf(data.mass, data.pdf, mUL=mUL, mLL=mLL, **p)
    return -2.0 * np.log(prob) * ristra.get(data.weight)

  # }}}

  # Allocate the sample variables {{{

  print(f"Cut: {current_cut}")
  print(f"Mass branch: {mass_branch}")
  print(f"Mass weight: {mass_weight}")
  rd = Sample.from_pandas(odf)
  _proxy = np.float64(rd.df[mass_branch]) * 0.0 - 999
  rd.chop(current_cut)
  rd.allocate(mass=mass_branch, pdf=f'0*{mass_branch}', weight=mass_weight)
  # print(rd)

  # }}}

  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd}, method='minuit',
                         verbose=verbose, strategy=1, tol=0.05)
  if res:
    print(res)
    fpars = ipanema.Parameters.clone(res.params)
  else:
    print("Could not fit it!. Cloning pars to res")
    fpars = ipanema.Parameters.clone(pars)
    print(fpars)

  fig, axplot, axpull = complot.axes_providers.axes_plotpull()
  hdata = complot.hist(ristra.get(rd.mass), weights=rd.df.eval(mass_weight),
                       bins=56, density=False)
  axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                  fmt='.k')

  norm = hdata.norm  # *(hdata.bins[1]-hdata.bins[0])
  mass = ristra.linspace(mLL, mUL, 1000)
  signal = 0 * mass

  # plot signal: nbkg -> 0 and nexp -> 0
  _p = ipanema.Parameters.clone(fpars)
  if 'fcomb' in _p:
    _p['fcomb'].set(value=0, min=-np.inf, max=np.inf)
  _x = ristra.get(mass)
  _y = ristra.get(pdf(mass, signal, **_p.valuesdict(),
                  mLL=mLL, mUL=mUL, norm=norm))
  # print(_y)
  axplot.plot(_x, _y, color="C1", label=rf'{model}')

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  x = ristra.get(mass)
  y = ristra.get(pdf(mass, signal, **_p.valuesdict(),
                 mLL=mLL, mUL=mUL, norm=norm))
  axplot.plot(x, y, color='C0')
  _pulls = complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts,
                                    *hdata.yerr)
  axpull.fill_between(hdata.bins, _pulls, 0, facecolor="C0", alpha=0.5)
  if 'kplus' in figs:
    axpull.set_xlabel(r'$m(\text{J}\!/\!\uppsi \text{pK}^+)$ [MeV/$c^2$]')
  else:
    axpull.set_xlabel(r'$m(\text{J}\!/\!\uppsi \text{pK}^-)$ [MeV/$c^2$]')
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_yticks([-5, 0, 5])
  if mass_weight:
    axplot.set_ylabel("Weighted candidates")
  else:
    axplot.set_ylabel("Candidates")
  # axplot.legend(loc="upper left")
  if figs:
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "fit.pdf"))
    axplot.set_yscale('log')
    axplot.set_ylim(1e0, 1.5 * np.max(y))
    fig.savefig(os.path.join(figs, "logfit.pdf"))
    plt.close()

  # compute sWeights if asked {{{

  if sweights:
    # separate paramestes in yields and shape parameters
    _yields = ipanema.Parameters.find(fpars, "fsigBs(.*)?") + ["fcomb"]
    _pars = list(fpars)
    [_pars.remove(_y) for _y in _yields]
    _yields = ipanema.Parameters.build(fpars, _yields)
    _pars = ipanema.Parameters.build(fpars, _pars)
    print(_yields, _pars)

    def __pdf(*x, **y): return pdf(rd.mass,
                                   rd.pdf, mLL=mLL, mUL=mUL, *x, **y)
    sw = splot.compute_sweights(__pdf, _pars, _yields)
    for k, v in sw.items():
      _sw = np.copy(_proxy)
      _sw[list(rd.df.index)] = v * np.float64(rd.df.eval(mass_weight))
      sw[k] = _sw
    print(sw)
    return (fpars, sw)

  # }}}

  return (fpars, False)

# }}}


# command line interface {{{

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--sample')
  p.add_argument('--input-params', default=False)
  p.add_argument('--output-params')
  p.add_argument('--output-figures')
  p.add_argument('--mass-model')
  p.add_argument('--mass-weight')
  p.add_argument('--mass-bin', default=False)
  p.add_argument('--nosep')
  p.add_argument('--sweights')
  p.add_argument('--mode')
  p.add_argument('--version')
  p.add_argument('--year')
  args = vars(p.parse_args())

  if args['sweights']:
    sweights = True
  else:
    sweights = False

  if args['input_params']:
    input_pars = ipanema.Parameters.load(args['input_params'])
  else:
    input_pars = False

  if args['year'] in ('2015', '2016', '2017', '2018'):
    family = 'run2'
  else:
    family = 'run1'

  if args['nosep'] == 'kplus':
    kaon = 'plus'
    mass_branch = f'B_pKMuMuK{kaon}_M_1'
    mass_range = (5520, 5800)
    cut = CUT_KPLUS
  elif args['nosep'] == 'kminus':
    kaon = 'minus'
    mass_branch = f'B_pKMuMuK{kaon}_M_1'
    mass_range = (5520, 5800)
    cut = CUT_KMINUS
  else:
    print('merda')

  branches = [mass_branch]
  branches += ['hminus_ProbNNp', 'hplus_ProbNNp', 'B_LOKI_DTF_CTAU']

  if args['mass_weight']:
    mass_weight = args['mass_weight']
    branches += [mass_weight]
  else:
    mass_weight = f'{mass_branch}/{mass_branch}'

  sample = Sample.from_root(args['sample'], branches=branches)
  print(sample.df)

  pars, sw = mass_fitter(sample.df, mass_range=mass_range,
                         mass_branch=mass_branch, mass_weight=mass_weight,
                         figs=args['output_figures'], sweights=sweights,
                         model=args['mass_model'], extra_cut=cut,
                         input_pars=input_pars, verbose=True)
  pars.dump(args['output_params'])
  if sw:
    np.save(args['sweights'], sw)

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
