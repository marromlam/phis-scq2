# background_subtraction
#
#

__all__ = ['cb_exponential', 'ipatia_exponential']
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


# Modules {{{

import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
import re
import yaml
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and
import pandas as pd

import complot
import ipanema
import matplotlib.pyplot as plt
import numpy as np
import uproot3 as uproot
from ipanema import Sample, ristra, splot

import config
from utils.helpers import cuts_and, trigger_scissors
from utils.strings import printsec, printsubsec
from ipanema.tools.misc import get_vars_from_string

with open('selection/gb_weights/config.yml') as file:
  GBW_CONFIG = yaml.load(file, Loader=yaml.FullLoader)


def get_sizes(size, BLOCK_SIZE=8):
  """
  i need to check if this worls for 3d size and 3d block
  """
  a = size % BLOCK_SIZE
  if a == 0:
    gs, ls = size, BLOCK_SIZE
  elif size < BLOCK_SIZE:
    gs, ls = size, 1
  else:
    a = np.ceil(size / BLOCK_SIZE)
    gs, ls = a * BLOCK_SIZE, BLOCK_SIZE
  return int(gs), int(ls)


ipanema.initialize(config.user["backend"], 1)
print(config.user["backend"])
prog = ipanema.compile(
    """
    #define USE_DOUBLE 1
    #include <exposed/kernels.ocl>
    """
)

# }}}


# Hypatia + Combinatorial {{{

def ipatia_exponential(
    mass, merr, signal, fsigBs=0, fsigBd=0, fcomb=0, muBs=5400, s0Bs=1, s1Bs=0,
    s2Bs=0, muBd=5300, s0Bd=1, s1Bd=0, s2Bd=0, lambd=0, zeta=0, beta=0,
    aL=0, nL=0, aR=0, nR=0, b=0, norm=1, mLL=None, mUL=None,
):
  # prepare calibrations --
  sBs = s0Bs  # + merr * (s1Bs + merr * s2Bs)
  sBd = s0Bd  # + merr * (s1Bd + merr * s2Bd)
  # print(sBs, sBd)

  # main peak
  prog.py_ipatia(
      signal, mass, np.float64(muBs), np.float64(sBs), np.float64(lambd),
      np.float64(zeta), np.float64(beta), np.float64(aL), np.float64(nL),
      np.float64(aR), np.float64(nR), global_size=(len(mass)),
  )
  pdfBs = 1.0 * signal.get()
  signal = 0 * signal
  # second peak
  prog.py_ipatia(signal, mass, np.float64(muBd), np.float64(sBd),
                 np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
                 np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
                 )
  pdfBd = 1.0 * signal.get()
  # combinatorial background
  backgr = ristra.exp(mass * b).get()
  # normalize
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x * 0
  prog.py_ipatia(
      _y, _x, np.float64(muBs), np.float64(sBs), np.float64(lambd),
      np.float64(zeta), np.float64(beta), np.float64(aL), np.float64(nL),
      np.float64(aR), np.float64(nR), global_size=(len(_x)),
  )
  nBs = np.trapz(ristra.get(_y), ristra.get(_x))
  _y = _x * 0
  prog.py_ipatia(
      _y, _x, np.float64(muBd), np.float64(sBd), np.float64(lambd),
      np.float64(zeta), np.float64(beta), np.float64(aL), np.float64(nL),
      np.float64(aR), np.float64(nR), global_size=(len(_x)),
  )
  nBd = np.trapz(ristra.get(_y), ristra.get(_x))
  nbackgr = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))

  # combinatorial background
  if fcomb > 0:
    prog.kernel_exponential(
        signal, mass, np.float64(b), np.float64(mLL), np.float64(mUL),
        global_size=(len(mass)),
    )
    pComb = ristra.get(signal)
  else:
    pComb = 0

  # compute pdf value
  ans = fsigBs * pdfBs / nBs + fsigBd * pdfBd / nBd + fcomb * pComb
  return norm * ans

# }}}


# Crystal Ball + Combinatorial {{{

def cb_exponential(
    mass, merr, signal, fsigBs=0, fsigBd=0, fcomb=0, muBs=5400, s0Bs=1, s1Bs=0,
    s2Bs=0, muBd=5300, s0Bd=1, s1Bd=0, s2Bd=0, aL=0, nL=0, aR=0, nR=0, b=0,
    norm=1, mLL=None, mUL=None
):
  # prepare calibrations --
  sBs = s0Bs + merr * (s1Bs + merr * s2Bs)
  sBd = s0Bd + merr * (s1Bd + merr * s2Bd)
  # print(sBs, sBd)
  # print(
  #     fsigBs, fsigBd, fcomb,
  #     np.float64(muBs), sBs, np.float64(aL), np.float64(nL),
  #     np.float64(aR), np.float64(nR), np.float64(mLL), np.float64(mUL),
  # )

  # main peak -- double-sided crystal-ball
  prog.kernel_double_crystal_ball(
      signal, mass, np.float64(muBs), sBs, np.float64(aL), np.float64(nL),
      np.float64(aR), np.float64(nR), np.float64(mLL), np.float64(mUL),
      global_size=(len(mass),)
  )
  pBs = ristra.get(signal)

  # second peak with same tails as main one
  if fsigBd > 0:
    prog.kernel_double_crystal_ball(
        signal, mass, np.float64(muBd), sBd, np.float64(aL), np.float64(nL),
        np.float64(aR), np.float64(nR), np.float64(mLL), np.float64(mUL),
        global_size=(len(mass),)
    )
    pBd = ristra.get(signal)
  else:
    pBd = 0

  # combinatorial background
  if fcomb > 0:
    prog.kernel_exponential(
        signal, mass, np.float64(b), np.float64(mLL), np.float64(mUL),
        global_size=(len(mass)),
    )
    pComb = ristra.get(signal)
  else:
    pComb = 0

  ans = fsigBs * pBs + fsigBd * pBd + fcomb * pComb
  return norm * ans

# }}}


# Double Gaussian + Combinatorial {{{

def dgauss_exponential(
    mass, merr, signal, fsigBs=0, fcomb=0, muBs=5400, s1Bs=0, s2Bs=0,
    frac=0.5, b=0, norm=1, mLL=None, mUL=None
):
  """
  Mass model to fit Bd, Bs and Bu mass shapes.
  It returns the normalized model

  .. math::
    p.d.f.(m) = f_{sig} DoubleGaussian + (1-f_{sig}) * Exponential

  """
  # main peak -- double-sided crystal-ball
  prog.kernel_double_gaussian(
      signal, mass, np.float64(muBs), np.float64(s1Bs), np.float64(s2Bs),
      np.float64(frac), np.float64(mLL), np.float64(mUL),
      global_size=(len(mass),)
  )
  pBs = ristra.get(signal)

  # combinatorial background
  if fcomb > 0:
    prog.kernel_exponential(
        signal, mass, np.float64(b), np.float64(mLL), np.float64(mUL),
        global_size=(len(mass)),
    )
    pComb = ristra.get(signal)
  else:
    pComb = 0

  ans = fsigBs * pBs + fcomb * pComb
  return norm * ans

# }}}


# Bs mass fit function {{{

def mass_fitter(odf, mass_range=False, mass_branch="B_ConstJpsi_M_1",
                mass_weight="B_ConstJpsi_M_1/B_ConstJpsi_M_1", cut=False, figs=False,
                model="dscb", has_bd=False, trigger=False, input_pars=False,
                sweights=False, verbose=False, prefit=False, free_tails=False,
                mode=False):

  if 'Bs' in mode:
    mode_label = f"B_s^0"
  elif 'Bd' in mode:
    mode_label = f"B_d^0"
  elif 'Bu' in mode:
    mode_label = f"B_u^+"
  else:
    print("this mode is not familiar")
    exit()

  # mass range cut
  if not mass_range:
    mass_range = min(odf[mass_branch]), max(odf[mass_branch])
    mass_range = np.floor(mass_range[0]), np.ceil(mass_range[1])
  mLL, mUL = mass_range
  mass_cut = f"{mass_branch} > {mLL} & {mass_branch} < {mUL}"

  # mass cut and trigger cut
  cut = f"({trigger}) & ({cut})" if trigger else cut
  current_cut = mass_cut if not cut else f"({mass_cut}) & ({cut})"

  # Allocate the sample variables {{{

  if verbose:
    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
  rd = Sample.from_pandas(odf)
  _proxy = np.float64(rd.df[mass_branch]) * 0.0
  # rd = Sample.from_pandas(odf.head(int(1e5)))
  rd.chop(current_cut)
  if verbose:
    print("Candidates:", rd.df.shape)
  rd.allocate(mass=mass_branch, merr="B_ConstJpsi_MERR_1")
  rd.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)

  # for a fast 100k events fit
  rd100k = Sample.from_pandas(odf.head(int(1e5)))
  rd100k.chop(current_cut)
  rd100k.allocate(mass=mass_branch, merr="B_ConstJpsi_MERR_1")
  rd100k.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)

  # }}}

  # Chose model {{{

  with_calib = False
  if model == 'hypatia':
    pdf = ipatia_exponential
  elif model == "crystalball" or model == 'dscb':
    pdf = cb_exponential
  elif model == "dscbcalib":
    pdf = cb_exponential
    with_calib = True
  elif model == "dgauss":
    pdf = dgauss_exponential

  def fcn(params, data):
    p = params.valuesdict()
    prob = pdf(mass=data.mass, merr=data.merr, signal=data.pdf, **p)
    return -2.0 * np.log(prob) * ristra.get(data.weight)

  # }}}

  # Build parameters {{{

  _mu = np.mean(rd.mass.get())
  _sigma = 10  # min(np.std(rd.mass.get()), 20)
  _dsigma = 0
  _res = 0.
  print(mode)
  if mode == 'Bd2JpsiKstar':
    _sigma = 7
    _dsigma = 5
    _res = 0.5
    _fsig = 0.8
  elif mode == 'Bs2JpsiPhi':
    _mu = 5367
    _sigma = 7
    _dsigma = 15
    _sigma = 5
    _dsigma = 9
    _res = 0.5
    _fsig = 0.8
  elif mode == 'Bu2JpsiKplus':
    _sigma = 8
    _dsigma = 15
    _res = 0.44
    _fsig = 0.8
  elif 'Prompt' in mode:
    _sigma = 8
    _dsigma = 15
    _res = 0.44
    _fsig = 0.1

  pars = ipanema.Parameters()

  # Create common set of Bs parameters (all models must have and use)
  pars.add(dict(name="fsigBs", value=0.5, min=0.0, max=1,
                free=True,
                latex=r"f_{B_s}"))
  pars.add(dict(name="muBs", value=_mu, min=_mu - 2 * _sigma, max=_mu + 2 * _sigma,
                free=True,
                latex=r"\mu_{B_s}"))
  if with_calib:
    pars.add(dict(name="s0Bs", value=0, min=0, max=30,
                  free=False,
                  latex=r"p_0^{B_s}"))
    pars.add(dict(name="s1Bs", value=0.7, min=-5, max=10,
                  free=True,
                  latex=r"p_1^{B_s}"))
    pars.add(dict(name="s2Bs", value=0.00, min=-1, max=1,
                  free=True,
                  latex=r"p_2^{B_s}"))
  else:
    if model == 'dgauss':
      pars.add(dict(name="s1Bs", value=_sigma, min=1, max=20,
                    free=True, latex=r"p_1^{B_s}"))
      # pars.add(dict(name="s2Bs", value=_dsigma, min=-20, max=20, # WARN
      pars.add(dict(name="s2Bs", value=_dsigma, min=1, max=20,
                    free=True,
                    latex=r"p_2^{B_s}"))
    else:
      pars.add(dict(name="s0Bs", value=_sigma, min=1, max=30,
                    free=True,
                    latex=r"p_0^{B_s}"))
      pars.add(dict(name="s1Bs", value=0.0, min=0, max=10,
                    free=False, latex=r"p_1^{B_s}"))
      pars.add(dict(name="s2Bs", value=0.0, min=-1, max=1,
                    free=False,
                    latex=r"p_2^{B_s}"))

  if input_pars:
    _pars = ipanema.Parameters.clone(input_pars)
    _pars.lock()
    if model == 'dgauss':
      _pars.remove("fsigBs", "muBs", "s1Bs", "s2Bs", "frac")
    else:
      if with_calib:
        _pars.remove("fsigBs", "muBs")
        pars.remove("s1Bs", "s2Bs")
        _pars.unlock("s1Bs", "s2Bs")
      else:
        _pars.remove("fsigBs", "muBs", "s0Bs")
    _pars.unlock("b")
    pars = pars + _pars
  else:
    # This is the prefit stage. Here we will lock the nsig to be 1 and we
    # will not use combinatorial background.
    pars["fsigBs"].value = 1 if prefit else _fsig
    pars["fsigBs"].free = False if prefit else True
    if "hypatia" in model:
      # Hypatia tails {{{
      pars.add(dict(name="lambd", value=-3.5, min=-6, max=2,
                    free=True,
                    latex=r"\lambda",))
      pars.add(dict(name="zeta", value=1e-5,
                    free=False,
                    latex=r"\zeta"))
      pars.add(dict(name="beta", value=0.0,
                    free=False,
                    latex=r"\beta"))
      pars.add(dict(name="aL", value=1.23, min=0.5, max=10,
                    free=True,
                    latex=r"a_l"))
      pars.add(dict(name="nL", value=2.05, min=0, max=60,
                    free=True,
                    latex=r"n_l"))
      pars.add(dict(name="aR", value=1.03, min=0.5, max=10,
                    free=True,
                    latex=r"a_r"))
      pars.add(dict(name="nR", value=3.02, min=0, max=60,
                    free=True,
                    latex=r"n_r"))
      # }}}
    elif "crystalball" in model or 'dscb' in model:
      # Crystal Ball tails {{{
      pars.add(dict(name="aL", value=2.5, min=0.0, max=5,
                    free=True,
                    latex=r"a_l"))
      pars.add(dict(name="nL", value=1.20, min=0, max=500,
                    free=True,
                    latex=r"n_l"))
      pars.add(dict(name="aR", value=2.5, min=0.0, max=5,
                    free=True,
                    latex=r"a_r"))
      pars.add(dict(name="nR", value=1.20, min=0, max=500,
                    free=True,
                    latex=r"n_r"))
      # }}}
    elif "dscbcalib" in model:
      # Crystal Ball with per event resolution tails {{{
      pars.add(dict(name="aL", value=1.4, min=-1, max=10.5,
                    free=True,
                    latex=r"a_l"))
      pars.add(dict(name="nL", value=1, min=-1, max=50,
                    free=True,
                    latex=r"n_l"))
      pars.add(dict(name="aR", value=1.4, min=-1, max=10.5,
                    free=True,
                    latex=r"a_r"))
      pars.add(dict(name="nR", value=1, min=-1, max=50,
                    free=True,
                    latex=r"n_r"))
      # }}}
    elif "dgauss" in model:
      # Crystal Ball with per event resolution tails {{{
      # pars.add(dict(name="frac", value=_res, min=0, max=10, # WARN
      pars.add(dict(name="frac", value=_res, min=0, max=1,
                    free=True,
                    latex=r"f"))
      # }}}
    # Combinatorial background
    pars.add(dict(name='b', value=-0.001, min=-1, max=1,
                  free=False if prefit else True,
                  latex=r'b'))
    pars.add(dict(name='fcomb', formula="1-fsigBs",
                  latex=r'f_{\text{comb}}'))

  # finally, set mass lower and upper limits
  pars.add(dict(name="mLL", value=mLL, free=False, latex=r"m_{ll}"))
  pars.add(dict(name="mUL", value=mUL, free=False, latex=r"m_{ul}"))
  # if verbose:
  #     print(pars)
  print(pars)

  # }}}

  # Fit {{{

  # 1st: do a fast fit to 100k events
  # if verbose:
  #     print("Running fast fit")
  # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd100k},
  #                        method='minuit', verbose=False, strategy=1)
  # for k, v in pars.items():
  #     if v.free:
  #         pars[k].value=res.params[k].value
  #         pars[k].init=res.params[k].value

  # 2nd: fit to the full sample
  res = False
  if verbose:
    print("Running full fit")
  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                         method='minuit', verbose=verbose, strategy=2)

  if verbose:
    print(res)
  if res:
    fpars = ipanema.Parameters.clone(res.params)
  else:
    fpars = ipanema.Parameters.clone(pars)
  print(fpars)
  # fpars = ipanema.Parameters.clone(pars)

  if free_tails:
    for k, v in pars.items():
      if v.free:
        pars[k].value = res.params[k].value
        pars[k].init = res.params[k].value
      if k.startswith('a') or k.startswith('n') or k == 'lambd' or k == 'zeta' or k == 'beta':
        # if k.startswith('a') or k.startswith('n') or k=='lambd':
        pars[k].free = True
    if verbose:
      print("Running fit with free tails")
    print(pars)
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=verbose, strategy=1)
    print(res)

  # }}}

  # plot the fit result {{{

  # fall back to averaged resolution when plotting
  _p = fpars.valuesdict()
  if model == 'dgauss':
    merr = rd.merr
  elif model == 'dgauss':
    merr = _p["s0Bs"] + _p["s1Bs"] * rd.merr + _p["s2Bs"] * rd.merr * rd.merr
  merr = np.median(ristra.get(rd.merr))

  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(
      ristra.get(rd.mass), weights=rd.df.eval(mass_weight), bins=100, density=False
  )
  axplot.errorbar(
      hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
  )

  mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
  signal = 0 * mass
  merr = ristra.allocate(np.ones_like(mass.get()) * merr)

  # plot signal: nbkg -> 0 and nexp -> 0
  _p = ipanema.Parameters.clone(fpars)
  if "fsigBd" in _p:
    _p["fsigBd"].set(value=0, min=-np.inf, max=np.inf)
  if "fcomb" in _p:
    _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
  _x, _y = ristra.get(mass), ristra.get(
      pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  axplot.plot(_x, _y, color="C1", label=rf"${mode_label}$ {model}")

  # plot backgrounds: nsig -> 0
  if has_bd:
    _p = ipanema.Parameters.clone(fpars)
    if "fsigBs" in _p:
      _p["fsigBs"].set(value=0, min=-np.inf, max=np.inf)
    if "fcomb" in _p:
      _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.plot(_x, _y, "-.", color="C2", label="Bd")

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  x, y = ristra.get(mass), ristra.get(
      pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  axplot.plot(x, y, color="C0")
  pulls = complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr)
  if np.amax(np.abs(pulls)) > 4:
    print(f"ERROR: fit did not converge: max. pull = {np.amax(pulls)}")
  axpull.fill_between(
      hdata.bins, pulls,
      0,
      facecolor="C0",
      alpha=0.5,
  )

  # label and save the plot
  axpull.set_xlabel(r"$m(J\!/\psi X)$ [MeV/$c^2$]")
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_yticks([-5, 0, 5])
  axplot.set_ylabel("Candidates")
  axplot.legend(loc="upper left")
  if figs:
    os.makedirs(figs, exist_ok=True)
    n_files = len(next(os.walk(figs))[2])
    if not n_files:
      fig.savefig(os.path.join(figs, "fit.pdf"))
    else:
      fig.savefig(os.path.join(figs, f"fit{n_files+1}.pdf"))
  axplot.set_yscale("log")
  try:
    axplot.set_ylim(1e0, 1.5 * np.max(y))
  except:
    print("Axes were not scaled, because log of zero events")
  if figs:
    if not n_files:
      fig.savefig(os.path.join(figs, "logfit.pdf"))
    else:
      fig.savefig(os.path.join(figs, f"logfit{n_files+1}.pdf"))
  plt.close()

  # }}}

  # compute sWeights if asked {{{

  if sweights:
    # separate paramestes in yields and shape parameters
    _yields = ipanema.Parameters.find(fpars, "fsig.*") + ["fcomb"]
    _pars = list(fpars)
    [_pars.remove(_y) for _y in _yields]
    _yields = ipanema.Parameters.build(fpars, _yields)
    _pars = ipanema.Parameters.build(fpars, _pars)

    # comput sweights
    sw = splot.compute_sweights(
        lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y),
        _pars, _yields, rd.weight
    )
    for k, v in sw.items():
      _sw = np.copy(_proxy)
      _sw[list(rd.df.index)] = v
      sw[k] = _sw
    return (fpars, sw)

  # }}}

  return (fpars, False)

# }}}


# command-line interface {{{

if __name__ == '__main__':
  DESCRIPTION = """
    Backround subtraction script for the selection pipeline. It computes
    sWeights for Bs, Bd and Bu modes using Hypatia and DSCB lineshapes.
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--sample',
                 help='Dataset to fit and get sweights for')
  p.add_argument('--input-params', default=False,
                 help='In case of doing a prefit, provide here parameters')
  p.add_argument('--output-params',
                 help='Output mass fit parameters')
  p.add_argument('--output-figures',
                 help='Mass fit plots')
  p.add_argument('--mass-model',
                 help='Lineshape to fit the mass with')
  p.add_argument('--mass-branch', default='B_ConstJpsi_M_1',
                 help='Variable for the mass')
  p.add_argument('--mass-weight', default=False,
                 help='Weight for the likelihood')
  p.add_argument('--mass-bin', default=False,
                 help='mX bin')
  p.add_argument('--trigger', default=None,
                 help='Split in trigger')
  p.add_argument('--sweights', default=False,
                 help='Path to the sWeighted sample, if wanted')
  p.add_argument('--mode',
                 help='Mode to of the dataset')
  p.add_argument('--version',
                 help='Version of the dataset')
  p.add_argument('--year',
                 help='Year of the dataset')
  p.add_argument('--override_prefit', default=False,
                 help='Year of the dataset')
  args = vars(p.parse_args())

  if args["sweights"]:
    sweights = True
  else:
    sweights = False

  if args["input_params"]:
    input_pars = ipanema.Parameters.load(args["input_params"])
  else:
    input_pars = False

  mass_branch = args['mass_branch']

  if args["mass_weight"]:
    mass_weight = args["mass_weight"]
  else:
    mass_weight = f"{mass_branch}/{mass_branch}"

  cut = False
  is_prefit = False
  if "tails" in args["output_params"]:
    cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"
    is_prefit = True
  print("is_prefit =", is_prefit)
  is_prefit = bool(args['override_prefit']) if args['override_prefit'] else is_prefit
  print("is_prefit =", is_prefit)

  branches = GBW_CONFIG['all_branches'][args['mode']]
  _branches = sum([get_vars_from_string(k) for k in branches.values()], [])
  df = pd.concat([
      Sample.from_root(sample, branches=_branches).df
      for sample in args["sample"].split(',')])
  [df.eval(f"{k}={v}", inplace=True) for k, v in branches.items()]
  df = df[list(branches.keys())]

  if 'Bu' in args['mode']:
    mass_range = (5240, 5320)
  elif 'Bd' in args['mode']:
    mass_range = (5230, 5330)
  elif 'Bs' in args['mode']:
    mass_range = (5320, 5420)
  else:
    mass_range = False

  # TODO: maybe in the future we want to split by trigger category
  pars, sw = mass_fitter(
      df,
      mass_range=mass_range,  # we compute it from the mass branch range
      mass_branch=args['mass_branch'],  # branch to fit
      mass_weight=mass_weight,  # weight to apply to the likelihood
      trigger=False,  # trigger splitter
      figs=args["output_figures"],  # where to save the fit plots
      model=args["mass_model"],  # mass model to use
      cut=cut,  # extra cuts, if required
      sweights=sweights,  # whether to comput or not the sWeights
      input_pars=input_pars,  # whether to use prefit tail parameters or not
      verbose=True,  # level of verobisty
      prefit=is_prefit,  # level of verobisty
      mode=args['mode']
  )

  pars.dump(args["output_params"])
  if sw:
    df['sw'] = sw[list(sw.keys())[0]]
    with uproot.recreate(args['sweights']) as f:
      _branches = {}
      for k, v in df.items():
        if 'int' in v.dtype.name:
          _v = np.int32
        elif 'bool' in v.dtype.name:
          _v = np.int32
        else:
          _v = np.float64
        _branches[k] = _v
      mylist = list(dict.fromkeys(_branches.values()))
      f["DecayTree"] = uproot.newtree(_branches)
      f["DecayTree"].extend(df.to_dict(orient='list'))

# }}}


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
