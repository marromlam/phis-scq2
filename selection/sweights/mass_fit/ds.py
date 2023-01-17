__all__ = ["hypatia_exponential"]
__author__ = ["Marcos Romero"]
__email__ = ["mromerol@cern.ch"]


# Modules {{{

# from logging import exception
import os
import typing
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
# import re
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
# from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and
from scipy import interpolate
import pandas as pd

import complot
import ipanema
import matplotlib.pyplot as plt
import numpy as np
import uproot3 as uproot
from ipanema import Sample, ristra, splot

import config
from utils.helpers import cuts_and, trigger_scissors
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
# from utils.strings import printsec, printsubsec


# initialize ipanema3 and compile lineshapes
ipanema.initialize(config.user["backend"], 1)

prog = ipanema.compile(
    """
# define USE_DOUBLE 1
# include <exposed/kernels.ocl>
"""
)

# }}}


# hypatia + exponential {{{

def hypatia_exponential(
        mass, prob,
        fsigBs=0, fsigBd=0, fcomb=0, freco=0,  # fbackground=0,
        muBs=0, sigmaBs=10,
        muBd=0, sigmaBd=1,
        lambd=0, zeta=0,
        beta=0, aL=0, nL=0, aR=0, nR=0, alpha=0, norm=1, mLL=None, mUL=None,
        *args, **kwargs):
  """
  Hypatia exponential Bs and Bd lineshape for DsPi
  """
  pdfBd = 0
  nBd = 1
  # Bs hypatia {{{
  prog.py_ipatia(prob, mass, np.float64(muBs), np.float64(sigmaBs),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(mass)),)
  pdfBs = 1.0 * prob.get()
  prob = 0 * prob
  # }}}
  # Bd hypatia {{{
  if fsigBd:
    prog.py_ipatia(prob, mass, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(mass)),)
    pdfBd = 1.0 * prob.get()
    prob = 0 * prob
  # }}}

  # normalize
  _x = ristra.linspace(mLL, mUL, 2000)
  _y = _x * 0
  # Bs hypatia integral {{{
  prog.py_ipatia(_y, _x, np.float64(muBs), np.float64(sigmaBs),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(_x)),)
  nBs = np.trapz(ristra.get(_y), ristra.get(_x))
  # }}}
  # Bd hypatia integral {{{
  if fsigBd:
    prog.py_ipatia(_y, _x, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)),)
    nBd = np.trapz(ristra.get(_y), ristra.get(_x))
  # }}}

  prog.kernel_exponential(prob, mass, np.float64(alpha),
                          np.float64(mLL), np.float64(mUL),
                          global_size=(len(mass)),)
  combinatorial = 1.0 * prob.get()

  # compute pdf value
  ans = fsigBs * pdfBs / nBs                              # Bs
  ans += freco * fsigBd * pdfBd / nBd                             # Bd
  ans += fcomb * combinatorial                            # comb
  return norm * ans

# }}}


# Bs mass fit function {{{

def mass_fitter(
        odf: pd.DataFrame,
        mass_range: typing.Optional[tuple] = None,
        mass_branch: str = "B_ConstJpsi_M_1",
        mass_weight: str = "B_ConstJpsi_M_1/B_ConstJpsi_M_1",
        cut: typing.Optional[str] = None,
        figs: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        has_reco: bool = False,
        trigger: typing.Optional[str] = "combined",
        input_pars: typing.Optional[str] = None,
        sweights: bool = False,
        verbose: bool = False) -> typing.Tuple[ipanema.Parameters, typing.Optional[dict]]:

  # mass range cut
  if not mass_range:
    mass_range = (min(odf[mass_branch]), max(odf[mass_branch]))
  mLL, mUL = mass_range
  mass_cut = f"{mass_branch} > {mLL} & {mass_branch} < {mUL}"

  # mass cut and trigger cut
  current_cut = trigger_scissors(trigger, cuts_and(mass_cut, cut))

  # Select model and set parameters {{{
  #    Select model from command-line arguments and create corresponding set
  #    of paramters

  # Chose model {{{

  model = 'hypatia'
  if model == 'hypatia':
    signal_pdf = hypatia_exponential
  else:
    signal_pdf = hypatia_exponential
    raise ValueError(f"{model} cannot be assigned as mass model")

  def roo_hist_pdf(hist):
    h, e = hist
    c = 0.5 * (e[1:] + e[:-1])
    fun = interpolate.interp1d(c, h, fill_value=(h[0], 0.), bounds_error=False)
    # fun = interpolate.interp1d(c, h, fill_value="extrapolate", bounds_error=False)
    # _x = np.linspace(5100, 5600, 1000)
    # norm = np.trapz(fun(_x), _x)
    # fun = interpolate.interp1d(_x, fun(_x) / norm)
    _x = np.linspace(mLL, mUL, 2000)
    norm = np.trapz(fun(_x), _x)
    if norm == 0:
      return interpolate.interp1d([mLL, mUL], [0, 0])
    return interpolate.interp1d(_x, fun(_x) / norm)

  bs_dskx = uproot.open("/scratch49/forMarcos/bs_dskx.root")
  bs_dskx = bs_dskx[bs_dskx.keys()[0]].numpy()
  pdf_bs_dskx = roo_hist_pdf(bs_dskx)

  bs_dsx = uproot.open("/scratch49/forMarcos/bs_dsx.root")
  bs_dsx = bs_dsx[bs_dsx.keys()[0]].numpy()
  pdf_bs_dsx = roo_hist_pdf(bs_dsx)

  DsK_mass_template = uproot.open("/scratch49/forMarcos/DsK_mass_template.root")

  h_Bs2DsstPi = DsK_mass_template['m_Bs2DsstPi'].numpy()
  pdf_Bs2DsstPi = roo_hist_pdf(h_Bs2DsstPi)
  h_Bs2DsRho = DsK_mass_template['m_Bs2DsRho'].numpy()
  pdf_Bs2DsRho = roo_hist_pdf(h_Bs2DsRho)
  h_Lb2LcPi = DsK_mass_template['m_Lb2LcPi'].numpy()
  pdf_Lb2LcPi = roo_hist_pdf(h_Lb2LcPi)
  h_Bd2DPiX = DsK_mass_template['m_Bd2DPiX'].numpy()
  pdf_Bd2DPiX = roo_hist_pdf(h_Bd2DPiX)

  # _x = np.linspace(mLL, mUL, 1000)
  # plt.plot(_x, pdf_Bd2DPiX(_x), label="pdf-Bd2DPiX")
  # plt.plot(_x, pdf_Bs2DsRho(_x), label="pdf-Bs2DsRho")
  # plt.plot(_x, pdf_Bs2DsstPi(_x), label="pdf-Bs2DsstPi")
  # plt.plot(_x, pdf_Lb2LcPi(_x), label="pdf-Lb2LcPi")
  # plt.plot(_x, pdf_bs_dskx(_x), label="pdf-bs-dskx")
  # plt.plot(_x, pdf_bs_dsx(_x), label="pdf-bs-dsx")
  # plt.legend()
  # plt.savefig("tu_madre_es_puta.pdf")
  # plt.close()
  # # h_Bs2DsstPi = DsK_mass_template['m_Bs2DsstPi'].numpy()
  # # h_Bs2DsRho =  DsK_mass_template['m_Bs2DsRho'].numpy()
  # # h_Lb2LcPi = DsK_mass_template['m_Lb2LcPi'].numpy()
  # # h_Bd2DPiX = DsK_mass_template['m_Bd2DPiX'].numpy()

  def pdf(mass, prob, norm=1, *args, **kwargs):
    mass_h = ristra.get(mass)
    _prob = ristra.get(signal_pdf(mass=mass, prob=prob, norm=norm, *args, **kwargs))
    _prob += norm * kwargs['freco'] * kwargs['fDsK'] * pdf_bs_dskx(mass_h)
    _prob += norm * kwargs['freco'] * kwargs['fDsX'] * pdf_bs_dsx(mass_h)
    _prob += norm * kwargs['freco'] * kwargs['fDsstPi'] * pdf_Bs2DsstPi(mass_h)
    _prob += norm * kwargs['freco'] * kwargs['fDsRho'] * pdf_Bs2DsRho(mass_h)
    _prob += norm * kwargs['freco'] * kwargs['fLb'] * pdf_Lb2LcPi(mass_h)
    _prob += norm * kwargs['freco'] * kwargs['fBdDsPi'] * pdf_Bd2DPiX(mass_h)
    return _prob

  def fcn(params, data):
    p = params.valuesdict()
    prob = ristra.get(pdf(mass=data.mass, prob=data.pdf, **p, norm=1))
    return -2.0 * np.log(prob) * ristra.get(data.weight)

  # }}}

  # Generate all parameters if there is no input pars
  if not input_pars:
    # all paramaters {{{
    pars = ipanema.Parameters()
    pars.add(dict(name="fsigBs", value=0.43, min=0, max=1, free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^- \pi^+}"))
    pars.add(dict(name="fsigBd", value=0.019, min=0, max=1, free=True,
                  latex=r"f_{B_d^0 \rightarrow D_s^- \pi^+}"))
    pars.add(dict(name='fDsK', value=0.008, min=0, max=1, free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^- K^+}"))
    pars.add(dict(name='fBdDsPi', value=0.00, min=0, max=1, free=True,
                  latex=r"f_{B_d^0 \rightarrow D_s^- \pi^+ x}"))
    pars.add(dict(name='fLb', value=0.0087, min=0, max=1, free=True,
                  latex=r"f_{\Lambda_b \rightarrow \Lambda_c^- \pi^+}"))
    pars.add(dict(name='fDsstPi', value=0.7, min=0, max=1, free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^{*-} \pi^+}"))
    pars.add(dict(name='fDsX', value=0.0, min=0, max=1, free=True,
                  latex=r"f_{B_s^0 \rightarrow D_s^- \pi^+ x}"))
    pars.add(dict(name='fDsRho',
                  formula="1-fsigBd-fDsK-fBdDsPi-fLb-fDsstPi-fDsX",  # min=0, max=1,
                  latex=r"f_{B_s^0 \rightarrow D_s^- \rho^+}"))
    pars.add(dict(name='freco',
                  value=0.45, min=0.0, max=1, free=True,
                  latex=r"f_{\textrm{Part. Reco.}}"))
    pars.add(dict(name='fcomb',
                  formula="1-fsigBs-freco",  # min=0.0, max=1,
                  latex=r"f_{\textrm{Combinatorial}}"))

    # model base parameters
    pars.add(dict(name="muBs", value=5366, min=5330, max=5410,
                  latex=r"\mu_{B_s^0}"))
    pars.add(dict(name="sigmaBs", value=18, min=0.1, max=100, free=True,
                  latex=r"\sigma_{B_s^0}"))
    pars.add(dict(name="muBd", formula=f"muBs-87.19",
                  latex=r"\mu_{B_d^0}"))
    pars.add(dict(name="sigmafrac", value=1.1155, min=0., max=2., free=False,
                  latex=r"\sigma_{B_d^0}/\sigma_{B_s^0}"))
    pars.add(dict(name="sigmaBd", formula="sigmaBs*sigmafrac",
                  latex=r"\sigma_{B_d^0}"))
    # Combinatorial background
    pars.add(dict(name='alpha', value=-0.00313, min=-1, max=1, free=True,
                  latex=r'b'))
    # now we fix them
    pars.lock('fsigBd', 'fDsK', 'fBdDsPi', 'fLb', 'fDsstPi', 'fDsX',
              'fDsRho', 'alpha', 'sigmafrac')
    pars['freco'].set(value=0, free=False)
    pars['fsigBs'].set(value=1, free=False)
    # finally, set mass lower and upper limits
    pars.add(dict(name="mLL", value=mLL, free=False,
                  latex=r"m_{ll}"))
    pars.add(dict(name="mUL", value=mUL, free=False,
                  latex=r"m_{ul}"))
    # }}}

  if input_pars:
    pars = ipanema.Parameters.clone(input_pars)
    pars.lock()
    if not has_reco:
      # This would be MC2Bs2DsPi mass fit (not prefit)
      # pars.remove("alpha", "aL", "nL", "aR", "nR", "beta")
      pars.unlock("muBs", "sigmaBs", "alpha")
      pars.unlock("aL", "nL", "aR", "nR")
      if 'lambd' in pars:
        pars.unlock("lambd", "beta")
      pars['freco'].set(value=0, free=False)
      pars['fsigBs'].set(value=0.5, free=True)
    else:
      # If it is Bs2DsPi, then either full mass window or narrow, tails are
      # fixed. Beta is also fixed to MC
      if 'lambd' in pars:
        pars.unlock("lambd", "beta")
      pars.unlock("muBs", "sigmaBs", "alpha", "sigmafrac")
      pars.unlock('fsigBd', 'fDsK', 'fBdDsPi', 'fLb', 'fDsstPi', 'fDsX', 'fDsRho')
      pars['freco'].set(value=0.45, free=True)
      pars['freco'].min = 0.0
      pars['fsigBs'].set(value=0.43, free=True)
      pars['fcomb'].min = 0
      pars['fcomb'].max = 1
      if sweights:
        pars.lock('fsigBd', 'fDsK', 'fBdDsPi', 'fLb', 'fDsstPi', 'fDsX', 'fDsRho')
        pars.unlock('fsigBs', 'freco')
        # pars.lock('sigmaBs')
        pars.lock('sigmafrac', 'beta')
      # pars['freco'].set(value=0)
  else:
    # add parameters depending on model
    if "hypatia" in model:
      # Hypatia tails {{{
      pars.add(dict(name="lambd", value=-2.9, min=-10, max=10.1, free=True,
                    latex=r"\lambda"))
      pars.add(dict(name="zeta", value=1e-6, free=False,
                    latex=r"\zeta"))
      pars.add(dict(name="beta", value=0.0, min=-1, max=1, free=False,
                    latex=r"\beta"))
      pars.add(dict(name="aL", value=2, min=0.1, max=5.5, free=True,
                    latex=r"a_l"))
      pars.add(dict(name="nL", value=1.6, min=0, max=6, free=True,
                    latex=r"n_l"))
      pars.add(dict(name="aR", value=2, min=0.1, max=5.5, free=True,
                    latex=r"a_r"))
      pars.add(dict(name="nR", value=0.68, min=0, max=6, free=True,
                    latex=r"n_r"))
      # }}}
    else:
      raise ValueError(f"{model} cannot be assigned as mass_model")

  pars['mLL'].set(value=mLL)
  pars['mUL'].set(value=mUL)
  print(pars)

  # }}}

  # Allocate the sample variables {{{

  print(f"Cut: {current_cut}")
  print(f"Mass branch: {mass_branch}")
  print(f"Mass weight: {mass_weight}")
  rd = Sample.from_pandas(odf)
  print(f"Events (before cut): {rd.shape}")
  _proxy = np.float64(rd.df[mass_branch]) * 0.0 - 999
  rd.chop(current_cut)
  print(f"Events (after cut): {rd.shape}")
  rd.allocate(mass=mass_branch)
  rd.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)
  rd.weight *= ipanema.ristra.sum(rd.weight) / ipanema.ristra.sum(rd.weight**2)

  # }}}

  # fit {{{

  res = False
  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                         method='minuit', verbose=verbose, strategy=2)
  if sweights:
    res.params['beta'].free = True
    res.params['lambd'].free = True
    res = ipanema.optimize(fcn, res.params, fcn_kwgs={'data': rd},
                           method='minuit', verbose=verbose, strategy=2)

  if res:
    print(res)
    fpars = ipanema.Parameters.clone(res.params)
  else:
    print("Could not fit it!. Cloning pars to res")
    fpars = ipanema.Parameters.clone(pars)
  for v in fpars.values():
    v.set(value=v.value, min=-np.inf, max=np.inf, free=False)
  print(fpars)

  # }}}

  # plot the fit result {{{

  _mass = ristra.get(rd.mass)
  _weight = rd.df.eval(mass_weight)

  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(_mass, weights=_weight, bins=100, density=False)
  axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                  fmt=".k")

  proxy_mass = ristra.linspace(min(_mass), max(_mass), 1000)
  proxy_prob = 0 * proxy_mass

  # plot subcomponents
  for icolor, pspecie in enumerate(fpars.keys()):
    _color = f"C{icolor+1}"
    if pspecie.startswith('f') and pspecie != 'freco':
      _label = rf"${fpars[pspecie].latex.split('f_{')[-1][:-1]}$"
      _p = ipanema.Parameters.clone(fpars)
      for f in _p.keys():
        if f.startswith('f'):
          if f != pspecie and f != 'freco':
            _p[f].set(value=0, min=-np.inf, max=np.inf)
      _prob = pdf(proxy_mass, proxy_prob, **_p.valuesdict(), norm=hdata.norm)
      axplot.plot(ristra.get(proxy_mass), ristra.get(_prob), color=_color,
                  label=_label)

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  _prob = pdf(proxy_mass, proxy_prob, **_p.valuesdict(), norm=hdata.norm)
  axplot.plot(ristra.get(proxy_mass), _prob, color="C0")
  pulls = complot.compute_pdfpulls(ristra.get(proxy_mass), ristra.get(_prob),
                                   hdata.bins, hdata.counts, *hdata.yerr)
  axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)

  # label and save the plot
  axpull.set_xlabel(r"$m(D_s \pi)$ [MeV/$c^2$]")
  axplot.set_ylabel(rf"Candidates")
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_yticks([-5, 0, 5])
  # axpull.set_yticks([-2, 0, 2])
  axpull.hlines(3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axpull.hlines(-3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  axplot.legend(loc="upper right", prop={'size': 8})
  v_mark = 'LHC$b$'  # watermark plots
  tag_mark = 'THIS THESIS'
  watermark(axplot, version=v_mark, tag=tag_mark, scale=1.2)
  if figs:
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, f"fit{mLL}.pdf"))
    fig.savefig(os.path.join(f"fit.pdf"))
  axplot.set_yscale("log")
  try:
    axplot.set_ylim(1e0, 1.5 * np.max(_prob))
  except:
    print("axes not scaled")
  if figs:
    fig.savefig(os.path.join(figs, f"logfit{mLL}.pdf"))
    fig.savefig(os.path.join(f"logfit.pdf"))
  plt.close()

  # }}}

  # compute sWeights if asked {{{

  if sweights:
    # separate paramestes in yields and shape parameters
    _yields = ['fsigBs', 'freco', 'fcomb']
    print(_yields)
    _pars = list(fpars)
    [_pars.remove(_y) for _y in _yields]
    _yields = ipanema.Parameters.build(fpars, _yields)
    _pars = ipanema.Parameters.build(fpars, _pars)

    # WARNING: Breaking change!  -- February 4th
    #          sw(p, y, len(data)) * wLb != sw(p, y, wLb.sum())
    #          which one is correct? Lera does the RS and I did LS
    # sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y), _pars, _yields, ristra.get(rd.weight).sum())
    sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.pdf, *x, **y), _pars, _yields, rd.weight)
    for k, v in sw.items():
      _sw = np.copy(_proxy)
      _sw[list(rd.df.index)] = v
      sw[k] = _sw
    # print("sum of wLb", np.sum( rd.df.eval(mass_weight).values ))
    print(sw)
    return (res.params, sw)

  # }}}

  return (res.params, False)


# }}}


# command-line interface {{{

if __name__ == '__main__':
  p = argparse.ArgumentParser(description="arreglatelas!")
  p.add_argument('--sample')
  p.add_argument('--input-params', default=False)
  p.add_argument('--output-params')
  p.add_argument('--output-figures')
  p.add_argument('--mass-model', default='hypatia')
  p.add_argument('--mass-weight')
  p.add_argument('--mass-branch', default='B_PVFitDs_M_1')
  p.add_argument('--mass-bin', default=False)
  p.add_argument('--trigger')
  p.add_argument('--sweights')
  p.add_argument('--mode')
  p.add_argument('--version')
  args = vars(p.parse_args())

  # args['sample'] = "/scratch49/forMarcos/Bs2DsPi/Bs2DsPi_2017_selected_v1r0.root"
  # args['input_params'] = False
  # args['output_params'] = "mierda.json"
  # args['output_figures'] = "shit_figs/"
  # args['mass_model'] = 'hypatia'
  # args['mass_branch'] = 'B_PVFitDs_M_1'
  # args['mass_bin'] = False
  # args['trigger'] = None
  # args['sweights'] = "shit.npy"
  # args['mode'] = 'Bs2DsPi'
  # args['version'] = 'v1r0'

  if args["sweights"]:
    sweights = True
  else:
    sweights = False

  if args["input_params"]:
    input_pars = ipanema.Parameters.load(args["input_params"])
    # mass_range = (5300, 5600)  # narrow window
    mass_range = (5100, 5600)  # wide window
  else:
    input_pars = False
    mass_range = (5300, 5600)  # narrow window
    mass_range = (5100, 5600)  # wide window

  mass_branch = args['mass_branch']
  branches = [mass_branch]

  if args["mass_weight"]:
    mass_weight = args["mass_weight"]
    branches += [mass_weight]
  else:
    mass_weight = f"{mass_branch}/{mass_branch}"

  cut = None
  if "prefit" in args["output_params"]:
    cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"
    branches += ["B_BKGCAT"]
  # cut = None

  if 'MC' in args['mode']:
    has_reco = False
  else:
    has_reco = True

  sample = Sample.from_root(args["sample"], branches=branches)

  if has_reco:
    input_pars, sw = mass_fitter(
        sample.df,
        mass_range=mass_range,
        mass_branch=mass_branch,
        mass_weight=mass_weight,
        trigger=args["trigger"],
        figs=args["output_figures"],
        model=args["mass_model"],
        cut=cut,
        has_reco=has_reco,
        sweights=False,
        input_pars=input_pars,
        verbose=False,
    )
    mass_range = (5300, 5600)  # narrow window
  pars, sw = mass_fitter(
      sample.df,
      mass_range=mass_range,
      mass_branch=mass_branch,
      mass_weight=mass_weight,
      trigger=args["trigger"],
      figs=args["output_figures"],
      model=args["mass_model"],
      cut=cut,
      has_reco=has_reco,
      sweights=sweights,
      input_pars=input_pars,
      verbose=True,
  )
  pars.dump(args["output_params"])
  if args['sweights']:
    np.save(args["sweights"], sw)

# }}}


# vim: fdm=marker
