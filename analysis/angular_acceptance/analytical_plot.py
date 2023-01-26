from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, version_guesser
from analysis.angular_acceptance.analytical_corrected import \
    get_angular_prediction
from analysis import badjanak
import config
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
from scipy.special import comb, lpmv
from scipy.interpolate import interp1d, interpn
from scipy.integrate import romb, simpson
from scipy import special, stats
from ipanema.core.python import ndmesh
from ipanema import (IPANEMALIB, Parameters, Sample, get_confidence_bands,
                     initialize, optimize, ristra, uncertainty_wrapper)
import uproot3 as uproot
import uncertainties.unumpy as unp
import uncertainties as unc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hjson
import complot
import argparse
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]
__all__ = []


import os

import ipanema

ipanema.initialize(config.user["backend"], 1)

# from ipanema import plotting

# initialize('cuda',1)

# badjanak.get_kernels()


order_cosK, order_cosL, order_hphi = config.angacc["legendre_order"]
nob = config.angacc["analytic_bins"][0]


if __name__ == "__main__":
  DESCRIPTION = """
    Computes the legendre-based angular acceptance with corrections in mHH, pB,
    pTB variables using an a reweight.
    """
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument("--samples", help="Bs2JpsiPhi MC sample")
  p.add_argument("--params", help="Bs2JpsiPhi data sample")
  p.add_argument("--angular-acceptance", help="Bs2JpsiPhi MC generator parameters")
  p.add_argument("--output", help="Bs2JpsiPhi MC angular acceptance")
  p.add_argument("--nknots", help="Bs2JpsiPhi MC angular acceptance tex")
  p.add_argument("--mode", help="Mode to compute angular acceptance with")
  p.add_argument("--year", help="Year to compute angular acceptance with")
  p.add_argument("--version", help="Version of the tuples")
  p.add_argument("--trigger", help="Trigger to compute angular acceptance with")
  p.add_argument("--angacc", help="Trigger to compute angular acceptance with")
  p.add_argument("--timeacc", help="Trigger to compute angular acceptance with")
  args = vars(p.parse_args())

  # get list of knots
  from trash_can.knot_generator import create_time_bins

  tLL = 0.3
  tUL = 15
  knots = create_time_bins(int(args["nknots"]), tLL, tUL).tolist()
  # knots = all_knots[args['nknots']]

  #
  MODE = args["mode"]
  TRIGGER = args["trigger"]
  YEAR = args["year"]
  samples = args["samples"]
  gen_params = args["params"].split(",")
  gen_params = [Parameters.load(p) for p in gen_params]
  output = args["output"].split(",")
  angaccs = args["angular_acceptance"].split(",")

  noe = (
      (order_hphi + 1) * (order_cosL + 1) * (order_cosK + 1)
  )  # number of coefficients

  gen = Parameters.load(args["params"])

  pars_dict = np.ascontiguousarray([gen.valuesdict() for i in range(nob ** 3)])

  printsubsec("Prepare arrays")
  sample = Sample.from_root(args["samples"])
  sample.chop(trigger_scissors(TRIGGER))
  # sample = sample.df

  # TODO: These two functions sort elements in the HD-fitter way. This is clearly
  #       uneeded, but one should rewrite the whole code, and loses comparison
  #       capabilitis wrt. HD-fitter.
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

  cosK = 0.5 * (np.linspace(-1, 1, 21)[1:] + np.linspace(-1, 1, 21)[:-1])
  cosL = 0.5 * (np.linspace(-1, 1, 21)[1:] + np.linspace(-1, 1, 21)[:-1])
  hphi = 0.5 * (np.linspace(-1, 1, 21)[1:] + np.linspace(-1, 1, 21)[:-1])

  N = 21
  cosKd = ristra.allocate(
      0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1])
  )
  cosLd = ristra.allocate(
      0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1])
  )
  hphid = ristra.allocate(
      0.5 * (np.linspace(-1, 1, N)[1:] + np.linspace(-1, 1, N)[:-1])
  )

  # sw switcher
  sw = "sWeight/gbWeight"
  sw = "sWeight"
  # if 'MC_Bs2JpsiPhi' in MODE:
  # sw += '/gb_weights'

  # Run all plots
  printsec("Ploting")
  for k, var in enumerate(["cosK", "cosL", "hphi"]):
    if var == "cosK":
      proj = (1, 2)
      bounds = 1
      tex = r"\mathrm{cos}\,\theta_K"
      dir = (0, 1)
      x = np.copy(cosK)
      xh = np.copy(ristra.get(cosKd))
    elif var == "cosL":
      proj = (0, 1)
      bounds = 1
      tex = r"\mathrm{cos}\,\theta_{\mu}"
      dir = (1, 2)
      x = np.copy(cosL)
      xh = np.copy(ristra.get(cosLd))
    elif var == "hphi":
      proj = (2, 0)
      bounds = np.pi
      tex = r"\phi_h\, \mathrm{[rad]}"
      dir = (2, 0)
      x = np.copy(hphi)
      xh = np.copy(ristra.get(hphid))

    fig, ax = complot.axes_plot()
    printsubsec(f"{var} efficiency")

    # 1st plot the full-range angular acceptance ------------------------------

    # hist data
    print(" * Prepare data for the whole range")
    df = sample.cut(f"time>={tLL} & time<={tUL}")
    print(df.shape)
    histdd = np.histogramdd(
        df[["cosK", "cosL", "hphi"]].values,
        bins=(nob, nob, nob),
        weights=df.eval(f"polWeight*{sw}"),
        range=[(-1, 1), (-1, 1), (-np.pi, np.pi)],
    )
    mccounts, (ecosK, ecosL, ehphi) = histdd

    # generate predictions
    print(" * Prepare theory for the whole range")
    m1 = ndmesh(ecosK[:-1], ecosL[:-1], ehphi[:-1])
    m2 = ndmesh(ecosK[1:], ecosL[1:], ehphi[1:])
    arry = np.ascontiguousarray(np.zeros((nob ** 3, 2 * 3)))
    arry[:, 0::2] = np.stack((m.ravel() for m in m1), axis=-1)
    arry[:, 1::2] = np.stack((m.ravel() for m in m2), axis=-1)
    gencounts = get_angular_prediction(pars_dict, *arry.T, tLL=tLL, tUL=tUL)
    gencounts = np.sum(df.eval(sw)) * gencounts.reshape(nob, nob, nob)

    # create arrays
    mc = np.float64([data_3d(i) for i in range(nob ** 3)])
    mc = mc.reshape(nob, nob, nob)
    th = np.float64([prediction_3d(i) for i in range(nob ** 3)])
    th = th.reshape(nob, nob, nob)

    # project data and theory into variable
    _mc = np.sum(mc, proj)
    _th = np.sum(th, proj)
    y = _mc / _th
    _umc = np.sqrt(_mc)
    _uth = 0 * np.sqrt(_th)
    uy = np.sqrt((1 / _th) ** 2 * _umc ** 2 + (-_mc / _th ** 2) ** 2 * _uth ** 2)
    ux = 0.5 * (x[1] - x[0]) * np.ones_like(x)

    # get efficiency, project it and normalize
    upeff = Parameters.load(angaccs[0])
    upeff = [p.uvalue for p in upeff.values()]
    print(upeff)
    eff = badjanak.analytical_angular_efficiency(
        np.array([x.n for x in upeff]),
        cosKd,
        cosLd,
        hphid,
        None,
        order_cosK,
        order_cosL,
        order_hphi,
    )
    eff = np.sum(eff * th, dir) / np.sum(th, dir)
    norm = 1 / (np.trapz(y, x) / np.trapz(eff, xh))
    scale = np.max(y)

    # compute confidence bands
    yunc = uncertainty_wrapper(
        lambda p: np.mean(
            th
            * badjanak.analytical_angular_efficiency(
                p, cosKd, cosLd, hphid, None, order_cosK, order_cosL, order_hphi
            ),
            dir,
        )
        / np.mean(th, dir),
        upeff,
    )
    yl, yh = get_confidence_bands(yunc)

    # plot
    ax.errorbar(
        bounds * x,
        y / scale,
        yerr=uy / scale,
        xerr=bounds * ux,
        fmt=".",
        color="C0",
    )
    ax.plot(bounds * xh, eff / norm / scale, label="full range", color="C0")
    ax.fill_between(
        bounds * x, yh / norm / scale, yl / norm / scale, color="C0", alpha=0.2
    )

    # 2nd: run over all time-binned angular acceptances ------------------------
    if len(angaccs) > 1:
      for i, angacc_ in enumerate(angaccs[1:]):
        # get lower and upper limits
        ll, ul = knots[i: i + 2]

        print(f" * Prepare data in ({ll},{ul})")
        df = sample.cut(f"time>={ll} & time<={ul}")
        print(df.shape)
        histdd = np.histogramdd(
            df[["cosK", "cosL", "hphi"]].values,
            bins=(nob, nob, nob),
            weights=df.eval(f"polWeight*{sw}"),
            range=[(-1, 1), (-1, 1), (-np.pi, np.pi)],
        )
        mccounts, (ecosK, ecosL, ehphi) = histdd

        # generate predictions
        print(f" * Prepare theory in ({ll},{ul})")
        gencounts = get_angular_prediction(pars_dict, *arry.T, tLL=ll, tUL=ul)
        gencounts = np.sum(df.eval(sw)) * gencounts.reshape(nob, nob, nob)

        # create arrays
        mc = np.float64([data_3d(i) for i in range(nob ** 3)]).reshape(
            nob, nob, nob
        )
        th = np.float64([prediction_3d(i) for i in range(nob ** 3)]).reshape(
            nob, nob, nob
        )

        # project data and theory into variable
        _mc = np.sum(mc, proj)
        _th = np.sum(th, proj)
        y = _mc / _th
        _umc = np.sqrt(_mc)
        _uth = 0 * np.sqrt(_th)
        uy = np.sqrt(
            (1 / _th) ** 2 * _umc ** 2 + (-_mc / _th ** 2) ** 2 * _uth ** 2
        )
        ux = 0.5 * (x[1] - x[0]) * np.ones_like(x)

        # get efficiency, project it and normalize
        upeff = Parameters.load(angacc_)
        print(upeff)
        upeff = [p.uvalue for p in upeff.values()]
        eff = badjanak.analytical_angular_efficiency(
            np.array([x.n for x in upeff]),
            cosKd,
            cosLd,
            hphid,
            None,
            order_cosK,
            order_cosL,
            order_hphi,
        )
        eff = np.sum(eff * th, dir) / np.sum(th, dir)
        norm = 1 / (np.trapz(y, x) / np.trapz(eff, xh))
        # scale = np.max(y)

        # compute confidence bands
        yunc = uncertainty_wrapper(
            lambda p: np.mean(
                th
                * badjanak.analytical_angular_efficiency(
                    p,
                    cosKd,
                    cosLd,
                    hphid,
                    None,
                    order_cosK,
                    order_cosL,
                    order_hphi,
                ),
                dir,
            )
            / np.mean(th, dir),
            upeff,
        )
        yl, yh = get_confidence_bands(yunc)

        # plot
        ax.errorbar(
            bounds * x,
            y / scale,
            yerr=uy / scale,
            xerr=bounds * ux,
            fmt=".",
            color=f"C{i+1}",
        )
        ax.plot(
            bounds * xh,
            eff / norm / scale,
            label=f"$t \in ({ll},{ul})$",
            color=f"C{i+1}",
        )
        ax.fill_between(
            bounds * x,
            yh / norm / scale,
            yl / norm / scale,
            color=f"C{i+1}",
            alpha=0.2,
        )
        ax.legend()

    # label and save
    ax.set_ylim(0.85, 1.05)
    ax.set_xlabel(f"${tex}$")
    ax.set_ylabel(rf"$\varepsilon(\Omega)$ [a.u.]")
    _mark = args['version'].split('@')[0]  # watermark plots
    watermark(ax, version=f"${_mark}$", scale=1.02)
    # fig.savefig(f"{var}.pdf")
    fig.savefig(output[k])
