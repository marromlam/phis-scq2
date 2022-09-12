__all__ = ["cb_exponential3"]
__author__ = ["Marcos Romero"]
__email__ = ["mromerol@cern.ch"]



# Modules {{{

import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
import re
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and

import complot
import ipanema
import matplotlib.pyplot as plt
import numpy as np
import uproot3 as uproot
from ipanema import Sample, ristra, splot

import config
from utils.helpers import cuts_and, trigger_scissors
from utils.strings import printsec, printsubsec


def get_sizes(size, BLOCK_SIZE=256):
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

from analysis.csp_factors.efficiency import (create_mass_bins,
                                             create_time_bins,
                                             create_sigmam_bins)



# initialize ipanema3 and compile lineshapes
ipanema.initialize(config.user["backend"], 1)
# prog = ipanema.compile("""
# #define USE_DOUBLE 1
# #include <lib99ocl/core.c>
# #include <lib99ocl/complex.c>
# #include <lib99ocl/stats.c>
# #include <lib99ocl/special.c>
# #include <lib99ocl/lineshapes.c>
# #include <exposed/kernels.ocl>
# """)

prog = ipanema.compile(
    """
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>


WITHIN_KERNEL
ftype __dcb_with_calib(const ftype mass, const ftype merr,
    const ftype mu, const ftype sigma,
    const ftype aL, const ftype nL, const ftype aR, const ftype nR,
    const ftype mLL, const ftype mUL)
{
  const ftype t = (mass - mu)/sigma;
  const ftype tLL = (mLL - mu)/sigma;
  const ftype tUL = (mUL - mu)/sigma;

  if (get_global_id(0) == 0){
  printf("mass=%f, mu=%f, sigma=%f, aL=%f, nL=%f, aR=%f, nR=%f, mLL=%f, mUL=%f\\n",
           mass, mu, sigma, aL, nL, aR, nR, mLL, mUL);
  }

  const ftype AR = rpow(nR / aR, nR) * exp(-0.5 * aR * aR);
  ftype BR = nR / aR - aR;
  const ftype AL = rpow(nL / aL, nL) * exp(-0.5 * aL * aL);
  ftype BL = nL / aL - aL;
  // BR = BR > 0. ? BR : 1e-8;
  // BL = BL > 0. ? BL : 1e-8;

  if (get_global_id(0) == 0){
      printf("%f \\n", BL);
      printf("%f \\n", BR);
  }

  if (get_global_id(0) == 0){
    printf("%f, %f, %f, %f\\n", AR, BR, AL, BL);
  }

  // TODO : protect nR/nL === 1 (integral diverges) 
  ftype den = sqrt(M_PI/2.)*(erf(aL/sqrt(2.)) + erf(aR/sqrt(2.)));
  if (get_global_id(0) == 0){
      printf("(%f,%f) :: den 1 =%f\\n", mLL, mUL, den);
  }
  den += (AR*(rpow(aR + BR,1 - nR) - rpow(BR + tUL,1 - nR)))/(-1 + nR);
  if (get_global_id(0) == 0){
      printf("(%f,%f) :: den 2 =%f\\n", mLL, mUL, den);
  }
  den += (AL*(rpow(aL + BL,1 - nL) - rpow(BL - tLL,1 - nL)))/(-1 + nL);
  if (get_global_id(0) == 0){
      printf("(%f,%f) :: den 3 =%f\\n", mLL, mUL, den);
      printf("BL - tLL  =%f\\n", BL-tLL);
  }

  ftype num = 0;
  if ( t > aR )
  {
    num = ( AR / rpow(BR + t, nR) );
  }
  else if ( t < -aL )
  {
    num = ( AL / rpow(BL - t, nL) );
  }
  else
  {
    num = ( exp(-0.5 * t * t) );
  }
  if (get_global_id(0) == 0){
      printf("num/den = %f/%f\\n", num, den);
  }
  return num / den;
}

WITHIN_KERNEL
ftype __dcb_with_calib_numerical(const ftype mass, const ftype merr,
    const ftype mu, const ftype sigma,
    const ftype aL, const ftype nL, const ftype aR, const ftype nR,
    const ftype mLL, const ftype mUL)
{
  ftype num = double_crystal_ball(mass, mu, sigma, aL, nL, aR, nR);
  if (get_global_id(0) == 0) {
    printf("num = %f\\n", num);
  }

  ftype den = 0.;
  ftype oldden = 0.0;
  ftype __x, __tnm, __sum, __del;
  ftype __s;
  int __it, __j;

  for (int n=1; n<=20; n++)
  {
    // TRAPZ SOUBROUTINE {{{

    if (n == 1)
    {
      den = 0.5 * (mUL-mLL) * (
        double_crystal_ball(mLL, mu, sigma, aL, nL, aR, nR) + 
        double_crystal_ball(mUL, mu, sigma, aL, nL, aR, nR)
      );
    }
    else
    {
      for (__it=1, __j=1; __j<n-1; __j++) __it <<= 1;
      __tnm = __it;
      __del = (mUL-mLL)/__tnm;
      __x = mLL + 0.5 * __del;

      for (__sum=0.0, __j=1; __j<=__it; __j++, __x+=__del)
      {
        __sum += double_crystal_ball(__x, mu, sigma, aL, nL, aR, nR);
        den = 0.5 * ( den + (mUL-mLL)*__sum / __tnm );
      }
    }
    if (get_global_id(0) == 0) {
      printf("den[%d] = %f\\n", n, den);
    }

    // }}}  // END TRAPZD SOUBROUTINE




    if (n > 5)  // avoid spurious early convergence
    {
      if (fabs(den-oldden) < 1e-5*fabs(oldden) || (den == 0.0 && oldden == 0.0))
      {
        return num / den;
      }
    }
    oldden = den;
  }

  return num / den;
}

KERNEL
void DoubleCrystallBallWithCalibration(
    GLOBAL_MEM ftype *prob, GLOBAL_MEM ftype *data,
    const ftype fBs, const ftype fBd,
    const ftype muBs, const ftype s0, const ftype s1, const ftype s2,
    const ftype muBd, const ftype sBd,
    const ftype aL, const ftype nL, const ftype aR, const ftype nR,
    const ftype b,
    const ftype mLL, const ftype mUL)
{
  const int idx = get_global_id(0);
  const ftype mass = data[2*idx + 0];
  const ftype merr = data[2*idx + 1];
  const ftype sBs = s0 + s1*merr + s2*merr*merr;

  ftype probBs = 0.0;
  ftype probBd = 0.0;
  if (fBs > 1e-14) {
    probBs = __dcb_with_calib_numerical(mass, merr, muBs, sBs, aL, nL, aR, nR, mLL, mUL);
  }
  if (fBd > 1e-14) {
    probBd = __dcb_with_calib(mass, merr, muBd, sBd, aL, nL, aR, nR, mLL, mUL);
  }
  const ftype probComb = (exp(-b*mass)) / ((exp(-b*mLL) - exp(-b*mUL))/b);

  prob[idx] = probBs; // + fBd * probBd + (1-fBs-fBd) * probComb; 
}


KERNEL
void DoubleCrystallBallPerEventSigma(
    GLOBAL_MEM ftype *prob, GLOBAL_MEM ftype *mass,
    const ftype muBs, GLOBAL_MEM const ftype *sigmaBs,
    const ftype aL, const ftype nL, const ftype aR, const ftype nR)
{
  const int idx = get_global_id(0);

  prob[idx] = double_crystal_ball(mass[idx], muBs, sigmaBs[idx], aL, nL, aR, nR);
}

"""
)

# }}}


# ipatia + exponential {{{


def ipatia_exponential(
    mass, signal,
    fsigBs=0, fsigBd=0, fcomb=0,
    muBs=0, sigmaBs=10,
    muBd=0, sigmaBd=1, frac=0, f_hyp_gauss=0,
    lambd=0, zeta=0,
    beta=0, aL=0, nL=0, aR=0, nR=0, b=0, norm=1, mLL=None, mUL=None,):
    # Bs hypatia {{{
    prog.py_ipatia( signal, mass, np.float64(muBs), np.float64(sigmaBs),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBs = 1.0 * signal.get()
    signal = 0 * signal
    # }}}
    # Bd hypatia {{{
    gaussBd = 0
    # prog.kernel_gaussian(signal, mass,
    #     np.float64(muBd), np.float64(sigmaBd), np.float64(mLL), np.float64(mUL),
    #     global_size=(len(mass)),
    # )
    # gaussBd = 0.0 * signal.get()
    # print(gaussBd)
    # signal = 0 * signal
    prog.py_ipatia( signal, mass, np.float64(muBd), np.float64(sigmaBd),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
    )
    pdfBd = 1.0 * signal.get()
    signal = 0 * signal
    # }}}
    # print(pdfBd)

    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x * 0
    # Bs hypatia integral {{{
    prog.py_ipatia( _y, _x, np.float64(muBs), np.float64(sigmaBs),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
    )
    nBs = np.trapz(ristra.get(_y), ristra.get(_x))
    # }}}
    # Bd hypatia integral {{{
    prog.py_ipatia( _y, _x, np.float64(muBd), np.float64(sigmaBd),
        np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
        np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
    )
    nBd = np.trapz(ristra.get(_y), ristra.get(_x))
    # }}}

    # background {{{
    prog.kernel_exponential(
        signal, mass, np.float64(b), np.float64(mLL), np.float64(mUL),
        global_size=(len(mass)),)
    pComb = ristra.get(signal)
    # }}}

    ans = fsigBs * pdfBs / nBs                                           # Bs
    # ans += fsigBd*( (1-f_hyp_gauss)*gaussBd + f_hyp_gauss*pdfBd/nBd )  # Bd
    ans += fsigBd*pdfBd/nBd
    ans += fcomb * pComb
    return norm * ans


# }}}


# Bs mass fit function {{{


def mass_fitter(
    odf,
    mass_range=False,
    mass_branch="B_ConstJpsi_M_1",
    mass_weight="B_ConstJpsi_M_1/B_ConstJpsi_M_1",
    cut=False,
    figs=False,
    model=False,
    has_bd=False,
    trigger="combined",
    input_pars=False,
    sweights=False,
    verbose=False,
):

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

    with_calib = False
    if model == 'hypatia':
        pdf = ipatia_exponential
    elif model == "crystalball":
        pdf = cb_exponential3
    elif model == "cbcalib":
        pdf = cb_exponential3
        # with_calib = True

    def fcn(params, data):
        p = params.valuesdict()
        prob = pdf(mass=data.mass, signal=data.pdf, **p)
        return -2.0 * np.log(prob) * ristra.get(data.weight)

    # }}}

    pars = ipanema.Parameters()
    # Create common set of Bs parameters (all models must have and use)
    pars.add(dict(name="fsigBs", value=0.9, min=0.0, max=1, free=True,
                  latex=r"f_{B_s}"))
    pars.add(dict(name="fsigBd", value=0.0, min=0.0, max=1, free=False,
                  latex=r"f_{B_d}"))

    pars.add(dict(name="muBs", value=5367, min=5350, max=5390,
                  latex=r"\mu_{B_s}"))
    pars.add(dict(name="muBd", formula=f"muBs-87.19",
                  latex=r"\mu_{B_d}"))

    pars.add(dict(name="sigmaBs", value=20, min=0.1, max=50, free=True,
                  latex=r"\sigma_{B_s}"))
    pars.add(dict(name="frac", value=1.1155, min=0., max=2., free=False,
                  latex=r"f_{\sigma}"))
    pars.add(dict(name="sigmaBd", formula="frac*sigmaBs",
                  latex=r"\sigma_{B_s}"))
    pars.add(dict(name="f_hyp_gauss",
                  value=1.,
                  # value=0.6439,
                  min=0., max=2., free=False,
                  latex=r"f_{\sigma}"))

    if input_pars:
        _pars = ipanema.Parameters.clone(input_pars)
        _pars.lock()
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
        # pars["fsigBs"].value = 1
        if "hypatia" in model:
            # Hypatia tails {{{
            pars.add(dict(name="lambd", value=-1.5, min=-4, max=-1.1, free=True,
                    latex=r"\lambda",
                )
            )
            pars.add(dict(name="zeta", value=1e-6, free=False,
                          latex=r"\zeta"))
            pars.add(dict(name="beta", value=0.0, free=False,
                          latex=r"\beta"))
            pars.add(
                dict(name="aL", value=1.23, min=0.5, max=5.5, free=True, latex=r"a_l")
            )
            pars.add(dict(name="nL", value=1.05, min=0, max=6, free=True, latex=r"n_l"))
            pars.add(
                dict(name="aR", value=1.03, min=0.5, max=5.5, free=True, latex=r"a_r")
            )
            pars.add(dict(name="nR", value=1.02, min=0, max=6, free=True, latex=r"n_r"))
            # }}}
        elif "crystalball" in model:
            # Crystal Ball tails {{{
            pars.add(
                dict(name="aL", value=1.4, min=0.5, max=3.5, free=True, latex=r"a_l")
            )
            pars.add(dict(name="nL", value=20, min=0, max=500, free=True, latex=r"n_l"))
            pars.add(
                dict(name="aR", value=1.4, min=0.5, max=3.5, free=True, latex=r"a_r")
            )
            pars.add(dict(name="nR", value=20, min=0, max=500, free=True, latex=r"n_r"))
            # }}}
        elif "cbcalib" in model:
            # Crystal Ball with per event resolution tails {{{
            pars.add(
                dict(name="aL", value=1.4, min=-1, max=10.5, free=True, latex=r"a_l")
            )
            pars.add(dict(name="nL", value=1, min=-1, max=50, free=True, latex=r"n_l"))
            pars.add(
                dict(name="aR", value=1.4, min=-1, max=10.5, free=True, latex=r"a_r")
            )
            pars.add(dict(name="nR", value=1, min=-1, max=50, free=True, latex=r"n_r"))
            # }}}
        # Combinatorial background
        pars.add(dict(name='b',         value=-0.05, min=-1,  max=1,     free=False,  latex=r'b'))
        pars.add(dict(name="fcomb", formula="1-fsigBs-fsigBd",
                      latex=r"f_{comb}"))

    if has_bd:
        # Create common set of Bd parameters
        DMsd = 5366.89 - 5279.63
        DMsd = 87.26
        pars.add(
            dict(name="fsigBd", value=0.01, min=0, max=1, free=True, latex=r"N_{B_d}")
        )
        pars.add(dict(name="muBd", formula=f"muBs-{DMsd}", latex=r"\mu_{B_d}"))
        # pars.add(dict(name='sigmaBd',   value=1,    min=5,    max=20,    free=True,  latex=r'\sigma_{B_d}'))
        if with_calib:
            pars.add(
                dict(
                    name="s0Bd",
                    value=7,
                    min=5,
                    max=20,
                    free=False,
                    latex=r"\sigma_{B_d}",
                )
            )
        else:
            pars.add(
                dict(
                    name="s0Bd",
                    value=7,
                    min=5,
                    max=20,
                    free=False,
                    latex=r"\sigma_{B_d}",
                )
            )
            # pars.add(dict(name='s0Bd', formula="s0Bs",                           latex=r'\sigma_{B_d}'))
        # Combinatorial background
        pars.pop("fcomb")
        pars.add(dict(name="fcomb", formula="1-fsigBs-fsigBd", latex=r"N_{comb}"))

    # finally, set mass lower and upper limits
    pars.add(dict(name="mLL", value=mLL, free=False, latex=r"m_{ll}"))
    pars.add(dict(name="mUL", value=mUL, free=False, latex=r"m_{ul}"))
    print(pars)

    # }}}

    # Allocate the sample variables {{{

    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
    rd = Sample.from_pandas(odf)
    _proxy = np.float64(rd.df[mass_branch]) * 0.0
    rd.chop(current_cut)
    rd.allocate(mass=mass_branch)
    rd.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)
    # print(rd)

    # }}}

    # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='nelder', verbose=verbose)
    # for name in ['nsig', 'mu', 'sigma']:
    #     pars[name].init = res.params[name].value
    # res = False
    try:
        res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                               method='minuit', verbose=True, strategy=2,
                               tol=0.05)
    except:
        res = False
    # res = False

    if res:
        print(res)
        fpars = ipanema.Parameters.clone(res.params)
    else:
        print("Could not fit it!. Cloning pars to res")
        fpars = ipanema.Parameters.clone(pars)
        print(fpars)

    # plot the fit result {{{

    # fall back to averaged resolution when plotting
    _p = fpars.valuesdict()
    # merr = _p["s0Bs"] + _p["s1Bs"] * rd.merr + _p["s2Bs"] * rd.merr * rd.merr
    # merr = np.median(ristra.get(rd.merr))

    fig, axplot, axpull = complot.axes_plotpull()
    hdata = complot.hist(
        ristra.get(rd.mass), weights=rd.df.eval(mass_weight), bins=60, density=False
    )
    axplot.errorbar(
        hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
    )

    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0 * mass
    # merr = ristra.allocate(np.ones_like(mass.get()) * merr)

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if "fsigBd" in _p:
        _p["fsigBd"].set(value=0, min=-np.inf, max=np.inf)
    if "fcomb" in _p:
        _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.plot(_x, _y, color="C1", label=rf"$B_s^0$ {model}")

    # plot backgrounds: nsig -> 0
    _p = ipanema.Parameters.clone(fpars)
    if "fsigBs" in _p:
        _p["fsigBs"].set(value=0, min=-np.inf, max=np.inf)
    if "fcomb" in _p:
        _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.plot(_x, _y, "-.", color="C2", label="Bd")

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.plot(x, y, color="C0")
    axpull.fill_between(
        hdata.bins,
        complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr),
        0,
        facecolor="C0",
        alpha=0.5,
    )

    # label and save the plot
    axpull.set_xlabel(r"$m(J/\psi KK)$ [MeV/$c^2$]")
    axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel(rf"Candidates")
    axplot.legend(loc="upper left")
    if figs:
        os.makedirs(figs, exist_ok=True)
        fig.savefig(os.path.join(figs, f"fit.pdf"))
    axplot.set_yscale("log")
    try:
        axplot.set_ylim(1e0, 1.5 * np.max(y))
    except:
        print("axes not scaled")
    if figs:
        fig.savefig(os.path.join(figs, f"logfit.pdf"))
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

        # WARNING: Breaking change!  -- February 4th
        #          sw(p, y, len(data)) * wLb != sw(p, y, wLb.sum())
        #          which one is correct? Lera does the RS and I did LS
        # sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y), _pars, _yields, ristra.get(rd.weight).sum())
        sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.pdf, *x, **y), _pars, _yields, rd.weight)
        for k,v in sw.items():
            _sw = np.copy(_proxy)
            _sw[list(rd.df.index)] = v
            sw[k] = _sw
        # print("sum of wLb", np.sum( rd.df.eval(mass_weight).values ))
        return (fpars, sw)

    # }}}

    return (fpars, False)


# }}}


# command-line interface {{{

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="arreglatelas!")
    p.add_argument('--sample')
    p.add_argument('--input-params', default=False)
    p.add_argument('--output-params')
    p.add_argument('--output-figures')
    p.add_argument('--mass-model')
    p.add_argument('--mass-weight')
    p.add_argument('--mass-branch', default='B_ConstJpsi_M_1')
    p.add_argument('--mass-bin', default=False)
    p.add_argument('--trigger')
    p.add_argument('--sweights')
    p.add_argument('--mode')
    p.add_argument('--version')
    args = vars(p.parse_args())

    args['sample'] = "/scratch49/forMarcos/Bs2DsPi/Bs2DsPi_2017_selected_v1r0.root"
    args['input_params'] = False
    args['output_params'] = "mierda.json"
    args['output_figures'] = "shit_figs/"
    args['mass_model'] = 'hypatia'
    args['mass_branch'] = 'B_PVFitDs_M_1'
    args['mass_bin'] = False
    args['trigger'] = False
    args['sweights'] = False
    args['mode'] = 'Bs2DsPi'
    args['version'] = 'v1r0'


    if args["sweights"]:
        sweights = True
    else:
        sweights = False

    if args["input_params"]:
        input_pars = ipanema.Parameters.load(args["input_params"])
    else:
        input_pars = False

    mass_branch = args['mass_branch']
    branches = [ mass_branch ]

    if args["mass_weight"]:
        mass_weight = args["mass_weight"]
        branches += [mass_weight]
    else:
        mass_weight = f"{mass_branch}/{mass_branch}"

    cut = False
    if "prefit" in args["output_params"]:
        cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"
        branches += ["B_BKGCAT"]

    sample = Sample.from_root(args["sample"], branches=branches)

    mass_range = (5300, 5600)  # narrow window
    # mass_range = (5100, 5600)  # wide window
    if args["mass_bin"]:
        if "Bd2JpsiKstar" in args["mode"]:
            mass = [826, 861, 896, 931, 966]
        elif "Bs2JpsiPhi" in args["mode"] or "Bs2JpsiKK" in args['mode']:
            mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
        if args["mass_bin"] == "all":
            mLL = mass[0]
            mUL = mass[-1]
        else:
            bin = int(args["mass_bin"][-1])
            mLL = mass[bin - 1]
            mUL = mass[bin]
        if "LSB" in args["mass_bin"]:
            mass_range = (5202, 5367 + 50)
        elif "RSB" in args["mass_bin"]:
            mass_range = (5367 - 80, 5548)
        cut = f"({cut}) & X_M>{mLL} & X_M<{mUL}" if cut else f"X_M>{mLL} & X_M<{mUL}"

    pars, sw = mass_fitter(
        sample.df,
        mass_range=mass_range,
        mass_branch=mass_branch,
        mass_weight=mass_weight,
        trigger=args["trigger"],
        figs=args["output_figures"],
        model=args["mass_model"],
        cut=cut,
        sweights=sweights,
        input_pars=input_pars,
        verbose=True,
    )
    pars.dump(args["output_params"])
    if sw:
        np.save(args["sweights"], sw)

# }}}


# vim:foldmethod=marker ft=python
