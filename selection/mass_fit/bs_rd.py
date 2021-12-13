__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']
__all__ = ['mass_fitter']


# Modules {{{

import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, Sample, splot)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and

import complot

def get_sizes(size, BLOCK_SIZE=256):
    '''
    i need to check if this worls for 3d size and 3d block
    '''
    a = size % BLOCK_SIZE
    if a == 0:
      gs, ls = size, BLOCK_SIZE
    elif size < BLOCK_SIZE:
      gs, ls = size, 1
    else:
      a = np.ceil(size/BLOCK_SIZE)
      gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
    return int(gs), int(ls)


# initialize ipanema3 and compile lineshapes
ipanema.initialize(os.environ['IPANEMA_BACKEND'], 1)
# prog = ipanema.compile("""
# #define USE_DOUBLE 1
# #include <lib99ocl/core.c>
# #include <lib99ocl/complex.c>
# #include <lib99ocl/stats.c>
# #include <lib99ocl/special.c>
# #include <lib99ocl/lineshapes.c>
# #include <exposed/kernels.ocl>
# """)

prog = ipanema.compile("""
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

""")

# }}}


# ipatia + exponential {{{

def ipatia_exponential(mass, signal, nsigBs=0, nsigBd=0, nexp=0,
                       muBs=0, sigmaBs=10, muBd=5000, sigmaBd=50,lambd=0, zeta=0, beta=0, aL=0, nL=0, aR=0, nR=0,
                       b=0, norm=1, mLL=None, mUL=None):
    # ipatia
    prog.py_ipatia(signal, mass, np.float64(muBs), np.float64(sigmaBs),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR), np.float64(nR),
                   global_size=(len(mass)))
    pdfBs = 1.0 * signal.get()
    signal = 0*signal
    prog.py_ipatia(signal, mass, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR), np.float64(nR),
                   global_size=(len(mass)))
    pdfBd = 1.0 * signal.get()
    backgr = ristra.exp(mass*b).get()
    # normalize
    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x*0
    prog.py_ipatia(_y, _x, np.float64(muBs), np.float64(sigmaBs),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)))
    nBs = np.trapz(ristra.get(_y), ristra.get(_x))
    _y = _x*0
    prog.py_ipatia(_y, _x, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)))
    nBd = np.trapz(ristra.get(_y), ristra.get(_x))
    nbackgr = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
    # compute pdf value
    ans = norm * ( nsigBs*pdfBs/nBs + nsigBd*pdfBd/nBd + nexp*backgr/nbackgr )
    return ans

# }}}


# CB Bs2JpsiPhi mass model {{{

def cb_exponential(mass, merr, signal, nsigbs=0, nsigbd=0, nexp=0,
                   mubs=0, sigmabs=10, mubd=5000, sigmabd=50, al=0, nl=0, ar=0, nr=0,
                   b=0, norm=1, mLL=None, mUL=None):
  # compute backgrounds
  pexp = ristra.get(ristra.exp(mass*b))
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(mubs), np.float64(sigmabs),
                              np.float64(al), np.float64(nl), np.float64(ar),
                              np.float64(nr), global_size=(len(mass)))
  pbs = ristra.get(signal)
  prog.py_double_crystal_ball(signal, mass, np.float64(mubd), np.float64(sigmabd),
                              np.float64(al), np.float64(nl), np.float64(ar),
                              np.float64(nr), global_size=(len(mass)))
  pbd = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x*0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mubs), np.float64(sigmabs),
                              np.float64(al), np.float64(nl), np.float64(ar),
                              np.float64(nr), global_size=(len(_x)))
  nbs = np.trapz(ristra.get(_y), ristra.get(_x))
  prog.py_double_crystal_ball(_y, _x, np.float64(mubd), np.float64(sigmabd),
                              np.float64(al), np.float64(nl), np.float64(ar),
                              np.float64(nr), global_size=(len(_x)))
  nbd = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexp = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
  # compute pdf value
  ans = nsigbs*pbs/nbs + nsigbd*pbd/nbd + nexp*pexp/nexp
  return norm*ans


def cb_exponential2(mass, merr, signal, nsigBs=0, nsigBd=0, nexp=0,
                   muBs=0, s0=0, s1=1, s2=1, muBd=5200, sigmaBd=50, aL=0, nL=0, aR=0, nR=0,
                   b=0, norm=1, mLL=None, mUL=None):
  # compute backgrounds
  pExp = ristra.get(ristra.exp(mass*b))
  # get signal
  sigmaBs = s0 + merr * (s1 + merr * s2)
  prog.DoubleCrystallBallPerEventSigma(signal, mass, np.float64(muBs), sigmaBs,
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pBs = ristra.get(signal)
  prog.py_double_crystal_ball(signal, mass, np.float64(muBd), np.float64(sigmaBd),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pBd = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x*0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(muBs), np.float64(np.mean(sigmaBs.get())),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  nBs = np.trapz(ristra.get(_y), ristra.get(_x))
  prog.py_double_crystal_ball(_y, _x, np.float64(muBd), np.float64(sigmaBd),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  nBd = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nExp = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
  # compute pdf value
  ans = nsigBs*pBs/nBs + nsigBd*pBd/nBd + nExp*pExp/nExp
  return norm*ans


def cb_exponential_withcalib(mass, signal, nsigBs=0, nsigBd=0, nexp=0,
                   muBs=0, s0=0, s1=1, s2=1, muBd=5000, sigmaBd=50, aL=0, nL=0, aR=0, nR=0,
                   b=0, norm=1, mLL=None, mUL=None):
  g_size, l_size = get_sizes(len(signal), 128)
  prog.DoubleCrystallBallWithCalibration(
    signal, mass,
    np.float64(nsigBs), np.float64(nsigBd),
    np.float64(muBs), np.float64(s0), np.float64(s1), np.float64(s2),
    np.float64(muBd), np.float64(sigmaBd),
    np.float64(aL), np.float64(nL), np.float64(aR), np.float64(nR),
    np.float64(b),
    np.float64(mLL), np.float64(mUL), global_size=g_size, local_size=l_size )
  print(signal)
  return norm * ristra.get(signal)
# }}}


# Bs mass fit function {{{

def mass_fitter(odf,
                mass_range=False, mass_branch='B_ConstJpsi_M_1',
                mass_weight='B_ConstJpsi_M_1/B_ConstJpsi_M_1',
                cut=False,
                figs = False, model=False, has_bd=False,
                trigger='combined', input_pars=False, sweights=False, verbose=False):

    # mass range cut
    if not mass_range:
      mass_range = ( min(odf[mass_branch]), max(odf[mass_branch]))
    mass_cut = f'B_ConstJpsi_M_1 > {mass_range[0]} & B_ConstJpsi_M_1 < {mass_range[1]}'
    mLL, mUL = mass_range
    # mass_cut = f'B_ConstJpsi_M_1 > 5220 & B_ConstJpsi_M_1 < 5330'

    # mass cut and trigger cut 
    current_cut = trigger_scissors(trigger, cuts_and(mass_cut, cut))


    # Select model and set parameters {{{
    #    Select model from command-line arguments and create corresponding set of
    #    paramters

    # Chose model {{{

    with_calib = False
    if model == 'ipatia':
      pdf = ipatia_exponential 
    elif model == 'crystalball':
      pdf = cb_exponential 
    elif model == 'cbcalib':
      pdf = cb_exponential2
      with_calib = True


    def fcn(params, data):
        p = params.valuesdict()
        prob = pdf(mass=data.mass, merr=data.merr, signal=data.pdf, **p)
        return -2.0 * np.log(prob) * ristra.get(data.weight)

    # }}}

    pars = ipanema.Parameters()
    # Create common set of Bs parameters (all models must have and use)
    pars.add(dict(name='nsigBs',    value=1, min=0.2,  max=1,    free=False,  latex=r'N_{B_s}'))
    pars.add(dict(name='muBs',      value=5367,  min=5200, max=5500,             latex=r'\mu_{B_s}'))
    if with_calib:
      pars.add(dict(name='s0',   value=0,    min=0,    max=100,  free=False,  latex=r'\sigma_{B_s}'))
      pars.add(dict(name='s1',   value=0.5,    min=0,    max=10,  free=True,  latex=r'\sigma_{B_s}'))
      pars.add(dict(name='s2',   value=0.5,    min=0,    max=10,  free=True,  latex=r'\sigma_{B_s}'))
    else:
      pars.add(dict(name='sigmaBs',   value=5,    min=5,    max=100,  free=True,  latex=r'\sigma_{B_s}'))

    if input_pars:
      _pars = ipanema.Parameters.clone(input_pars)
      _pars.lock()
      if with_calib:
        _pars.remove('nsigBs', 'muBs', 's0', 's1', 's2')
      else:
        _pars.remove('nsigBs', 'muBs', 'sigmaBs')
      _pars.unlock('b')
      pars = pars + _pars
    else:
      if 'ipatia' in model:
        # Hypatia tails {{{
        pars.add(dict(name='lambd',   value=-1.5,  min=-4,   max=-1.1, free=True,  latex=r'\lambda'))
        pars.add(dict(name='zeta',    value=1e-5,                      free=False, latex=r'\zeta'))
        pars.add(dict(name='beta',    value=0.0,                       free=False, latex=r'\beta'))
        pars.add(dict(name='aL',      value=1.23,  min=0.5, max=3.5,   free=True,  latex=r'a_l'))
        pars.add(dict(name='nL',      value=1.05,  min=0,   max=4,     free=True,  latex=r'n_l'))
        pars.add(dict(name='aR',      value=1.03,  min=0.5, max=3.5,   free=True,  latex=r'a_r'))
        pars.add(dict(name='nR',      value=1.02,  min=0,   max=4,     free=True,  latex=r'n_r'))
        # }}}
      elif "crystalball" in model:
        # Crystal Ball tails {{{
        pars.add(dict(name='aL',      value=1.4,  min=0.5, max=3.5,    free=True,  latex=r'a_l'))
        pars.add(dict(name='nL',      value=20,     min=0,   max=500,   free=True,  latex=r'n_l'))
        pars.add(dict(name='aR',      value=1.4,  min=0.5, max=3.5,    free=True,  latex=r'a_r'))
        pars.add(dict(name='nR',      value=20,     min=0,   max=500,   free=True,  latex=r'n_r'))
      elif "cbcalib" in model:
        # Crystal Ball tails {{{
        pars.add(dict(name='aL',      value=0.4,  min=0.1, max=3.5,    free=True,  latex=r'a_l'))
        pars.add(dict(name='nL',      value=2,     min=0,   max=50,   free=True,  latex=r'n_l'))
        pars.add(dict(name='aR',      value=0.4,  min=0.1, max=3.5,    free=True,  latex=r'a_r'))
        pars.add(dict(name='nR',      value=2,     min=0,   max=50,   free=True,  latex=r'n_r'))
        # }}}
      # Combinatorial background
      pars.add(dict(name='b',         value=-4e-3, min=-1,  max=1,     free=False,  latex=r'b'))
      pars.add(dict(name='nexp',      formula="1-nsigBs",                          latex=r'N_{comb}'))

    if has_bd:
      # Create common set of Bd parameters
      DMsd = 5366.89 - 5279.63
      DMsd = 87.26
      pars.add(dict(name='nsigBd',    value=0.01,  min=0.,  max=1,     free=True,  latex=r'N_{B_d}'))
      pars.add(dict(name='muBd',      formula=f"muBs-{DMsd}",                      latex=r'\mu_{B_d}'))
      # pars.add(dict(name='sigmaBd',   value=1,    min=5,    max=20,    free=True,  latex=r'\sigma_{B_d}'))
      if with_calib:
        pars.add(dict(name='sigmaBd',   value=7,    min=5,    max=20,    free=False,  latex=r'\sigma_{B_d}'))
      else:
        pars.add(dict(name='sigmaBd', formula="sigmaBs",                           latex=r'\sigma_{B_d}'))
      # Combinatorial background
      pars.pop('nexp')
      pars.add(dict(name='nexp',     formula="1-nsigBs-nsigBd", latex=r'N_{comb}'))
    pars.add(dict(name='mLL',  value=mLL, free=False, latex=r'N_{comb}'))
    pars.add(dict(name='mUL',  value=mUL, free=False, latex=r'N_{comb}'))
    print(pars)

    # }}}



    # Allocate the sample variables {{{

    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
    rd = Sample.from_pandas(odf)
    _proxy = np.float64(rd.df[mass_branch]) * 0.0
    rd.chop(current_cut)
    rd.allocate(mass=mass_branch, merr='B_ConstJpsi_MERR_1')
    rd.allocate(pdf=f'0*{mass_branch}', weight=mass_weight)
    # print(rd)

    # }}}

    # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='nelder', verbose=verbose)
    # for name in ['nsig', 'mu', 'sigma']:
    #     pars[name].init = res.params[name].value
    # res = False
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='minuit',
                           verbose=True, strategy=1, tol=0.05)
    if res:
      print(res)
      fpars = ipanema.Parameters.clone(res.params)
    else:
      print("Could not fit it!. Cloning pars to res")
      fpars = ipanema.Parameters.clone(pars)
      print(fpars)

    # plot the fit result {{{

    # fall back to averaged resolution when plotting
    if with_calib:
      _p = fpars.valuesdict()
      merr = _p['s0'] + _p['s1'] * rd.merr + _p['s2'] * rd.merr * rd.merr
      merr = np.median(ristra.get(rd.merr)) 

    fig, axplot, axpull = complot.axes_plotpull()
    hdata = complot.hist(ristra.get(rd.mass), weights=rd.df.eval(mass_weight),
                           bins=50, density=False)
    axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                    fmt='.k')

    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0*mass
    merr = ristra.allocate(np.ones_like(mass.get()) * merr)

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if 'nsigBd' in _p:
      _p['nsigBd'].set(value=0, min=-np.inf, max=np.inf)
    if 'nexp' in _p:
      _p['nexp'].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ristra.get(mass), ristra.get(pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm))
    axplot.plot(_x, _y, color="C1", label=rf'$B_s^0$ {model}')

    # plot backgrounds: nsig -> 0
    if has_bd:
      _p = ipanema.Parameters.clone(fpars)
      if 'nsigBs' in _p: _p['nsigBs'].set(value=0,min=-np.inf, max=np.inf )
      if 'nexp' in _p:   _p['nexp'].set(value=0, min=-np.inf, max=np.inf)
      _x, _y = ristra.get(mass), ristra.get(pdf(mass, merr, signal, **_p.valuesdict(),
                                                norm=hdata.norm))
      axplot.plot(_x, _y, '-.', color="C2", label='Bd')

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm))
    axplot.plot(x, y, color='C0')
    axpull.fill_between(hdata.bins, complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr), 0, facecolor="C0", alpha=0.5)

    # label and save the plot
    axpull.set_xlabel(r'$m(J/\psi KK)$ [MeV/$c^2$]')
    axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel(rf"Candidates")
    axplot.legend(loc="upper left")
    if figs:
      os.makedirs(figs, exist_ok=True)
      fig.savefig(os.path.join(figs, f"fit.pdf"))
    axplot.set_yscale('log')
    axplot.set_ylim(1e0,1.5*np.max(y))
    if figs:
      fig.savefig(os.path.join(figs, f"logfit.pdf"))
    plt.close()

    # }}}


    # compute sWeights if asked {{{

    if sweights:
        # separate paramestes in yields and shape parameters
        _yields = ipanema.Parameters.find(fpars, "nsig.*") + ["nexp"]
        _pars = list(fpars)
        [_pars.remove(_y) for _y in _yields]
        _yields = ipanema.Parameters.build(fpars, _yields)
        _pars = ipanema.Parameters.build(fpars, _pars)

        sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y), _pars, _yields)
        for k,v in sw.items():
          _sw = np.copy(_proxy)
          _sw[list(rd.df.index)] = v * np.float64(rd.df.eval(mass_weight))
          sw[k] = _sw
        print(sw)
        return (fpars, sw)

    # }}}

    return (fpars, False)

# }}}


# command-line interface {{{

if __name__ == '__main__':
  p = argparse.ArgumentParser(description="mass fit")
  p.add_argument('--sample')
  p.add_argument('--input-params', default=False)
  p.add_argument('--output-params')
  p.add_argument('--output-figures')
  p.add_argument('--mass-model')
  p.add_argument('--mass-weight')
  p.add_argument('--mass-bin', default=False)
  p.add_argument('--trigger')
  p.add_argument('--sweights')
  p.add_argument('--mode')
  args = vars(p.parse_args())
  
  if args['sweights']:
    sweights = True
  else:
    sweights = False

  if args['input_params']:
    input_pars = ipanema.Parameters.load(args['input_params'])
  else:
    input_pars = False
  
  branches = ['B_ConstJpsi_M_1', 'B_ConstJpsi_MERR_1', 'hlt1b', 'X_M']

  if args['mass_weight']:
    mass_weight = args['mass_weight']
    branches += [mass_weight]
  else:
    mass_weight = 'B_ConstJpsi_M_1/B_ConstJpsi_M_1'

  cut = False
  if "prefit" in args['output_params']:
    cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"
    branches += ['B_BKGCAT']
  
  sample = Sample.from_root(args['sample'], branches=branches)
  

  mass_range=(5202, 5548)
  if args['mass_bin']:
    if 'Bd2JpsiKstar' in args['mode']:
      mass = [826, 861, 896, 931, 966]
    elif 'Bs2JpsiPhi' in args['mode']:
      mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    if args['mass_bin'] == 'all':
      mLL = mass[0]
      mUL = mass[-1]
    else:
      bin = int(args['mass_bin'][-1])
      mLL = mass[bin-1]
      mUL = mass[bin]
    if "LSB" in args['mass_bin']:
      mass_range=(5202, 5367+50)
    elif "RSB" in args['mass_bin']:
      mass_range=(5367-80, 5548)
    cut = f"({cut}) & X_M>{mLL} & X_M<{mUL}" if cut else f"X_M>{mLL} & X_M<{mUL}"

  pars, sw = mass_fitter(sample.df,
                         mass_range=mass_range, mass_branch='B_ConstJpsi_M_1',
                         mass_weight=mass_weight, trigger=args['trigger'],
                         figs = args['output_figures'], model=args['mass_model'],
                         cut=cut, sweights=sweights,
                         has_bd=True if args['mode']=='Bs2JpsiPhi' else False,
                         input_pars=input_pars, verbose=False)
  pars.dump(args['output_params'])
  if sw: np.save(args['sweights'], sw)

# }}}


# vim:foldmethod=marker
