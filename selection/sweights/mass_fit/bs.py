__all__ = ["cb_exponential3"]
__author__ = ["Marcos Romero"]
__email__ = ["mromerol@cern.ch"]

from analysis.csp_factors.efficiency import (create_mass_bins,
                                             create_time_bins,
                                             create_cosL_bins,
                                             create_cosK_bins,
                                             create_hphi_bins,
                                             create_sigmam_bins)
from utils.helpers import cuts_and, trigger_scissors
import config
import ipanema
# from ipanema import (ristra, Sample, splot)
import complot
from utils.helpers import trigger_scissors, cuts_and
from utils.strings import printsec, printsubsec
import matplotlib.pyplot as plt
import re
import numpy as np
import uproot3 as uproot
import argparse
import os


# some helpers {{{

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


def get_bin_from_version(version, subbin, mode):
  # parse tuple version name first {{{
  v = version.split('@')
  if len(v) > 1:
    v, c = v
  else:
    v, c = v[0], ''
  v = v.split('~')
  if len(v) > 1:
    v, w = v
  else:
    v, w = v[0], False

  if not w:
    if 'Bs' in mode:
      w = 'mX6'
    else:
      w = ''
  print(v, w, c)

  pattern = [
      "(mX(1|2|3|4|5|6))?",
      "(time(1|2|3|4))?",
      "(cosL(1|2|3|4))?",
      "(cosK(1|2|3|4))?",
      "(hphi(1|2|3|4))?",
      "(sigmam(1|2|3|4|5|6|7|8|9))?",
  ]
  pattern = rf"\A{''.join(pattern)}\Z"
  p = re.compile(pattern)
  if subbin != 'all':
    try:
      q = p.search(w).groups()
      mX_bins = int(q[1]) if q[0] else 1
      time_bins = int(q[3]) if q[2] else 1
      cosK_bins = int(q[5]) if q[4] else 1
      cosL_bins = int(q[7]) if q[6] else 1
      hphi_bins = int(q[9]) if q[8] else 1
      sigmam_bins = int(q[11]) if q[10] else 1
    except:
      raise ValueError(f'Cannot interpret {w} as a sWeight config')
  else:
    mX_bins = 1
    time_bins = 1
    cosK_bins = 1
    cosL_bins = 1
    hphi_bins = 1
    sigmam_bins = 1
  # pattern = rf"\A{''.join(pattern)}\Z"
  print(f"from version = {version} :: {mX_bins}, {time_bins}, {sigmam_bins}")
  # }}}

  # extract current subbin {{{
  p = re.compile(pattern)
  if subbin != 'all':
    try:
      q = p.search(subbin).groups()
      mX = int(q[1]) if q[0] else 1
      time = int(q[3]) if q[2] else 1
      cosK = int(q[5]) if q[4] else 1
      cosL = int(q[7]) if q[6] else 1
      hphi = int(q[9]) if q[8] else 1
      sigmam = int(q[11]) if q[10] else 1
    except:
      raise ValueError(f'Cannot interpret {w} as a subbin')
  else:
    mX = 1
    time = 1
    cosK = 1
    cosL = 1
    hphi = 1
    sigmam = 1
  print(f"from subbin = {subbin} :: {mX}, {time}, {sigmam}")
  # create bins limits
  mXLL, mXUL = create_mass_bins(int(mX_bins))[mX - 1:mX + 1]
  timeLL, timeUL = create_time_bins(int(time_bins))[time - 1:time + 1]
  cosKLL, cosKUL = create_cosK_bins(int(cosK_bins))[cosK - 1:cosK + 1]
  cosLLL, cosLUL = create_cosL_bins(int(cosL_bins))[cosL - 1:cosL + 1]
  hphiLL, hphiUL = create_hphi_bins(int(hphi_bins))[hphi - 1:hphi + 1]
  sigmamLL, sigmamUL = create_sigmam_bins(int(sigmam_bins))[sigmam - 1:sigmam + 1]
  # list of cuts
  list_of_cuts = [
      f"X_M > {mXLL} & X_M < {mXUL}",
      f"time > {timeLL} & time < {timeUL}",
      f"helcosthetaK > {cosKLL} & helcosthetaK < {cosKUL}",
      f"helcosthetaL > {cosLLL} & helcosthetaL < {cosLUL}",
      f"helphi > {hphiLL} & helphi < {hphiUL}",
      f"B_ConstJpsi_MERR_1 > {sigmamLL} & B_ConstJpsi_MERR_1 < {sigmamUL}",
  ]
  cuts = f"( {' ) & ( '.join(list_of_cuts)} )"
  return cuts

# }}}

# }}}


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
    mass,
    merr,
    signal,
    fsigBs=0,
    fsigBd=0,
    fcomb=0,
    muBs=0,
    s0Bs=1,
    s1Bs=0,
    s2Bs=0,
    muBd=5300,
    s0Bd=1,
    s1Bd=0,
    s2Bd=0,
    lambd=0,
    zeta=0,
    beta=0,
    aL=0,
    nL=0,
    aR=0,
    nR=0,
    b=0,
    norm=1,
    mLL=None,
    mUL=None,
):
  # ipatia
  sigmaBs = s0Bs
  prog.py_ipatia(signal, mass, np.float64(muBs), np.float64(sigmaBs),
                 np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
                 np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(mass)),
                 )
  pdfBs = 1.0 * signal.get()
  # normalize
  _x = ipanema.ristra.linspace(ipanema.ristra.min(mass), ipanema.ristra.max(mass), 1000)
  _y = _x * 0
  prog.py_ipatia(_y, _x, np.float64(muBs), np.float64(sigmaBs),
                 np.float64(lambd), np.float64(zeta), np.float64(beta), np.float64(aL),
                 np.float64(nL), np.float64(aR), np.float64(nR), global_size=(len(_x)),
                 )
  nBs = np.trapz(ipanema.ristra.get(_y), ipanema.ristra.get(_x))

  # second peak with same tails as main one
  if fsigBd > 0:
    prog.kernel_gaussian(
        signal,
        mass,
        np.float64(muBd),
        np.float64(s0Bd),
        np.float64(mLL),
        np.float64(mUL),
        global_size=(len(mass)),
    )
    pBd = ipanema.ristra.get(signal)
  else:
    pBd = 0
  # combinatorial background
  if fcomb > 0:
    prog.kernel_exponential(
        signal,
        mass,
        np.float64(b),
        np.float64(mLL),
        np.float64(mUL),
        global_size=(len(mass)),
    )
    pComb = ipanema.ristra.get(signal)
  else:
    pComb = 0
  # compute pdf value
  ans = fsigBs * pdfBs / nBs + fsigBd * pBd + fcomb * pComb
  return norm * ans


# }}}


# CB Bs2JpsiPhi mass model {{{


# def cb_exponential(
#     mass,
#     merr,
#     signal,
#     nsigBs=0,
#     nsigBd=0,
#     nexp=0,
#     muBs=0,
#     sigmaBs=10,
#     muBd=5000,
#     sigmaBd=50,
#     aL=0,
#     nL=0,
#     aR=0,
#     nR=0,
#     b=0,
#     norm=1,
#     mLL=None,
#     mUL=None,
# ):
#   # compute backgrounds
#   pExp = ristra.get(ristra.exp(mass * b))
#   # get signal
#   prog.py_double_crystal_ball(
#       signal,
#       mass,
#       np.float64(muBs),
#       np.float64(sigmaBs),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(mass)),
#   )
#   pBs = ristra.get(signal)
#   prog.py_double_crystal_ball(
#       signal,
#       mass,
#       np.float64(muBd),
#       np.float64(sigmaBd),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(mass)),
#   )
#   pBd = ristra.get(signal)
#   # normalize arrays
#   _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
#   _y = _x * 0
#   # normalize cb-shape
#   prog.py_double_crystal_ball(
#       _y,
#       _x,
#       np.float64(muBs),
#       np.float64(sigmaBs),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(_x)),
#   )
#   nBs = np.trapz(ristra.get(_y), ristra.get(_x))
#   prog.py_double_crystal_ball(
#       _y,
#       _x,
#       np.float64(muBd),
#       np.float64(sigmaBd),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(_x)),
#   )
#   nBd = np.trapz(ristra.get(_y), ristra.get(_x))
#   # normalize exp
#   nExp = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
#   # compute pdf value
#   ans = nsigBs * pBs / nBs + nsigBd * pBd / nBd + nexp * pExp / nExp
#   return norm * ans
#
#
# def cb_exponential2(
#     mass,
#     merr,
#     signal,
#     nsigBs=0,
#     nsigBd=0,
#     nexp=0,
#     muBs=0,
#     s0=0,
#     s1=1,
#     s2=1,
#     muBd=5200,
#     sigmaBd=50,
#     aL=0,
#     nL=0,
#     aR=0,
#     nR=0,
#     b=0,
#     norm=1,
#     mLL=None,
#     mUL=None,
# ):
#   # compute backgrounds
#   pComb = ristra.get(ristra.exp(mass * b))
#   # get signal
#   sigmaBs = s0 + merr * (s1 + merr * s2)
#   # print(sigmaBs)
#   prog.DoubleCrystallBallPerEventSigma(
#       signal,
#       mass,
#       np.float64(muBs),
#       sigmaBs,
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(mass)),
#   )
#   pBs = ristra.get(signal)
#   prog.py_double_crystal_ball(
#       signal,
#       mass,
#       np.float64(muBd),
#       np.float64(sigmaBd),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(mass)),
#   )
#   pBd = ristra.get(signal)
#   # normalize arrays
#   _x = ristra.linspace(mLL, mUL, 1000)
#   _y = _x * 0
#   # normalize cb-shape
#   prog.py_double_crystal_ball(
#       _y,
#       _x,
#       np.float64(muBs),
#       np.float64(np.mean(sigmaBs.get())),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(_x)),
#   )
#   nBs = np.trapz(ristra.get(_y), ristra.get(_x))
#   prog.py_double_crystal_ball(
#       _y,
#       _x,
#       np.float64(muBd),
#       np.float64(sigmaBd),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       global_size=(len(_x)),
#   )
#   nBd = np.trapz(ristra.get(_y), ristra.get(_x))
#   # normalize exp
#   nComb = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
#   # compute pdf value
#   ans = nsigBs * pBs / nBs + nsigBd * pBd / nBd + nexp * pComb / nComb
#   return norm * ans


# TODO: THIS IS THE FINAL ONE
def cb_exponential3(
        mass, merr, signal, fsigBs=0, fsigBd=0, fcomb=0, muBs=5400, s0Bs=1,
        s1Bs=0, s2Bs=0, muBd=5300, s0Bd=1, s1Bd=0, s2Bd=0, aL=0, nL=0, aR=0,
        nR=0, b=0, norm=1, mLL=None, mUL=None,):
  # prepare calibrations
  sBs = s0Bs + merr * (s1Bs + merr * s2Bs)
  # sBd = s0Bd + 0 * s1Bd + 0 * s2Bd
  sBd = s0Bs + merr * (s1Bs + merr * s2Bs)

  # main peak
  prog.kernel_double_crystal_ball(signal, mass, np.float64(muBs), sBs,
                                  np.float64(aL), np.float64(nL),
                                  np.float64(aR), np.float64(nR),
                                  np.float64(mLL), np.float64(mUL),
                                  global_size=(len(mass)),)
  pBs = ipanema.ristra.get(signal)
  # second peak with same tails as main one
  if fsigBd > 0:
    prog.kernel_double_crystal_ball(signal, mass, np.float64(muBd), sBd,
                                  np.float64(aL), np.float64(nL),
                                  np.float64(aR), np.float64(nR),
                                  np.float64(mLL), np.float64(mUL),
                                  global_size=(len(mass)),)
    # prog.kernel_gaussian(signal, mass, np.float64(muBd), np.float64(sBd),
    #                      np.float64(mLL), np.float64(mUL),
    #                      global_size=(len(mass)),)
    pBd = ipanema.ristra.get(signal)
  else:
    pBd = 0
  # combinatorial background
  if fcomb > 0:
    prog.kernel_exponential(signal, mass, np.float64(b), np.float64(mLL),
                            np.float64(mUL), global_size=(len(mass)),)
    pComb = ipanema.ristra.get(signal)
  else:
    pComb = 0
  ans = fsigBs * pBs + fsigBd * pBd + fcomb * pComb
  return norm * ans


# def cb_exponential_withcalib(
#     mass,
#     signal,
#     nsigBs=0,
#     nsigBd=0,
#     nexp=0,
#     muBs=0,
#     s0=0,
#     s1=1,
#     s2=1,
#     muBd=5000,
#     sigmaBd=50,
#     aL=0,
#     nL=0,
#     aR=0,
#     nR=0,
#     b=0,
#     norm=1,
#     mLL=None,
#     mUL=None,
# ):
#   g_size, l_size = get_sizes(len(signal), 128)
#   prog.DoubleCrystallBallWithCalibration(
#       signal,
#       mass,
#       np.float64(nsigBs),
#       np.float64(nsigBd),
#       np.float64(muBs),
#       np.float64(s0),
#       np.float64(s1),
#       np.float64(s2),
#       np.float64(muBd),
#       np.float64(sigmaBd),
#       np.float64(aL),
#       np.float64(nL),
#       np.float64(aR),
#       np.float64(nR),
#       np.float64(b),
#       np.float64(mLL),
#       np.float64(mUL),
#       global_size=g_size,
#       local_size=l_size,
#   )
#   # print(signal)
#   return norm * ristra.get(signal)


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

  is_bkg60 = False
  if 'B_BKGCAT == 60' in cut:
    print('this sample is bkgcat60 only')
    is_bkg60 = True

  # Select model and set parameters {{{
  #    Select model from command-line arguments and create corresponding set
  #    of paramters

  # Chose model {{{

  with_calib = False
  if model == 'ipatia':
    pdf = ipatia_exponential
  elif model == "crystalball":
    pdf = cb_exponential3
  elif model == "cbcalib":
    pdf = cb_exponential3
    with_calib = True

  def fcn(params, data):
    p = params.valuesdict()
    prob = pdf(mass=data.mass, merr=data.merr, signal=data.pdf, **p)
    return -2.0 * np.log(prob) * ipanema.ristra.get(data.weight)

  # }}}

  pars = ipanema.Parameters()
  # Create common set of Bs parameters (all models must have and use)
  pars.add(
      dict(name="fsigBs", value=0.9, min=0.05, max=1, free=True, latex=r"f_{B_s}")
  )
  pars.add(dict(name="muBs", value=5367, min=5350, max=5390, latex=r"\mu_{B_s}"))
  if with_calib:
    # if not input_pars:
    pars.add(
        dict(name="s0Bs", value=0, min=0, max=50, free=False, latex=r"p_0^{B_s}")
    )
    pars.add(
        dict(name="s1Bs", value=0.67, min=-5, max=10, free=True, latex=r"p_1^{B_s}")
    )
    pars.add(
        dict(name="s2Bs", value=0.00, min=-1, max=2, free=True, latex=r"p_2^{B_s}")
    )
  else:
    pars.add(
        dict(name="s0Bs", value=2, min=2, max=20, free=True, latex=r"p_0^{B_s}")
    )
    pars.add(
        dict(name="s1Bs", value=0.0, min=0, max=10, free=False, latex=r"p_1^{B_s}")
    )
    pars.add(
        dict(name="s2Bs", value=0.0, min=-1, max=1, free=False, latex=r"p_2^{B_s}")
    )

  if input_pars:
    _pars = ipanema.Parameters.clone(input_pars)
    _pars.lock()
    if with_calib:
      _pars.remove("fsigBs", "muBs")
      pars.remove("s1Bs", "s2Bs")
      _pars.unlock("s1Bs", "s2Bs")
    else:
      _pars.remove("fsigBs", "muBs")
      pars.remove("s0Bs")
      _pars.unlock("s0Bs")
    _pars.unlock("b")
    if 'lambd' in _pars:
      pars.remove("lambd")
      _pars.unlock("lambd")
    pars = pars + _pars
  else:
    # This is the prefit stage. Here we will lock the nsig to be 1 and we
    # will not use combinatorial background.
    pars["fsigBs"].value = 1
    pars["fsigBs"].free = False
    if "ipatia" in model:
      # Hypatia tails {{{
      pars.add(
          dict(
              name="lambd",
              value=-1.5,
              min=-4,
              max=-1.1,
              free=True,
              latex=r"\lambda",
          )
      )
      pars.add(dict(name="zeta", value=1e-5, free=False, latex=r"\zeta"))
      pars.add(dict(name="beta", value=0.0, free=False, latex=r"\beta"))
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
    pars.add(dict(name='b', value=-0.05, min=-1, max=1, free=False, latex=r'b'))
    pars.add(dict(name='fcomb', formula="1-fsigBs", latex=r'f_{comb}'))

  if has_bd:
    # Create common set of Bd parameters
    DMsd = 5366.89 - 5279.63
    DMsd = 87.26
    pars.add(
        dict(name="fsigBd", value=0.01, min=0, max=1, free=True, latex=r"f_{B_d^0}")
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
    pars.add(dict(name="fcomb", formula="1-fsigBs-fsigBd",
                  min=0, max=1, latex=r"f_{comb}"))

  if is_bkg60:
    pars.unlock('aL', 'nL', 'aR', 'nR')
  # finally, set mass lower and upper limits
  pars.add(dict(name="mLL", value=mLL, free=False, latex=r"m_{ll}"))
  pars.add(dict(name="mUL", value=mUL, free=False, latex=r"m_{ul}"))

  print("Parameters before fit:")
  print(pars)

  # }}}

  # Allocate the sample variables {{{

  print(f"Cut: {current_cut}")
  print(f"Mass branch: {mass_branch}")
  print(f"Mass weight: {mass_weight}")
  rd = ipanema.Sample.from_pandas(odf)
  _proxy = np.float64(rd.df[mass_branch]) * 0.0 - 999
  # if is_bkg60:
  #   _proxy += 1
  print(rd)
  rd.chop(current_cut)
  print(rd)
  rd.allocate(mass=mass_branch, merr="B_ConstJpsi_MERR_1")
  rd.allocate(pdf=f"0*{mass_branch}", weight=mass_weight)
  rd.weight *= ipanema.ristra.sum(rd.weight) / ipanema.ristra.sum(rd.weight**2)
  # mass_range = (min(rd.df[mass_branch]), max(rd.df[mass_branch]))
  # mLL, mUL = mass_range
  # print(rd)

  # }}}

  # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='nelder', verbose=verbose)
  # for name in ['nsig', 'mu', 'sigma']:
  #     pars[name].init = res.params[name].value
  # res = False
  try:
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=verbose, strategy=1,
                           tol=0.05)
  except:
    res = False
  # res = False
  if not res:
    print("There was a problem with the fit. Let's try again.")
    pars['fsigBs'].set(value=0.5, min=0.0, max=1)
    pars['s1Bs'].set(value=0.0, init=0.0, min=-20, max=20, free=False)
    pars['s2Bs'].set(value=0.0, init=0.0, min=-1, max=1, free=False)
    pars['s0Bs'].set(value=10.0, init=10.0, min=0.5, free=True)
    if 'lambd' in pars:
      pars['lambd'].set(value=_pars['lambd'].value, init=-1.5, free=False)
    if 'fsigBd' in pars:
      pars['fsigBd'].set(value=0, init=0, free=False)

    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=True, strategy=1,
                           tol=0.1)
    if with_calib:
      pars = ipanema.Parameters.clone(res.params)
      pars.lock()
      pars["s1Bs"].set(value=1.0, init=1.0, min=-20, max=20, free=True)
      pars["s2Bs"].set(value=0.1, init=0.1, min=-1, max=1, free=True)
      pars["s0Bs"].set(value=0.0, init=0.0, min=0.0, free=False)
      res = ipanema.optimize(fcn, pars, fcn_kwgs={"data": rd},
                             method="minuit", verbose=True, strategy=2,
                             tol=0.05)
    pars = ipanema.Parameters.clone(res.params)
    for k, v in pars.items():
      v.init = v.value
    print("Partial fits succeded! Now performing full fit")
    pars.unlock('fsigBs', 'muBs', 'b')
    if 'fsigBd' in pars:  # and len(rd.mass) > 5000:
      pars.unlock('fsigBd')
    if 'lambd' in pars:  # and len(rd.mass) > 200:
      pars.unlock('lambd')
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd},
                           method='minuit', verbose=True, strategy=2,
                           tol=0.05)

  if res:
    print(res)
    fpars = ipanema.Parameters.clone(res.params)
  else:
    print("Could not fit it!. Cloning pars to res")
    fpars = ipanema.Parameters.clone(pars)
    print(fpars)

  # plot the fit result {{{

  # fall back to averaged resolution when plotting
  all_pdfs = {}
  _p = fpars.valuesdict()
  merr = _p["s0Bs"] + _p["s1Bs"] * rd.merr + _p["s2Bs"] * rd.merr * rd.merr
  merr = np.median(ipanema.ristra.get(rd.merr))

  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(
      ipanema.ristra.get(rd.mass), weights=rd.df.eval(mass_weight), bins=60, density=False
  )
  axplot.errorbar(
      hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr, fmt=".k"
  )

  mass = ipanema.ristra.linspace(ipanema.ristra.min(rd.mass), ipanema.ristra.max(rd.mass), 1000)
  signal = 0 * mass
  merr = ipanema.ristra.allocate(np.ones_like(mass.get()) * merr)

  # plot signal: nbkg -> 0 and nexp -> 0
  _p = ipanema.Parameters.clone(fpars)
  if "fsigBd" in _p:
    _p["fsigBd"].set(value=0, min=-np.inf, max=np.inf)
  if "fcomb" in _p:
    _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
  _x, _y = ipanema.ristra.get(mass), ipanema.ristra.get(
      pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  all_pdfs['pdf_fsigBs'] = _y
  axplot.plot(_x, _y, color="C1", label=rf"$B_s^0$ {model}")

  # plot backgrounds: nsig -> 0
  if has_bd:
    _p = ipanema.Parameters.clone(fpars)
    if "fsigBs" in _p:
      _p["fsigBs"].set(value=0, min=-np.inf, max=np.inf)
    if "fcomb" in _p:
      _p["fcomb"].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ipanema.ristra.get(mass), ipanema.ristra.get(
        pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
    )
    axplot.plot(_x, _y, "-.", color="C2", label="Bd")
    all_pdfs['pdf_fsigBd'] = _y

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(fpars)
  x, y = ipanema.ristra.get(mass), ipanema.ristra.get(
      pdf(mass, merr, signal, **_p.valuesdict(), norm=hdata.norm)
  )
  all_pdfs['pdf_total'] = y
  all_pdfs['mass_mumuKK'] = x
  all_pdfs['bins'] = hdata.bins
  all_pdfs['counts'] = hdata.counts
  all_pdfs['yerr'] = hdata.yerr
  all_pdfs['xerr'] = hdata.xerr

  axplot.plot(x, y, color="C0")
  pulls = complot.compute_pdfpulls(x, y, hdata.bins, hdata.counts, *hdata.yerr)
  axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)
  all_pdfs['pulls'] = pulls

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
    axplot.set_ylim(1e-1, 1.5 * np.max(y))
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

    sw = ipanema.splot.compute_sweights(
        lambda *x, **y: pdf(rd.mass, rd.merr, rd.pdf, *x, **y),
        _pars, _yields, rd.weight)

    for k, v in sw.items():
      _sw = np.copy(_proxy)
      _sw[list(rd.df.index)] = v
      sw[k] = _sw
    sw = {**sw, **all_pdfs}
    return (fpars, sw)

  # }}}

  return (fpars, False)


# }}}


# command-line interface {{{

if __name__ == '__main__':
  DESCRIPTION = """
  Macro to compute sWeights for Bs RD and Bs MC samples.
  """
  p = argparse.ArgumentParser(description=DESCRIPTION)
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
  p.add_argument('--version')
  args = vars(p.parse_args())

  if args["sweights"]:
    sweights = True
  else:
    sweights = False

  mass_model = args['mass_model']
  if "sigma" in args['version']:
    if mass_model == 'cbcalib':
      mass_model = 'crystalball'

  has_bd = True if args["mode"] == "Bs2JpsiPhi" else False

  if args["input_params"]:
    input_pars = ipanema.Parameters.load(args["input_params"])
  else:
    input_pars = False

  branches = ["B_ConstJpsi_M_1", "B_ConstJpsi_MERR_1", "hlt1b", "X_M",
              'time', 'helcosthetaK', 'helcosthetaL', 'helphi']

  if args["mass_weight"]:
    mass_weight = args["mass_weight"]
    branches += [mass_weight]
  else:
    mass_weight = "B_ConstJpsi_M_1/B_ConstJpsi_M_1"

  cut = False
  if 'MC' in args['mode']:
    branches += ["B_BKGCAT"]
  if "prefit" in args["output_params"]:
    cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"

  if 'bkgcat60' in args['version'] and 'MC' in args['mode'] and args['mass_bin'] == 'all':
    cut = "B_BKGCAT == 60"
    if mass_model == 'cbcalib':
      mass_model = 'crystalball'

  sample = ipanema.Sample.from_root(args["sample"], branches=branches)

  # mass_range = (5202, 5548)
  mass_range = (5200, 5550)
  #   mass_range = (5270, 5450) # DANGER
  bin_cut = get_bin_from_version(args['version'], args['mass_bin'], args['mode'])

  if input_pars:
    from analysis.samples.chop_tuples import vsub_dict
    if "LSBsmall" in args['version']:
      mass_cut = vsub_dict['LSBsmall']
      mass_range = False
      print("mass cut from (L|R)SB", mass_cut)
      cut = cuts_and(mass_cut, cut)
    elif "RSBsmall" in args['version']:
      mass_cut = vsub_dict['RSBsmall']
      mass_range = False
      has_bd = False
      print("mass cut from (L|R)SB", mass_cut)
      cut = cuts_and(mass_cut, cut)
    elif "LSB" in args['version']:
      # mass_cut = vsub_dict['LSB']
      # cut = cuts_and(mass_cut, cut)
      # mass_range = False
      # mass_range = (mass_range[0], 5367 + 30)
      _2sig = input_pars['muBs'].value + 2 * 30
      mass_range = (mass_range[0], _2sig)
    elif "RSB" in args['version']:
      # mass_cut = vsub_dict['RSB']
      # cut = cuts_and(mass_cut, cut)
      # mass_range = False
      # mass_range = (5367 - 30, mass_range[1])
      _2sig = input_pars['muBs'].value - 2 * 30
      mass_range = (_2sig, mass_range[1])
      # has_bd = False
  cut = cuts_and(bin_cut, cut)
  print(cut)

  print(sample.df)
  pars, sw = mass_fitter(
      sample.df,
      mass_range=mass_range,
      mass_branch="B_ConstJpsi_M_1",
      mass_weight=mass_weight,
      trigger=args["trigger"],
      figs=args["output_figures"],
      model=mass_model,
      cut=cut,
      sweights=sweights,
      has_bd=has_bd,
      input_pars=input_pars,
      verbose=False,
  )
  for k, v in pars.items():
    if v.free:
      print(f"{k:>10} : {v.uvalue:.6uP}")
  pars.dump(args["output_params"])
  if sw:
    np.save(args["sweights"], sw)

# }}}


# vim: fdm=marker
