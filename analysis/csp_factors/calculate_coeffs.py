__all__ = ['blatt_weisskopf', 'barrier_factor_bw', 'breit_wigner', 'flatte']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


import numpy as np
import argparse
from scipy import interpolate
from ipanema import Parameters

from analysis.csp_factors.efficiency import create_mass_bins

lo = 2 * 493.677  # EvtGen table
hi = 1060.

# TODO: look for these numbers in PDG
MKp = 493.677
MK0 = 497.614  # EvtGen tables
Mpip = 139.57018  # EvtGen tables
Mpi0 = 134.9766  # EvtGen tables

m0_flatte = 949.9
gPiPi = 167.0
gKK = 3.05 * gPiPi

m0_bwigner = 1019.4610
g0_bwigner = 4.266
mKp = MKp

Mmom = 5366.77
Msister = 3096.916
Mdau2 = MKp
Mdau1 = Mdau2


def Blatt_Weisskopf(q, q0, L=1, d0=3e-3):
  """
  Get Blatt-Weisskopf coefficient
  """
  if (L < 1.):
    return 1.
  d = d0 / L
  z = q * d * q * d
  z0 = q0 * d * q0 * d
  if (L == 1):
    return (1 + z0) / (1 + z)
  elif (L == 2):
    return ((z0 - 3) * (z0 - 3) + 9 * z0) / ((z - 3) * (z - 3) + 9 * z)
  elif (L == 3):
    return (z0 * (z0 - 15) * (z0 - 15) + 9 * (z0 - 5)) / (z * (z - 15) * (z - 15) + 9 * (z - 5))


def barrier_factor_BW(q, L=1, isB=0):
  """
  Yet another version for the Get Blatt-Weisskopf coefficient

  Please take a look
  """
  if L < 1:
    return 1.0

  if isB == 0:
    d = 3.e-03 / L
  else:
    d = 5.e-03

  z = q * d * q * d
  if (L == 1):
    return np.sqrt(1 / (1 + z))
  if (L == 2):
    return np.sqrt((1. / ((z - 3) * (z - 3) + 9 * z)))


# lineshapes {{{

def breit_wigner(m, M0, Gamma0, m1, m2, J=1):
  """
  Breit Wigner propagator
  """
  def get_q(M, m1, m2):
    M2 = M * M
    m12 = m1 * m1
    m22 = m2 * m2
    q2 = .25 * (M2 * M2 - 2 * M2 * (m12 + m22) + (m12 * m12 + m22 * m22) - 2 * m12 * m22) / M2
    return np.sqrt(q2)

  def q(x): return get_q(x, m1, m2)
  q0 = get_q(M0, m1, m2)
  Gamma = Gamma0 * np.power(q(m) / q0, 2 * J + 1) * M0 / m * Blatt_Weisskopf(q(m), q0, J)
  return 1. / (M0 * M0 - m * m - 1j * M0 * Gamma)


def flatte(m, m0, gpipi, gKK, Mpip, Mpi0, MKp, MK0):
  @np.vectorize
  def get_rho(mu, m0):
    rho_sq = 1 - 4 * m0 * m0 / (mu * mu)
    if rho_sq < 0:
      return 1j * np.abs(rho_sq)**0.5
    else:
      return np.abs(rho_sq)**0.5
  ans = (
      gpipi * ((2. / 3.) * get_rho(m, Mpip) + (1. / 3.) * get_rho(m, Mpi0)) +
      gKK * ((1. / 2.) * get_rho(m, MKp) + (1. / 2.) * get_rho(m, MK0))
  )
  ans = m0 * m0 - m * m - 1j * m0 * ans
  return 1 / ans


# def NR_spline():
#     m_knots = np.array([.990, 1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060])
#     SR_knots = np.array([1.66216, 1.03973, 0.177684, -
#                         0.376651, -0.184457, -1.19145, 0.899762, -2.59912])
#     SI_knots = np.array([1.67003, 1.25299, 1.17785, 1.12897,
#                         1.28353, 1.00101, 1.00347, 0.693512])
#     N_knots = len(SR_knots)
#     SR = TSpline3("SR", m_knots, SR_knots, N_knots)
#     SI = TSpline3("SI", m_knots, SI_knots, N_knots)
#
#     return SR, SI
#
#
# SR_S_new, SI_S_new = NR_spline()

# }}}


def f0_Syr(m): return flatte(m, m0_flatte, gPiPi, gKK, Mpip, Mpi0, MKp, MK0)


def phi2KK_EvtGen(m): return breit_wigner(
    m, m0_bwigner, g0_bwigner, mKp, mKp, 1)


def get_q(M, m1, m2):
  M2 = M * M
  m12 = m1 * m1
  m22 = m2 * m2
  q2 = .25 * (M2 * M2 - 2 * M2 * (m12 + m22) + (m12 * m12 + m22 * m22) - 2 * m12 * m22) / M2
  return np.sqrt(q2)


def qq(m): return get_q(m, Mdau1, Mdau2)
def pp(m): return get_q(Mmom, Msister, m)


def Bs2f0Jpsi_BW(m): return barrier_factor_BW(pp(m), 1, 1)
def phi2KK_BW(m): return barrier_factor_BW(qq(m), 1, 0)


def Bs2JpsiKK_ps_S(m): return np.sqrt(pp(m) * qq(m)) * pp(m) * Bs2f0Jpsi_BW(m)
def Bs2JpsiKK_ps_P(m): return np.sqrt(pp(m) * qq(m)) * qq(m) * phi2KK_BW(m)


def evCsp(mLL, mUL, threshold, cut_off, eff=False,
          swave=f0_Syr, pwave=phi2KK_EvtGen,
          PSs=Bs2JpsiKK_ps_S, PSp=Bs2JpsiKK_ps_P):
  """
  Core of the CSP calculation
  """

  # it there is no efficiency, then just create a flat step function
  if not eff:
    @np.vectorize
    def eff(m):
      if m >= mLL and m <= mUL:
        return 1
      return 0
    threshold = mLL
    cut_off = mUL

  def sw(m): return swave(m)
  def pw(m): return pwave(m)
  def pwconj(m): return np.conjugate(pwave(m))
  def swconj(m): return np.conjugate(swave(m))

  def f1(m): return pw(m) * pwconj(m) * PSp(m) * PSp(m)
  def f2(m): return sw(m) * swconj(m) * PSs(m) * PSs(m)
  def f3(m): return swconj(m) * pw(m) * PSs(m) * PSp(m)

  _x = np.linspace(threshold, cut_off, int(1e4))
  _c = np.trapz(f1(_x) * np.abs(eff(_x)), _x)
  _d = np.trapz(f2(_x) * np.abs(eff(_x)), _x)
  # exit()
  if np.imag(_c) > 1e-14 or np.imag(_d) > 1e-14:
    print("WARNING: Precision in the integral is not good")
  _c = np.real(_c)
  _d = np.real(_d)
  csp = np.trapz(f3(_x) * eff(_x), _x)
  csp /= np.sqrt(_d * _c)
  x = np.real(csp)
  y = np.imag(csp)

  csp_factor = np.sqrt(x**2 + y**2)
  delta = -np.arctan2(y, x)
  # print("CSP:", csp_factor, delta)

  return csp_factor, delta


# def num_int_Tspline(SR_S, SI_S, pw, lo, hi, PSs=Bs2JpsiKK_ps_S, PSp=Bs2JpsiKK_ps_P):
#     bins = 1000
#     bin_width = (hi - lo)/float(bins)
#     integral_ss = 0
#     integral_ps = 0
#
#     for i in range(bins+1):
#         x = lo + bin_width*i
#         PSs_tmp = PSs
#         PSs_n = 1.
#         PSp_tmp = PSp
#         PSp_n = PSp_tmp.subs(mass, x)
#         pw_tmp = pw
#         pw_n = pw_tmp.subs(mass, x)
#         xgev = x*1e-03
#         y_ss = ((SR_S.Eval(xgev)*SR_S.Eval(xgev) +
#                 SI_S.Eval(xgev)*SI_S.Eval(xgev))*PSs_n*PSs_n)
#         y_ps = ((SR_S.Eval(xgev) - I*SI_S.Eval(xgev))*pw_n*PSs_n*PSp_n).n()
#         integral_ss += bin_width*y_ss
#         integral_ps += bin_width*y_ps
#
#     return integral_ss, integral_ps
#
#
# def Csp_Tspline(SR_S, SI_S, pw, lo, hi, PSs=Bs2JpsiKK_ps_S, PSp=Bs2JpsiKK_ps_P):
#     pwconj = pw.conjugate()
#     pwconj = pwconj.subs(mass, mass.conjugate())
#     c = Integral(pw*pwconj*PSp*PSp, (mass, mul, muh))
#     c = c.subs([(mul, lo), (muh, hi)])
#     c = c.n()
#     d, csp = num_int_Tspline(SR_S, SI_S, pw, lo, hi)
#     cte = 1/Sqrt(d*c)
#     csp = csp*cte
#
#     x = re(csp)
#     y = im(csp)
#
#     CSP = sqrt(x**2 + y**2)
#     theta = -atan2(y, x)
#
#     return CSP, theta


def calculate_csp_factors(mKK_knots, histos=False, as_params=False,
                          threshold=2 * (493.677 + 2 * 0), cut_off=1200):
  # input_dir1, input_dir2 = histos, histos

  # TODO: get this from the created funtion
  # __x = np.linspace(mKK_knots[0], mKK_knots[-1], 100)

  if isinstance(histos, np.ndarray):
    eff = []
    x = []
    y = []
    for bx, by in zip(*histos):
      x.append(np.array(bx))
      y.append(np.array(by))
      eff.append(interpolate.interp1d(bx, by, fill_value='extrapolate'))
  else:
    eff = [False for i in range(len(mKK_knots) - 1)]

  resolution_last_bin = True
  if not isinstance(histos, np.ndarray):
    resolution_last_bin = False
  # this is useful for non S-wave MCs only
  resolution_last_bin = False

  if len(eff) > 3 and resolution_last_bin:
    print('adding shifts')
    __f = []
    _x = []
    _y = []
    # WARNING: This two lines are hardcoded
    some_cut = 1060.
    shifts = [8, 16, 18]
    for i in shifts:
      __x = x[-2] + i
      __y = np.delete(y[-2], np.argwhere(__x > some_cut))
      __x = np.delete(__x, np.argwhere(__x > some_cut))
      _x.append(__x)
      _y.append(__y)
      # __f.append(interpolate.interp1d( __x, __y, fill_value='extrapolate'))
      __f.append(
          interpolate.interp1d(__x, __y, fill_value=(__y[0], __y[-1]), bounds_error=False)
      )

    __x = _x[0].tolist() + _x[1].tolist() + _x[2].tolist()
    __x.sort()
    __x = np.array(__x)
    __y = np.maximum(__f[0](__x) + (1 - __f[0](__x)) * __f[1](__x), __f[2](__x))
    __f = interpolate.interp1d(__x, __y, fill_value=(
        __y[0], __y[-1]), bounds_error=False)
    eff[-1] = __f

  # Start the computation
  coeffs = []
  print("Calculating coefficients")
  for i in range(len(mKK_knots) - 1):
    mLL, mUL = mKK_knots[i], mKK_knots[i + 1]
    coeffs.append(evCsp(mLL, mUL, threshold, cut_off, eff[i], f0_Syr))

  # cook results
  csps = []
  deltas = []
  for rp, ip in coeffs:
    csps.append(np.float64(rp))
    deltas.append(np.float64(ip))
  print(csps)
  print(deltas)
  if as_params:
    pars = Parameters()
    for ib, mb in enumerate(mKK_knots):
      pars.add(dict(
          name=f"mKK{ib}", latex=rf"m_{{KK}}^{{({ib})}}",
          value=mb, free=False
      ))
    for ib, mb in enumerate(csps):
      pars.add(dict(
          name=f"CSP{ib+1}", latex=rf"C_{{SP}}^{{({ib+1})}}",
          value=mb, free=False
      ))
    for ib, mb in enumerate(deltas):
      pars.add(dict(
          name=f"deltaSP{ib+1}", latex=rf"\delta_{{SP}}^{{({ib+1})}}",
          value=mb, free=False
      ))

    print(pars)
    return pars

  return csps, deltas


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--histos', help='Path to the MC_BsJpsiPhi files')
  p.add_argument('--year', help='Year of the selection in yaml')
  p.add_argument('--output', help='Name of output json file')
  p.add_argument('--mode', help='Name of output json file')
  p.add_argument('--nbins', help='Name of output json file')
  args = vars(p.parse_args())
  histos = 'merda.root'
  mode = args['mode']
  year = args['year']
  output = args['output']
  histos = np.load(args['histos'], allow_pickle=True)
  # histos = False

  nbins = int(args['nbins'])
  # mKK_knots = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  # b                           1020
  # mKK_knots = [990, 1014.78, 1018.41, 1020, 10223.42, 1032, 1050]
  mKK = create_mass_bins(nbins)
  print(mKK)
  csp = calculate_csp_factors(mKK, histos, as_params=True)
  csp.dump(output)


# vim: fdm=marker
