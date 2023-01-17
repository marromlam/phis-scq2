# from SomeUtils.numericFunctionClass import *
from numericFunctionClass import *
from math import *
# from Urania.SympyBasic import *
import numpy as np
# from Urania import PDG
#from ROOT import TSpline3, TCanvas, TColor
from ROOT import *
import pickle as cPickle
import sys
import argparse
import json
# import sympy as sp
# from sympy import I

from scipy import interpolate
# import scipy
# from scipy.integrate import quad
#
# def complex_quadrature(func, a, b, **kwargs):
#     def real_func(x):
#         return scipy.real(func(x))
#     def imag_func(x):
#         return scipy.imag(func(x))
#     real_integral = quad(real_func, a, b, **kwargs)
#     imag_integral = quad(imag_func, a, b, **kwargs)
#     return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

import scipy
# from scipy import array

def quad_routine(func, a, b, x_list, w_list):
    c_1 = (b-a)/2.0
    c_2 = (b+a)/2.0
    eval_points = map(lambda x: c_1*x+c_2, x_list)
    func_evals = list(map(func, eval_points))    # Python 3: make a list here
    return c_1 * np.sum(np.array(func_evals) * np.array(w_list))

def quad_gauss_7(func, a, b):
    x_gauss = [-0.949107912342759, -0.741531185599394, -0.405845151377397, 0, 0.405845151377397, 0.741531185599394, 0.949107912342759]
    w_gauss = np.array([0.129484966168870, 0.279705391489277, 0.381830050505119, 0.417959183673469, 0.381830050505119, 0.279705391489277,0.129484966168870])
    return quad_routine(func,a,b,x_gauss, w_gauss)

def quad_kronrod_15(func, a, b):
    x_kr = [-0.991455371120813,-0.949107912342759, -0.864864423359769, -0.741531185599394, -0.586087235467691,-0.405845151377397, -0.207784955007898, 0.0, 0.207784955007898,0.405845151377397, 0.586087235467691, 0.741531185599394, 0.864864423359769, 0.949107912342759, 0.991455371120813]
    w_kr = [0.022935322010529, 0.063092092629979, 0.104790010322250, 0.140653259715525, 0.169004726639267, 0.190350578064785, 0.204432940075298, 0.209482141084728, 0.204432940075298, 0.190350578064785, 0.169004726639267, 0.140653259715525,  0.104790010322250, 0.063092092629979, 0.022935322010529]
    return quad_routine(func,a,b,x_kr, w_kr)

class Memoize:                     # Python 3: no need to inherit from object
    def __init__(self, func):
        self.func = func
        self.eval_points = {}
    def __call__(self, *args):
        if args not in self.eval_points:
            self.eval_points[args] = self.func(*args)
        return self.eval_points[args]

def complex_quadrature(func,a,b):
    ''' Output is the 15 point estimate; and the estimated error '''
    func = Memoize(func) #  Memoize function to skip repeated function calls.
    g7 = quad_gauss_7(func,a,b)
    k15 = quad_kronrod_15(func,a,b)
    err = (200*np.absolute(g7-k15))**1.5

    if g7 is np.nan or k15 is np.nan:
        return k15
    elif 1e10*(g7 - k15) < 0.2e-15:
        print("tolerance reached")
        return k15
    elif a-b < 0.2e-16:
        print("cant split more")
        return complex_quadrature(func, (a+b)/2, b)
    else:
        g7 = quad_gauss_7(func,(a+b)/2,b)
        k15 = quad_kronrod_15(func,a,(a+b)/2)
        return g7+k15


    # I don't have much faith in this error estimate taken from wikipedia
    # without incorporating how it should scale with changing limits
    # return [k15, (200*np.absolute(g7-k15))**1.5]



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--histos', help='Path to the MC_BsJpsiPhi files')
    parser.add_argument('--year', help='Year of the selection in yaml')
    parser.add_argument('--output', help='Name of output json file')
    parser.add_argument('--mode', help='Name of output json file')
    return parser


# myJ = sp.Symbol("L", positive=True)
Nbins = 1000
lo = 2*493.677  # EvtGen table
hi = 1060.

# mass = sp.Symbol("mu", positive=True)
# m0 = sp.Symbol("m0", positive=True)
# Mmom = sp.Symbol("Mmom", positive=True)
# Msister = sp.Symbol("Msister", positive=True)
# gpipi = sp.Symbol("gpipi", positive=True)
# gKK = sp.Symbol("gKK", positive=True)

# MKp = PDG.Kplus.mass
# Mpip = PDG.piplus.mass  # 139.57018#EvtGen tables
# Mpi0 = PDG.pi0.mass  # 134.9766#EvtGen tables
# MK0 = PDG.K0.mass  # 497.614#EvtGen tables
# TODO: look for these numbers in PDG
MKp = 493.677
MK0 = 497.614#EvtGen tables
Mpip = 139.57018#EvtGen tables
Mpi0 = 134.9766#EvtGen tables

# Mdau1 = sp.Symbol("mdau1", positive=True)
# Mdau2 = sp.Symbol("mdau2", positive=True)
#
# Gamma0 = sp.Symbol("Gamma0", positive=True)
# muh = sp.Symbol("muh", positive=True)
# mul = sp.Symbol("mul", positive=True)


def Blatt_Weisskopf(q, q0, L=1, d0=3e-3):
    """
    Get Blatt-Weisskopf coefficient
    """
    if (L < 1.):
        return 1.
    d = d0/L
    z = q*d*q*d
    z0 = q0*d*q0*d
    if (L == 1):
        return (1+z0)/(1+z)
    elif (L == 2):
        return ((z0-3)*(z0-3) + 9*z0) / ((z-3)*(z-3) + 9*z)
    elif (L == 3):
        return (z0*(z0-15)*(z0-15) + 9*(z0-5)) / (z*(z-15)*(z-15) + 9*(z-5))


def barrier_factor_BW(q, L=1, isB=0):
    """
    Yet another version for the Get Blatt-Weisskopf coefficient

    Please take a look
    """
    if L < 1:
        return 1.0

    if isB == 0:
        d = 3.e-03/L
    else:
        d = 5.e-03

    z = q*d*q*d
    if (L == 1):
        return np.sqrt(1/(1+z))
    if (L == 2):
        return np.sqrt((1. / ((z-3)*(z-3) + 9*z)))


# lineshapes {{{

def Breit_Wigner(m, M0, Gamma0, m1, m2, J=1):
    """
    Breit Wigner propagator
    """
    def get_q(M, m1, m2):
        M2 = M*M
        m12 = m1*m1
        m22 = m2*m2
        q2 = .25*(M2*M2 - 2*M2*(m12+m22) + (m12*m12+m22*m22)-2*m12*m22) / M2
        return np.sqrt(q2)
    q = lambda x: get_q(x, m1, m2)
    q0 = get_q(M0, m1, m2)
    Gamma = Gamma0 * np.power(q(m)/q0, 2*J+1)*M0/m*Blatt_Weisskopf(q(m), q0, J)
    return 1./(M0*M0-m*m-1j*M0*Gamma)


def flatte(m, m0, gpipi, gKK, Mpip, Mpi0, MKp, MK0):
    @np.vectorize
    def get_rho(mu, m0):
        rho_sq = 1 - 4*m0*m0/(mu*mu)
        if rho_sq < 0:
            return 1j*np.abs(rho_sq)**0.5
        else:
            return np.abs(rho_sq)**0.5
    ans = (
        gpipi*((2./3.)*get_rho(m, Mpip) + (1./3.)*get_rho(m, Mpi0)) +
        gKK*(  (1./2.)*get_rho(m, MKp)  + (1./2.)*get_rho(m, MK0))
    )
    ans = m0*m0 - m*m - 1j*m0*ans
    return 1/ans


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




m0_flatte = 949.9
gPiPi = 167.0
gKK = 3.05 * gPiPi

f0_Syr = lambda m: flatte(m, m0_flatte, gPiPi, gKK, Mpip, Mpi0, MKp, MK0)


m0_bwigner = 1019.4610
g0_bwigner = 4.266
mKp = MKp
# EvtGen = Breit_Wigner(mass, m0_bwigner, g0_bwigner, mKp, mKp, 1)

# Step by step. First we tell her the two daughters are the same
# phi2KK_EvtGen = EvtGen.subs([(m0, 1019.4610), (Gamma0, 4.266), (Mdau1, Mdau2)])
# Now substitute the daughter by a number
# phi2KK_EvtGen = phi2KK_EvtGen.subs(Mdau2, MKp)
phi2KK_EvtGen = lambda m: Breit_Wigner(m, m0_bwigner, g0_bwigner, mKp, mKp, 1)

def get_q(M, m1, m2):
    M2 = M*M
    m12 = m1*m1
    m22 = m2*m2
    q2 = .25*(M2*M2 - 2*M2*(m12+m22) + (m12*m12+m22*m22)-2*m12*m22) / M2
    return np.sqrt(q2)


Mmom = 5366.77
Msister = 3096.916
Mdau2 = MKp
Mdau1 = Mdau2
print(Mdau1, Mdau2, Msister)
# mkk_mass = np.linspace(mLL, mUL, 1000)
q = lambda m: get_q(m, Mdau1, Mdau2)
p = lambda m: get_q(Mmom, Msister, m)

Bs2f0Jpsi_BW = lambda m: barrier_factor_BW(p(m), 1, 1)
phi2KK_BW = lambda m: barrier_factor_BW(q(m), 1, 0)

Bs2JpsiKK_ps_S = lambda m: np.sqrt(p(m)*q(m)) * p(m) * Bs2f0Jpsi_BW(m)
Bs2JpsiKK_ps_P = lambda m: np.sqrt(p(m)*q(m)) * q(m) * phi2KK_BW(m)






def evCsp(mLL, mUL, threshold, cut_off, eff=False,
          swave=f0_Syr, pwave=phi2KK_EvtGen,
          PSs=Bs2JpsiKK_ps_S, PSp=Bs2JpsiKK_ps_P):
    """
    Core of the CSP calculation
    """
    # eff = 

    # it there is no efficiency, then just create a flat step function
    if not eff:
        @np.vectorize
        def eff(m):
            # if m < mLL or m > mUL:
            #     return 0
            return 1


    sw = lambda m: swave(m)
    pw = lambda m: pwave(m)
    pwconj = lambda m: np.conjugate(pwave(m))
    swconj = lambda m: np.conjugate(swave(m))

    f1 = lambda m: pw(m) * pwconj(m) * PSp(m) * PSp(m)
    f2 = lambda m: sw(m) * swconj(m) * PSs(m) * PSs(m)
    f3 = lambda m: swconj(m) * pw(m) * PSs(m) * PSp(m)


    _x = np.linspace(threshold, cut_off, int(1e4))
    print("eff =", eff(_x))
    # _c = complex_quadrature(lambda m: f1(m) * eff(m), lo, cut_off)
    _c = np.trapz(f1(_x) * eff(_x), _x)
    # _d = complex_quadrature(lambda m: f2(m) * eff(m), lo, cut_off)
    _d = np.trapz(f2(_x) * eff(_x), _x)
    print(_c, _d)
    # exit()
    if np.imag(_c) > 1e-14 or np.imag(_d) > 1e-14:
        print("WARNING: Preciion in the integral is not good")
    _c = np.real(_c)
    _d = np.real(_d)
    # csp = complex_quadrature(lambda m: f3(m) * eff(m), lo, cut_off)
    csp = np.trapz(f3(_x) * eff(_x), _x)
    csp /= np.sqrt(_d * _c)
    x = np.real(csp)
    y = np.imag(csp)

    csp_factor = np.sqrt(x**2 + y**2)
    delta = -np.arctan2(y, x)
    print("CSP:", csp_factor, delta)

    return csp_factor, delta


def load_phis(mKK_bins, input_dir):
    phis = []
    for i in range(len(mKK_bins)-1):
        m0 = mKK_bins[i]
        m1 = mKK_bins[i+1]
        phis.append(cPickle.load( open(input_dir+"eff_hist_"+str(m0)+"_"+str(m1), 'rb') ))

    return phis


def analytical_csp(sw, pw, lo, hi, PSs=Bs2JpsiKK_ps_S, PSp=Bs2JpsiKK_ps_P):
    pwconj = pw.conjugate()
    # mass is real, let's make life easier
    pwconj = pwconj.subs(mass, mass.conjugate())
    c = sp.Integral(pw*pwconj*PSp*PSp, (mass, mul, muh))
    c = c.subs([(mul, lo), (muh, hi)])
    c = c.n()
    d = sp.Integral(sw*sw.conjugate()*PSs*PSs, (mass, mul, muh))
    d = d.subs([(mul, lo), (muh, hi)])
    d = d.n()
    cte = 1/sp.sqrt(d*c)
    csp = sp.Integral(sw.conjugate()*cte*pw*PSs*PSp, (mass, mul, muh))
    csp = csp.subs([(mul, lo), (muh, hi)])
    csp = csp.n()
    x = float(sp.re(csp))
    y = float(sp.im(csp))

    CSP = np.sqrt(x**2 + y**2)
    #theta = -atan(y/x)
    theta = -np.arctan2(y, x)
    if theta < 0:
        theta = theta+2*pi

    return CSP, theta


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


def do_shifted_phi(phi, delta):
    x0, y0 = [], []
    for i in phi.References:
        if i+delta > 1060:
            continue
        x0.append(i+delta)
        y0.append(phi(i))
    return NF(x0, y0)
    # return interpolate.interp1d(x0, y0)




mKK_knots = [990, 1008, 1016, 1020, 1024, 1032, 1050]
mass_knots = [990, 1008, 1016, 1020, 1024, 1032, 1050]

def calculate_csp_factors_with_efficiency(histos, mKK_knots, as_params=False):
    input_dir1, input_dir2 = histos, histos
    # TODO: get this from the created funtion
    __x = np.linspace(980, 1050, 100)

    # load efficiency histograms
    phis_NEW = load_phis(mKK_knots, input_dir1)
    phis_NEW_SWAVE = load_phis(mKK_knots, input_dir2)

    # load efficiency
    x = [
        np.float64(phis_NEW[i].References) for i in range(len(phis_NEW))
    ]
    y = [
        np.float64(list(phis_NEW[i].DataBase.values())) for i in range(len(phis_NEW))
    ]

    eff = [
        interpolate.interp1d(x[i], y[i], fill_value='extrapolate') for i in range(len(phis_NEW))
    ]
    # print(phis_NEW[4](1020), __f[-2](1020))
    # exit()
    resolution_last_bin = True
    if len(eff) > 3 and resolution_last_bin:
        # print(phis_NEW)
        # exit()
        __f = []
        _x = []; _y = []
        some_cut = 1060.
        shifts = [8, 16, 18]
        for i in shifts:
            __x = x[-2] + i
            __y = np.delete(y[-2], np.argwhere(__x > some_cut))
            __x = np.delete(__x, np.argwhere(__x > some_cut))
            _x.append(__x)
            _y.append(__y)
            __f.append(interpolate.interp1d(__x, __y, fill_value='extrapolate'))
        phi0_NEW = do_shifted_phi(phis_NEW[4], 8)
        phi1_NEW = do_shifted_phi(phis_NEW[4], 16)
        phi2_NEW = do_shifted_phi(phis_NEW[4], 18)

        print(phi0_NEW(1020), __f[0](1020))
        print(phi1_NEW(1020), __f[1](1020))

        __x = _x[0].tolist() + _x[1].tolist() + _x[2].tolist()
        __x.sort()
        __x = np.array(__x)
        xlist = phi0_NEW.References + phi1_NEW.References + phi2_NEW.References
        xlist.sort()
        ylist = []
        for m in xlist:
            e1 = phi0_NEW(m)
            e2 = phi1_NEW(m)
            e3 = phi2_NEW(m)

            e0 = e1 + (1-e1)*e2
            ylist.append(max(e0, e3))
        g_NEW = NF(xlist, ylist)
        # print(__x)
        print("sdfadsfdsafadsfasdfdsfsdfa")
        # print( __f[1](__x)+(1-__f[1](__x))*__f[2](__x) )
        print( __f[2](__x) )
        print(__x)
        __y = np.maximum(__f[0](__x)+(1-__f[0](__x))*__f[1](__x), __f[2](__x))
        __f = interpolate.interp1d(__x, __y, fill_value=(__y[0], __y[-1]), bounds_error=False)
        eff[-1] = __f
        print("->-", g_NEW(1060), __f(1060))
        # exit()
    # exit()
    # if len(mass_knots) > 4:
    #     eff = []
    #     for x, y in 

    threshold = lo
    cut_off = 1200.

    CSP = []
    print("f0, phi with eff.")
    for i in range(len(mass_knots)-1):
        mLL, mUL = mass_knots[i], mass_knots[i+1]
        CSP.append(evCsp(mLL, mUL, threshold, cut_off, eff[i], f0_Syr))
    print(CSP)
    # CSP.append(evCsp(mLL, mUL, threshold, cut_off, g_NEW, f0_Syr))
    # CSP.append([0,0])
    print("JpsiPhi CSP_factors = {1: "+str(round(CSP[0][0], 4))+", 2: "+str(round(CSP[1][0], 4))+", 3: "+str(round(
        CSP[2][0], 4))+", 4: "+str(round(CSP[3][0], 4))+", 5: "+str(round(CSP[4][0], 4))+", 6: "+str(round(CSP[5][0], 4))+"}")
    exit()

    cSP = 0
    deltaS = 0
    return cSP, deltaS

    # fh = open(output, 'w')
    # fh.write('{\n')
    # fh.write('\t"'+str(year)+'":{')
    # fh.write('\n\t\t"CspFactors" : [')
    #
    # first = True
    # for i in range(6):
    #     if not first:
    #         fh.write(',')
    #     first = False
    #     fh.write('\n\t\t\t{')
    #     fh.write('\n\t\t\t\t"Name": "Csp'+str(i)+'",')
    #     fh.write('\n\t\t\t\t"Value": '+str(round(CSP_SWAVE[i][0], 4))+',')
    #     fh.write('\n\t\t\t\t"Error": 0.0,')
    #     fh.write('\n\t\t\t\t"Bin_ll":'+str(mKK_bins[i])+',')
    #     fh.write('\n\t\t\t\t"Bin_ul":'+str(mKK_bins[i+1]))
    #     fh.write('\n\t\t\t}')
    # fh.write('\n\t\t]')
    # fh.write('\n\t}')
    # fh.write('\n}')
    # fh.close()

    # # exit(0)
    #
    # '''
    # Code below for systematics from previous round. To be adjusted during systematics studies.
    # '''
    #
    # print("=========================")
    # print("Csp factors for systematics")
    # print("=========================")
    #
    # print("f0, phi NOT SMEARED")
    # C1 = Csp(f0_Syr, phi2KK_EvtGen, 990, 1008)
    # C2 = Csp(f0_Syr, phi2KK_EvtGen, 1008, 1016)
    # C3 = Csp(f0_Syr, phi2KK_EvtGen, 1016, 1020)
    # C4 = Csp(f0_Syr, phi2KK_EvtGen, 1020, 1024)
    # C5 = Csp(f0_Syr, phi2KK_EvtGen, 1024, 1032)
    # C6 = Csp(f0_Syr, phi2KK_EvtGen, 1032, 1050)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # cut_off_spl = 1060.
    # print("NR, phi SMEARED NEW SPLINE, NEW HIST, CUT OFF at", cut_off_spl)
    # C1 = evCsp_Tspline(
    #     phis_NEW_SWAVE[0], SR_S_new, SI_S_new, 990, 1008, cut_off_spl, phi2KK_EvtGen)
    # C2 = evCsp_Tspline(
    #     phis_NEW_SWAVE[1], SR_S_new, SI_S_new, 1008, 1016, cut_off_spl, phi2KK_EvtGen)
    # C3 = evCsp_Tspline(
    #     phis_NEW_SWAVE[2], SR_S_new, SI_S_new, 1016, 1020, cut_off_spl, phi2KK_EvtGen)
    # C4 = evCsp_Tspline(
    #     phis_NEW_SWAVE[3], SR_S_new, SI_S_new, 1020, 1024, cut_off_spl, phi2KK_EvtGen)
    # C5 = evCsp_Tspline(
    #     phis_NEW_SWAVE[4], SR_S_new, SI_S_new, 1024, 1032, cut_off_spl, phi2KK_EvtGen)
    # C6 = evCsp_Tspline(
    #     phis_NEW_SWAVE[5], SR_S_new, SI_S_new, 1032, 1050, cut_off_spl, phi2KK_EvtGen)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # print("NR, phi NOT SMEARED NEW")
    # C1 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 990, 1008)
    # C2 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1008, 1016)
    # C3 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1016, 1020)
    # C4 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1020, 1024)
    # C5 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1024, 1032)
    # C6 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1032, 1050)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # sigma_m = 2.1
    # sigma_gKK = 0.13
    # sigma_gpipi = 8.
    #
    # f0_Syr1_m0_up_gKK_up_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_up_gpipi_up = f0_Syr1_m0_up_gKK_up_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up up up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_down_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_down_gpipi_up = f0_Syr1_m0_up_gKK_down_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up down up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_up_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_up_gpipi_down = f0_Syr1_m0_up_gKK_up_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up up down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_down_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_down_gpipi_down = f0_Syr1_m0_up_gKK_down_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up down down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_up_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_up_gpipi_up = f0_Syr1_m0_down_gKK_up_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down up up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_down_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_down_gpipi_up = f0_Syr1_m0_down_gKK_down_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down down up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_up_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_up_gpipi_down = f0_Syr1_m0_down_gKK_up_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down up down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_down_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_down_gpipi_down = f0_Syr1_m0_down_gKK_down_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down down down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr2 = Flatte_0.subs([(m0, 945.4), (gKK, 3.47*gpipi)])
    # f0_Syr2 = f0_Syr2.subs(gpipi, 167.)
    #
    # print("f0, phi SMEARED LATEST HISTS, 2nd solution")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr2)
    # C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr2)
    # C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr2)
    # C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr2)
    # C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr2)
    # C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr2)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # '''
    # FIN
    # '''

    # exit(0)
    #
    # '''                                                                                                                                                                                                                                      
    # Code below for systematics from previous round. Provided for reference.                                                                                                                                                
    # '''
    #
    # print("f0, phi NOT SMEARED")
    # C1 = Csp(f0_Syr, phi2KK_EvtGen, 990, 1008)
    # C2 = Csp(f0_Syr, phi2KK_EvtGen, 1008, 1016)
    # C3 = Csp(f0_Syr, phi2KK_EvtGen, 1016, 1020)
    # C4 = Csp(f0_Syr, phi2KK_EvtGen, 1020, 1024)
    # C5 = Csp(f0_Syr, phi2KK_EvtGen, 1024, 1032)
    # C6 = Csp(f0_Syr, phi2KK_EvtGen, 1032, 1050)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # print("NR, phi SMEARED NEW SPLINE, NEW HIST, CUT OFF at", cut_off)
    # C1 = evCsp_Tspline(phis_NEW[0], SR_S_new, SI_S_new,
    #                    990, 1008, cut_off, phi2KK_EvtGen)
    # C2 = evCsp_Tspline(phis_NEW[1], SR_S_new, SI_S_new,
    #                    1008, 1016, cut_off, phi2KK_EvtGen)
    # C3 = evCsp_Tspline(phis_NEW[2], SR_S_new, SI_S_new,
    #                    1016, 1020, cut_off, phi2KK_EvtGen)
    # C4 = evCsp_Tspline(phis_NEW[3], SR_S_new, SI_S_new,
    #                    1020, 1024, cut_off, phi2KK_EvtGen)
    # C5 = evCsp_Tspline(phis_NEW[4], SR_S_new, SI_S_new,
    #                    1024, 1032, cut_off, phi2KK_EvtGen)
    # C6 = evCsp_Tspline(g_NEW, SR_S_new, SI_S_new, 1032,
    #                    1050, cut_off, phi2KK_EvtGen)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # print("NR, phi NOT SMEARED NEW")
    # C1 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 990, 1008)
    # C2 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1008, 1016)
    # C3 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1016, 1020)
    # C4 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1020, 1024)
    # C5 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1024, 1032)
    # C6 = Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1032, 1050)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # sigma_m = 2.1
    # sigma_gKK = 0.13
    # sigma_gpipi = 8.
    #
    # f0_Syr1_m0_up_gKK_up_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_up_gpipi_up = f0_Syr1_m0_up_gKK_up_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up up up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_down_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_down_gpipi_up = f0_Syr1_m0_up_gKK_down_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up down up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_up_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_up_gpipi_down = f0_Syr1_m0_up_gKK_up_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up up down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_up_gKK_down_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9+sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_up_gKK_down_gpipi_down = f0_Syr1_m0_up_gKK_down_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, up down down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_up_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_up_gpipi_up = f0_Syr1_m0_down_gKK_up_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down up up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_down_gpipi_up = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_down_gpipi_up = f0_Syr1_m0_down_gKK_down_gpipi_up.subs(
    #     gpipi, 167.+sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down down up")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_up_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05+sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_up_gpipi_down = f0_Syr1_m0_down_gKK_up_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down up down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr1_m0_down_gKK_down_gpipi_down = Flatte_0.subs(
    #     [(m0, 949.9-sigma_m), (gKK, (3.05-sigma_gKK)*gpipi)])
    # f0_Syr1_m0_down_gKK_down_gpipi_down = f0_Syr1_m0_down_gKK_down_gpipi_down.subs(
    #     gpipi, 167.-sigma_gpipi)
    #
    # print("f0, phi SMEARED LATEST HISTS, down down down")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")
    #
    # f0_Syr2 = Flatte_0.subs([(m0, 945.4), (gKK, 3.47*gpipi)])
    # f0_Syr2 = f0_Syr2.subs(gpipi, 167.)
    #
    # print("f0, phi SMEARED LATEST HISTS, 2nd solution")
    # C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr2)
    # C2 = evCsp(phis_NEW[1], cut_off, f0_Syr2)
    # C3 = evCsp(phis_NEW[2], cut_off, f0_Syr2)
    # C4 = evCsp(phis_NEW[3], cut_off, f0_Syr2)
    # C5 = evCsp(phis_NEW[4], cut_off, f0_Syr2)
    # C6 = evCsp(g_NEW, cut_off, f0_Syr2)
    # print("CSP_factors = {1: "+str(round(C1[0], 4))+", 2: "+str(round(C2[0], 4))+", 3: "+str(round(
    #     C3[0], 4))+", 4: "+str(round(C4[0], 4))+", 5: "+str(round(C5[0], 4))+", 6: "+str(round(C6[0], 4))+"}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--histos', help='Path to the MC_BsJpsiPhi files')
    p.add_argument('--year', help='Year of the selection in yaml')
    p.add_argument('--output', help='Name of output json file')
    p.add_argument('--mode', help='Name of output json file')
    args = p.parse_args()
    histos = 'merda.root'
    mode = "Bs2JpsiPhi"
    year = 2016
    output = 'nana.json'
    csp = calculate_csp_factors_with_efficiency(histos, mKK_knots, as_params=True)
