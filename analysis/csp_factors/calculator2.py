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
import sympy as sp
from sympy import I

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir1', help='Path to the MC_BsJpsiPhi files')
    parser.add_argument('--input-dir2', help='Path to the MC_BsJpsiKK_Swave files')
    parser.add_argument('--year', help='Year of the selection in yaml')
    parser.add_argument('--output', help='Name of output json file')
    return parser

path_to_hist = "."

myJ = sp.Symbol("L", positive = True)
Nbins = 1000
lo = 2*493.677#EvtGen table
hi = 1060.

mass = sp.Symbol("mu", positive=True)
m0 = sp.Symbol("m0", positive=True)
Mmom = sp.Symbol("Mmom", positive=True)
Msister = sp.Symbol("Msister", positive=True)
gpipi = sp.Symbol("gpipi", positive=True)
gKK = sp.Symbol("gKK", positive=True)
MKp = 493.677
MK0 = 497.614#EvtGen tables
Mpip = 139.57018#EvtGen tables
Mpi0 = 134.9766#EvtGen tables

Mdau1 = sp.Symbol("mdau1", positive=True)
Mdau2 = sp.Symbol("mdau2", positive=True)

Gamma0 = sp.Symbol("Gamma0", positive=True)
muh = sp.Symbol("muh", positive=True)
mul = sp.Symbol("mul", positive=True)

def Blatt_Weisskopf( q,  q0, L = 1) :
    if (L<1.): return 1.
    d = 3.e-03/L
    z = q*d*q*d
    z0 = q0*d*q0*d
    if (L==1): return (1+z0)/(1+z)
    if (L==2): return ((z0-3)*(z0-3) + 9*z0) / ((z-3)*(z-3) + 9*z)
    if (L==3): return (z0*(z0-15)*(z0-15) + 9*(z0-5)) / (z*(z-15)*(z-15) + 9*(z-5))
    return ( sp.Pow(z0*z0 -45*z0+105,2) +25*z0*(2*z0-21)*(2*z0-21)) /(sp.Pow(z*z -45*z+105,2) +25*z*(2*z-21)*(2*z-21))
                
def get_q(  M,  m1,  m2 ) :
    M2 = M*M
    m12 = m1*m1
    m22 = m2*m2
    q2 = .25*( M2*M2 - 2*M2*(m12+m22) +(m12*m12+m22*m22)-2*m12*m22) /M2
    return sp.sqrt(q2)                                               

q = get_q(mass,Mdau1, Mdau2)
p = get_q(Mmom,Msister,mass)

def Blatt_Weisskopf_VC( q, L = 1, isB = 0):
    if (L<1.): 
        return 1.

    if (isB==0):
        d = 3.e-03/L
    else:
        d = 5.e-03
        
    z = q*d*q*d
    if (L==1): 
        return sp.sqrt(1/(1+z))
    if (L==2): 
        return sp.sqrt((1./ ((z-3)*(z-3) + 9*z)))
                           
    return sp.sqrt(1./(sp.Pow(z*z -45*z+105,2) +25*z*(2*z-21)*(2*z-21)))

def Breit_Wigner( M,  M0,  Gamma0,  m1,  m2,  J = 1):
    q = get_q(M,m1,m2)
    q0 = get_q(M0,m1,m2)
    Gamma = Gamma0*sp.Pow(q/q0,2*J+1)*M0/M*Blatt_Weisskopf(q,q0,J)
    return 1./(M0*M0-M*M-I*M0*Gamma)
    
def NR_spline():
    m_knots = np.array([.990, 1.000, 1.010, 1.020, 1.030, 1.040, 1.050, 1.060])
    SR_knots = np.array([1.66216, 1.03973, 0.177684, -0.376651, -0.184457, -1.19145, 0.899762, -2.59912])
    SI_knots = np.array([1.67003, 1.25299, 1.17785, 1.12897, 1.28353, 1.00101, 1.00347, 0.693512])
    N_knots = len(SR_knots)
    SR = TSpline3("SR", m_knots, SR_knots, N_knots)
    SI = TSpline3("SI", m_knots, SI_knots, N_knots)
    
    return SR,SI

SR_S_new, SI_S_new = NR_spline()

def get_rho_VC(mu, m0): 
    rho_sq = 1 - 4*m0*m0/(mu*mu)
    return sp.sqrt(rho_sq)

Flatte_0 = 1./(m0*m0 - mass*mass - I*m0*(gpipi*((2./3.)*get_rho_VC(mass,Mpip)+ (1./3.)*get_rho_VC(mass,Mpi0)) + gKK*((1./2.)*get_rho_VC(mass,MKp) + (1./2.)*get_rho_VC(mass,MK0))))                

f0_Syr = Flatte_0.subs([(m0, 949.9), (gKK,3.05*gpipi)])
f0_Syr = f0_Syr.subs(gpipi,167.)
                         
EvtGen = Breit_Wigner(mass,m0,Gamma0,Mdau1,Mdau2,1)

phi2KK_EvtGen = EvtGen.subs( [(m0,1019.4610), (Gamma0,4.266),(Mdau1,Mdau2)])## Step by step. First we tell her the two daughters are the same
phi2KK_EvtGen = phi2KK_EvtGen.subs(Mdau2,MKp)## Now substitute the daughter by a number   

Bs2f0Jpsi_BW  = Blatt_Weisskopf_VC(p, 1, 1)
phi2KK_BW     = Blatt_Weisskopf_VC(q, 1, 0)

Bs2JpsiKK_ps_S = sp.sqrt(p*q)*p*Bs2f0Jpsi_BW
Bs2JpsiKK_ps_P = sp.sqrt(p*q)*q*phi2KK_BW
Bs2JpsiKK_ps_S = Bs2JpsiKK_ps_S.subs([(Mmom,5366.77),(Msister,3096.916),(Mdau1,Mdau2)])
Bs2JpsiKK_ps_P = Bs2JpsiKK_ps_P.subs([(Mmom,5366.77),(Msister,3096.916),(Mdau1,Mdau2)])
Bs2JpsiKK_ps_S = sp.simplify(Bs2JpsiKK_ps_S.subs(Mdau2,MKp))
Bs2JpsiKK_ps_P = sp.simplify(Bs2JpsiKK_ps_P.subs(Mdau2,MKp))

def evCsp(phi, cut_off , sw = f0_Syr, pw = phi2KK_EvtGen,  PSs = Bs2JpsiKK_ps_S, PSp = Bs2JpsiKK_ps_P):
    pwconj = pw.conjugate()
    pwconj= pwconj.subs( mass, mass.conjugate()) 
    swconj= sw.conjugate().subs( mass, mass.conjugate())
    f1 = (pw*pwconj*PSp*PSp).subs([(mul,lo),(muh,cut_off)])
    f2 = (sw*swconj*PSs*PSs).subs([(mul,lo),(muh,cut_off)])
    f3 = (swconj*pw*PSs*PSp).subs([(mul,lo),(muh,cut_off)])
    c = 0
    d = 0
    csp = 0
    for i in range(Nbins+1):
        mvar = lo + (cut_off-lo)*i*1./Nbins
        eff = phi(mvar)
        dc = sp.re(f1.subs(mass,mvar).n())
        dd = sp.re(f2.subs(mass,mvar).n())
        dcsp = f3.subs(mass,mvar).n()
       
        c+= dc*eff
        d+= dd*eff
        csp += dcsp*eff

    c = c.n()
    d = d.n()
    csp = csp*1./sp.sqrt(d*c)
    csp = csp.n()
    x = sp.re(csp)
    y = sp.im(csp)

    CSP = sp.sqrt(x**2 + y**2)
    theta = -sp.atan2(y,x)

    return CSP, theta

class step:
    def __init__(self, m0,m1):
        self.m0 = m0
        self.m1 = m1
    def __call__(self,m):
        if m < self.m0 or m > self.m1: return 0
        else: return 1

def load_phis(mKK_bins, input_dir):
    phis = []
    for i in range(len(mKK_bins)-1):
        m0 = mKK_bins[i]
        m1 = mKK_bins[i+1]
        # phis.append(cPickle.load(file(input_dir+"eff_hist_"+str(m0)+"_"+str(m1))))
        phis.append(cPickle.load( open(input_dir+"eff_hist_"+str(m0)+"_"+str(m1), 'rb') ))
        
    return phis
        
def Csp(sw, pw, lo, hi, PSs = Bs2JpsiKK_ps_S, PSp = Bs2JpsiKK_ps_P):
    pwconj = pw.conjugate()
    pwconj= pwconj.subs( mass, mass.conjugate()) ## mass is real, let's make life easier
    c = Integral(pw*pwconj*PSp*PSp,(mass,mul,muh))
    c = c.subs([(mul,lo),(muh,hi)])
    c = c.n()
    d = Integral(sw*sw.conjugate()*PSs*PSs,(mass,mul,muh))
    d = d.subs([(mul,lo),(muh,hi)])
    d = d.n()
    cte = 1/Sqrt(d*c)
    csp = Integral(sw.conjugate()*cte*pw*PSs*PSp,(mass,mul,muh))
    csp = csp.subs([(mul,lo),(muh,hi)])
    csp = csp.n()
    x = re(csp)
    y = im(csp)

    CSP = sqrt(x**2 + y**2)
    #theta = -atan(y/x)
    theta = -atan2(y,x)
    if theta < 0 : theta = theta+2*pi

    return CSP, theta
        
def num_int_Tspline(SR_S, SI_S, pw, lo, hi, PSs = Bs2JpsiKK_ps_S, PSp = Bs2JpsiKK_ps_P):
    bins = 1000
    bin_width = (hi - lo)/float(bins)
    integral_ss = 0
    integral_ps = 0
    
    for i in range(bins+1):
        x = lo + bin_width*i
        PSs_tmp = PSs
        PSs_n = 1.
        PSp_tmp = PSp
        PSp_n = PSp_tmp.subs(mass,x)
        pw_tmp = pw
        pw_n = pw_tmp.subs(mass,x)
        xgev = x*1e-03
        y_ss = ((SR_S.Eval(xgev)*SR_S.Eval(xgev) + SI_S.Eval(xgev)*SI_S.Eval(xgev))*PSs_n*PSs_n)
        y_ps = ((SR_S.Eval(xgev) - I*SI_S.Eval(xgev))*pw_n*PSs_n*PSp_n).n()
        integral_ss += bin_width*y_ss
        integral_ps += bin_width*y_ps
        
    return integral_ss, integral_ps

def Csp_Tspline(SR_S, SI_S, pw, lo, hi, PSs = Bs2JpsiKK_ps_S, PSp = Bs2JpsiKK_ps_P):
    pwconj = pw.conjugate()
    pwconj= pwconj.subs(mass, mass.conjugate())
    c = Integral(pw*pwconj*PSp*PSp,(mass,mul,muh))
    c = c.subs([(mul,lo),(muh,hi)])
    c = c.n()
    d, csp = num_int_Tspline(SR_S, SI_S, pw, lo, hi)
    cte = 1/Sqrt(d*c)
    csp = csp*cte
    
    x = re(csp)
    y = im(csp)

    CSP = sqrt(x**2 + y**2)
    theta = -atan2(y,x)

    return CSP, theta

# from SomeUtils.numericFunctionClass import *
def do_shifted_phi(phi,delta):
    x0,y0 = [], []
    for i in phi.References:
        if i+delta > 1060:
            continue
        x0.append(i+delta)
        y0.append(phi(i) )
    return NF(x0, y0)

def evCsp_Tspline(phi, SR_S, SI_S, ll, ul, cut_off, pw = phi2KK_EvtGen,  PSs = Bs2JpsiKK_ps_S, PSp = Bs2JpsiKK_ps_P):
    pwconj = pw.conjugate()
    pwconj= pwconj.subs( mass, mass.conjugate()) ## mass is real, let's make life easier
    f1 = (pw*pwconj*PSp*PSp).subs([(mul,lo),(muh,cut_off)])
    c = 0
    d = 0
    csp = 0
    SR_check, SI_check = NR_spline()
    for i in range(Nbins+1):
        mvar = lo + (cut_off-lo)*i*1./Nbins
        eff = phi(mvar)
        pw_tmp = pw
        pw_n = pw_tmp.subs(mass,mvar)
        PSs_n = 1.#PSs_tmp.subs(mass,x)
        PSp_tmp = PSp
        PSp_n = PSp_tmp.subs(mass,mvar)
        dc = re(f1.subs(mass,mvar).n())
        xgev = mvar*1e-03
        
        dd = ((SR_S.Eval(xgev)*SR_S.Eval(xgev) + SI_S.Eval(xgev)*SI_S.Eval(xgev))*PSs_n*PSs_n) #re(f2.subs(mass,mvar).n())
        dcsp = ((SR_S.Eval(xgev) - I*SI_S.Eval(xgev))*pw_n*PSs_n*PSp_n).n()
        
        c+= dc*eff
        d+= dd*eff
        csp += dcsp*eff
        
    c = c.n()
    csp = csp*1./Sqrt(d*c)
    csp = csp.n()
    x = re(csp)
    y = im(csp)

    CSP = sqrt(x**2 + y**2)
    theta = -atan2(y,x)
    
    return CSP, theta

mKK_knots = [990, 1008, 1016, 1020, 1024, 1032, 1050]
mass_knots = [990, 1008, 1016, 1020, 1024, 1032, 1050]

def calculate_csp_factors_with_efficiency(histos, mKK_knots):
    input_dir1, input_dir2 = histos, histos
    mKK_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    
    phis_NEW = load_phis(mKK_bins, input_dir1)
    phis_NEW_SWAVE = load_phis(mKK_bins, input_dir2)
    
    phi0_NEW = do_shifted_phi(phis_NEW[4], 8)
    phi1_NEW = do_shifted_phi(phis_NEW[4], 16)
    phi2_NEW = do_shifted_phi(phis_NEW[4], 18)
    
    xlist = phi0_NEW.References + phi1_NEW.References + phi2_NEW.References
    xlist.sort()
    ylist = []
    for m in xlist:
        e1 = phi0_NEW(m)
        e2 = phi1_NEW(m)
        e3 = phi2_NEW(m)

        e0 = e1 + (1-e1)*e2
        ylist.append(max(e0,e3))
    g_NEW = NF(xlist,ylist)
    
    phi0_NEW_SWAVE = do_shifted_phi(phis_NEW_SWAVE[4], 8)
    phi1_NEW_SWAVE = do_shifted_phi(phis_NEW_SWAVE[4], 16)
    phi2_NEW_SWAVE = do_shifted_phi(phis_NEW_SWAVE[4], 18)
    
    xlist = phi0_NEW_SWAVE.References + phi1_NEW_SWAVE.References + phi2_NEW_SWAVE.References
    xlist.sort()
    ylist = []
    for m in xlist:
        e1 = phi0_NEW_SWAVE(m)
        e2 = phi1_NEW_SWAVE(m)
        e3 = phi2_NEW_SWAVE(m)

        e0 = e1 + (1-e1)*e2
        ylist.append(max(e0,e3))
    g_NEW_SWAVE = NF(xlist,ylist)
    
    cut_off = 1200.
    
    CSP = []
    print("f0, phi with eff.")
    CSP.append(evCsp(phis_NEW[0], cut_off, f0_Syr))
    CSP.append(evCsp(phis_NEW[1], cut_off, f0_Syr))
    CSP.append(evCsp(phis_NEW[2], cut_off, f0_Syr))
    CSP.append(evCsp(phis_NEW[3], cut_off, f0_Syr))
    CSP.append(evCsp(phis_NEW[4], cut_off, f0_Syr))
    CSP.append(evCsp(g_NEW, cut_off, f0_Syr))
    print("JpsiPhi CSP_factors = {1: "+str(round(CSP[0][0],4))+", 2: "+str(round(CSP[1][0],4))+", 3: "+str(round(CSP[2][0],4))+", 4: "+str(round(CSP[3][0],4))+", 5: "+str(round(CSP[4][0],4))+", 6: "+str(round(CSP[5][0],4))+"}")
    
    CSP_SWAVE = []
    print("f0, phi with eff., SWAVE ")
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr))
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr))
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr))
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr))
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr))
    CSP_SWAVE.append(evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr))
    #CSP_SWAVE.append(evCsp(g_NEW_SWAVE, cut_off, f0_Syr))
    print("JpsiKK CSP_factors = {1: "+str(round(CSP_SWAVE[0][0],4))+", 2: "+str(round(CSP_SWAVE[1][0],4))+", 3: "+str(round(CSP_SWAVE[2][0],4))+", 4: "+str(round(CSP_SWAVE[3][0],4))+", 5: "+str(round(CSP_SWAVE[4][0],4))+", 6: "+str(round(CSP_SWAVE[5][0],4))+"}")
    exit()
    #print(evCsp(g_NEW_SWAVE, cut_off, f0_Syr))

    fh = open(output, 'w' )
    fh.write('{\n')
    fh.write('\t"'+str(year)+'":{')
    fh.write('\n\t\t"CspFactors" : [' )
        
    first = True
    for i in range(6):
        if not first : fh.write( ',' )
        first = False
        fh.write( '\n\t\t\t{' )
        fh.write( '\n\t\t\t\t"Name": "Csp'+str(i)+'",')
        fh.write( '\n\t\t\t\t"Value": '+str(round(CSP_SWAVE[i][0],4))+',')
        fh.write( '\n\t\t\t\t"Error": 0.0,')
        fh.write( '\n\t\t\t\t"Bin_ll":'+str(mKK_bins[i])+',')
        fh.write( '\n\t\t\t\t"Bin_ul":'+str(mKK_bins[i+1]))
        fh.write( '\n\t\t\t}' )
    fh.write('\n\t\t]')
    fh.write('\n\t}')
    fh.write('\n}')
    fh.close()

    #exit(0)

    '''
    Code below for systematics from previous round. To be adjusted during systematics studies.
    '''

    print("=========================")
    print("Csp factors for systematics")
    print("=========================")

    print("f0, phi NOT SMEARED")
    C1 =  Csp(f0_Syr, phi2KK_EvtGen, 990, 1008)
    C2 =  Csp(f0_Syr, phi2KK_EvtGen, 1008, 1016)
    C3 =  Csp(f0_Syr, phi2KK_EvtGen, 1016, 1020)
    C4 =  Csp(f0_Syr, phi2KK_EvtGen, 1020, 1024)
    C5 =  Csp(f0_Syr, phi2KK_EvtGen, 1024, 1032)
    C6 =  Csp(f0_Syr, phi2KK_EvtGen, 1032, 1050)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    cut_off_spl = 1060.
    print("NR, phi SMEARED NEW SPLINE, NEW HIST, CUT OFF at", cut_off_spl)
    C1 = evCsp_Tspline(phis_NEW_SWAVE[0],SR_S_new, SI_S_new, 990, 1008, cut_off_spl, phi2KK_EvtGen)
    C2 = evCsp_Tspline(phis_NEW_SWAVE[1],SR_S_new, SI_S_new, 1008, 1016, cut_off_spl, phi2KK_EvtGen)
    C3 = evCsp_Tspline(phis_NEW_SWAVE[2],SR_S_new, SI_S_new, 1016, 1020, cut_off_spl, phi2KK_EvtGen)
    C4 = evCsp_Tspline(phis_NEW_SWAVE[3],SR_S_new, SI_S_new, 1020, 1024, cut_off_spl, phi2KK_EvtGen)
    C5 = evCsp_Tspline(phis_NEW_SWAVE[4],SR_S_new, SI_S_new, 1024, 1032, cut_off_spl, phi2KK_EvtGen)
    C6 = evCsp_Tspline(phis_NEW_SWAVE[5],SR_S_new, SI_S_new, 1032, 1050, cut_off_spl, phi2KK_EvtGen)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    print("NR, phi NOT SMEARED NEW")
    C1 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 990, 1008)
    C2 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1008, 1016)
    C3 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1016, 1020)
    C4 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1020, 1024)
    C5 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1024, 1032)
    C6 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1032, 1050)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    sigma_m = 2.1
    sigma_gKK = 0.13
    sigma_gpipi = 8.

    f0_Syr1_m0_up_gKK_up_gpipi_up = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_up_gpipi_up = f0_Syr1_m0_up_gKK_up_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up up up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_down_gpipi_up = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_down_gpipi_up = f0_Syr1_m0_up_gKK_down_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up down up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_up_gpipi_down = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_up_gpipi_down = f0_Syr1_m0_up_gKK_up_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up up down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_down_gpipi_down = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_down_gpipi_down = f0_Syr1_m0_up_gKK_down_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up down down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_up_gpipi_up = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_up_gpipi_up = f0_Syr1_m0_down_gKK_up_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down up up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_down_gpipi_up = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_down_gpipi_up = f0_Syr1_m0_down_gKK_down_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down down up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_up_gpipi_down = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_up_gpipi_down = f0_Syr1_m0_down_gKK_up_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down up down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_down_gpipi_down = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_down_gpipi_down = f0_Syr1_m0_down_gKK_down_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down down down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr2 = Flatte_0.subs([(m0, 945.4), (gKK,3.47*gpipi)])
    f0_Syr2 = f0_Syr2.subs(gpipi,167.)

    print("f0, phi SMEARED LATEST HISTS, 2nd solution")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr2)
    C2 = evCsp(phis_NEW_SWAVE[1], cut_off, f0_Syr2)
    C3 = evCsp(phis_NEW_SWAVE[2], cut_off, f0_Syr2)
    C4 = evCsp(phis_NEW_SWAVE[3], cut_off, f0_Syr2)
    C5 = evCsp(phis_NEW_SWAVE[4], cut_off, f0_Syr2)
    C6 = evCsp(phis_NEW_SWAVE[5], cut_off, f0_Syr2)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    '''
    FIN
    '''

    exit(0)

    '''                                                                                                                                                                                                                                      
    Code below for systematics from previous round. Provided for reference.                                                                                                                                                
    '''

    print("f0, phi NOT SMEARED")
    C1 =  Csp(f0_Syr, phi2KK_EvtGen, 990, 1008)
    C2 =  Csp(f0_Syr, phi2KK_EvtGen, 1008, 1016)
    C3 =  Csp(f0_Syr, phi2KK_EvtGen, 1016, 1020)
    C4 =  Csp(f0_Syr, phi2KK_EvtGen, 1020, 1024)
    C5 =  Csp(f0_Syr, phi2KK_EvtGen, 1024, 1032)
    C6 =  Csp(f0_Syr, phi2KK_EvtGen, 1032, 1050)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    print("NR, phi SMEARED NEW SPLINE, NEW HIST, CUT OFF at", cut_off)
    C1 = evCsp_Tspline(phis_NEW[0],SR_S_new, SI_S_new, 990, 1008, cut_off, phi2KK_EvtGen)
    C2 = evCsp_Tspline(phis_NEW[1],SR_S_new, SI_S_new, 1008, 1016, cut_off, phi2KK_EvtGen)
    C3 = evCsp_Tspline(phis_NEW[2],SR_S_new, SI_S_new, 1016, 1020, cut_off, phi2KK_EvtGen)
    C4 = evCsp_Tspline(phis_NEW[3],SR_S_new, SI_S_new, 1020, 1024, cut_off, phi2KK_EvtGen)
    C5 = evCsp_Tspline(phis_NEW[4],SR_S_new, SI_S_new, 1024, 1032, cut_off, phi2KK_EvtGen)
    C6 = evCsp_Tspline(g_NEW,SR_S_new, SI_S_new, 1032, 1050, cut_off, phi2KK_EvtGen)     
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    print("NR, phi NOT SMEARED NEW")
    C1 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 990, 1008)
    C2 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1008, 1016)
    C3 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1016, 1020)
    C4 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1020, 1024)
    C5 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1024, 1032)
    C6 =  Csp_Tspline(SR_S_new, SI_S_new, phi2KK_EvtGen, 1032, 1050)     
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    sigma_m = 2.1
    sigma_gKK = 0.13
    sigma_gpipi = 8.

    f0_Syr1_m0_up_gKK_up_gpipi_up = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_up_gpipi_up = f0_Syr1_m0_up_gKK_up_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up up up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_up_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_down_gpipi_up = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_down_gpipi_up = f0_Syr1_m0_up_gKK_down_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up down up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_down_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_up_gpipi_down = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_up_gpipi_down = f0_Syr1_m0_up_gKK_up_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up up down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_up_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_up_gKK_down_gpipi_down = Flatte_0.subs([(m0, 949.9+sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_up_gKK_down_gpipi_down = f0_Syr1_m0_up_gKK_down_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, up down down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_up_gKK_down_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_up_gpipi_up = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_up_gpipi_up = f0_Syr1_m0_down_gKK_up_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down up up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_up_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_down_gpipi_up = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_down_gpipi_up = f0_Syr1_m0_down_gKK_down_gpipi_up.subs(gpipi,167.+sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down down up")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_down_gpipi_up)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_up_gpipi_down = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05+sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_up_gpipi_down = f0_Syr1_m0_down_gKK_up_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down up down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_up_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr1_m0_down_gKK_down_gpipi_down = Flatte_0.subs([(m0, 949.9-sigma_m), (gKK,(3.05-sigma_gKK)*gpipi)])
    f0_Syr1_m0_down_gKK_down_gpipi_down = f0_Syr1_m0_down_gKK_down_gpipi_down.subs(gpipi,167.-sigma_gpipi)

    print("f0, phi SMEARED LATEST HISTS, down down down")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    C6 = evCsp(g_NEW, cut_off, f0_Syr1_m0_down_gKK_down_gpipi_down)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

    f0_Syr2 = Flatte_0.subs([(m0, 945.4), (gKK,3.47*gpipi)])
    f0_Syr2 = f0_Syr2.subs(gpipi,167.)

    print("f0, phi SMEARED LATEST HISTS, 2nd solution")
    C1 = evCsp(phis_NEW_SWAVE[0], cut_off, f0_Syr2)
    C2 = evCsp(phis_NEW[1], cut_off, f0_Syr2)
    C3 = evCsp(phis_NEW[2], cut_off, f0_Syr2)
    C4 = evCsp(phis_NEW[3], cut_off, f0_Syr2)
    C5 = evCsp(phis_NEW[4], cut_off, f0_Syr2)
    C6 = evCsp(g_NEW, cut_off, f0_Syr2)
    print("CSP_factors = {1: "+str(round(C1[0],4))+", 2: "+str(round(C2[0],4))+", 3: "+str(round(C3[0],4))+", 4: "+str(round(C4[0],4))+", 5: "+str(round(C5[0],4))+", 6: "+str(round(C6[0],4))+"}")

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    histos = 'merda.root'
    mode = "Bs2JpsiPhi"
    year = 2016
    output = 'nana.json'
    calculate_csp_factors_with_efficiency(histos, mKK_knots)

