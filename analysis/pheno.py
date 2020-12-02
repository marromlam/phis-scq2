#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% Modules -------------------------------------------------------------------
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy.physics.quantum.spin import Rotation
from sympy.abc import i, k, l, m
import math
import uproot
import os
import platform

# %% Some functions ------------------------------------------------------------
MAX_SPIN = 1
POSIBLE_STATES = sp.Sum(2*m+1,(m,0,MAX_SPIN)).doit()
POSIBLE_STATES
POSIBLE_COMBS = (POSIBLE_STATES*(POSIBLE_STATES+1))/2
POSIBLE_COMBS

x = sp.Symbol('x'); thetaL = sp.Symbol(r"\theta_{\mu}",real=True)
y = sp.Symbol('y'); thetaK = sp.Symbol(r"\theta_K",real=True)
z = sp.Symbol('z'); phi    = sp.Symbol("\phi",real=True)
t = sp.Symbol('t')

amps = {}
deltas = {}
phiss = {}
lambdas = {}

states = []
for spin in range(0,MAX_SPIN+1):
    for polarization in range(-min(1,spin),min(1,spin)+1,1):
        states.append([spin,polarization])
states

combinations = []
for state in range(0,len(states)):
    for spin in range(0,state+1):
        combinations.append([states[state],states[spin]])


sort_mat = sp.zeros(POSIBLE_COMBS,POSIBLE_COMBS)            # sort as in ananote
sort_mat[0,5]= 1
sort_mat[1,9]= 1
sort_mat[2,2]= 1
sort_mat[3,7]= 1
sort_mat[4,8]= 1
sort_mat[5,4]= 1
sort_mat[6,0]= 1
sort_mat[7,6]= 1
sort_mat[8,1]= 1
sort_mat[9,3]= 1



def py_name(k):
    s = k[0]; p = k[1]
    wave = ["S", "P", "D", "F"]
    polarization = ["per", "lon", "par"]
    return wave[s]+polarization[p+1]


def latex_name(k):
    s = k[0]; p = k[1]
    wave = ["S", "P", "D", "F"]
    polarization = ["\perp", "0", "\parallel"]
    if s == 0 : return "{S}"
    if s == 1 : return "{"+polarization[p+1]+"}"
    else: return "{"+str(s)+polarization[p+1]+"}"


def Cww(s2,s1):
    if s1 == s2: return 1
    elif s1 == 0 and s2 == 1: return pars["CSP"]
    elif s1 == 0 and s2 == 2: return pars["CSD"]
    elif s1 == 1 and s2 == 2: return pars["CPD"]

def eta(k):
    r = k[0]; c = k[1]+1
    A = sp.Matrix([[0,-1,0],[-1,1,1]])
    return A[r,c]

def A(k,amps=amps,deltas=deltas,phiss=phiss,lambdas=lambdas):
    return sp.exp(sp.I*deltas[py_name(k)])



def B(k,amps=amps,deltas=deltas,phiss=phiss,lambdas=lambdas):
    return eta(k)*lambdas[py_name(k)]*A(k)*sp.exp(-sp.I*(phiss[py_name(k)] ))

def check_odd_even(eta1,eta2,J):
    return eta1*eta2*(-1)**J

def re_im(expression):
    return [sp.re(expression),sp.im(expression)]

def getA(k,amps=amps):
    spin = k[0]; polarization = k[1]
    return amps[py_name(k)]

def get_CP(cpst):
    if cpst == 1: return 0
    else: return 1

#%% Create dicts with all variables --------------------------------------------
for state in states:
    amps[py_name(state)] = sp.Symbol("A_"+latex_name(state),real=True)
    deltas[py_name(state)] = sp.Symbol("\delta_"+latex_name(state),real=True)
    phiss[py_name(state)] = sp.Symbol("\phi_"+latex_name(state),real=True)
    lambdas[py_name(state)] = sp.Symbol("\lambda_"+latex_name(state),real=True)

pars={
    "G":sp.Symbol("\Gamma",real=True),
    "DG":sp.Symbol("\Delta \Gamma",real=True),
    "DM":sp.Symbol("\Delta m",real=True),
    "CSP":sp.Symbol("C_{SP}",real=True),
    "CSD":sp.Symbol("C_{SP}",real=True),
    "CPD":sp.Symbol("C_{SP}",real=True),
}



#%% Normalization --------------------------------------------------------------
def getN():
    Amplitudes = sp.Matrix(POSIBLE_COMBS, 1, lambda i,j: 0)
    for c in range(0,len(combinations)):
        Amplitudes[c] = (getA(combinations[c][0])*
                         getA(combinations[c][1])*
                         Cww(combinations[c][0][0],combinations[c][1][0]))
    return Amplitudes

getN()

#%% Angular Part ---------------------------------------------------------------
def getOmega():
    return ((9/(16*sp.pi))*(4*sp.pi/3))*sp.Matrix([
    sp.sin(thetaL)**2/(4*sp.pi),
    -sp.sqrt(6)*sp.sin(phi)*sp.sin(thetaK)*sp.sin(thetaL)*sp.cos(thetaL)/(4*sp.pi),
    3*(-sp.sin(phi)**2*sp.sin(thetaL)**2 + 1)*sp.sin(thetaK)**2/(8*sp.pi),
    sp.sqrt(3)*sp.sin(thetaL)**2*sp.cos(thetaK)/(2*sp.pi),
    -3*sp.sqrt(2)*sp.sin(phi)*sp.sin(thetaK)*sp.sin(thetaL)*sp.cos(thetaK)*sp.cos(thetaL)/(4*sp.pi),
    3*sp.sin(thetaL)**2*sp.cos(thetaK)**2/(4*sp.pi),
    sp.sqrt(6)*sp.sin(thetaK)*sp.sin(thetaL)*sp.cos(phi)*sp.cos(thetaL)/(4*sp.pi),
    3*sp.sin(phi)*sp.sin(thetaK)**2*sp.sin(thetaL)**2*sp.cos(phi)/(4*sp.pi),
    3*sp.sqrt(2)*sp.sin(thetaK)*sp.sin(thetaL)*sp.cos(phi)*sp.cos(thetaK)*sp.cos(thetaL)/(4*sp.pi),
    3*(sp.sin(phi)**2*sp.sin(thetaL)**2 - sp.sin(thetaL)**2 + 1)*sp.sin(thetaK)**2/(8*sp.pi)
    ])

getOmega()

def WignerD(l,m,s,x):
    return Rotation.d(l,m,s,x)



def helicityAmplitude(j,l,thetaL,thetaK,abs_alpha=1):
    H = sp.Matrix([0,0])
    H[0] = (
      sp.sqrt((2*j + 1)/(4*sp.pi))*sp.exp(-sp.I*l*phi)*
      WignerD(j, -l,               0, thetaK)*
      WignerD(1, +l, +abs(abs_alpha), thetaL)
    )
    H[1] = (
       sp.sqrt((2*j + 1)/(4*sp.pi))*sp.exp(-sp.I*l*phi)*
        WignerD(j, -l,               0, thetaK)*
        WignerD(1, +l, -abs(abs_alpha), thetaL)
    )
    return H



def transAmp(j,l,thetaL=thetaL,thetaK=thetaK):
  if (l == 0) | (j==0):
    A = helicityAmplitude(j,0,thetaL,thetaK).T
  else:
    A = (  helicityAmplitude(j,-1,thetaL,thetaK).T+
         l*helicityAmplitude(j,1,thetaL,thetaK).T)/(sp.sqrt(2))
  return A



def mul(x,y):
  xx = re_im(x[0]*y[0])
  yy = re_im(x[1]*y[1])
  return (xx[0] + yy[0]) + (xx[1] + yy[1])#*sp.I



def getAngularPart():
  fk = sp.Matrix(POSIBLE_COMBS, 1, lambda i,j: 0)
  for k in range(0,len(combinations)):
    combnow=combinations[k]
    s1 = combnow[0]; s2 = combnow[1]; lnow = [s1[0],s2[0]]
    if k == 1:
      s2 = combnow[0]; s1 = combnow[1]; # WARNING!
    cpst = check_odd_even(eta(s1),eta(s2),sum(lnow))
    if s1 == s2:
      const = 1
    else:
      const = 2
    fk[k] = (
              sp.factor( sp.expand_trig( ((9/(16*sp.pi))*(4*sp.pi/3))*mul(
                transAmp(*s1).doit() ,transAmp(*s2).conjugate().doit()
              ).rewrite(sp.cos) ) ) * const
            ).simplify()
  return fk



for item in getAngularPart()-getOmega():
  print(item.simplify())
getAngularPart()


#%% Coeffs a, b, c and d -------------------------------------------------------
def getABCD():
    abcd = sp.Matrix(POSIBLE_COMBS, 4, lambda i,j: 0)
    for k in range(0,len(combinations)):
        combnow=combinations[k]
        s1 = combnow[0]; s2 = combnow[1]; lnow = [s1[0],s2[0]]
        cpst = check_odd_even(eta(s1),eta(s2),sum(lnow))
        item0 = cpst*( A(s1)*(A(s2).conjugate()) + B(s1)*(B(s2).conjugate()) ).expand().simplify()
        item1 = cpst*( A(s1)*(B(s2).conjugate()) + B(s1)*(A(s2).conjugate()) ).simplify()
        item2 = cpst*( A(s1)*(A(s2).conjugate()) - B(s1)*(B(s2).conjugate()) ).simplify()
        item3 = -sp.I*cpst*( A(s1)*(B(s2).conjugate()) - B(s1)*(A(s2).conjugate()) ).simplify()
        abcd[k,0] = +(re_im(item0)[get_CP(cpst)].rewrite(sp.exp, sp.cos)).trigsimp().simplify()
        abcd[k,1] = -(re_im(item1)[get_CP(cpst)].rewrite(sp.exp, sp.cos)).trigsimp().simplify()
        abcd[k,2] = +(re_im(item2)[get_CP(cpst)].rewrite(sp.exp, sp.cos)).trigsimp().simplify()
        abcd[k,3] = +(re_im(item3)[get_CP(cpst)].rewrite(sp.exp, sp.cos)).trigsimp().simplify()
    return abcd/2

getABCD()

sort_mat*getOmega()
sort_mat*sp.Matrix([1,0,1,0,0,1,0,0,0,1])
#%% PDF expressions ------------------------------------------------------------
def getBpdf():
    N = getN(); ABCD = getABCD()
    A = ABCD[:,0]; B = ABCD[:,1]; C = ABCD[:,2]; D = ABCD[:,3]
    F = getOmega()
    F = F.subs(sp.sin(thetaK),sp.sqrt(1-sp.cos(thetaK)*sp.cos(thetaK)))
    F = F.subs(sp.sin(thetaL),sp.sqrt(1-sp.cos(thetaL)*sp.cos(thetaL)))
    F = F.subs(sp.cos(thetaK),x).subs(sp.cos(thetaL),y).subs(phi,z)
    Z = [1,0,1,0,0,1,0,0,0,1]
    PDF1w = 0
    PDF0w = 0
    for k in range(0,POSIBLE_COMBS):
        PDF1w += F[k]*N[k]*sp.exp(-pars["G"]*t)*(
                                                  A[k]*sp.cosh(pars["DG"]*t/2)+
                                                  B[k]*sp.sinh(pars["DG"]*t/2)+
                                                  C[k]*sp.cos(pars["DM"]*t)+
                                                  D[k]*sp.sin(pars["DM"]*t)
                                                )
        PDF0w += Z[k]*N[k]*sp.exp(-pars["G"]*t)*(
                                                  A[k]*sp.cosh(pars["DG"]*t/2)+
                                                  B[k]*sp.sinh(pars["DG"]*t/2)+
                                                  C[k]*sp.cos(pars["DM"]*t)+
                                                  D[k]*sp.sin(pars["DM"]*t)
                                                )
    return PDF1w, F, PDF0w

def getBbarpdf():
    N = getN(); ABCD = getABCD()
    A = ABCD[:,0]; B = ABCD[:,1]; C = ABCD[:,2]; D = ABCD[:,3]
    F = getOmega()
    F = F.subs(sp.sin(thetaK),sp.sqrt(1-sp.cos(thetaK)*sp.cos(thetaK)))
    F = F.subs(sp.sin(thetaL),sp.sqrt(1-sp.cos(thetaL)*sp.cos(thetaL)))
    F = F.subs(sp.cos(thetaK),x).subs(sp.cos(thetaL),y).subs(phi,z)
    Z = [1,0,1,0,0,1,0,0,0,1]
    PDF1w = 0
    PDF0w = 0
    for k in range(0,POSIBLE_COMBS):
        PDF1w += F[k]*N[k]*sp.exp(-pars["G"]*t)*(
                                                  A[k]*sp.cosh(pars["DG"]*t/2)+
                                                  B[k]*sp.sinh(pars["DG"]*t/2)-
                                                  C[k]*sp.cos(pars["DM"]*t)-
                                                  D[k]*sp.sin(pars["DM"]*t)
                                                )
        PDF0w += Z[k]*N[k]*sp.exp(-pars["G"]*t)*(
                                                  A[k]*sp.cosh(pars["DG"]*t/2)+
                                                  B[k]*sp.sinh(pars["DG"]*t/2)-
                                                  C[k]*sp.cos(pars["DM"]*t)-
                                                  D[k]*sp.sin(pars["DM"]*t)
                                                )
    return PDF1w, F, PDF0w

numB, fB, denB = getBpdf()
tLL = sp.Symbol('t_{LL}')
tUL = sp.Symbol('t_{UL}')
sp.integrate(numB,(t,tLL,tUL))
numB
subs_dict

 denB.subs(subs_dict).simplify()


*#%% Prepare substitution dict --------------------------------------------------
pars_values     = {"DG": 0*0.08543, "DM": 0.5, "G": 0.66137,
                   "CSP": 1, "CSD": 1, "CPD": 1}
amps_values     = {"Plon": np.sqrt(0.520935),
                   "Slon": 0.0,
                   "Ppar": np.sqrt(1-0.248826-0.520935),
                   "Pper": np.sqrt(0.248826)}
phiss_values    = {"Slon": 0.0,
                   "Plon": 0.0,
                   "Ppar": 0.0,
                   "Pper": 0.0}
deltas_values   = {"Plon": 0.0,
                   "Slon": 3.07,
                   "Ppar": 3.30,
                   "Pper": 3.07}
lambdas_values  = {"Plon": 1,
                   "Slon": 1,
                   "Pper": 1,
                   "Ppar": 1}



pars_values     = {"DG": 0,
                   "CSP": 1, "CSD": 1, "CPD": 1}
phiss_values    = {"Slon": 0.0,
                   "Plon": 0.0,
                   "Ppar": 0.0,
                   "Pper": 0.0}
lambdas_values  = {"Plon": 1,
                   "Slon": 1,
                   "Pper": 1,
                   "Ppar": 1}

subs_dict = {}                                         # full dict of parameters

for key in amps:
    subs_dict.update({phiss[key]:phiss_values[key]})
    #subs_dict.update({lambdas[key]:lambdas_values[key]})
for key in pars:
  try:
    subs_dict.update({pars[key]:pars_values[key]})
  except:
    0



getOmega()


_pdfwB
_wB
_pdfB





"""


hasta aqui puede ser util


"""




#%% Define pdf lambdas ---------------------------------------------------------
_pdfwB, _wB, _pdfB = getBpdf()
_pdfwB      = _pdfwB.subs(subs_dict)
_wB         = sort_mat*_wB.subs(subs_dict)
_pdfB       = _pdfB.subs(subs_dict)
_pdfwBbar, _wBbar, _pdfBbar = getBbarpdf()
_pdfwBbar   = _pdfwBbar.subs(subs_dict)
_wBbar      = sort_mat*_wBbar.subs(subs_dict)
_pdfBbar    = _pdfBbar.subs(subs_dict)

pdfwB       = sp.lambdify( (x,y,z,t), _pdfwB, ("numpy"))
wB          = sp.lambdify( (x,y,z),   _wB, ("numpy"))
pdfB        = sp.lambdify( (x,y,z,t), _pdfB, ("numpy"))
pdfwBbar    = sp.lambdify( (x,y,z,t), _pdfwBbar, ("numpy"))
wBbar       = sp.lambdify( (x,y,z),   _wBbar, ("numpy"))
pdfBbar     = sp.lambdify( (x,y,z,t), _pdfBbar, ("numpy"))
 0.088488/0.022122
0.068405/4
pars['DM']

_pdfwB.subs(pars['DM'],0)



shit = [_pdfwB[i].subs(subs_dict) for i in range(0,10)]
shit = sort_mat*sp.Matrix([shit]).T
shit.subs(pars['DM'],0)

shit.subs(pars['DM'],0)



sort_mat*getOmega() / (   9/(16*sp.pi)  )



_pdfwBbar

shit = [_pdfwBbar[i].subs(subs_dict) for i in range(0,10)]
shit = sort_mat*sp.Matrix([shit]).T
shit.subs(pars['DM'],0)



double ak;
case 1:  ak = 1;
case 2:  ak = 1;
case 3:  ak = 1;
case 4:  ak = 0;
case 5:  ak = cos(delta_pa);
case 6:  ak = 0;
case 7:  ak = 1;
case 8:  ak = 0;
case 9:  ak = sin(delta_pe-delta_S);
case 10: ak = 0;

double ck;
case 1:  ck = 0;
case 2:  ck = 0;
case 3:  ck = 0;
case 4:  ck = sin(delta_pe-delta_pa);
case 5:  ck = 0;
case 6:  ck = sin(delta_pe);
case 7:  ck = 0;
case 8:  ck = cos(delta_S-delta_pa);
case 9:  ck = 0;
case 10: ck = cos(delta_S));


AkA[0] = A_0_abs*A_0_abs;
AkA[1] = A_pa_abs*A_pa_abs;
AkA[2] = A_pe_abs*A_pe_abs;
AkA[3] = A_pe_abs*A_pa_abs*sin(delta_pe-delta_pa);
AkA[4] = A_0_abs*A_pa_abs*cos(delta_pa);
AkA[5] = A_0_abs*A_pe_abs*sin(delta_pe);
AkA[6] = A_S_abs*A_S_abs;
AkA[7] = CSP*A_S_abs*A_pa_abs*cos(delta_pa-delta_S);
AkA[8] = CSP*A_S_abs*A_pe_abs*sin(delta_pe-delta_S);
AkA[9] = CSP*A_S_abs*A_0_abs*cos(delta_S);

fkA[0] = c*helcosthetaK*helcosthetaK*helsinthetaL*helsinthetaL;
fkA[1] = c*0.5*helsinthetaK*helsinthetaK*(1.-helcosphi*helcosphi*helsinthetaL*helsinthetaL);
fkA[2] = c*0.5*helsinthetaK*helsinthetaK*(1.-helsinphi*helsinphi*helsinthetaL*helsinthetaL);
fkA[3] = c*helsinthetaK*helsinthetaK*helsinthetaL*helsinthetaL*helsinphi*helcosphi;
fkA[4] = c*sqrt(2.)*helsinthetaK*helcosthetaK*helsinthetaL*helcosthetaL*helcosphi;
fkA[5] =-c*sqrt(2.)*helsinthetaK*helcosthetaK*helsinthetaL*helcosthetaL*helsinphi;
fkA[6] = c*helsinthetaL*helsinthetaL/3.;
fkA[7] = c*2.*helsinthetaK*helsinthetaL*helcosthetaL*helcosphi/sqrt(6.);
fkA[8] =-c*2.*helsinthetaK*helsinthetaL*helcosthetaL*helsinphi/sqrt(6.);
fkA[9] = c*2.*helcosthetaK*helsinthetaL*helsinthetaL/sqrt(3.);

case 1 =   helcosthetaK*helcosthetaK*helsinthetaL*helsinthetaL;
case 2 =   0.5*helsinthetaK*helsinthetaK*(1.-helcosphi*helcosphi*helsinthetaL*helsinthetaL);
case 3 =   0.5*helsinthetaK*helsinthetaK*(1.-helsinphi*helsinphi*helsinthetaL*helsinthetaL);
case 4 =   helsinthetaK*helsinthetaK*helsinthetaL*helsinthetaL*helsinphi*helcosphi;
case 5 =   sqrt(2.)*helsinthetaK*helcosthetaK*helsinthetaL*helcosthetaL*helcosphi;
case 6 =  -sqrt(2.)*helsinthetaK*helcosthetaK*helsinthetaL*helcosthetaL*helsinphi;
case 7 =   helsinthetaL*helsinthetaL/3.;
case 8 =   2.*helsinthetaK*helsinthetaL*helcosthetaL*helcosphi/sqrt(6.);
case 9 =  -2.*helsinthetaK*helsinthetaL*helcosthetaL*helsinphi/sqrt(6.);
case 10=   2.*helcosthetaK*helsinthetaL*helsinthetaL/sqrt(3.);





abcd_



sort_mat*abcd_

ABCD
ABCD = getABCD()
N = getN();
N
sort_mat*(ABCD.subs(subs_dict))
abcd_ = ABCD.subs(subs_dict)
n_ = getN()


shit = sp.Matrix([[n_,abcd_]])
shit
for i in range(0,len(n_)):
  print(n_[i], abcd_[i])





#%% Compute weights ------------------------------------------------------------

pdf_omega_B         = pdfwB(xtrue,ytrue,ztrue,ttrue)
omega_B             = wB(xtrue,ytrue,ztrue)[:,0]
pdf_no_omega_B      = pdfB(xtrue,ytrue,ztrue,ttrue)
pdf_omega_Bbar      = pdfwBbar(xtrue,ytrue,ztrue,ttrue)
omega_Bbar          = wBbar(xtrue,ytrue,ztrue)[:,0]
pdf_no_omega_Bbar   = pdfBbar(xtrue,ytrue,ztrue,ttrue)

pdf_omega_B
pdf_omega_B[0]/np.array([ 0.12013033,  0.02594035,  0.03719554,  0.04008305,  0.02504965])
print('time     pdfB    pdfBbar')
for k in range(0,5):
  print("%.8f\t %.8f\t %.8f\t " %(ttrue[k],pdf_omega_B[k],pdf_omega_Bbar[k]) )
  [0.42005, 0.0358913, 0.0443011],
  [2.56827, 0.0076924, 0.0112894],
  [2.16904, 0.0110455, 0.0091436],
  [1.42379, 0.0119338, 0.0048330],
  [2.38205, 0.0074331, 0.0084366]

scq = np.array([ttrue[:5],pdf_omega_B[:5],pdf_omega_Bbar[:5]]).T

hei = np.array([
[0.42005, 0.0358913, 0.0443011],
[2.56827, 0.0076924, 0.0112894],
[2.16904, 0.0110455, 0.0091436],
[1.42379, 0.0119338, 0.0048330],
[2.38205, 0.0074331, 0.0084366]])

sort_mat*omega_B[:,0].T


omega_B[:,:5].T


omega_B[:5,0:10].size
for item in omega_B[:,:5].T:
  meh = ''
  for subitem in item:
    meh += "%+.4f" % subitem + "\t"
  print(meh)

xtrue[4],ytrue[4],ztrue[4]
hei-scq


print(subs_dict)



np.array([xtrue[:5],ytrue[:5],ztrue[:5],ttrue[:5]]).T

"""
  pdf 0.0806929
  pdf 0.0198286
  pdf 0.0202845
  pdf 0.0172775
  pdf 0.0161505

  ----------pdfB------
  pdf 0.03601
  pdf 0.00825965
  pdf 0.011074
  pdf 0.00557102
  pdf 0.00763715

  ----------pdfBbar------
  pdf 0.0438152
  pdf 0.0113512
  pdf 0.00929542
  pdf 0.01172
  pdf 0.00862243
"""
















pdf_omega/2

pdf_omega = np.array(pdf_omega).astype(np.float64)[:5]
ang_accep = vcalcTimeAcceptance(ttrue[:5])
ang_accep*pdf_omega


omega = np.array(omega).astype(np.float64)
pdf_no_omega = np.array(pdf_no_omega).astype(np.float64)



pdf_omega

wi = omega/pdf_omega
wi

pdf_omega[:4]
pdf_no_omega[:4]
wi[:,:4]

time[0]

subs_dict

########
(sp.exp(-pars["G"]*t)*sp.cosh(pars["DG"]*t/2))

(sp.exp(-pars["G"]*t)*sp.cosh(pars["DG"]*t/2)).subs(subs_dict).subs(t,time[0])
(sp.exp(-pars["G"]*t)*sp.sinh(pars["DG"]*t/2)).subs(subs_dict).subs(t,time[0])
(sp.exp(-pars["G"]*t)*sp.cos(pars["DM"]*t) ).subs(subs_dict).subs(t,time[0])
(sp.exp(-pars["G"]*t)*sp.sin(pars["DM"]*t) ).subs(subs_dict).subs(t,time[0])





#########

# Testing functions

def conv_exp_VC(t, gamma, omega, sigma):
    from scipy.special import wofz
    if(t>5*sigma):
        return 2.*(np.sqrt(0.5*np.pi))*np.exp(-gamma*t+0.5*gamma*gamma*sigma*sigma-0.5*omega*omega*sigma*sigma)*(np.cos(omega*(t-gamma*sigma*sigma)) + (1.j)*np.sin(omega*(t-gamma*sigma*sigma)));
    else:
        z = (-(1.j)*(t-sigma*sigma*gamma) - omega*sigma*sigma)/(sigma*np.sqrt(2.));
        fad = wofz(z);
        return np.sqrt(0.5*np.pi)*np.exp(-0.5*t*t/sigma*sigma)*(np.real(fad) - (1.j)*np.imag(fad));








def test(t):
    hyper_p = conv_exp_VC(t-0, 0.66137 + 0.5*0.0,   0., 0)
    hyper_m = conv_exp_VC(t-0, 0.66137 - 0.5*0.0,   0., 0);
    trig    = conv_exp_VC(t-0,           0.66137, 17.8, 0);

    ta = np.real(0.5*(hyper_m + hyper_p));
    tb = np.real(0.5*(hyper_m - hyper_p));
    tc = np.real(trig);
    td = np.imag(trig);
    return ta,tb,tc,td



treco[0]
ttrue[0]
test(ttrue[0])/(2*np.sqrt(np.pi/2))
np.exp(-0.66137*0.4200498)*np.cosh(0.4200498*(0.0)/2)
np.exp(-0.66137*0.4200498)*np.sinh(0.4200498*(0.0)/2)
np.exp(-0.66137*0.4200498)*np.cos(0.4200498*(17.8))
np.exp(-0.66137*0.4200498)*np.sin(0.4200498*(17.8))

np.exp(-0.65789*0.3)
2*np.sqrt(np.pi/2)
# Implement DTA

knots = [ 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 ]
coeffs1 = [1., 1.053, 1.096, 0.97, 1.052, 1.051, 1.027, 1.09, 1.047]





def u(i):
    if (i<=0):        i = 0;
    elif (i>=n):      i = n
    return knots[i]

def modelEfficiency(t, knots, i, mu, sigma, listcoeffs):
  if i == len(knots)-1:
    a,b,c,d = listcoeffs[i-1:i+3]
    ti = knots[i]; tf = 15.
  else:
    a,b,c,d = listcoeffs[i:i+4]
    ti = knots[i]; tf = knots[i+1]
  c0, c1, c2, c3 = getEfficiency4Coeffs(knots,i,a,b,c,d)
  f = c0 + c1*t +c2*t*t + c3*t*t*t
  return f

def modelEffFit(t,knots,i,mu,sigma,gamma,listcoeffs1,listcoeffs2,tLL=0.3,tUL=15):
  if i == len(knots)-1:
    a1,b1,c1,d1 = listcoeffs1[i-1:i+3]
    a2,b2,c2,d2 = listcoeffs2[i-1:i+3]
    ti = knots[i]; tf = 15.
  else:
    a1,b1,c1,d1 = listcoeffs1[i:i+4]
    a2,b2,c2,d2 = listcoeffs2[i:i+4]
    ti = knots[i]; tf = knots[i+1]
  r0, r1, r2, r3 = getEfficiency4Coeffs(knots,i,a1,b1,c1,d1)
  b0, b1, b2, b3 = getEfficiency4Coeffs(knots,i,a2,b2,c2,d2)
  t = t - mu
  erf_value = 1 - erf((gamma*sigma - t/sigma)/np.sqrt(2.0));
  fpdf  = 0.5*np.exp( 0.5*gamma*(sigma*sigma*gamma - 2*t) ) * (erf_value);
  fpdf *= (r0 + t*(r1 + t*(r2 + t*r3)));
  fpdf *= (b0 + t*(b1 + t*(b2 + t*b3)));
  return (fpdf)


def getKnot(i, knots, n):
    if (i<=0):        i = 0;
    elif (i>=n):      i = n
    return knots[i]







def getTimeBin(t, knots=[ 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 ], n=7):
  _i = 0;
  _n = n-1;
  while (_i <= _n ):
    if( t < knots[_i] ):
      break;
    _i+=1;
  if (0 == _i):
    print("WARNING: t=%f below first knot!\n",t);
  return _i - 1;



for item in ttrue:
  if item < 0.3:
    print('shit')


ttrue


vcalcTimeAcceptance = np.vectorize(calcTimeAcceptance)
vcalcTimeAcceptance(ttrue)

def calcTimeAcceptance( t, coeffs=coeffs1, knots=[ 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 ], n=7):
  tLL = 0.30; tUL = 15.00;
  bin = getTimeBin(t, knots, n);
  if (t < tLL): return 0.0;
  if (t > tUL): return 0.0;

  c0,c1,c2,c3 = getEfficiency4Coeffs(coeffs)[bin]

  return (c0 + t*(c1 + t*(c2 + t*c3)));




def getAcceptanceSingle( t, coeffs=coeffs1,  sigma=0.045, gamma=0.6, mu=0, knots=[ 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 ], n=7):
  from scipy.special import erf
  tLL = 0.30;  tUL = 15.00;
  ti  = 0.0;  tf  =  0.0;
  erf_value = 1 - erf((gamma*sigma - t/sigma)/np.sqrt(2.0));
  fpdf = 1.0;
  fpdf *= 0.5*np.exp( 0.5*gamma*(sigma*sigma*gamma - 2.0*t) ) * (erf_value);
  fpdf *= calcTimeAcceptance( t, coeffs);
  return fpdf


# spline x conv(exp.gauss)
vgetAcceptanceSingle = np.vectorize(getAcceptanceSingle)
vgetAcceptanceSingle(ttrue)
plt.plot(t,vgetAcceptanceSingle(t))

# spline
vcalcTimeAcceptance = np.vectorize(calcTimeAcceptance)
plt.plot(t,vcalcTimeAcceptance(t))





def getEfficiency4Coeffs(listcoeffs,knots=[ 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 ]):
  n = len(knots)
  result = []                                             # list of bin coeffs C
  def u(j): return getKnot(j,knots,n-1)
  for i in range(0,n-1):
        a, b, c, d = listcoeffs[i:i+4]                      # bspline coeffs b_i
        C = []                                     # each bin 4 coeffs c_{bin,i}
        C.append(-((b*u(-2 + i)*pow(u(1 + i),2))/
        ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i)))) +
         (a*pow(u(1 + i),3))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
         (c*pow(u(-1 + i),2)*u(1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         (b*u(-1 + i)*u(1 + i)*u(2 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         (c*u(-1 + i)*u(i)*u(2 + i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (b*u(i)*pow(u(2 + i),2))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (d*pow(u(i),3))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
         (c*pow(u(i),2)*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
        C.append((2*b*u(-2 + i)*u(1 + i))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
         (3*a*pow(u(1 + i),2))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
         (b*pow(u(1 + i),2))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
         (c*pow(u(-1 + i),2))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         (b*u(-1 + i)*u(1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         (2*c*u(-1 + i)*u(1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         (b*u(-1 + i)*u(2 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         (b*u(1 + i)*u(2 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         (c*u(-1 + i)*u(i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (c*u(-1 + i)*u(2 + i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
         (2*b*u(i)*u(2 + i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (c*u(i)*u(2 + i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
         (b*pow(u(2 + i),2))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (c*pow(u(i),2))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
         (3*d*pow(u(i),2))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) -
         (2*c*u(i)*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
        C.append(-((b*u(-2 + i))/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*
              (-u(i) + u(1 + i)))) + (3*a*u(1 + i))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
         (2*b*u(1 + i))/
          ((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) -
         (b*u(-1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         (2*c*u(-1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         (b*u(1 + i))/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*
            (-u(-1 + i) + u(2 + i))) + (c*u(1 + i))/
          ((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         (b*u(2 + i))/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*
            (-u(-1 + i) + u(2 + i))) + (c*u(-1 + i))/
          ((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (b*u(i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
         (c*u(i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         (2*b*u(2 + i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
         (c*u(2 + i))/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) +
         (2*c*u(i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) -
         (3*d*u(i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
         (c*u(3 + i))/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
        C.append(-(a/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i)))) +
         b/((-u(-2 + i) + u(1 + i))*(-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))) +
         b/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) -
         c/((-u(-1 + i) + u(1 + i))*(-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))) +
         b/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         c/((-u(i) + u(1 + i))*(-u(-1 + i) + u(2 + i))*(-u(i) + u(2 + i))) -
         c/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))) +
         d/((-u(i) + u(1 + i))*(-u(i) + u(2 + i))*(-u(i) + u(3 + i))))
        result.append(C)
  m = C[1] + 2*C[2]*u(n) + 3*C[3]*u(n)**2
  C = [C[0] + C[1]*u(n) + C[2]*u(n)**2 + C[3]*u(n)**3 - m*u(n),m,0,0]
  result.append(C)
  return result






#######



getEfficiency4Coeffs(coeffs1)[0]










pdf_omega[:4]/peilian
peilian_true = np.array(
[0.0323402,
0.00765504,
0.00814191,
0.00676178]
)

peilian_reco = np.array([
0.0321921,
0.00790871,
0.00821465,
0.00697316])


sort_mat*getN().subs(subs_dict)


The Aks are almost same, except these related to A_0.
Ak0 0.520935
Ak1 0.230239
Ak2 0.248826
Ak3 0.239352
Ak4 0.346323
Ak5 0.360031
Ak6 0
Ak7 0
Ak8 0
Ak9 0

for item in omega[:,0]:
    print(item)

omega[11,12]


shit = getABCD()

sort_mat*np.array(shit.subs(subs_dict)).astype(np.float64)
(sort_mat*shit.subs(subs_dict)).astype(np.float64)

foo = sort_mat*getABCD().subs(subs_dict)


pdf_omega[:5]


/np.array([0.0323402, 0.00765504, 0.00814191])

0.0323402 0.00765504 0.00814191

foo = sort_mat*getABCD().subs(subs_dict)

bar = pdf_omega[:,:1]
for k in range(0,bar.shape[0]):
    mystr = ''
    for l in range(0,bar.shape[1]):
        mystr += '%+.8f' % bar[k,l] + "   "
    print(mystr)




transAmp()

sort_mat*getOmega().subs(subs_dict)


abcd 1 -0.997551 0 0.0699428
abcd 1 -0.997551 0 0.0699428
abcd 1 0.997551 0 -0.0699428
abcd 0 -0.068101 -0.227978 -0.971282
abcd -0.98748 0.985061 0 -0.0690671
abcd -0 0.0697637 0.0715315 0.994996
abcd 1 0.997551 0 -0.0699428
abcd 0 -0.0159454 0.973666 -0.227419
abcd -0 -0 -0 -0
abcd 0 0.00500312 -0.997438 0.0713563

The Aks are almost same, except these related to A_0.
Ak0 0.520935
Ak1 0.230239
Ak2 0.248826
Ak3 0.239352
Ak4 0.346323
Ak5 0.360031
Ak6 0
Ak7 0
Ak8 0
Ak9 0


"""
fk0 0.101411    0.0791723   0.00349643
fk1 0.00384494  0.0231311   0.0754164
fk2 0.00170212  0.0268456   0.0699348
fk3 0.000927445 0.0494593   0.0161135
fk4 0.00517684  0.00567202  -0.0174411
fk5 -0.0249939  -0.00526201 0.0243559
fk6 0.0353604   0.0594565   0.0125124
fk7 -0.00305689 0.00491531  -0.0329937
fk8 0.0147588   -0.00456    0.0460745
fk9 -0.119765   0.13722     0.0132286

pdf: 0.0529364  0.042374  0.042374
"""





omega[:,:3]


tristan_weights     = tdpB_value/pdfB_value
tristan_weights.shape
angular_weights     = np.sum(tristan_weights,1)
#angular_weights    *= _sw/np.sum(_sw)
#angular_weights    = np.array(sort_mat).dot(angular_weights)
angular_weights     = angular_weights/angular_weights[0]

# Print weights
for k in range(0,len(angular_weights)):
    print("w_{%i} = %+.8f" %(k,angular_weights[k]))
















SantiagoOld = sp.Matrix([
[1.48925465941 , 0.0362972306736],
[1.88366485512 , 0.029241139738],
[1.97403452379 , 0.0360320083889],
[2.02497404136 , 0.0332331418842],
[2.06520278131 , 0.0350109820231],
[2.15427963309 , 0.0412760268784],
[2.05004082063 , 0.0392203084279],
[2.02106527264 , 0.0351107794942],
])


# Santiago + set_strategy
Santiago2 = sp.Matrix([
[1.48903336172,0.0361467879022],
[1.8835639914 ,0.0291771648884],
[1.97383056071,0.0358679371568],
[2.02479932088,0.0331407047466],
[2.06502463592,0.0348438644959],
[2.15410703664,0.0411996073515],
[2.04982101907,0.0389947433575],
[2.02084798065,0.0349134804885],
])



Santiago1 = sp.Matrix([
[1.4896199559 , 0.036168059797],
[1.8839357951 , 0.029190166723],
[1.9744091017 , 0.035888386115],
[2.0252981501 , 0.033158331330],
[2.0655862288 , 0.034863568239],
[2.1546121119 , 0.041218038222],
[2.0504254170 , 0.039015550674],
[2.0214414640 , 0.034934138910],
])

Heidelberg = sp.Matrix([
[1.4890206 , 0.0365253],
[1.8836051 , 0.0293337],
[1.9738032 , 0.0362811],
[2.0248062 , 0.0333680],
[2.0650108 , 0.0352683],
[2.1541471 , 0.0413867],
[2.0497802 , 0.0395725],
[2.0208309 , 0.0354138],
])


Santiago = Santiago2
subt = np.array((Santiago[:,0] - Heidelberg[:,0]))
stat = np.minimum(np.array(Santiago[:,1]),np.array(Heidelberg[:,1]))
sp.Matrix(subt/stat)







shit = getABCD()
sort_mat*shit[:,:]
shit[:,:].subs(subs_dict)





print(sp.latex(shit))

sp.Matrix(tdpB_value[:,0:5])
#[1,0,1,0,0,1,0,0,0,1]


#plt.hist(data["time"],100);plt.show()
#plt.hist(data["helcosthetaK"],100);plt.show()
#plt.hist(data["helcosthetaL"],100);plt.show()
#plt.hist(data["helphi"],100);plt.show()


## Debug FK
(sort_mat*getOmega()*(4*sp.pi/3))[9].subs(sp.sin(phi)**2,1-sp.cos(phi)**2).simplify()
correct_minus_sign = [1,1,1,1,1,-1,1,1,-1,1]
correct_minus_sign[7]



((sort_mat*getOmega()*(4*sp.pi/3))[4]).expand().simplify()



## Debug AK
from sympy import mathematica_code as mcode
(sort_mat*shit[:,0])[4].trigsimp()



#%% some stuff
"""
SIMON RESULTS:
  w1/w0:  1.0269630 +- 0.0008287
  w2/w0:  1.0268884 +- 0.0008278
  w3/w0:  0.0000433 +- 0.0005392
  w4/w0: -0.0002540 +- 0.0003694
  w5/w0: -0.0000474 +- 0.0003480
  w6/w0:  1.0103428 +- 0.0005275
  w7/w0:  0.0006896 +- 0.0004849
  w8/w0: -0.0005949 +- 0.0004707
  w9/w0: -0.0017298 +- 0.0010612
"""




"""
tristan_weights = np.zeros_like(tdpBbar(1))
tristan_weights = np.zeros_like(tdpB(1))
events = 0
for k in range(0,5):
    _x, _y, _z, _t = data[k,0], data[k,1], data[k,2], data[k,3]
    pdfBbar_value = pdfBbar(data[:,0], data[:,1], data[:,2], data[:,3])
    tristan_weights += tdpBbar(_t) * 1./pdfBbar_value
    events += 1
    print(pdfBbar_value)

angular_weights = tristan_weights/events
"""
