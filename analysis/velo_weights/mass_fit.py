__all__ = ['cb', 'cb_argus', 'cb_physbkg']

from timeit import main
import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, Sample)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
import complot

DOCAz_BINS = 15

##############################################################################
# selection
selection = 'francesca'
# mass variable to use
BuM = 'Bu_M'
BuM = 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr'
# mass range cut
mass_range_cut = True
##############################################################################

# initialize ipanema3 and compile lineshapes
ipanema.initialize(config.user['backend'], 1)
prog = ipanema.compile("""
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>
""")


# PDF models ------------------------------------------------------------------
#    Select pdf model to fit data. {{{

# CB + ARGUS + EXP {{{
def cb_argus(mass, signal, nsig, nbkg, nexp, mu, sigma, aL, nL, aR, nR, b, m0, c, p,
             norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_Argus(signal, mass, np.float64(m0), np.float64(c), np.float64(p),
                global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signalpy_double_crystal_ball
  prog.py_double_crystal_ball(signal, mass, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize argus
  prog.py_Argus(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans
# }}}


def cb(mass, signal, nsig, nexp, mu, sigma, aL, nL, aR, nR, b, norm=1):
  # compute peak
  prog.kernel_double_crystal_ball(
      signal, mass, np.float64(mu), 0. * mass + sigma, np.float64(aL),
      np.float64(nL), np.float64(aR), np.float64(nR),
      np.float64(ristra.min(mass)), np.float64(ristra.max(mass)),
      global_size=(len(mass)))
  peak = nsig * ristra.get(signal)
  # compute background
  if nexp > 0:
    prog.kernel_exponential(
        signal, mass, np.float64(b),
        np.float64(ristra.min(mass)), np.float64(ristra.max(mass)),
        global_size=(len(mass)))
    comb = nexp * ristra.get(signal)
  else:
    comb = 0
  return norm * (peak + comb)
# }}}


# CB + PHYSBKG + EXP {{{
def cb_physbkg(mass, signal, nsig, nbkg, nexp, mu, sigma, aL, nL, aR, nR, b, m0, c,
               p, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_physbkg(signal, mass, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1001)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize rgus
  prog.py_physbkg(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans
# }}}


# CB + SHOULDER + EXP {{{
def cb_shoulder(mass, signal, sig, bkg, mu, sigma, aL, nL, b, p, s, trans, norm=1):
  # compute backgrounds
  pexp = ristra.get(ristra.exp(mass * b))
  prog.py_shoulder(signal, mass, np.float64(p), np.float64(s),
                   np.float64(trans), global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_CrystalBall(signal, mass, np.float64(mu), np.float64(sigma),
                      np.float64(aL), np.float64(nL), global_size=(len(mass))
                      )
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_CrystalBall(_y, _x, np.float64(mu), np.float64(sigma),
                      np.float64(aL), np.float64(nL), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize argus
  prog.py_shoulder(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                   global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexp = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = norm * (sig * (pcb / npb) + (1. - sig - bkg) * (pexp / nexp) + bkg * (pargus / nargus))
  return ans
# }}}


# DOUBLE GAUSSIAN + ARGUS + EXP {{{
def dgauss_argus(mass, signal, nsig, nbkg, nexp, mu, sigma, dmu, dsigma, res, b, m0, c,
                 p, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_Argus(signal, mass, np.float64(m0), np.float64(c), np.float64(p),
                global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_double_gaussian(signal, mass, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res), global_size=(len(mass))
                          )
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_gaussian(_y, _x, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res),
                          global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize argus
  prog.py_Argus(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans
# }}}


# DOUBLE GAUSSIAN + PHYSBKG + EXP {{{
def dgauss_physbkg(mass, signal, nsig, nbkg, nexp, mu, sigma, dmu, dsigma, res, b, m0,
                   c, p, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_physbkg(signal, mass, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_double_gaussian(signal, mass, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res), global_size=(len(mass))
                          )
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_gaussian(_y, _x, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res),
                          global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize argus
  prog.py_physbkg(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans
# }}}


# DOUBLE GAUSSIAN {{{
def dgauss(mass, signal, nsig, nexp, mu, sigma, dmu, dsigma, res, b,
           norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_double_gaussian(signal, mass, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res), global_size=(len(mass))
                          )
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_gaussian(_y, _x, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res),
                          global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo)
  return norm * ans
# }}}


# DOUBLE GAUSSIAN + SHOULDER {{{
def dgauss_shoulder(mass, signal, nsig, nbkg, nexp, mu, sigma, dmu, dsigma, res, b,
                    p, s, trans, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_shoulder(signal, mass, np.float64(p), np.float64(s),
                   np.float64(trans), global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_double_gaussian(signal, mass, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res),
                          global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_gaussian(_y, _x, np.float64(mu), np.float64(sigma),
                          np.float64(dmu), np.float64(dsigma),
                          np.float64(1.), np.float64(res),
                          global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize argus
  prog.py_shoulder(_y, _x, np.float64(p), np.float64(s), np.float64(trans),
                   global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans
# }}}


# IPATIA + ARGUS {{{
def ipatia(mass, signal, nsig, nexp, mu, sigma, lambd, zeta, beta, aL, nL, aR, nR,
           b, norm=1):
  # ipatia
  prog.py_ipatia(signal, mass, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(mass)))
  backgr = ristra.exp(mass * b)
  # normalize
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  prog.py_ipatia(_y, _x, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(_x)))
  nsignal = np.trapz(ristra.get(_y), ristra.get(_x))
  nbackgr = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = norm * (nsig * signal / nsignal + nexp * backgr / nbackgr)
  return ans
# }}}

# }}}


# WARNING: Please move this function to ipanema3 package
def equibins1d(x, nbin):
  """
  This functions takes a random variable x and creates nbin bins with the same
  number of candidates in each of them (if possible).

  Parameters
  ----------
  x : ndarray
    Random variable to histogram.
  nbins : int
    Number of bins.

  Returns
  -------
  ndarray
    Set of edges to histogram x with.
  """
  n = len(x)
  return np.interp(np.linspace(0, n, nbin + 1), np.arange(n), np.sort(x))


"""
The boss:

Enton, brevemente o que hai que facer e, para distintos valores de DOCAZ,
caluclar a efficiencia

eff = num/den

onde
num -> numero de B+ ->JpsiK+ que matchan no velo
( entendo que eso significa pedir na ntupla Bu_PVConst_veloMatch> 0
ou se non Bu_PVConst_veloMatch_stdmethod>0)

den -> "" sin o match requirement

O numero de sinal sacalo dun mass fit a B+ mais bkg. O bkg pode ser unha
exponencial tipicamente.
A maiores pode ser que necesites un pico pequeno para B+ >Jpsi pi+ e moi
probablemente un
shoulder para a sideband esquerda (teria q ver o plot pra decircho seguro).
Se tes dudas pasa x aqui.
"""


printsec("Loading data")
# Selection -------------------------------------------------------------------
if selection == 'naive':
  cut = "Bu_IPCHI2_OWNPV<9 & Jpsi_M>3075 & Jpsi_M<3120"
elif selection == 'diego':
  # chat with Diego
  cut = "Bu_IPCHI2_OWNPV<9 & Jpsi_M>3075 & Jpsi_M<3120"
  cut += " & Jpsi_ENDVERTEX_CHI2<9 & Bu_ENDVERTEX_CHI2<27 & log10(Bu_TAU)>-3.1"
elif selection == 'francesca':
  """
  mattermost with Francesca:
  //Fiducial cuts!
  if(muplus_LOKI_ETA<2.0  || muplus_LOKI_ETA > 4.5)    continue;         DONE
  if(muminus_LOKI_ETA<2.0 || muminus_LOKI_ETA > 4.5)    continue;        DONE
  if(Kplus_LOKI_ETA<2.0   || Kplus_LOKI_ETA > 4.5)    continue;          DONE
  if(fabs(Bu_PVConst_PV_Z[0])>100.) continue;                            DONE

  if(muplus_PIDmu < 0.) continue;                                         DONE
  if(muminus_PIDmu < 0.) continue;                                        DONE
  if(muplus_PT < 500.)    continue;                                       DONE
  if(muminus_PT < 500.)     continue;                                     DONE
  if(Jpsi_ENDVERTEX_CHI2 > 16.)continue;                                  DONE
  if(Jpsi_M < 3030.|| Jpsi_M > 3150.)continue;                            DONE
  if(Kplus_PIDK < 0.) continue;
  if(Kplus_PT < 500.) continue;                                           DONE

  if(Kplus_TRACK_CHI2NDOF > 4.) continue;
  if(muplus_TRACK_CHI2NDOF > 4.) continue;
  if(muplus_TRACK_CHI2NDOF > 4.) continue;

  //if((Bu_PVConstPVReReco_chi2[0]/Bu_PVConstPVReReco_nDOF[0])>5.) continue;
  if((Bu_PVConst_chi2[0]/Bu_PVConst_nDOF[0])>5.) continue;
  if(Bu_MINIPCHI2>25.) continue;
  if((Bu_MINIPCHI2NEXTBEST < 50.) && (Bu_MINIPCHI2NEXTBEST != -1)) continue;
  if((Bu_LOKI_MASS_JpsiConstr_NoPVConstr < 5170.) || (Bu_LOKI_MASS_JpsiConstr_NoPVConstr > 5400.))continue;
  if((Bu_PVConst_ctau[0]/0.299792458)< 0.3) continue;
  if((Bu_PVConst_ctau[0]/0.299792458)> 14.) continue;

  //if(Bu_hasBestDTFCHI2 < 0) continue;  >>>>>>>>>>>>>>>>>>>> FIX IT FOR DATA EXPECIALLY

  if(Bu_L0MuonDecision_TOS!= 1 && Bu_L0DiMuonDecision_TOS!=1) continue;   DONE
  if(Bu_Hlt1DiMuonHighMassDecision_TOS!=1) continue;                      DONE
  if(Bu_Hlt2DiMuonDetachedJPsiDecision_TOS!=1) continue;                  DONE
  """
  # fiducial cuts
  cuts = ['Jpsi_LOKI_ETA>2 & Jpsi_LOKI_ETA<4.5',
          'muplus_LOKI_ETA>2 & muplus_LOKI_ETA<4.5',
          'muminus_LOKI_ETA>2 & muminus_LOKI_ETA<4.5',
          'Kplus_LOKI_ETA>2 & Kplus_LOKI_ETA<4.5', 'abs(Bu_PVConst_PV_Z)<100'
          ]
  # Bu2JpsiKplus cuts
  cuts += [  # 'Bu_PVConstPVReReco_chi2/Bu_PVConstPVReReco_nDOF)<5',
      'Bu_IPCHI2_OWNPV<25',
      'Bu_MINIPCHI2<25',
      '(Bu_PVConst_chi2/Bu_PVConst_nDOF)<5',
      'Bu_MINIPCHI2NEXTBEST>50 | Bu_MINIPCHI2NEXTBEST==-1',
      'Bu_LOKI_DTF_CHI2NDOF<4',
      '(Bu_PVConst_ctau/0.299792458)>0.3',
      '(Bu_PVConst_ctau/0.299792458)<14',
      # 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr>5170',
      # 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr<5400'
  ]
  # Kplus cuts
  cuts += ['Kplus_TRACK_CHI2NDOF<4', 'Kplus_PT>500',  # 'Kplus_P>10000',
           'Kplus_PIDK>0'
           ]
  # muons
  cuts += ['muplus_TRACK_CHI2NDOF<4', 'muminus_TRACK_CHI2NDOF<4',
           'muplus_PT>500', 'muminus_PT>500', 'muplus_PIDmu>0',
           'muminus_PIDmu>0'
           ]
  # Jpsi cuts
  cuts += ['Jpsi_ENDVERTEX_CHI2<16', 'Bu_LOKI_FDS>3',
           'Jpsi_M>3030 & Jpsi_M<3150']
  # # trigger requirements
  cuts += ["Bu_L0MuonDecision_TOS==1 | Bu_L0DiMuonDecision_TOS==1",
           "Bu_Hlt1DiMuonHighMassDecision_TOS==1",
           "Bu_Hlt2DiMuonDetachedJPsiDecision_TOS==1"]
  cut = "(" + ") & (".join(cuts) + ")"
else:
  cut = "1"

if mass_range_cut:
  cut = f"({cut}) & ({BuM}>5170 & {BuM}<5400)"


if __name__ == '__main__':
  # Parse command line arguments ----------------------------------------------
  p = argparse.ArgumentParser(description='Get efficiency in DOCAz bin.')
  p.add_argument('--sample', help='Bu2JpsiKplus RD sample')
  p.add_argument('--params', help='Mass fit parameters')
  p.add_argument('--params-match', help='Mass fit parameters VELO matching')
  p.add_argument('--plot-mass', help='Plot of the mass fit')
  p.add_argument('--plot-logmass', help='Plot of the log mass fit')
  p.add_argument('--plot-mass-match', help='Plot of the mass fit VELO match')
  p.add_argument('--plot-logmass-match',
                 help='Plot of the log mass fit VELO match')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--docaz-bin', help='Different flag to ... ')
  p.add_argument('--mass-model', help='Different flag to ... ')
  args = vars(p.parse_args())

  MODEL = args['mass_model']
  DOCAZ_BIN = int(args['docaz_bin'])
  SAMPLE = args["sample"]

  # Load sample ---------------------------------------------------------------
  #    To speed up, please use the reduce_tuples script to avoid jagged arrays
  #    which are very slow in python.
  sample = Sample.from_root(SAMPLE)
  print(f"Original sample has {sample.shape}")
  # sample.chop(cut)
  # sample.chop("Bu_PVConst_J_psi_1S_muminus_0_DOCAz<=5")
  # if MODEL == "dgauss":
  #   cut = f"({cut}) & ({mass}>5200)"
  print(f"Selection cut: {cut}")
  print(f"Selected sample has {sample.shape}")

  if "hplus" in args['params']:
    DOCAz = "Bu_PVConst_Kplus_DOCAz"
    sample.df.eval(f"DOCAz={DOCAz}", inplace=True)
  elif "hminus" in args['params']:
    DOCAz = "Bu_PVConst_Kplus_DOCAz"
    sample.df.eval(f"DOCAz={DOCAz}", inplace=True)
  elif "muplus" in args['params']:
    DOCAz = "Bu_PVConst_J_psi_1S_muplus_0_DOCAz"
    sample.df.eval(f"DOCAz={DOCAz}", inplace=True)
  elif "muminus" in args['params']:
    DOCAz = "Bu_PVConst_J_psi_1S_muminus_0_DOCAz"
    sample.df.eval(f"DOCAz={DOCAz}", inplace=True)
  else:
    print("bug here")
    exit()

  print(sample.df)
  # Select model and set parameters -------------------------------------------
  #    Select model from command-line arguments and create corresponding set of
  #    paramters
  pars = ipanema.Parameters()
  # Create common set of parameters (all models must have and use)
  pars.add(dict(name='nsig', value=0.90, min=0, max=1, free=True,
                latex=r'N_{signal}'))
  pars.add(dict(name='mu', value=5280, min=5200, max=5400,
                latex=r'\mu'))
  pars.add(dict(name='sigma', value=18, min=1, max=100, free=True,
                latex=r'\sigma'))
  if "cb" in MODEL.split('_'):  # {{{
    # crystal ball tails
    pars.add(dict(name='aL', value=1, latex=r'a_l', min=-50, max=50,
                  free=True))
    pars.add(dict(name='nL', value=2, latex=r'n_l', min=-500, max=500,
                  free=True))
    pars.add(dict(name='aR', value=1, latex=r'a_r', min=-50, max=500,
                  free=True))
    pars.add(dict(name='nR', value=2, latex=r'n_r', min=-500, max=500,
                  free=True))
    if "argus" in MODEL.split('_'):
      pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                    latex=r'N_{part.reco.}'))
      pars.add(dict(name='c', value=20, min=-1000, max=100, free=True,
                    latex=r'c'))
      pars.add(dict(name='p', value=1, min=0.1, max=50, free=True,
                    latex=r'p'))
      pars.add(dict(name='m0', value=5155, min=5100, max=5220, free=True,
                    latex=r'm_0'))
      pdf = cb_argus
      print("Using CB + argus pdf")
    elif "physbkg" in MODEL.split('_'):
      pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                    latex=r'N_{background}'))
      pars.add(dict(name='c', value=0.001, min=-1000, max=100, free=True,
                    latex=r'c'))
      pars.add(dict(name='p', value=1, min=0.01, max=50, free=True,
                    latex=r'p'))
      pars.add(dict(name='m0', value=5175, min=5150, max=5200, free=True,
                    latex=r'm_0'))
      pdf = cb_physbkg
      print("Using CB + physbkg pdf")
    else:
      pdf = cb
    # }}}
  elif "dgauss" in MODEL.split('_'):
    pars.add(dict(name='dmu', value=0, min=-10, max=10, free=False,
                  latex=r'\Delta\mu'))
    pars.add(dict(name='dsigma', value=0, min=-30, max=30, free=True,
                  latex=r'\Delta\sigma'))
    pars.add(dict(name='res', value=0.0, latex=r'\delta', min=0, max=10,
                  free=True))
    if "argus" in MODEL.split('_'):
      # ARGUS PARAMETERS {{{
      pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                    latex=r'N_{background}'))
      pars.add(dict(name='c', value=-20, min=-1000, max=0, free=True,
                    latex=r'c'))
      pars.add(dict(name='p', value=1, min=0.01, max=50, free=True,
                    latex=r'p'))
      pars.add(dict(name='m0', value=5150, min=5100, max=5190, free=True,
                    latex=r'm_0'))
      # }}}
      pdf = dgauss_argus
    elif "shoulder" == MODEL.split('_'):
      # SHOULDER PARAMETERS {{{
      #    This model uses a double gaussian for the peak and a exponential
      #    background. You should shrink the mass window to avoid the background
      #    in the left-sideband. {{{
      pars.add(dict(name='dmu', value=0, min=-10, max=10, free=False,
                    latex=r'\Delta\mu'))
      pars.add(dict(name='dsigma', value=1, min=-50, max=50, free=True,
                    latex=r'\Delta\sigma'))
      pars.add(dict(name='res', value=0, latex=r'\delta', min=-10, max=10,
                    free=True))
      # }}}
      pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                    latex=r'N_{background}'))
      pars.add(dict(name='p', value=5090, min=5000, max=5200, free=True,
                    latex=r'p'))
      pars.add(dict(name='s', value=20, min=1, max=50, free=True,
                    latex=r's'))
      pars.add(dict(name='trans', value=25, min=-10000, max=10000, free=True,
                    latex=r'trans'))
      # }}}
      pdf = dgauss_shoulder
    else:
      pdf = dgauss
  elif "ipatia" in MODEL.split('_'):
    # ipatia tails {{{
    pars.add(dict(name='lambd', value=-1, min=-20, max=0, free=True,
                  latex=r'\lambda'))
    pars.add(dict(name='zeta', value=0.0, latex=r'\zeta', free=False))
    pars.add(dict(name='beta', value=0.0, latex=r'\beta', free=False))
    pars.add(dict(name='aL', value=1, latex=r'a_l', free=True))
    pars.add(dict(name='nL', value=30, latex=r'n_l', free=True))
    pars.add(dict(name='aR', value=1, latex=r'a_r', free=True))
    pars.add(dict(name='nR', value=30, latex=r'n_r', free=True))
    pdf = ipatia
    # }}}

  # EXPONENCIAL Parameters {{{
  pars.add(dict(name='b', value=-0.0014, min=-1, max=1, latex=r'b'))
  pars.add(dict(name='nexp', value=0.02, min=0, max=1, free=True,
                formula=f"1-nsig{'-nbkg' if 'nbkg' in pars else ''}",
                latex=r'N_{exp}'))
  # }}}

  # check if
  if not pdf:
    print("Unknown model. Exiting.")
    exit()

  print("Initial set of parameters:")
  print(pars)

  # Define fcn function
  def fcn(params, data):
    p = params.valuesdict()
    prob = pdf(data.mass, data.pdf, **p)
    return -2.0 * np.log(prob) * ristra.get(data.weight)

  # sample
  # sample = ipanema.Sample.from_root(args["sample"])

  # Get DOCAz bins
  # docaz = [0., 0.0119, 0.0247, 0.0394, 0.0584, 0.0875, 0.1427, 0.2767, 5.]
  docaz = [0.0, 0.3, 0.58, 0.91, 1.35, 1.96, 3.01, 7.0]
  docaz = [0.01909757, 0.02995095, 0.04697244, 0.07366745, 0.11553356,
           0.1811927, 0.28416671, 0.44566212, 0.69893733, 1.09615194,
           1.71910844, 2.69609871, 4.22832446, 6.63133278]

  # these ones are somehow expoenly distributed and work just fine
  docaz = [0, 0.01652674, 0.02591909, 0.04064923, 0.06375068, 0.09998097,
           0.15680137, 0.2459135, 0.38566914, 0.60484961, 0.94859302,
           1.48768999, 2.33316235, 3.65912694, 6, 10]
  docaz = np.round(equibins1d(sample.df['DOCAz'], 15) * 1e4) / 1e4
  print(docaz)
  docaz = docaz[DOCAZ_BIN - 1:DOCAZ_BIN + 1]
  print(docaz)
  #docaz = np.array([0.0, 10.])
  nevt = []

  # %% Start Bu mass to get efficiency shape ----------------------------------
  printsec(f"Fitting Bu mass in DOCAz range = [{docaz[0]}, {docaz[1]})")

  doca_cut = [f"(DOCAz>={docaz[0]})",
              f"(DOCAz<{docaz[1]})"]
  doca_cut = f"({' & '.join(doca_cut)})"
  # do both fits
  for match in [True, False]:
    _pars = ipanema.Parameters.clone(pars)
    if match:
      rd = Sample.from_root(SAMPLE)
      rd.df.eval(f"DOCAz={DOCAz}", inplace=True)
      rd.chop(f'{cut} & {doca_cut} & (Bu_PVConst_veloMatch>0)')
      rd.allocate(mass=f'{BuM}', pdf=f'0*{BuM}', weight=f'{BuM}/{BuM}')
      printsubsec(f"Fitting {len(rd.mass)} VELO-matched events.")
    else:
      rd = Sample.from_root(SAMPLE)
      rd.df.eval(f"DOCAz={DOCAz}", inplace=True)
      # a / (a + b)
      rd.chop(f'{cut} & {doca_cut} & ~(Bu_PVConst_veloMatch>0)')
      # a / b
      # rd.chop(f'{cut} & {doca_cut}')
      rd.allocate(mass=f'{BuM}', pdf=f'0*{BuM}', weight=f'{BuM}/{BuM}')
      _pars.lock()
      _pars.unlock("nsig", "b")
      printsubsec(f"Fitting {len(rd.mass)} events.")
    print(rd)

    # fit
    fit_valid = 0
    tries = 0
    max_tries = 2
    while not fit_valid and tries < max_tries:
      res = None
      try:
        res = ipanema.optimize(fcn, _pars, fcn_kwgs={'data': rd},
                               method='minuit', verbose=False)
        fit_valid = True
      except:
        print(f"Try #{tries} failed...")
        tries += 1
        fit_valid = False
        # _pars.lock()
        _pars["nsig"].set(value=1, init=1)

    # errors suck
    if res:
      print(res)
      fpars = ipanema.Parameters.clone(res.params)
    else:
      print("could not fit it!. Cloning pars to res")
      fpars = ipanema.Parameters.clone(pars)

    # plot --------------------------------------------------------------------
    fig, axplot, axpull = complot.axes_providers.axes_plotpull()
    hdata = complot.hist(ristra.get(rd.mass), weights=None,
                         bins=50, density=False)
    axplot.errorbar(hdata.bins, hdata.counts,
                    yerr=hdata.yerr,
                    xerr=hdata.xerr, fmt='.k')

    norm = hdata.norm * (hdata.bins[1] - hdata.bins[0])
    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0 * mass

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if 'nbkg' in _p:
      _p['nbkg'].set(value=0)
    _p['nexp'].set(value=0)
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm))
    axplot.plot(_x, _y, color="C1", label='signal')

    # plot backgrounds: nsig -> 0
    _p = ipanema.Parameters.clone(fpars)
    _p['nexp'].set(value=_p['nexp'].value)
    _p['nsig'].set(value=0)
    _x, _y = ristra.get(mass), ristra.get(
        pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm))
    axplot.plot(_x, _y, '-.', color="C2", label='background')

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(),
                                            norm=hdata.norm))
    axplot.plot(x, y, color='C0')
    axpull.fill_between(hdata.bins,
                        complot.compute_pdfpulls(x, y, hdata.bins,
                                                 hdata.counts, *hdata.yerr),
                        0, facecolor="C0", alpha=0.5)
    axpull.set_xlabel(r'$m(B^+)$ [MeV/$c^2$]')
    axpull.set_ylim(-3.5, 3.5)
    axpull.set_yticks([-2.5, 0, 2.5])
    axplot.set_ylabel(rf"Candidates{' matching VELO' if match else ''}")
    fig.savefig(args[f"plot_mass{'_match' if match else ''}"])
    fig.savefig(f"mass{DOCAZ_BIN}{'_match' if match else ''}.pdf")
    axplot.set_yscale('log')
    axplot.set_ylim(1e0, 1.5 * np.max(y))
    fig.savefig(args[f"plot_logmass{'_match' if match else ''}"])
    fig.savefig(f"logmass{DOCAZ_BIN}{'_match' if match else ''}.pdf")
    plt.close()

    # If we want to use the VELO-matched params for the peak as
    if res:
      pars = ipanema.Parameters.clone(res.params)

    # Dumping fit parameters --------------------------------------------------
    print(len(rd.mass) * fpars['nsig'].uvalue)
    if fpars['nsig'].stdev:
      fpars['nsig'].set(value=fpars['nsig'].uvalue.n * len(rd.mass),
                        stdev=fpars['nsig'].stdev * len(rd.mass),
                        min=-np.inf, max=np.inf)
      fpars['nexp'].set(value=fpars['nexp'].uvalue.n * len(rd.mass),
                        min=-np.inf, max=np.inf)
      print(fpars)
      if 'nbkg' in fpars:
        fpars['nbkg'].set(value=fpars['nbkg'].uvalue.n * len(rd.mass),
                          stdev=fpars['nbkg'].stdev * len(rd.mass),
                          min=-np.inf, max=np.inf)
    fpars.dump(args[f"params{'_match' if match else ''}"])
    nevt.append(fpars['nsig'].uvalue)
    del rd

  # Print result
  print(nevt)
  print(f"Efficiency in this doca bin is {nevt[0]/(nevt[1]+1*nevt[0])}")


# vim: set ts=2 sw=2 sts=2 et
