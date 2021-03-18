DESCRIPTION = """
    Computes the legendre-based angular acceptance with corrections in mHH, pB,
    pTB variables using an a reweight.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []



from ipanema import wrap_unc, uncertainty_wrapper, get_confidence_bands
from ipanema import initialize, ristra, Parameters, Sample, optimize, IPANEMALIB
from utils.helpers import  version_guesser, trigger_scissors
from utils.strings import printsubsec
from ipanema.core.python import ndmesh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot3 as uproot
from scipy.special import lpmv
from scipy.interpolate import interp1d, interpn
import argparse
initialize('cuda',1)
import badjanak
badjanak.get_kernels()
from scipy.special import comb
from scipy.integrate import romb, simpson
from ipanema import plotting
import uncertainties.unumpy as unp
import uncertainties as unc
from scipy import stats, special

order_cosK = 2
order_cosL = 4
order_hphi = 2
nob = 20

import hjson
all_knots = hjson.load(open('config.json'))['time_acceptance_knots']


# load generator level parameters and generate a toy
gen = Parameters.load('analysis/params/generator_old/2015/MC_Bs2JpsiPhi.json')

# load MC sample
"""
Sample.from_root('/scratch46/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi/v0r0.root').branches
uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root')
mc = Sample.from_root('/scratch46/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi/v0r0.root').df.query('Jpsi_Hlt1DiMuonHighMassDecision_TOS==0')
"""

printsubsec("Prepare data")
mc = pd.concat((
uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw.root')['DecayTree'].pandas.df(['helcosthetaK','helcosthetaL','helphi', 'time', 'sw', 'gb_weights', 'hlt1b', 'cosThetaKRef_GenLvl', 'cosThetaMuRef_GenLvl', 'phiHelRef_GenLvl']),
uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2015_UpDown_DST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].pandas.df(['PolWeight'])
), axis=1)

mc.eval('cosK = helcosthetaK', inplace=True)
mc.eval('cosL = helcosthetaL', inplace=True)
mc.eval('hphi = helphi', inplace=True)
mc.eval('gencosK = cosThetaKRef_GenLvl', inplace=True)
mc.eval('gencosL = cosThetaMuRef_GenLvl', inplace=True)
mc.eval('genhphi = phiHelRef_GenLvl', inplace=True)
mc.eval('polWeight = PolWeight', inplace=True)

# get only one trigger cat
mc = mc.query('hlt1b == 0')
weights = np.array(mc.eval(f'polWeight*sw/gb_weights'))
alpha = np.sum(weights)/np.sum(weights**2)

# create margic function
@np.vectorize
def get_angular_prediction(params, cosKLL=-1, cosKUL=-0.9, cosLLL=-1,
                           cosLUL=-0.9, hphiLL=-np.pi, hphiUL=-0.9*np.pi):
    """
    Calculates the angular prediction in a given 3d bin in cosK, cosL and hphi.

    Parameters
    ----------
    params:

    """
    # TODO: This function is too slow. It was coded this way just to check
    #       good agreement with HD-fitter. Once this is done (it is), it should
    #       be rewritten in openCL.
    var = ristra.allocate(np.ascontiguousarray(np.float64([0.0]*3+[0.3]+[1020.]+[0.0]*5)))
    pdf = ristra.allocate(np.float64([0.0]))
    badjanak.delta_gamma5_mc(var, pdf, **params, tLL=0.3, tUL=15)
    num = pdf.get()
    badjanak.delta_gamma5_mc(var, pdf, **params, cosKLL=cosKLL,
                             cosKUL=cosKUL, cosLLL=cosLLL, cosLUL=cosLUL,
                             hphiLL=hphiLL, hphiUL=hphiUL, tLL=0.3, tUL=15)
    den = pdf.get()
    return num/den



printsubsec("Prepare data")
histdd = np.histogramdd(mc[['cosK','cosL','hphi']].values, bins=(nob,nob,nob),
                        weights=mc.eval(f'polWeight*sw/gb_weights'),
                        range=[(-1,1),(-1,1),(-np.pi,np.pi)])
mccounts, (ecosK, ecosL, ehphi) = histdd
noe = (order_hphi+1)*(order_cosL+1)*(order_cosK+1)



printsubsec("Prepare theory")
# WARNING: Prepare mesh with get_angular_prediction, which is a very slow
#          function
m1 = ndmesh(ecosK[:-1], ecosL[:-1], ehphi[:-1])
m2 = ndmesh(ecosK[1:], ecosL[1:], ehphi[1:])
arry = np.ascontiguousarray(np.zeros((nob**3,2*3)))
arry[:,0::2] = np.stack((m.ravel() for m in m1 ), axis=-1)
arry[:,1::2] = np.stack((m.ravel() for m in m2 ), axis=-1)
pars_dict = np.ascontiguousarray([gen.valuesdict() for i in range(nob**3)])

# generate predictions
gencounts = get_angular_prediction(pars_dict, *arry.T).reshape(nob,nob,nob)

# cook arrays with bin centers
bcosK = 0.5*(ecosK[1:]+ecosK[:-1])
bcosL = np.copy(bcosK)
bhphi = np.copy(bcosK)

# TODO: These two functions sort elements in the HD-fitter way. This is clearly
#       uneeded, but one should rewrite the whole code, and loses comparison
#       capabilitis wrt. HD-fitter.
def data_3d(i, N=20):
  c = i//(N*N); d = i%(N*N);
  a = d//N; b = d%N
  return mccounts[c, b, a]

def prediction_3d(i, N=20):
  c = i//(N*N); d = i%(N*N);
  a = d//N; b = d%N
  return gencounts[a, b, c]


#%%
printsubsec('Fitting legendre polynomials coefficients')

# I messed up, so
order_cosK, order_cosL, order_hphi = order_cosL, order_cosK, order_hphi
print(' * Build parameters dict')
pars = Parameters()
for i in range(noe):
  pars.add({"name":f"b{i+1}", "value":0.0, "latex":f"b_{i+1}", "free":False})
  cosK_bin = i % (order_cosK+1)
  hphi_bin = ((i - cosK_bin)/(order_cosK+1)) % (order_hphi+1)
  cosL_bin = ((i - cosK_bin)/(order_cosK+1) - hphi_bin)/(order_hphi+1);
  if ((cosK_bin % 2 == 0 or cosK_bin == 0) and
      (hphi_bin % 2 == 0 or hphi_bin == 0) and
      (cosL_bin % 2 == 0 or cosL_bin == 0)):
    pars[f'b{i+1}'].free = True
    pars[f'b{i+1}'].set(value=0.0)
pars['b1'].set(value=1.0)
#for k in deleted_ones:
#  pars[k].free=False
#print(pars)

print(' * Allocating arrays')
chi2_d = ristra.allocate(0*bcosK).astype(np.float64)
cosK_d = ristra.allocate(bcosK).astype(np.float64)
cosL_d = ristra.allocate(bcosL).astype(np.float64)
hphi_d = ristra.allocate(bhphi).astype(np.float64)
data_3d_d = ristra.allocate(
                np.float64([data_3d(i) for i in range(nob**3)])
            ).astype(np.float64)
prediction_3d_d = np.sum(weights)*ristra.allocate(
                        np.float64([prediction_3d(i) for i in range(nob**3)])
                  ).astype(np.float64)

# TODO: This function should be moved to badjanak
badjanak.get_kernels()
def fcn_d(pars):
  pars_d = ristra.allocate(np.array(pars))
  chi2_d = ristra.allocate(0*data_3d_d).astype(np.float64)
  cosK_l = len(cosK_d); cosL_l = len(cosL_d); hphi_l = len(hphi_d)
  badjanak.__KERNELS__.magia_borras(chi2_d, cosK_d, cosL_d, hphi_d, data_3d_d,
                                    prediction_3d_d, pars_d, np.int32(cosK_l),
                                    np.int32(cosL_l), np.int32(hphi_l),
                                    np.int32(order_cosK), np.int32(order_cosL),
                                    np.int32(order_hphi),
                                    global_size=(cosL_l, hphi_l, cosK_l))
  return ristra.get(chi2_d).ravel()

print(' * Fitting...')
result = optimize(fcn_d, pars, method='minuit', verbose=False, timeit=True)
deleted_ones = []
#for n,p in result.params.items():
#  if p.stdev:
#    if abs(p.stdev/p.value) > 1:
#      deleted_ones.append(n)
#      p.set(value=0)
#      p.stdev=0
print(result)


#%%

def transform_cijk(cijk, order_x, order_y, order_z):
    # TODO: I'm pretty sure this can be done much more easily and faster, but
    #       it seems work to the next slave working on phi_s at SCQ
    noe = (order_x+1)*(order_y+1)*(order_z+1)
    corr_x = np.zeros(noe)
    corr_y = np.zeros(noe)
    corr_z = np.zeros(noe)
    coeffs = np.zeros(noe)

    count = 0
    for i in range(0, order_x+1):
      for j in range(0, order_y+1):
        for k in range(0, order_z+1):
          from_bin = i+(order_x+1)*j + (order_x+1)*(order_y+1)*k
          #print(from_bin, peff[count])
          for m in range(0, i//2+1):
            to_bin = (i-2*m)+(order_x+1)*j + (order_x+1)*(order_y+1)*k
            corr_x[to_bin] += (-1)**(m) * comb(i, m) * comb(2*i-2*m, i) * cijk[count] / ( 2.0**(i) )
          #print(to_bin, corr_x[to_bin], '\n')
          count += 1

    for i in range(0, order_x+1):
      for j in range(0, order_y+1):
        for k in range(0, order_z+1):
          from_bin = i+(order_x+1)*j + (order_x+1)*(order_y+1)*k;
          #print(from_bin, corr_x[from_bin])
          for m in range(0, j//2+1):
            to_bin = i+(order_x+1)*(j-2*m) + (order_x+1)*(order_y+1)*k
            corr_y[to_bin] += (1.0 if (m%2==0) else -1.0) * comb(j, m) * comb(2*j-2*m, j) * corr_x[from_bin] / ( 2.0**(j) )
          #print(to_bin, corr_y[to_bin], '\n')

    for i in range(0, order_x+1):
      for j in range(0, order_y+1):
        for k in range(0, order_z+1):
          from_bin = i+(order_x+1)*j + (order_x+1)*(order_y+1)*k;
          for m in range(0, k//2+1):
            #print(from_bin, corr_y[from_bin])
            to_bin = i+(order_x+1)*j + (order_x+1)*(order_y+1)*(k-2*m)
            corr_z[to_bin] += (1.0 if (m%2==0) else -1.0) * comb(k, m) * comb(2*k-2*m, k) * corr_y[from_bin] / ( 2.0**(k) )
            #print(to_bin, corr_z[to_bin], '\n')

    # correct because hphi is not in (-1,1)
    for j in range(0,order_cosL+1):
      for k in range(0,order_hphi+1):
        for l in range(0,order_cosK+1):
          lbin =  j + (order_cosL+1)*k + (order_cosL+1)*(order_hphi+1)*l
          coeffs[lbin] = corr_z[lbin]/np.pi**k;
          #print(corr_z[lbin], coeffs[lbin])
    return coeffs



"""
SIMON NUMBERS

peff2 = np.array([1.00439, 0, -0.0632034, 0, 0.0497235, 0, 0, 0, 0, 0, -0.0057634, 0, 0.0434837, 0, -0.0509029, 0, 0, 0, 0, 0, 0.000872136, 0, -0.00607617, 0, 0.00682896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0782333, 0, 0.181929, 0, -0.305658, 0, 0, 0, 0, 0, 0.00863651, 0, -0.107195, 0, 0.145518, 0, 0, 0, 0, 0, -0.00207202, 0, 0.0225941, 0, -0.0255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.173056, 0, -0.341704, 0, 0.628401, 0, 0, 0, 0, 0, -0.00219721, 0, 0.160761, 0, -0.260355, 0, 0, 0, 0, 0, 0.000966552, 0, -0.0253226, 0, 0.0340703])

peff = transform_cijk(np.array(result.params), 4, 4, 4)

"""

# MESS UP END
#    If you recall I messed up just before. Whatever I did wrong is correct from
#    the lines that follow up. So, basically, you do not need to touch from now
#    on but before.
# get them back to what user wants
order_cosK, order_cosL, order_hphi = order_cosL, order_cosK, order_hphi

upeff = np.array([p.uvalue for p in result.params.values()])
peff = uncertainty_wrapper(lambda p: transform_cijk(p, order_cosL, order_hphi, order_cosK), upeff)

print(peff)


upeff = np.array([
unc.ufloat(0.998596,0),
unc.ufloat(0,0),
unc.ufloat(-0.0207735,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0.00081787,0),
unc.ufloat(0,0),
unc.ufloat(-0.00153898,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(-0.0598308,0),
unc.ufloat(0,0),
unc.ufloat(-0.0635831,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(-0.000775511,0),
unc.ufloat(0,0),
unc.ufloat(0.0174514,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0.146413,0),
unc.ufloat(0,0),
unc.ufloat(0.124509,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0,0),
unc.ufloat(0.000647654,0),
unc.ufloat(0,0),
unc.ufloat(-0.0197212,0)
])
print([p.n for p in upeff])


exit()












#%%


def angeff_plot_crap(angacc, cosK, cosL, hphi, project=None, order_cosK=4, order_cosL=4, order_hphi=4):
  cosK_l = len(cosK)
  cosL_l = len(cosL)
  hphi_l = len(hphi)
  eff =  ristra.zeros(cosK_l*cosL_l*hphi_l)
  try:
    _angacc = ristra.allocate(np.array(angacc))
  except:
    _angacc = ristra.allocate(np.array([a.n for a in angacc]))
  badjanak.__KERNELS__.crap(eff, _angacc.astype(np.float64),
                            cosK.astype(np.float64), cosL.astype(np.float64), hphi.astype(np.float64),
                            np.int32(cosK_l), np.int32(cosL_l), np.int32(hphi_l),
                            np.int32(order_cosK), np.int32(order_cosL), np.int32(order_hphi),
                            global_size=(cosL_l, hphi_l, cosK_l))
                            #global_size=(1,))
  res = ristra.get(eff).reshape(cosL_l,hphi_l,cosK_l)
  if project==1:
    #return np.trapz(res.T, cosKd.get())[:,5]
    return np.sum(res, (0,1))
    #return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4).T, cosLd.get()), np.pi*hphid.get())
  if project==2:
    #return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4), cosKd.get()), np.pi*hphid.get())
    return np.sum(res, (2,1))
  if project==3:
    #return np.trapz(np.trapz(angeff_plot_crap(peff, cosKd, cosLd, hphid, None, 4, 4, 4).T, cosLd.get()).T, cosKd.get())
    return np.sum(res, (2,0))
  return res







#%% PLOT #######################################################################


"""
peff = np.array([0.984343, 0, -0.0200683, 0, 0, 0, 0.000863937, 0, -0.000526386, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0716313, 0, -0.00418124, 0, 0, 0, -0.00059725, 0, 0.00338139])

peff = np.array([1.00439, 0, -0.0632034, 0, 0.0497235, 0, 0, 0, 0, 0, -0.0057634, 0, 0.0434837, 0, -0.0509029, 0, 0, 0, 0, 0, 0.000872136, 0, -0.00607617, 0, 0.00682896, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0782333, 0, 0.181929, 0, -0.305658, 0, 0, 0, 0, 0, 0.00863651, 0, -0.107195, 0, 0.145518, 0, 0, 0, 0, 0, -0.00207202, 0, 0.0225941, 0, -0.0255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.173056, 0, -0.341704, 0, 0.628401, 0, 0, 0, 0, 0, -0.00219721, 0, 0.160761, 0, -0.260355, 0, 0, 0, 0, 0, 0.000966552, 0, -0.0253226, 0, 0.0340703])
"""


"""## TEST DIEGO THING
from scipy import stats, special

def chi2_probability(chi2, nDoF):
    return 1 - stats.chi2.cdf(chi2,nDoF)

def Prob(p):
    return np.sqrt(2)*special.erfcinv(p)

def bar(x):
  return fcn_d(x).sum()

def foo(x, dir=(1,0)):
  _x = transform_cijk(x, order_cosL, order_hphi, order_cosK)
  eff = angeff_plot_crap(_x, cosKd, cosLd, hphid, None, order_cosL, order_hphi, order_cosK)
  return np.sum(eff*th, dir) / np.sum(th, dir)



dof = 20**3 - sum([p.free for p in result.params.values()])
"""

#%% #################################################################
##########################################################
##########################################
############################
##############



#%% THE PLOTS ##################################################################

mc = data_3d_d.get().reshape(20,20,20)
th = prediction_3d_d.get().reshape(20,20,20)

cosK = 0.5*(np.linspace(-1,1,21)[1:]+np.linspace(-1,1,21)[:-1])
cosL = 0.5*(np.linspace(-1,1,21)[1:]+np.linspace(-1,1,21)[:-1])
hphi = 0.5*(np.linspace(-1,1,21)[1:]+np.linspace(-1,1,21)[:-1])

N = 21
cosKd = ristra.allocate(0.5*(np.linspace(-1,1,N)[1:]+np.linspace(-1,1,N)[:-1]))
cosLd = ristra.allocate(0.5*(np.linspace(-1,1,N)[1:]+np.linspace(-1,1,N)[:-1]))
hphid = ristra.allocate(0.5*(np.linspace(-1,1,N)[1:]+np.linspace(-1,1,N)[:-1]))

#upeff = np.array([p.uvalue for p in result.params.values()])

for k, var in enumerate(['cosK', 'cosL', 'hphi']):
  if var=='cosK':
    proj = (1,2); bounds = 1; tex = r'\mathrm{cos}\,\theta_K'
    dir = (0,1)
    x = np.copy(cosK); xh = np.copy(ristra.get(cosKd))
  elif var=='cosL':
    proj = (0,1); bounds = 1; tex = r'\mathrm{cos}\,\theta_{\mu}'
    dir = (1,2)
    x = np.copy(cosL); xh = np.copy(ristra.get(cosLd))
  elif var=='hphi':
    proj = (2,0); bounds = np.pi; tex = r'\phi_h\, \mathrm{[rad]}'
    dir = (2,0)
    x = np.copy(hphi); xh = np.copy(ristra.get(hphid))

  # project data and theory into variable
  _mc = np.sum(mc, proj); _th = np.sum(th, proj); y = _mc/_th

  # get efficiency, project it and normalize
  peff = transform_cijk(np.array(result.params), order_cosK, order_hphi, order_cosL)
  #peff = np.array(result.params)
  #peff = np.array([p.n for p in upeff])
  print(peff)
  eff = angeff_plot_crap(peff, cosKd, cosLd, hphid, None, order_cosK, order_cosL, order_hphi)
  eff = np.sum(eff*th, dir) / np.sum(th, dir)
  #norm = 1
  norm = 1/(np.trapz(y, x)/np.trapz(eff, xh))
  #norm = 1/np.max(eff)

  # prepare errorbars
  _umc = np.sqrt(_mc); _uth = np.sqrt(_th)
  uy = np.sqrt((1/_th)**2*_umc**2 + (-_mc/_th**2)**2*_uth**2)
  ux = 0.5*(x[1]-x[0])*np.ones_like(x)

  # compute confidence bands
  yunc = uncertainty_wrapper(lambda p: np.mean(th*angeff_plot_crap(p, cosKd, cosLd, hphid, None, order_cosK, order_cosL, order_hphi), dir) / np.mean(th, dir) , uncertainty_wrapper(lambda p: transform_cijk(np.array(p), order_cosK, order_hphi, order_cosL), upeff))
  yl, yh = get_confidence_bands(yunc)#/norm/np.max(y)

  # toy confidence bands (very slow, as you would expect)
  # pars = []; chi2 = []; accs = []
  # for rng in range(0,27*int(1e6)):
  #   rng_pars = np.ascontiguousarray([np.random.normal(p.n, 5*p.s) for p in upeff])
  #   rng_chi2 = bar(rng_pars)
  #   #print(rng_chi2)
  #   if chi2_probability(abs(rng_chi2-8351.725486964235), 27) > 0.317:
  #       print(rng_chi2)
  #       #pars.append(rng_pars)
  #       #chi2.append(rng_chi2)
  #       accs.append(foo(rng_pars, dir))
  # yh = np.max(np.array(accs),0)
  # yl = np.min(np.array(accs),0)
  # shit = np.mean(np.array(accs),0)

  # plot
  fig, ax = plotting.axes_plot()
  ax.errorbar(bounds*x, y/np.max(y), yerr=uy, xerr=bounds*ux, fmt='.k')
  ax.plot(bounds*xh, eff/norm/np.max(y))
  #ax.plot(bounds*xh, shit/norm/np.max(y))
  ax.fill_between(bounds*x, yh/norm/np.max(y), yl/norm/np.max(y), alpha=0.2)
  ax.set_ylim(0.85,1.05)
  ax.set_xlabel(f"${tex}$")
  ax.set_ylabel(rf"$\varepsilon({tex})$ [a.u.]")
  fig.savefig(f"{var}.pdf")
