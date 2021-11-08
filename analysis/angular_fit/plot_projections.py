from ipanema import wrap_unc, uncertainty_wrapper, get_confidence_bands
from ipanema import initialize, ristra, Parameters, Sample, optimize, IPANEMALIB, ristra
from utils.helpers import  version_guesser, trigger_scissors, cuts_and
from utils.strings import printsec, printsubsec
from ipanema.core.python import ndmesh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot3 as uproot
from scipy.special import lpmv
from scipy.interpolate import interp1d, interpn
import argparse
initialize('opencl',1)
import badjanak
badjanak.get_kernels()
from scipy.special import comb
# from scipy.integrate import romb, simpson
from ipanema import plotting, hist
import uncertainties.unumpy as unp
import uncertainties as unc
from scipy import stats, special
import os
import hjson
import ipanema

import config


import argparse


p = argparse.ArgumentParser(description="dfdf")
p.add_argument('--version')
p.add_argument('--trigger')
p.add_argument('--year')
args = vars(p.parse_args())


VERSION = args['version']
TRIGGER = args['trigger']
YEAR = args['year']

# VERSION = 'v1r0p1@pTB4'
# TRIGGER = 'unbiased'
# YEAR = '2016'

# pars = Parameters.load("analysis/params/generator/2016/MC_Bs2JpsiPhi_dG0.json").valuesdict()
pars = Parameters.load(f"output/params/physics_params/run2/Bs2JpsiPhi/{VERSION}_run2_run2Dual_vgc_amsrd_simul3_amsrd_combined.json").valuesdict()
# df = uproot.open("/scratch46/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5.root")['DecayTree'].pandas.df().query("hlt1b!=0 & time>0.3")
if TRIGGER == 'unbiased':
    df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b==0 & time>0.3")
else:
    df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b!=0 & time>0.3")
dtime = np.array(df['time'])
dcosL = np.array(df['cosL'])
dcosK = np.array(df['cosK'])
dhphi = np.array(df['hphi'])

# timeres = Parameters.load("output/params/time_resolution/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
csp = Parameters.load(f"output/params/csp_factors/{YEAR}/Bs2JpsiPhi/{VERSION}_vgc.json").valuesdict()
# flavor = Parameters.load("output/params/flavor_tagging/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
# angacc = Parameters.load("output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_correctedDual_none_unbiased.json").valuesdict()
angacc = Parameters.load(f"output/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_simul3_amsrd_{TRIGGER}.json").valuesdict()
# timeacc = Parameters.load("output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul3_biased.json")
timeacc = Parameters.load(f"output/params/time_acceptance/{YEAR}/Bd2JpsiKstar/{VERSION}_simul3_{TRIGGER}.json")
knots = np.array(Parameters.build(timeacc, timeacc.find("k.*"))).tolist()
timeacc = Parameters.build(timeacc, timeacc.find("(a|c|b)(A|B)?.*")).valuesdict()
badjanak.config['knots'] = knots
badjanak.config['debug'] = 0
badjanak.config['fast_integral'] = 0
badjanak.get_kernels()


  
def pdf_projector(params, edges, var='time', timeacc=False, angacc=False,
                  return_center=False):
  # create flags for acceptances
  use_timeacc = True if timeacc else False
  use_angacc = True if angacc else False

  # defaults
  cosKLL=-1; cosKUL=1; cosLLL=-1; cosLUL=1; hphiLL=-np.pi; hphiUL=+np.pi;
  tLL=0.3; tUL=15; acc = 1
  _x = 0.5 * (edges[:-1] + edges[1:])

  @np.vectorize
  def prob(pars, cosKLL=-1, cosKUL=-1, cosLLL=-1, cosLUL=-1, hphiLL=-np.pi, 
           hphiUL=-np.pi, tLL=0.3, tUL=15):
    var = np.float64([0.0]*3+[0.3]+[1020.]+[0.0]*5)
    var = ristra.allocate(np.ascontiguousarray(var))
    pdf = ristra.allocate(np.float64([0.0]))
    badjanak.delta_gamma5_mc(var, pdf, **pars, tLL=0.3, tUL=15)
    num = pdf.get()
    badjanak.delta_gamma5_mc(var, pdf, **pars, cosKLL=cosKLL, cosKUL=cosKUL,
                             cosLLL=cosLLL, cosLUL=cosLUL, hphiLL=hphiLL,
                             hphiUL=hphiUL, tLL=tLL, tUL=tUL)
    den = pdf.get()
    return num/den

  @np.vectorize
  def vtimeacc(x):
    if use_timeacc:
      return badjanak.bspline(np.float64([x]), [v for v in timeacc.values()])
    else:
      return 1

  @np.vectorize
  def vangacc(x, proj):
    if use_angacc:
      _x = np.linspace(-1, 1, 300)
      __x, __y, __z = ristra.ndmesh(_x, _x, np.pi*_x)
      __x = ristra.allocate( __x.reshape(len(_x)**3) )
      __y = ristra.allocate( __y.reshape(len(_x)**3) )
      __z = ristra.allocate( __z.reshape(len(_x)**3) )
      _arr = [__x, __y, __z]
      _arr[proj-1] *= x/_arr[proj-1]
      _ans = 1/badjanak.angular_efficiency_weights([v for v in angacc.values()], *_arr, proj)
      return np.mean(_ans)
    else:
      return 1
  
  if var == 'time':
    tLL, tUL = edges[:-1], edges[1:]
    acc = vtimeacc(_x)
  elif var == 'cosL':
    cosLLL, cosLUL = edges[:-1], edges[1:]
    acc = vangacc(_x, 1)
  elif var == 'cosK':
    cosKLL, cosKUL = edges[:-1], edges[1:]
    acc = vangacc(_x, 2)
  elif var == 'hphi':
    hphiLL, hphiUL = edges[:-1], edges[1:]
    acc = vangacc(_x, 3)
  else:
    raise ValueError(f"The pdf is not {var} dependent")

  _pdf = acc * prob(params, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL, tLL, tUL)
  _pdf /= np.trapz(_pdf, _x)
  if return_center:
    return _pdf, _x
  return _pdf


fig, axplot, axpull = plotting.axes_plotpull()

hvar = hist(dtime, bins=60, weights=df['sWeight'])
axplot.errorbar(hvar.bins, hvar.counts,
                yerr=[hvar.errh, hvar.errl],
                xerr=2*[hvar.edges[1:]-hvar.bins], fmt='.k')

# without acceptances
# var = np.linspace(0.3, 15, 1000)
# pdfvar, var = pdf_projector({**pars, **csp}, var, 'time', return_center=True)
# axplot.plot(var, hvar.norm * pdfvar, label='without acceptances', color='C2')

# with acceptances
var = np.linspace(0.3, 15, 1000)
pdfvar, var = pdf_projector(pars, var, 'time', timeacc=timeacc, return_center=True)
axplot.plot(var, hvar.norm * pdfvar, label=f'{VERSION}-{YEAR}-{TRIGGER}', color='C0')
axpull.fill_between(hvar.bins,
                    ipanema.histogram.pull_pdf(var, hvar.norm*pdfvar, hvar.bins,
                                               hvar.counts, hvar.errl,
                                               hvar.errh),
                    0, facecolor="C0", alpha=0.5)

# axpull.fill_between(hdata.bins,
#                     ipanema.histogram.pull_pdf(x, y, hdata.bins,
#                                                hdata.counts, hdata.errl,
#                                                hdata.errh),
#                     0, facecolor="C0", alpha=0.5)

axpull.set_xlabel(r'$t$ [ps]')
# axpull.set_ylim(-6.5, 6.5)
axpull.set_yticks([-5, 0, 5])
axplot.set_ylabel(rf"Weighted candidates")
axplot.legend(loc="upper right")
fig.savefig(f"{VERSION}_{YEAR}_{TRIGGER}.pdf")

exit()

plt.close()
hvar = hist(dcosK, bins=60)
plt.errorbar(hvar.bins, hvar.counts, yerr=np.sqrt(hvar.counts), fmt='.')
var = np.linspace(-1, 1, 1000)
pdfvar, var = pdf_projector(pars, var, 'cosK', return_center=True)
plt.plot(var, hvar.norm * pdfvar)
pdfvar, var = pdf_projector(pars, var, 'cosK', angacc=angacc, return_center=True)
var = np.linspace(-1, 1, 1000)
plt.plot(var, hvar.norm * pdfvar)
plt.show()


plt.close()
hvar = hist(dcosL, bins=60)
plt.errorbar(hvar.bins, hvar.counts, yerr=np.sqrt(hvar.counts), fmt='.')
var = np.linspace(-1, 1, 1000)
pdfvar, var = pdf_projector(pars, var, 'cosL', return_center=True)
plt.plot(var, hvar.norm * pdfvar)
var = np.linspace(-1, 1, 1000)
pdfvar, var = pdf_projector(pars, var, 'cosL', angacc=angacc, return_center=True)
plt.plot(var, hvar.norm * pdfvar)
plt.show()


