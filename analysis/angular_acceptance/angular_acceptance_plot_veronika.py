from ipanema.confidence import wrap_unc, get_confidence_bands
from ipanema import initialize, ristra, Parameters

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot

initialize('cuda',1)
import badjanak



# create a n-dimensional mesh
def ndmesh(*args):
   args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])



# wrapper arround ang_eff cuda kernel
#     it can project to 1,2,3 = cosK,cosK,hphi variables
def angeff_plot(angacc, cosK, cosL, hphi, project=None):
  eff = ristra.zeros_like(cosK)
  try:
    _angacc = ristra.allocate(np.array(angacc))
  except:
    _angacc = ristra.allocate(np.array([a.n for a in angacc]))
  badjanak.__KERNELS__.plot_moments(_angacc, eff, cosK, cosL, hphi, global_size=(eff.shape[0],))
  n = round(eff.shape[0]**(1/3))
  res = ristra.get(eff).reshape(n,n,n)
  if project==1:
    return np.sum(res,(1,0))
  if project==2:
    return np.sum(res,(1,2))
  if project==3:
    return np.sum(res,(2,0))
  return res



N = 100 # number of points to plot
cosK = np.linspace(-1,1,N)
cosL = np.linspace(-1,1,N)
hphi = np.linspace(-np.pi,+np.pi,N)

cosKh, cosLh, hphih = ndmesh(cosK, cosL, hphi)
cosKd = ristra.allocate( cosKh.reshape(N**3) )
cosLd = ristra.allocate( cosLh.reshape(N**3) )
hphid = ristra.allocate( hphih.reshape(N**3) )


#%% Run all plots
for var in ['cosK', 'cosL', 'hphi']:
  if var=='cosK':
    proj = 1; bounds = (-1,1); tex = r'\mathrm{cos}\theta_K'; x = cosK
  elif var=='cosL':
    proj = 3; bounds = (-1,1); tex = r'\mathrm{cos}\theta_L'; x = cosL
  elif var=='hphi':
    proj = 2; bounds = (-np.pi,np.pi); tex = r'\phi_h'; x = hphi
  else:
    print('huge error')
  for year in [2015,2016,2017,2018]:
    for trigger in ['biased', 'unbiased']:
      fig, ax = plt.subplots()
      # first plot the baseline and its confidence band (1sigma)
      angacc = Parameters.load(f'output/params/angular_acceptance/{year}/MC_Bs2JpsiPhi_dG0/v0r5_naive_{trigger}.json')
      eff = angeff_plot(angacc, cosKd, cosLd, hphid, proj)
      norm = np.trapz(eff, x)
      langacc = [p.uvalue for p in angacc.values()]
      yunc = wrap_unc(lambda p: angeff_plot(p, cosKd, cosLd, hphid, proj), langacc)
      yl, yh = get_confidence_bands(yunc)
      ax.fill_between(x, yl/norm,yh/norm, alpha=0.5)
      ax.plot(x, eff/norm, label='nominal')
      # then run over all time-binned angular acceptances
      for i in range(1,7):
        angacc = Parameters.load(f'output/params/angular_acceptance/{year}/MC_Bs2JpsiPhi_dG0/v0r5_naiveTime{i}_{trigger}.json')
        eff = angeff_plot(angacc, cosKd, cosLd, hphid, proj)
        eff /= np.trapz(eff, x)
        ax.plot(x,eff, '-.',label=f'$t$ bin = {i}')
      ax.set_xlabel(f'${tex}$')
      ax.legend()
      #ax.set_ylim(0.9,1.25)
      ax.set_title(f"{year} {trigger}")
      fig.savefig(f'output/figures/angular_acceptance/{year}/Bs2JpsiPhi/v0r5_naiveTime{var.title()}_{trigger}.pdf')
