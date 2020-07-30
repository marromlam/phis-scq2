from ipanema import Parameters, Sample
import os
import hjson
import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes
import numpy as np

binned_vars = hjson.load(open(f'config.json'))['binned_variables']
vars_ranges = hjson.load(open(f'config.json'))['binned_variables_ranges']
binned_vars
var = "sigmat"
version = "v0r5"





pars = [
  Parameters.load("output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_Corrected_unbiased.json")
]
for i in range(1,len(vars_ranges[var])):
  pars.append(
    Parameters.load(f"output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_CorrectedBinsigmat{i}_unbiased.json")
  )









#%%
nterms = 10 # trinagular number of the number of amplitudes T(4) = 10
pcols = 3
prows = (nterms-1)//pcols + (1 if (nterms-1)%pcols else 0)

fig, ax = plt.subplots(prows, pcols, sharex=True, figsize=[prows*4.8,2*4.8])
for i in range(1,nterms):
  ax[(i-1)//pcols ,(i-1)%pcols].fill_between(
    [vars_ranges[var][0], vars_ranges[var][-1]],
    2*[pars[0][f'w{i}'].value+pars[0][f'w{i}'].stdev],2*[pars[0][f'w{i}'].value-pars[0][f'w{i}'].stdev],
    alpha=0.8, label="Base")
  ax[(i-1)//pcols ,(i-1)%pcols].set_ylabel(f"$w_{i}$")
  for bin in range( len(vars_ranges[var])-1 ):
    ax[(i-1)//pcols ,(i-1)%pcols].errorbar(
      (vars_ranges[var][bin]+vars_ranges[var][bin+1])*0.5,
      pars[bin+1][f'w{i}'].value,
      xerr=vars_ranges[var][bin+1]-(vars_ranges[var][bin]+vars_ranges[var][bin+1])*0.5,
      yerr=pars[bin+1][f'w{i}'].stdev, fmt='.', color='k')
  #ax[(i-1)//pcols ,(i-1)%pcols].legend()
watermark(ax[0,0],version=f"${version}$",scale=1.01)
[make_square_axes(ax[ix,iy]) for ix,iy in np.ndindex(ax.shape)]
[tax.set_xlabel(f"${get_var_in_latex(f'{var}')}$") for tax in ax[prows-1,:]]
