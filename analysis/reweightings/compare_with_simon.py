# The main purpose of this file is to compare old Run2 weightings done by
# Simon with the ones phis-scq produces.
__all__ = []

#%matplotlib inline
import os
import matplotlib.pyplot as plt
import numpy as np

from ipanema import Sample
from ipanema import hist
import ipanema
from ipanema import histogram

# Copy of old tuples is at:
old_path = '/scratch03/marcos.romero/phisRun2/cooked_test_files/'
# and new ones are at:
new_path = '/scratch17/marcos.romero/phis_samples/'

# Create some paths
os.makedirs('output_new/figures/reweightings/2016', exist_ok=True )
os.makedirs('output_new/figures/reweightings/2015', exist_ok=True )



#%% Run shit -------------------------------------------------------------------

# Get all samples
old = {}; new = {}
for y in ['2015','2016']:
  old[y] = {}; t_o_p = os.path.join(old_path,y)
  new[y] = {}; t_n_p = os.path.join(new_path,y)
  for m in ['Bd2JpsiKstar', 'MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi',
            'Bs2JpsiPhi', 'MC_Bd2JpsiKstar']:
    old[y][m] = Sample.from_root(os.path.join(t_o_p,m,'test_kinWeight.root'))
    new[y][m] = Sample.from_root(os.path.join(t_n_p,m,'v0r0.root'))
    os.makedirs('output_new/figures/reweightings/'+y+f'/{m}', exist_ok=True )



for year in  [2015,2016]:
  #%% polWeight
  try:
    fig, axplot, axpull = ipanema.plotting.axes_plotpull()
    o, n = histogram.compare_hist([old[f'{year}'][f'MC_Bs2JpsiPhi'].df['polWeight'],new[f'{year}'][f'MC_Bs2JpsiPhi'].df['polWeight']], bins=4,range=(0,2))
    axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
    axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
    axpull.plot(n.bins,0*n.pulls, color='C2')
    axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
    axpull.set_xlabel('polWeight')
    axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}'][f'MC_Bs2JpsiPhi'].df['polWeight']-new[f'{year}'][f'MC_Bs2JpsiPhi'].df['polWeight']):0.3g}")
    axplot.set_ylabel('Events')
    axplot.legend()
    fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi/v0r0_polWeightComparison.pdf')
  except:
    print(f'Failed in output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi/v0r0_polWeightComparison.pdf')
  try:
    fig, axplot, axpull = ipanema.plotting.axes_plotpull()
    o, n = histogram.compare_hist([old[f'{year}'][f'MC_Bs2JpsiPhi_dG0'].df['polWeight'],new[f'{year}'][f'MC_Bs2JpsiPhi_dG0'].df['polWeight']], bins=4,range=(0,2))
    axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
    axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
    axpull.plot(n.bins,0*n.pulls, color='C2')
    axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
    axpull.set_xlabel('polWeight')
    axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}'][f'MC_Bs2JpsiPhi_dG0'].df['polWeight']-new[f'{year}'][f'MC_Bs2JpsiPhi_dG0'].df['polWeight']):0.3g}")
    axplot.set_ylabel('Events')
    axplot.legend()
    fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi_dG0/v0r0_polWeightComparison.pdf')
  except:
    print(f'Failed in output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi/v0r0_polWeightComparison.pdf')
  try:
    fig, axplot, axpull = ipanema.plotting.axes_plotpull()
    o, n = histogram.compare_hist([old[f'{year}'][f'MC_Bd2JpsiKstar'].df['polWeight'],new[f'{year}'][f'MC_Bd2JpsiKstar'].df['polWeight']], bins=4,range=(0,2))
    axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
    axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
    axpull.plot(n.bins,0*n.pulls, color='C2')
    axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
    axpull.set_xlabel('polWeight')
    axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}'][f'MC_Bd2JpsiKstar'].df['polWeight']-new[f'{year}'][f'MC_Bd2JpsiKstar'].df['polWeight']):0.3g}")
    axplot.set_ylabel('Events')
    axplot.legend()
    fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bd2JpsiKstar/v0r0_polWeightComparison.pdf')
  except:
    print(f'Failed in output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi/v0r0_polWeightComparison.pdf')


  #%% pdfWeight
  fig, axplot, axpull = ipanema.plotting.axes_plotpull()
  o, n = histogram.compare_hist([old[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['pdfWeight'],new[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['pdfWeight']],bins=50,range=(0,2))
  axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
  axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
  axpull.plot(n.bins,0*n.pulls, color='C2')
  axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
  axpull.set_xlabel('pdfWeight')
  axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['pdfWeight']-new[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['pdfWeight']):0.3g}")
  axplot.set_ylabel('Events')
  axplot.legend()
  fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi_dG0/v0r0_pdfWeightComparison.pdf')

  fig, axplot, axpull = ipanema.plotting.axes_plotpull()
  o, n = histogram.compare_hist([old[f'{year}']['MC_Bd2JpsiKstar'].df['pdfWeight'],new[f'{year}']['MC_Bd2JpsiKstar'].df['pdfWeight']],bins=50,range=(0,2))
  axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
  axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
  axpull.plot(n.bins,0*n.pulls, color='C2')
  axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
  axpull.set_xlabel('pdfWeight')
  axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}']['MC_Bd2JpsiKstar'].df['pdfWeight']-new[f'{year}']['MC_Bd2JpsiKstar'].df['pdfWeight']):0.3g}")
  axplot.set_ylabel('Events')
  axplot.legend()
  fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bd2JpsiKstar/v0r0_pdfWeightComparison.pdf')

  #%% kinWeight
  fig, axplot, axpull = ipanema.plotting.axes_plotpull()
  o, n = histogram.compare_hist([old[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['kinWeight'],new[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['kinWeight']],bins=50,range=(0,2))
  axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
  axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
  axpull.plot(n.bins,0*n.pulls, color='C2')
  axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
  axpull.set_xlabel('kinWeight')
  axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['kinWeight']-new[f'{year}']['MC_Bs2JpsiPhi_dG0'].df['kinWeight']):0.3g}")
  axplot.set_ylabel('Events')
  axplot.legend()
  fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bs2JpsiPhi_dG0/v0r0_kinWeightComparison.pdf')

  fig, axplot, axpull = ipanema.plotting.axes_plotpull()
  o, n = histogram.compare_hist([old[f'{year}']['MC_Bd2JpsiKstar'].df['kinWeight'],new[f'{year}']['MC_Bd2JpsiKstar'].df['kinWeight']],bins=50,range=(0,2))
  axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
  axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
  axpull.plot(n.bins,0*n.pulls, color='C2')
  axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
  axpull.set_xlabel('kinWeight')
  axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}']['MC_Bd2JpsiKstar'].df['kinWeight']-new[f'{year}']['MC_Bd2JpsiKstar'].df['kinWeight']):0.3g}")
  axplot.set_ylabel('Events')
  axplot.legend()
  fig.savefig(f'output_new/figures/reweightings/{year}/MC_Bd2JpsiKstar/v0r0_kinWeightComparison.pdf')

  fig, axplot, axpull = ipanema.plotting.axes_plotpull()
  o, n = histogram.compare_hist([old[f'{year}'][f'Bd2JpsiKstar'].df['kinWeight'],new[f'{year}'][f'Bd2JpsiKstar'].df['kinWeight']])
  axplot.plot(o.bins,o.counts,drawstyle='steps-mid',label='Heidelberg')
  axplot.plot(n.bins,n.counts,drawstyle='steps-mid',ls='-.',label='Santiago')
  axpull.plot(n.bins,0*n.pulls, color='C2')
  axpull.fill_between(n.bins,1*n.pulls,0*n.pulls, facecolor='C2')
  axpull.set_xlabel('kinWeight')
  axplot.set_title(f"max(Heidelberg - Santiago) = {np.amax(old[f'{year}'][f'Bd2JpsiKstar'].df['kinWeight']-new[f'{year}'][f'Bd2JpsiKstar'].df['kinWeight']):0.3g}")
  axplot.set_ylabel('Events')
  axplot.legend()
  fig.savefig(f'output_new/figures/reweightings/{year}/Bd2JpsiKstar/v0r0_kinWeightComparison.pdf')
