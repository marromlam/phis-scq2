# plot kinweights


#%matplotlib inline
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from ipanema import Sample
from ipanema import hist
import ipanema
from ipanema import histogram

def alpha(x, y=1):
  z = x/y
  return z*( (z.sum())/((z**2).sum()) )

def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--samples-path',
                      help='File to correct')
  parser.add_argument('--year',
                      help='File to reweight to')
  parser.add_argument('--flag', default='DecayTree',
                      help='Name of the target tree')
  parser.add_argument('--figures-path',
                      help='File to store the ntuple with weights')
  return parser

args = vars( argument_parser().parse_args() )

#%% Run shit -------------------------------------------------------------------

flag     = args['flag']
year     = args['year']
fig_path = args['figures_path']
path     = args['samples_path']

sample = {}
for m in ['Bd2JpsiKstar','MC_Bs2JpsiPhi_dG0','Bs2JpsiPhi','MC_Bd2JpsiKstar']:
  # sample[m] = Sample.from_root(
  #               os.path.join(path,f'{year}',m,f'{flag}_kinWeight.root')
  #             )
  sample[m] = Sample.from_root(
                os.path.join(path,f'{year}',m,f'{flag}.root')
              )
  os.makedirs(f'{fig_path}/{year}/{m}', exist_ok=True)

# Compute sWeights
if f'{flag}' == 'test': # MODIFY THIS
  sample['Bd2JpsiKstar'].df.eval('sWeight = @alpha(sw)',
                                          inplace=True)
  sample['Bs2JpsiPhi'].df.eval('sWeight = @alpha(sw)',
                                        inplace=True)
  sample['MC_Bs2JpsiPhi_dG0'].df.eval('sWeight = @alpha(sw,gb_weights)',
                                               inplace=True)
  sample['MC_Bd2JpsiKstar'].df.eval('sWeight = @alpha(sw)',
                                             inplace=True)



#%% Background-subtracted sample - not using kinWeight ------------------------

# Bs PT, Bd PT
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_PT'],
           sample['Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_d$ data', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel(r'$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{flag}_B_PT.pdf')

# Bs P, Bd P
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_P'],
           sample['Bd2JpsiKstar'].df['B_P']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(0,8e5), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_d$ data', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$p (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{flag}_B_P.pdf')

# Bs PT, Bs_MC PT
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_PT'],
           sample['MC_Bs2JpsiPhi_dG0'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_s$ MC', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{flag}_B_PT.pdf')

# Bs X_M, Bs_MC XM
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['X_M'],
           sample['MC_Bs2JpsiPhi_dG0'].df['X_M']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']],
  bins=70, range=(990,1050), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_s$ MC', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{flag}_X_M.pdf')

# Bd PT Bd_MC PT
x, y = histogram.compare_hist(
          [sample['Bd2JpsiKstar'].df['B_PT'],
           sample['MC_Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_d$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_d$ MC', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{flag}_B_PT.pdf')

# Bd X_M, Bd_MC XM
x, y= histogram.compare_hist(
          [sample['Bd2JpsiKstar'].df['X_M'],
           sample['MC_Bd2JpsiKstar'].df['X_M']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(840,960), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_d$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_d$ MC', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{flag}_X_M.pdf')



#%% Background-subtracted sample - using kinWeight ----------------------------

# Bs PT, Bd PT
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_PT'],
           sample['Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']*
           sample['Bd2JpsiKstar'].df['kinWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_d$ data', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d) \,\mathrm{data}}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{flag}_B_PT_kinWeight.pdf')

# Bs P, Bd P
x, y= histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_P'],
           sample['Bd2JpsiKstar'].df['B_P']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']*
           sample['Bd2JpsiKstar'].df['kinWeight']],
  bins=70, range=(0,8e5), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_d$ data', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$p (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{flag}_B_P_kinWeight.pdf')

# Bs PT, Bs_MC PT
x, y = histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['B_PT'],
           sample['MC_Bs2JpsiPhi_dG0'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']*
           sample['MC_Bs2JpsiPhi_dG0'].df['kinWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_s^0$ MC', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{flag}_B_PT_kinWeight.pdf')

# Bs X_M, Bs_MC X_M
x, y= histogram.compare_hist(
          [sample['Bs2JpsiPhi'].df['X_M'],
           sample['MC_Bs2JpsiPhi_dG0'].df['X_M']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']*
           sample['MC_Bs2JpsiPhi_dG0'].df['kinWeight']],
  bins=70, range=(990,1050), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_s^0$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_s^0$ MC', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{flag}_X_M_kinWeight.pdf')

# Bd PT, Bd_MC PT
x, y = histogram.compare_hist(
          [sample['Bd2JpsiKstar'].df['B_PT'],
           sample['MC_Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']*
           sample['MC_Bd2JpsiKstar'].df['kinWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_d$ data' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C9',label='$B_d$ MC', alpha=0.75)
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{flag}_B_PT_kinWeight.pdf')

# Bd X_M, Bd_MC X_M
x, y= histogram.compare_hist(
          [sample['Bd2JpsiKstar'].df['X_M'],
           sample['MC_Bd2JpsiKstar'].df['X_M']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']*
           sample['MC_Bd2JpsiKstar'].df['kinWeight']],
  bins=70, range=(840,960), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='C0',label='$B_d$ data', alpha=0.75 )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='C2',label='$B_d$ MC', alpha=0.75)
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{flag}_X_M_kinWeight.pdf')
