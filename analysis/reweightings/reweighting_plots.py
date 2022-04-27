# plot kinweights
__all__ = []

import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from ipanema import Sample
# from ipanema import hist
import ipanema
import complot
# from ipanema import histogram
from utils.helpers import trigger_scissors

from utils.plot import mode2tex, get_range, watermark, mode_tex, get_var_in_latex


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--original', help='File to correct')
    p.add_argument('--target', help='File to correct')
    p.add_argument('--output', help='File to store the ntuple with weights')
    p.add_argument('--year', help='File to reweight to')
    p.add_argument('--mode', help='Name of the target tree')
    p.add_argument('--version', help='Name of the target tree')
    p.add_argument('--branch', help='Name of the target tree')
    p.add_argument('--trigger', help='Name of the target tree')
    args = vars(p.parse_args())
  
    version = args['version']
    branch = args['branch']
    original_path = args['original']
    target_path = args['target']
    year = args['year']
    mode = args['mode']
    trigger = args['trigger']
    output = args['output']
  
    trig_cut = ''
    if trigger != 'combined':
      trig_cut = trigger_scissors(trigger)
  
    odf = Sample.from_root(f"{original_path}", cuts=trig_cut).df
    tdf = Sample.from_root(f"{target_path}", cuts=trig_cut).df
    otexmode = mode_tex(mode, False, version)
    ttexmode = mode_tex(mode, 'cdata', version)
    orang = get_range(branch, mode, False, version)
    trang = get_range(branch, mode, 'cdata', version)
  
    if mode == 'Bd2JpsiKstar':
      ncorr = ['sWeight', 'sWeight']
      wcorr = ['sWeight*kbsWeight', 'sWeight']
    elif mode == 'MC_Bd2JpsiKstar':
      ncorr = ['sWeight', 'sWeight*kbsWeight']
      wcorr = ['sWeight*kbsWeight*pdfWeight*polWeight', 'sWeight*kbsWeight']
      # ncorr[0] = f'{ncorr[0]}*gb_weights'
      # wcorr[0] = f'{wcorr[0]}*gb_weights'
    elif mode == 'MC_Bs2JpsiPhi':
      ncorr = ['sWeight', 'sWeight']
      wcorr = ['sWeight*kbsWeight*dg0Weight*pdfWeight*polWeight', 'sWeight']
    elif mode == 'MC_Bs2JpsiPhi_dG0':
      ncorr = ['sWeight', 'sWeight']
      wcorr = ['sWeight*kbsWeight*pdfWeight*polWeight', 'sWeight']
      # ncorr = [f'{corr}/gb_weights' for corr in ncorr]
      # wcorr = [f'{corr}/gb_weights' for corr in wcorr]
    elif mode == 'MC_Bu2JpsiKplus':
      ncorr = ['sWeight', 'sWeight']
      wcorr = ['sWeight*kbsWeight*polWeight', 'sWeight']
    else:
      ncorr = ['sWeight', 'sWeight']
      wcorr = ['sWeight', 'sWeight']
      print('to be developed')
  
    print("Original weights:", ncorr[0], wcorr[0])
    print("Target weights:", ncorr[1], wcorr[1])
  
    # Background-subtracted sample - not using kbsWeight 
    xncorr, yncorr, npull = complot.compare_hist(tdf[branch], odf[branch],
                                      tdf.eval(ncorr[1]), odf.eval(ncorr[0]),
                                      bins=60, range=trang,
                                      density=True)
    xwcorr, ywcorr, wpull = complot.compare_hist(tdf[branch], odf[branch],
                                      tdf.eval(wcorr[1]), odf.eval(wcorr[0]),
                                      bins=60, range=orang,
                                      density=True)
  
    fig, axplot, axpull = complot.axes_plotpull()
    axplot.fill_between(xncorr.bins, xncorr.counts,
                        step="mid", facecolor='k', alpha=0.2,
                        label=f"${ttexmode}$")
    axplot.fill_between(yncorr.bins,yncorr.counts,
                        step="mid", facecolor='none', edgecolor='C2', #hatch='xxx',
                        label=f"${otexmode}$ sWeighted")
    axplot.fill_between(ywcorr.bins,ywcorr.counts,
                        step="mid", facecolor='none', edgecolor='C0', #hatch='xxx',
                        label=f"${otexmode}$ reweighted")
  
    # axpull.fill_between(yncorr.bins, xncorr.counts/yncorr.counts, 1, facecolor='C2')
    # axpull.fill_between(ywcorr.bins, xncorr.counts/ywcorr.counts, 1, facecolor='C0')
    axpull.fill_between(yncorr.bins, npull, 0, facecolor='C2')
    axpull.fill_between(ywcorr.bins, wpull, 0, facecolor='C0')
  
    axpull.set_xlabel(rf"${get_var_in_latex(branch)}$")
    axpull.set_ylabel(f"$\\frac{{N({ttexmode})}}{{N({otexmode})}}$")
    if branch in ('time'):
      axplot.set_yscale('log')
      axplot.set_ylim(0, axplot.get_ylim()[1]*20)
    else:
      axplot.set_ylim(0, axplot.get_ylim()[1]*1.2)
    axpull.set_ylim(-6, 6)
    axpull.set_yticks([-5, 0, 5])
    axplot.legend()
    watermark(axplot, version=f"${args['version']}$", scale=1.25)
    fig.savefig(args['output'])
  
  
    # #%% Background-subtracted sample - using kbsWeight -----------------------------
    #
    # weight = 'sWeight*kbsWeight'
    # x,y = histogram.compare_hist(
    #   data = [target.df[f'{branch}'], original.df[f'{branch}']],
    #   weights=[target.df.eval('sWeight'), original.df.eval(weight)],
    #   bins=range[0], range=range[1:], density=True)
    #
    # fig, axplot, axpull = ipanema.plotting.axes_plotpull()
    # axplot.fill_between(x.bins,x.counts,
    #                     step="mid",color='k',alpha=0.2,
    #                     label=f"${mode_tex(mode,'comp')}$")
    # axplot.fill_between(y.bins,y.counts,
    #                     step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
    #                     label=f"${mode_tex(mode)}$")
    # axpull.fill_between(y.bins,x.counts/y.counts,1)
    # axpull.set_xlabel(branches_tex_dict[branch])
    # axpull.set_ylabel(f"$\\frac{{N( {mode_tex(mode,'comp')} )}}{{N( {mode_tex(mode)} )}}$")
    # axpull.set_ylim(-1,3)
    # #axplot.set_ylim(0,axplot.get_ylim()[1]*1.2)
    # axpull.set_yticks([-0.5, 1, 2.5])
    # axplot.legend()
    # watermark(axplot, version=f'\\textsf{{{version}}}')
    # fig.savefig(f'{kbsweighted}')
  
  
exit()













# Bs P, Bd P
x, y = histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['B_P'],
   sample['Bd2JpsiKstar'].df['B_P']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(0,8e5), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ data', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$p (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{version}_B_P.pdf')

# Bs PT, Bs_MC PT
x, y = histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['B_PT'],
   sample['MC_Bs2JpsiPhi_dG0'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_B_PT.pdf')

# Bs X_M, Bs_MC XM
x, y = histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['X_M'],
   sample['MC_Bs2JpsiPhi_dG0'].df['X_M']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']],
  bins=70, range=(990,1050), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_X_M.pdf')

# Bd PT Bd_MC PT
x, y = histogram.compare_hist(
  [sample['Bd2JpsiKstar'].df['B_PT'],
   sample['MC_Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_B_PT.pdf')

# Bd X_M, Bd_MC XM
x, y= histogram.compare_hist(
  [sample['Bd2JpsiKstar'].df['X_M'],
   sample['MC_Bd2JpsiKstar'].df['X_M']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']],
  bins=70, range=(840,960), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_X_M.pdf')



#%% Background-subtracted sample - using kbsWeight ----------------------------

# Bs PT, Bd PT
x, y = histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['B_PT'],
   sample['Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']*
           sample['Bd2JpsiKstar'].df['kbsWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ data', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d) \,\mathrm{data}}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{version}_B_PT_kbsWeight.pdf')

# Bs P, Bd P
x, y= histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['B_P'],
   sample['Bd2JpsiKstar'].df['B_P']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['Bd2JpsiKstar'].df['sWeight']*
           sample['Bd2JpsiKstar'].df['kbsWeight']],
  bins=70, range=(0,8e5), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ data', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$p (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{version}_B_P_kbsWeight.pdf')

# Bs PT, Bs_MC PT
x, y = histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['B_PT'],
   sample['MC_Bs2JpsiPhi_dG0'].df['B_PT']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']*
           sample['MC_Bs2JpsiPhi_dG0'].df['kbsWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s^0$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_B_PT_kbsWeight.pdf')

# Bs X_M, Bs_MC X_M
x, y= histogram.compare_hist(
  [sample['Bs2JpsiPhi'].df['X_M'],
   sample['MC_Bs2JpsiPhi_dG0'].df['X_M']],
  weights=[sample['Bs2JpsiPhi'].df['sWeight'],
           sample['MC_Bs2JpsiPhi_dG0'].df['sWeight']*
           sample['MC_Bs2JpsiPhi_dG0'].df['kbsWeight']],
  bins=70, range=(990,1050), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s^0$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_X_M_kbsWeight.pdf')

# Bd PT, Bd_MC PT
x, y = histogram.compare_hist(
  [sample['Bd2JpsiKstar'].df['B_PT'],
   sample['MC_Bd2JpsiKstar'].df['B_PT']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']*
           sample['MC_Bd2JpsiKstar'].df['kbsWeight']],
  bins=70, range=(0,4e4), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_B_PT_kbsWeight.pdf')

# Bd X_M, Bd_MC X_M
x, y= histogram.compare_hist(
  [sample['Bd2JpsiKstar'].df['X_M'],
   sample['MC_Bd2JpsiKstar'].df['X_M']],
  weights=[sample['Bd2JpsiKstar'].df['sWeight'],
           sample['MC_Bd2JpsiKstar'].df['sWeight']*
           sample['MC_Bd2JpsiKstar'].df['kbsWeight']],
  bins=70, range=(840,960), density=True)
fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,step='mid',
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_X_M_kbsWeight.pdf')
