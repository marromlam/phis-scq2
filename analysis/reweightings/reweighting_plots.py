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
  parser.add_argument('--original',
                      help='File to correct')
  parser.add_argument('--target',
                      help='File to correct')
  parser.add_argument('--year',
                      help='File to reweight to')
  parser.add_argument('--mode', default='DecayTree',
                      help='Name of the target tree')
  parser.add_argument('--version', default='DecayTree',
                      help='Name of the target tree')
  parser.add_argument('--branch', default='DecayTree',
                      help='Name of the target tree')
  parser.add_argument('--sweighted',
                      help='File to store the ntuple with weights')
  parser.add_argument('--kinweighted',
                      help='File to store the ntuple with weights')
  return parser

args = vars( argument_parser().parse_args() )

#%% Run shit -------------------------------------------------------------------

version = args['version']
branch = args['branch']
original_path = args['original']
target_path = args['target']
year = args['year']
mode = args['mode']
sweighted = args['sweighted']
kinweighted = args['kinweighted']

# sample = {}
# for m in ['Bd2JpsiKstar','MC_Bs2JpsiPhi_dG0','Bs2JpsiPhi','MC_Bd2JpsiKstar']:
#   # sample[m] = Sample.from_root(
#   #               os.path.join(path,f'{year}',m,f'{version}_kinWeight.root')
#   #             )
#   sample[m] = Sample.from_root(
#                 os.path.join(path,f'{year}',m,f'{version}.root')
#               )
#   os.makedirs(f'{fig_path}/{year}/{m}', exist_ok=True)

print(mode)
ranges_dict = dict(
B_PT = (70,0,4e4),
B_P = (70,0,8e5),
X_M = (70,840,960) if 'Bd2JpsiKstar' in mode else (70,990,1050),
)

def mode_tex(mode, mod=None, verbose=True):
  fmode = mode
  if mode.startswith('MC_') and mod=='comp':
    if mode in ('MC_BsJpsiPhi','MC_BsJpsiPhi_dG0','MC_Bs2JpsiKK_Swave'):
      fmode = 'Bs2JpsiPhi'
    elif mode in ('MC_BdJpsiKstar'):
      fmode = 'Bd2JpsiKstar'
  if mode=='Bd2JpsiKstar' and mod=='comp':
    fmode = 'Bs2JpsiPhi'
  if verbose:
    print(f'{mode}@{mod} => {fmode}')
  tex_str = ''
  # Particle in latex form
  if 'Bd2JpsiKstar' in fmode:
    tex_str += 'B_d'
  elif 'Bs2JpsiPhi' in fmode:
    tex_str += 'B_s^0'
  elif 'Bs2JpsiKK' in fmode:
    tex_str += 'B_s^0'
  # Add sim, toy or data info
  tex_str += r'\,\mathrm{'
  if 'MC' in fmode:
    tex_str += 'sim'
  elif 'TOY' in fmode:
    tex_str += 'toy'
  else:
    tex_str += 'data'
  tex_str += '}'
  # Add other info
  if 'dG0' in fmode:
    tex_str += r' \mathrm{w } \Delta \Gamma = 0'
  if 'Swave' in fmode:
    tex_str += r' \mathrm{w S-wave}'
  print(tex_str)
  return tex_str

modes_tex_dict = dict(
Bs2JpsiPhi = '$B_s^0$ data',
Bd2JpsiKstar = '$B_d$ data',
MC_Bs2JpsiPhi = '$B_s^0$ sim',
MC_Bd2JpsiKstar = '$B_d$ sim',
)
branches_tex_dict = dict(
B_PT = r'$p_T (B) \, [\mathrm{GeV}/c]$',
B_P = r'$p (B) \, [\mathrm{GeV}/c]$',
X_M = '$m (K^+K^-) \, [\mathrm{GeV}/c^2]$',
)
range = ranges_dict[branch]

original = Sample.from_root(f"{original_path}")
target   = Sample.from_root(f"{target_path}")


def watermark(ax, version='final', tag='', size=(20,8.25)):
  if version == 'final':
    version = 'LHC$b$'
    tag = ' COLLABORATION'
    size = [20,5.8]
  ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.03, ax.get_ylim()[1]*0.95,
          f'{version}', fontsize=size[0], color='black',
          ha='left', va='top', alpha=1.0)
  ax.text(ax.get_xlim()[1]*0.025, ax.get_ylim()[1]*0.85,
          f'{tag}', fontsize=size[1], color='black',
          ha='left', va='top', alpha=1.0)


#%% Background-subtracted sample - not using kinWeight -------------------------

weight = 'sWeight'
x,y = histogram.compare_hist(
        data = [target.df[f'{branch}'], original.df[f'{branch}']],
        weights=[target.df.eval(weight), original.df.eval(weight)],
        bins=range[0], range=range[1:], density=True)

fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,
                    step="mid",color='k',alpha=0.2,
                    label=f"${mode_tex(mode,'comp')}$")
axplot.fill_between(y.bins,y.counts,
                    step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
                    label=f"${mode_tex(mode)}$")
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel(branches_tex_dict[branch])
axpull.set_ylabel(f"$\\frac{{N( {mode_tex(mode,'comp')} )}}{{N( {mode_tex(mode)} )}}$")
axpull.set_ylim(-1,3)
axplot.set_ylim(0,axplot.get_ylim()[1]*1.2)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
watermark(axplot, version=f'\\textsf{{{version}}}')
fig.savefig(f'{sweighted}')



#%% Background-subtracted sample - using kinWeight -----------------------------

weight = 'sWeight*kinWeight'
x,y = histogram.compare_hist(
        data = [target.df[f'{branch}'], original.df[f'{branch}']],
        weights=[target.df.eval('sWeight'), original.df.eval(weight)],
        bins=range[0], range=range[1:], density=True)

fig, axplot, axpull = ipanema.plotting.axes_plotpull()
axplot.fill_between(x.bins,x.counts,
                    step="mid",color='k',alpha=0.2,
                    label=f"${mode_tex(mode,'comp')}$")
axplot.fill_between(y.bins,y.counts,
                    step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
                    label=f"${mode_tex(mode)}$")
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel(branches_tex_dict[branch])
axpull.set_ylabel(f"$\\frac{{N( {mode_tex(mode,'comp')} )}}{{N( {mode_tex(mode)} )}}$")
axpull.set_ylim(-1,3)
#axplot.set_ylim(0,axplot.get_ylim()[1]*1.2)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
watermark(axplot, version=f'\\textsf{{{version}}}')
fig.savefig(f'{kinweighted}')


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
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ data', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d) \,\mathrm{data}}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{version}_B_PT_kinWeight.pdf')

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
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ data', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$p (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_d \,\mathrm{data})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/Bd2JpsiKstar/{version}_B_P_kinWeight.pdf')

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
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s^0$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_B_PT_kinWeight.pdf')

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
                    facecolor='none', edgecolor='C0', label='$B_s^0$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_s^0$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_s^0 \,\mathrm{data})}{N(B_s^0 \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bs2JpsiPhi_dG0/{version}_X_M_kinWeight.pdf')

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
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,x.counts/y.counts,1)
axpull.set_xlabel('$p_T (B) \, [\mathrm{GeV}/c]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_B_PT_kinWeight.pdf')

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
                    facecolor='none', edgecolor='C0', label='$B_d$ data', alpha=1.0, hatch='//' )
axplot.fill_between(y.bins,y.counts,step='mid',
                    facecolor='none', edgecolor='r', label='$B_d$ MC', alpha=1.0, hatch='\\\\')
axpull.fill_between(y.bins,np.nan_to_num(x.counts/y.counts),1)
axpull.set_xlabel('$m (K^+K^-) \, [\mathrm{GeV}/c^2]$')
axpull.set_ylabel(r'$\frac{N(B_d \,\mathrm{data})}{N(B_d \,\mathrm{MC})}$')
axpull.set_ylim(-1,3)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
fig.savefig(f'{fig_path}/{year}/MC_Bd2JpsiKstar/{version}_X_M_kinWeight.pdf')
