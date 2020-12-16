import ipanema
import uproot
import matplotlib.pyplot as plt
from ipanema import plotting
import numpy as np

from utils.plot import watermark

sidecar = '/scratch17/marcos.romero/sidecar'


#%% mc vs rd
for mode in ['Bd2JpsiKstar', 'Bs2JpsiPhi']:
  fig, axplot, axpull = plotting.axes_plotpull()
  for i, year in enumerate([2015,2016,2017,2018]):
    # load sample
    rd = ipanema.Sample.from_root(f'{sidecar}/{year}/{mode}/v0r5.root')
    mc = ipanema.Sample.from_root(f'{sidecar}/{year}/MC_{mode}/v0r5.root')
    # histogram
    hrd = ipanema.hist(rd.df['log_B_IPCHI2_mva'], bins=np.linspace(-10,10,100),
                       weights=rd.df.eval('sw'), density=True)
    hmc = ipanema.hist(mc.df['log_B_IPCHI2_mva'], bins=hrd.edges,
                       weights=mc.df.eval('sw/gb_weights'), density=True)
    axplot.step(hrd.bins, hrd.counts, where='mid', label=f'{year} data',
                linestyle='-', color=f'C{i}')
    axplot.step(hmc.bins, hmc.counts, where='mid', label=f'{year} MC',
                linestyle='-.', color=f'C{i}')
    axpull.fill_between(hrd.bins,hrd.counts/hmc.counts,1,
                        facecolor=f'C{i}', alpha=0.4)
  axpull.set_xlabel("$ \\mathrm{log}\, \\chi^2_{IP}(B)$")
  axpull.set_ylabel(f"$\\frac{{N(data)}}{{N(MC)}}$")
  axplot.set_ylabel(f"Candidates")
  axpull.set_ylim(-0.5,2.0)
  axpull.set_xlim(-10,10)
  axpull.set_yticks([0.5, 1, 1.5])
  axplot.legend()
  #axplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  watermark(axplot, version=f'$v0r5$', scale=1.05)

  # savefig
  fig.savefig(f'tmp/logIPplots/run2_{mode}_mc_vs_rd.pdf')


#%% rd vs rd control
fig, axplot, axpull = plotting.axes_plotpull()
for i, year in enumerate([2015,2016,2017,2018]):
  # load sample
  rd = ipanema.Sample.from_root(f'{sidecar}/{year}/Bs2JpsiPhi/v0r5.root')
  mc = ipanema.Sample.from_root(f'{sidecar}/{year}/Bd2JpsiKstar/v0r5.root')
  # histogram
  hrd = ipanema.hist(rd.df['log_B_IPCHI2_mva'], bins=np.linspace(-10,10,100),
                     weights=rd.df.eval('sw'), density=True)
  hmc = ipanema.hist(mc.df['log_B_IPCHI2_mva'], bins=hrd.edges,
                     weights=mc.df.eval('sw'), density=True)
  axplot.step(hrd.bins, hrd.counts, where='mid', label=f'{year} data',
              linestyle='-', color=f'C{i}')
  axplot.step(hmc.bins, hmc.counts, where='mid', label=f'{year} MC',
              linestyle='-.', color=f'C{i}')
  axpull.fill_between(hrd.bins,hrd.counts/hmc.counts,1,
                      facecolor=f'C{i}', alpha=0.4)
axpull.set_xlabel("$ \\mathrm{log}\, \\chi^2_{IP}(B)$")
axpull.set_ylabel(f"$\\frac{{N(data)}}{{N(MC)}}$")
axplot.set_ylabel(f"Candidates")
axpull.set_ylim(-0.5,2.0)
axpull.set_xlim(-10,10)
axpull.set_yticks([0.5, 1, 1.5])
axplot.legend()
#axplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
watermark(axplot, version=f'$v0r5$', scale=1.05)

# savefig
fig.savefig(f'tmp/logIPplots/run2_controlrd_vs_rd.pdf')



#%% mc vs mc control
fig, axplot, axpull = plotting.axes_plotpull()
for i, year in enumerate([2015,2016,2017,2018]):
  # load sample
  rd = ipanema.Sample.from_root(f'{sidecar}/{year}/MC_Bs2JpsiPhi_dG0/v0r5.root')
  mc = ipanema.Sample.from_root(f'{sidecar}/{year}/MC_Bd2JpsiKstar/v0r5.root')
  # histogram
  hrd = ipanema.hist(rd.df['log_B_IPCHI2_mva'], bins=np.linspace(-10,10,100),
                     weights=rd.df.eval('sw'), density=True)
  hmc = ipanema.hist(mc.df['log_B_IPCHI2_mva'], bins=hrd.edges,
                     weights=mc.df.eval('sw'), density=True)
  axplot.step(hrd.bins, hrd.counts, where='mid', label=f'{year} Bs MC',
              linestyle='-', color=f'C{i}')
  axplot.step(hmc.bins, hmc.counts, where='mid', label=f'{year} Bd MC',
              linestyle='-.', color=f'C{i}')
  axpull.fill_between(hrd.bins,hrd.counts/hmc.counts,1,
                      facecolor=f'C{i}', alpha=0.4)
axpull.set_xlabel("$ \\mathrm{log}\, \\chi^2_{IP}(B)$")
axpull.set_ylabel(f"$\\frac{{N(data)}}{{N(MC)}}$")
axplot.set_ylabel(f"Candidates")
axpull.set_ylim(-0.5,2.0)
axpull.set_xlim(-10,10)
axpull.set_yticks([0.5, 1, 1.5])
axplot.legend()
#axplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
watermark(axplot, version=f'$v0r5$', scale=1.05)

# savefig
fig.savefig(f'tmp/logIPplots/run2_controlmc_vs_mc.pdf')
