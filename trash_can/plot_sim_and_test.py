import ipanema
import uproot
import os
import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex



sim09f = ipanema.Sample.from_root('MC_Bs2JpsiPhi_2017_str29r2_sim09f_dst_GB_weighted.root')
sim09g = ipanema.Sample.from_root('MC_Bs2JpsiPhi_2017_str29r2_sim09g_dst_GB_weighted.root')
rd2017 = ipanema.Sample.from_root('/scratch17/marcos.romero/phis_samples/2017/Bs2JpsiPhi/v0r5.root')
rd2017.df.eval("gb_weights=time/time",inplace=True)

vf = sim09f.df[['B_PT','B_P','X_M','hplus_P','hplus_PT','hminus_P','hminus_PT','gb_weights']]
vg = sim09g.df[['B_PT','B_P','X_M','hplus_P','hplus_PT','hminus_P','hminus_PT','gb_weights']]
rd = rd2017.df[['B_PT','B_P','X_M','hplus_P','hplus_PT','hminus_P','hminus_PT','gb_weights']]




#%% B_P
hf,hg = ipanema.histogram.compare_hist([vf['B_P'].values,vg['B_P'].values], range=get_range('B_P'), bins=100, density=True)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('B_P')}$")
plt.ylabel("Unweighted candidates")
plt.legend()
#%% B_PT
hf,hg = ipanema.histogram.compare_hist([vf['B_PT'].values,vg['B_PT'].values], range=get_range('B_PT'), bins=100, density=True)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('B_PT')}$")
plt.ylabel("Unweighted candidates")
plt.legend()
#%% X_M
hf,hg = ipanema.histogram.compare_hist([vf['X_M'].values,vg['X_M'].values], range=get_range('X_M'), bins=100, density=True)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('X_M')}$")
plt.ylabel("Unweighted candidates")
plt.legend()
#%% hplus_P
hf,hg = ipanema.histogram.compare_hist([vf['hplus_P'].values,vg['hplus_P'].values],  weights=[vf['gb_weights'].values,vg['gb_weights'].values], range=(0,20000), bins=100, density=True)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('hplus_P')}$")
plt.ylabel("GB-weighted candidates")
plt.legend()
#%% hplus_PT
hd,hf = ipanema.histogram.compare_hist([rd['hplus_PT'].values,vf['hplus_PT'].values], weights=[rd['gb_weights'].values,vf['gb_weights'].values], range=(0,2000), bins=100, density=True)
hd,hg = ipanema.histogram.compare_hist([rd['hplus_PT'].values,vg['hplus_PT'].values], weights=[rd['gb_weights'].values,vg['gb_weights'].values], range=(0,2000), bins=100, density=True)
plt.fill_between(hg.bins,hg.counts,label='data',color='k',alpha=0.3)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('hplus_PT')}$")
plt.ylabel("GB-weighted candidates")
plt.legend()
#%% hminus_PT
hf,hg = ipanema.histogram.compare_hist([vf['hminus_P'].values,vg['hminus_P'].values], weights=[vf['gb_weights'].values,vg['gb_weights'].values], range=(0,20000), bins=100, density=True)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('hminus_P')}$")
plt.ylabel("GB-weighted candidates")
plt.legend()
#%% hminus_PT
hd,hf = ipanema.histogram.compare_hist([rd['hminus_PT'].values,vf['hminus_PT'].values], weights=[rd['gb_weights'].values,vf['gb_weights'].values], range=(0,2000), bins=100, density=True)
hd,hg = ipanema.histogram.compare_hist([rd['hminus_PT'].values,vg['hminus_PT'].values], weights=[rd['gb_weights'].values,vg['gb_weights'].values], range=(0,2000), bins=100, density=True)
plt.fill_between(hg.bins,hg.counts,label='data',color='k',alpha=0.3)
plt.step(hg.bins,hg.counts,label='Sim09g')
plt.step(hf.bins,hf.counts,label='Sim09f')
plt.xlabel(f"${get_var_in_latex('hminus_PT')}$")
plt.ylabel("GB-weighted candidates")
plt.legend()
