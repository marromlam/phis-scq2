import matplotlib.pyplot as plt
import ipanema
from utils.plot import get_range, watermark, mode_tex

version = 'v0r5'

srd = ipanema.Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/Bs2JpsiPhi/{version}.root')
smc = ipanema.Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/{version}.root')
kin = ipanema.Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/{version}_angWeight.root')
kkp = ipanema.Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/MC_Bs2JpsiPhi/{version}_angWeight.root',treename='2015&2016&2017&2018')

niter = len(kkp.find('kkp.*')) # get last iteration number
strvar = 'B_PT'
rdvar = srd.df.eval(strvar)
mcvar = smc.df.eval(strvar)
rdwei = srd.df.eval('sWeight')
mckin = smc.df.eval('sWeight')*kin.df.eval('kinWeight')
mckkp = mckin*kkp.df.eval(f'pdfWeight{niter}*kkpWeight{niter}')

#%% ---
hrd, hmckin = ipanema.histogram.compare_hist([rdvar,mcvar], weights=[rdwei,mckin], density=True, range=get_range(strvar))
hrd, hmckkp = ipanema.histogram.compare_hist([rdvar,mcvar], weights=[rdwei,mckkp], density=True, range=get_range(strvar))

fig, axplot, axpull = ipanema.plotting.axes_plotpull();
axplot.fill_between(hrd.cmbins,hrd.counts,step="mid",color='k',alpha=0.2,
                    label=f"${mode_tex('Bs2JpsiPhi')}$")
axplot.fill_between(hmckkp.cmbins,hmckkp.counts,step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
                    label=f"${mode_tex('MC_Bs2JpsiPhi')}$")
# axplot.fill_between(hmc.cmbins,hmckkp.counts,step="mid",facecolor='none',edgecolor='C3',hatch='///',
#                     label=f"${mode_tex('MC_Bs2JpsiPhi')}$ kkp")
#axpull.fill_between(hrd.bins,hmckin.counts/hrd.counts,1,color='C0')
axpull.fill_between(hrd.bins,hmckkp.counts/hrd.counts,1,color='C0')
#axpull.set_xlabel(branches_tex_dict[branch])
axpull.set_ylabel(f"$\\frac{{N( {mode_tex('MC_Bs2JpsiPhi')} )}}{{N( {mode_tex('Bs2JpsiPhi')} )}}$")
axpull.set_ylim(-0.8,3.2)
#axplot.set_ylim(0,axplot.get_ylim()[1]*1.2)
axpull.set_yticks([-0.5, 1, 2.5])
axplot.legend()
watermark(axplot,version=f"${version}$",scale=1.25)


#%% ---


import matplotlib.pyplot as plt
import ipanema
from utils.plot import get_range, watermark, mode_tex

version = 'v0r5'
import numpy as np
from uncertainties import unumpy as unp
input_params = f'output_new/params/angular_acceptance/2017/Bs2JpsiPhi/{version}_corrected_biased.json'
w = ipanema.Parameters.load(input_params).uvaluesdict()
i=1
xp = np.array([0.3,15])
ypl = np.array(2*[w[f'w{i}'].n-w[f'w{i}'].s])
ypu = np.array(2*[w[f'w{i}'].n+w[f'w{i}'].s])


xb = np.array([0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00, 15])
yb = np.array([1.03, 1.08, 1.11, 1.05, 1.06, 1.01, 1.00])
yb = unp.uarray(yb,0.01*np.sqrt(yb))
yb




#%% ---



fig, axplot = ipanema.plotting.axes_square()
axplot.fill_between(xp, ypl, ypu, alpha=0.5,label='baseline')
axplot.errorbar(0.5*(xb[1:]+xb[:-1]),unp.nominal_values(yb),
             xerr=xb[1:]-0.5*(xb[1:]+xb[:-1]),
             yerr=unp.std_devs(yb),
             fmt='.',color='k',label='binned in time')
axplot.set_aspect('auto')
watermark(axplot, version=f"${version}$", scale=1.03)
axplot.legend()
xleft, xright = axplot.get_xlim(); ybottom, ytop = axplot.get_ylim()
axplot.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1.0)
axplot.set_xlabel("$t$ [ps]")
axplot.set_xlim(0,15.5)
axplot.set_ylabel(f"$w_{i}$")
fig.savefig(f'output_new/figures/angular_acceptance/2017/Bs2JpsiPhi/{version}_BaselineRun2TimeBinnedAngularWeight1_biased.pdf')
