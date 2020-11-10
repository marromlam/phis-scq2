#%% Import packages
import numpy as np
import ipanema
ipanema.initialize('cuda',1)
import badjanak
from badjanak import get_sizes
import matplotlib.pyplot as plt
import pandas as pd
import uproot

# %% wrapper




# %% Run toy -------------------------------------------------------------------
badjanak.config['debug'] = 0
badjanak.get_kernels()

# load parameters
pgen = ipanema.Parameters.load('output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul.json')
pgen += ipanema.Parameters.load('output/params/csp_factors/2018/Bs2JpsiPhi/v0r5.json')
p = badjanak.cross_rate_parser_new(**pgen.valuesdict(), tLL=0.3, tUL=15)

# prepare output array
out = []
wNorm = np.array([
[48.8914663704,358.037677975,1322.58087149,1025.44991048,418.693447303,201.557648496], #2015 biased
[149.50994111,1112.17755086,4056.22375747,3382.7026021,1389.62492793,600.036881838],
[289.442878715,1913.12535023,7093.27251931,5613.95446068,2241.74554212,1137.86083739], #2016 biased
[1109.54737658,7741.45392545,29209.6555018,23334.4041835,9347.37974402,4530.41734544]
])


bin=0
for ml,mh in zip(badjanak.config['x_m'][:-1],badjanak.config['x_m'][1:]):
  print(f'fill bin {np.int32(sum(wNorm[:,bin]))} events')
  out += np.int32((100*sum(wNorm[:,bin]))+1)*[10*[0.5*(ml+mh)]]
  bin +=1
out = ipanema.ristra.allocate(np.float64(out))


# generate
print(f'generating {len(out)} evets...')
badjanak.dG5toys(out, **p, use_angacc=0)
print('generation done!')



# some plots
# plt.hist(ipanema.ristra.get(out[:,0]));
# plt.hist(ipanema.ristra.get(out[:,1]));
# plt.hist(ipanema.ristra.get(out[:,2]));
# plt.hist(ipanema.ristra.get(out[:,3]));
# plt.hist(ipanema.ristra.get(out[:,4]));

# from array to dict of arrays and then to pandas.df
genarr = ipanema.ristra.get(out)

gendic = {
  'cosK':genarr[:,0],
  'cosL':genarr[:,1],
  'hphi':genarr[:,2],
  'time':genarr[:,3],
  'gencosK':genarr[:,0],
  'gencosL':genarr[:,1],
  'genhphi':genarr[:,2],
  'gentime':genarr[:,3],
  'mHH':genarr[:,4],
  'sigmat':genarr[:,5],
  'idB':genarr[:,6],
  'genidB':genarr[:,7],
  'tagOSeta':genarr[:,8],
  'tagSSeta':genarr[:,9],
  'polWeight':np.ones_like(genarr[:,0]),
  'sw':np.ones_like(genarr[:,0]),
  'gb_weights':np.ones_like(genarr[:,0])
}

# save tuple
tuple = "/scratch17/marcos.romero/sidecar/2015/TOY_Bs2JpsiPhi/20201009a0.root"
df = pd.DataFrame.from_dict(gendic)
with uproot.recreate(tuple) as fp:
  fp['DecayTree'] = uproot.newtree({var:'float64' for var in df})
  fp['DecayTree'].extend(df.to_dict(orient='list'))
fp.close()



# %%

# -------
real = ['cosK','cosL','hphi','time','mHH','sigmat','idB','idB','tagOSeta','tagSSeta']
s = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2015/TOY_Bs2JpsiPhi/20201009a0.root')
s.allocate(input=real, output='0*time', weight='time/time')
s.df


# -------
SWAVE = True
POLDEP = False
BLIND = False
DGZERO = False
pars = ipanema.Parameters()
list_of_parameters = [#
# S wave fractions
ipanema.Parameter(name='fSlon1', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{1}'),
ipanema.Parameter(name='fSlon2', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{2}'),
ipanema.Parameter(name='fSlon3', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{3}'),
ipanema.Parameter(name='fSlon4', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{4}'),
ipanema.Parameter(name='fSlon5', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{5}'),
ipanema.Parameter(name='fSlon6', value=SWAVE*0.5, min=0.00, max=0.80,
          free=SWAVE, latex=r'f_S^{6}'),
# P wave fractions
ipanema.Parameter(name="fPlon", value=0.5241, min=0.4, max=0.6,
          free=True, latex=r'f_0'),
ipanema.Parameter(name="fPper", value=0.25, min=0.1, max=0.3,
          free=True, latex=r'f_{\perp}'),
# Weak phases
ipanema.Parameter(name="pSlon", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_S - \phi_0",
          blindstr="BsPhisSDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
ipanema.Parameter(name="pPlon", value=-0.03, min=-5.0, max=5.0,
          free=True, latex=r"\phi_0",
          blindstr="BsPhiszeroFullRun2" if POLDEP else "BsPhisFullRun2",
          blind=BLIND, blindscale=2.0 if POLDEP else 1.0, blindengine="root"),
ipanema.Parameter(name="pPpar", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_{\parallel} - \phi_0",
          blindstr="BsPhisparaDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
ipanema.Parameter(name="pPper", value= 0.00, min=-1.0, max=1.0,
          free=POLDEP, latex=r"\phi_{\perp} - \phi_0",
          blindstr="BsPhisperpDelFullRun2",
          blind=BLIND, blindscale=2.0, blindengine="root"),
# S wave strong phases
ipanema.Parameter(name='dSlon1', value=+np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{1} - \delta_{\perp}"),
ipanema.Parameter(name='dSlon2', value=+np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{2} - \delta_{\perp}"),
ipanema.Parameter(name='dSlon3', value=+np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{3} - \delta_{\perp}"),
ipanema.Parameter(name='dSlon4', value=-np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{4} - \delta_{\perp}"),
ipanema.Parameter(name='dSlon5', value=-np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{5} - \delta_{\perp}"),
ipanema.Parameter(name='dSlon6', value=-np.pi/4*SWAVE, min=-3.0, max=+3.0,
          free=SWAVE, latex="\delta_S^{6} - \delta_{\perp}"),
# P wave strong phases
ipanema.Parameter(name="dPlon", value=0.00, min=-2*3.14, max=2*3.14,
          free=False, latex="\delta_0"),
ipanema.Parameter(name="dPpar", value=3.26, min=-2*3.14, max=2*3.14,
          free=True, latex="\delta_{\parallel} - \delta_0"),
ipanema.Parameter(name="dPper", value=3.1, min=-2*3.14, max=2*3.14,
          free=True, latex="\delta_{\perp} - \delta_0"),
# lambdas
ipanema.Parameter(name="lSlon", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_S/\lambda_0"),
ipanema.Parameter(name="lPlon", value=1., min=0.7, max=1.6,
          free=True,  latex="\lambda_0"),
ipanema.Parameter(name="lPpar", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_{\parallel}/\lambda_0"),
ipanema.Parameter(name="lPper", value=1., min=0.7, max=1.6,
          free=POLDEP, latex="\lambda_{\perp}/\lambda_0"),
# lifetime parameters
ipanema.Parameter(name="Gd", value= 0.65789, min= 0.0, max= 1.0,
          free=False, latex=r"\Gamma_d"),
ipanema.Parameter(name="DGs", value= (1-DGZERO)*0.08, min= 0.0, max= 1.7,
          free=1-DGZERO, latex=r"\Delta\Gamma_s",
          blindstr="BsDGsFullRun2",
          blind=BLIND, blindscale=1.0, blindengine="root"),
ipanema.Parameter(name="DGsd", value= 0.03,   min=-0.5, max= 0.5,
          free=True, latex=r"\Gamma_s - \Gamma_d"),
ipanema.Parameter(name="DM", value=17.757,   min=15.0, max=20.0,
          free=True, latex=r"\Delta m"),
]

pars.add(*list_of_parameters);
#pars = ipanema.Parameters.load('output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul.json')
pars += ipanema.Parameters.load('output/params/csp_factors/2018/Bs2JpsiPhi/v0r5.json')
# for par in genpars.keys():
#   if par in pars.keys():
#     pars[par].set(value=genpars[par].value)
#   else:
#     pars.add( genpars[par] )

#print(pars)
# pars = ipanema.Parameters.clone(pgen)
# pars['pSlon'].set(value=0)



# ------
def fcn_data(parameters, data):
  # here we are going to unblind the parameters to the fcn caller, thats why
  # we call parameters.valuesdict(blind=False), by default
  # parameters.valuesdict() has blind=True
  pdict = parameters.valuesdict()
  badjanak.delta_gamma5_mc(data.input, data.output, **pdict, tLL=0.3, tUL=15)
  return ( -2.0 *ipanema.ristra.log(data.output) ).get()

result = ipanema.optimize(fcn_data, method='minuit', params=pars, fcn_args=(s,),
                          verbose=False, timeit=False, tol=0.05 , strategy=1)
print(result)


# print fit vs gen and pulls
print(f"{'Parameters':>10}  {'Gen':>7}  {'Fit':>16}   {'Pull':>5}")
for k in result.params.keys():
  gen = pgen.valuesdict()[k]
  fit = result.params[k].value
  std = result.params[k].stdev
  if std:
    print(f"{k:>10}  {gen:+.4f}  {fit:+.4f}+/-{std:5.4f}   {(fit-gen)/std:+.2f}")
  else:
    0#print(f"{k:>10}  {gen:5.4f}  {fit:.4f}+/-{0:5.4f}   ")


#####
# %%
from ipanema import Sample
import matplotlib.pyplot as plt


# bdmc ok
"""
scq = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bd2JpsiKstar/v0r1.root')
hd = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bd2JpsiKstar/v0r0_kinWeight.root')
plt.plot(scq.df['pdfWeight']-hd.df['pdfWeight'])
"""


"""
# bdrd ok
scq = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/Bd2JpsiKstar/v0r1.root')
hd = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/Bd2JpsiKstar/v0r0_kinWeight.root')
plt.plot(scq.df['kinWeight']-hd.df['kinWeight'])
plt.ylim(-1e-15,1e-15)
"""



scq = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r1.root')
hd = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r0_kinWeight.root')
plt.plot(scq.df['kinWeight']-hd.df['kinWeight'])
scq = Sample.from_root('/scratch17/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi_dG0/v0r1.root')
hd = Sample.from_root('/scratch17/marcos.romero/sidecar/2015/MC_Bs2JpsiPhi_dG0/v0r0_kinWeight.root')
plt.plot(scq.df['kinWeight']-hd.df['kinWeight'])
plt.ylim(-1e-12,1e-12)



for i in range(0,len(scq.df['pdfWeight'])):
    a = scq.df['pdfWeight'].iloc[i]
    b = hd.df['pdfWeight'].iloc[i]
    if a-b>0.245:
        print(i)


scq.df['time'].iloc[2930619]

#plt.ylim(-1e-15,1e-15)

# %%

min(scq.df['time'])

scq.df['time'].iloc[2]
for i in (scq.df['pdfWeight']-hd.df['pdfWeight']).values:
  if i > 100:
    print(i)



scq

# %%
