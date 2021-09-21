#%% Import packages




std = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi/v0r5.root')
stdw = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi/v0r5_yearly_simul_kkpWeight.root')
std_df = pd.concat([std.df, stdw.df],axis=1).query('hlt1b==1')
dg0 = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5.root')
dg0w = ipanema.Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5_yearly_simul_kkpWeight.root')
dg0_df = pd.concat([dg0.df, dg0w.df],axis=1).query('hlt1b==1')

df = pd.concat([std_df, dg0_df], ignore_index=True)

weights = df['kkpWeight3']/df['pdfWeight']
normalisation = 1/weights.sum()



plt.hist(s.df['cosK'], weights=s.df['gencosK']/s.df['cosK'],range=(-1,1));
counts_g, bins = np.histogram(s.df['gencosL'] ,range=(-1,1), bins=50);
counts_r, bins = np.histogram(s.df['cosL'] ,range=(-1,1), bins=50);
bins = 0.5*(bins[1:]+bins[0:-1])
plt.step(bins, counts_r/counts_g)




edges
extent = (-1,1)
edges = np.linspace(*extent, 20)
centres = (edges[:-1] + edges[1:])/2
cut = pd.cut(df['cosK'], edges)
# We're actually plotting a histogram of the selected MC weighted by
# the inverse of the true PDF...
effs = weights.groupby(cut).sum()
errs = np.sqrt(np.square(weights).groupby(cut).sum())

effs *= normalisation
errs *= normalisation

fig, ax = plt.subplots()
ax.plot(centres, effs, 'ko')
ax.errorbar(centres, effs,
            xerr=np.diff(edges)/2,
            yerr=1*errs,
            fmt='.')
#ax.set_xlabel(angle.label)
ax.set_ylabel('Efficiency [A.U.]')
ax.set_xlim(*extent)
fig.tight_layout()



plt.plot(effs)

)


s.df['cosK']/s.df['gencosK']


#%% ----------------------------------------------------------------------------

import numpy as np
import ipanema
ipanema.initialize('cuda',1)
import badjanak
from badjanak import get_sizes
import matplotlib.pyplot as plt
import pandas as pd
import uproot
from ipanema import wrap_unc, get_confidence_bands, ristra
def ndmesh(*args):
   args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] for i, x in enumerate(args)])

N = 100
cosKh_ = np.linspace(-1,1,N)
cosLh_ = np.linspace(-1,1,N)
hphih_ = np.linspace(-np.pi,+np.pi,N)

cosKh, cosLh, hphih = ndmesh(cosKh_, cosLh_, hphih_)
cosK = ipanema.ristra.allocate( cosKh.reshape(N**3) )
cosL = ipanema.ristra.allocate( cosLh.reshape(N**3) )
hphi = ipanema.ristra.allocate( hphih.reshape(N**3) )

#angacc = ipanema.ristra.allocate(np.float64([1,1.0392,1.0374,-0.0106,0.0037,-0.0023,1.0213,-0.0044,-0.0008,-0.0351]))
angacc = ipanema.ristra.allocate(np.float64([1,1.0327,1.0327,0.0029,0.00309,-0.00024,1.019,0.00012,0.0001,0.0054]))
theory = ipanema.ristra.allocate(np.float64([1,1,1,0,0,0,1,0,0,0]))

effreal = ipanema.ristra.zeros_like(cosK)
effnomi = ipanema.ristra.zeros_like(cosK)


plotshit = badjanak.__KERNELS__.plot_moments

plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
effnomi.shape[0]**(1/3)
round(effnomi.shape[0]**(1/3))


from uncertainties import *
import numpy as np
from uncertainties import unumpy as unp


### funcs

x**2

f = lambda x: (x[1]*x[1])*np.array([1,2,3,4])


f([0,2])


(f([1,2+1e-8])-f([1,2-1e-8]))/(2*1e-8)

fast_jac(f, [1,1], f_size=4)

np.zeros([3, 2]).T


def fast_jac(f, vals, f_size=1):
  J = np.zeros([f_size, len(vals)])
  print(vals)
  for l in range(0,len(vals)):
    if vals[l]!= 0:
      h = np.sqrt(np.finfo(float).eps)*vals[l];
    else:
      h = 1e-14;
    print(h)
    xhp = np.copy(vals).astype(np.float64); xhp[l] += +h
    xhm = np.copy(vals).astype(np.float64); xhm[l] += -h;
    J[:,l] = (f(xhp) - f(xhm))/(2*h)
  return J.T





def propagate_term(der,unc):
  return der**2*unc**2

def wrap_unc(func, pars, **kwgs):

  f = lambda pars: func(pars, **kwgs)

  # get parameters and uncertainties
  vals = np.array([pars[k].nominal_value  for k in range(0,len(pars))])
  uncs = np.array([pars[k].std_dev        for k in range(0,len(pars))])

  # compute f nominal_value
  f_val = f(vals)
  if hasattr(f(vals), "__len__"):
    f_size = len(f(vals))
  else:
    f_size = 1
  print(f_size)

  # get numeric derivatives
  derivatives = fast_jac(f, vals, f_size)
  print(derivatives)
  f_unc = np.zeros([len(x)])

  # compute f std_dev
  for i in range(0,len(uncs)):
    f_unc[:] += propagate_term(derivatives[i],uncs[i])[0]
  f_unc = np.sqrt(f_unc)

  return unp.uarray(f_val,f_unc)


wrap_unc(f, [unc.ufloat(1),2])
wrap_unc(lambda p: angeff_plot(p, cosK, cosL, hphi, 1)[:5], lnom)
















def angeff_plot(angacc, cosK, cosL, hphi, project=None):
  eff = ristra.zeros_like(cosK)
  try:
    _angacc = ristra.allocate(np.array(angacc))
  except:
    _angacc = ristra.allocate(np.array([a.n for a in angacc]))
  plotshit(_angacc, eff, cosK, cosL, hphi, global_size=(eff.shape[0],))
  n = round(eff.shape[0]**(1/3))
  res = ristra.get(eff).reshape(n,n,n)
  if project==1:
    return np.sum(efftheoh,(1,0))
  if project==2:
    return np.sum(efftheoh,(1,2))
  if project==3:
    return np.sum(efftheoh,(2,0))

  return res

lnom[0].n

ristra.allocate(np.array(lnom))

[lnom]
nom = ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bd2JpsiKstar/v0r1_naive_biased.json')
lnom = [p.uvalue for p in nom.values()]
angeff_plot(nom, cosK, cosL, hphi, 1)
plt.plot(angeff_plot(nom, cosK, cosL, hphi, 1))
lnom
np.array([wrap_unc(lambda p: angeff_plot(p, cosK, cosL, hphi, 1)[i], lnom) for i in range(100)])

meh = lambda p: angeff_plot(p, cosK, cosL, hphi, 1)

meh(lnom)

lnom
fig, ax = plt.subplots()
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bd2JpsiKstar/v0r1_naive_biased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosK, label='2016 biased MCdG0')



fig, ax = plt.subplots()
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive_biased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosK, label='2016 biased MCdG0')
#nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bs2JpsiPhi/v0r5_naive_biased.json')))
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bs2JpsiPhi/v0r5_naiveTime1_biased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosK, label='2016 biased MC')
ax.legend()


fig, ax = plt.subplots()
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r1_naive_unbiased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosK, label='2016 biased MCdG0')
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2016/MC_Bs2JpsiPhi/v0r1_naive_unbiased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosK, label='2016 unbiased MC')
ax.legend()






#%% ----------------------------------------
for year in [5,6,7,8]:
  fig, ax = plt.subplots()
  nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/201{year}/MC_Bs2JpsiPhi_dG0/v0r5_naive_biased.json')))
  plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
  efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
  eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
  ax.plot(cosKh_,eff_cosK, label='nominal')
  if year == 9:
    plotshit(angacc, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
    effnomih = ipanema.ristra.get(effnomi).reshape(N,N,N)
    eff_cosK = np.sum(effnomih,(1,0))/len(cosKh_)**2
    ax.plot(cosKh_,eff_cosK, label='nominal old')

  for i in range(1,7):
    angacc = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/201{year}/MC_Bs2JpsiPhi_dG0/v0r5_naiveTime{i}_biased.json')))
    plotshit(angacc, effreal, cosK, cosL, hphi, global_size=(effreal.shape[0],))
    effrealh = ipanema.ristra.get(effreal).reshape(N,N,N)
    eff_cosK = np.sum(effrealh,(1,0))/len(cosKh_)**2
    ax.plot(cosKh_,eff_cosK, '-',label=f'$t$ bin = {i}')
  ax.set_xlabel(r'cos $\theta_K$')
  ax.legend()
  #ax.set_ylim(0.9,1.25)
  ax.set_title(f"201{year} biased")
  fig.savefig(f'201{year}_naive_timedep_cosK.pdf')




#%% cosL
fig, ax = plt.subplots()
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_run2_simul_biased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_cosL = np.sum(efftheoh,(0,2))/len(cosKh_)**2
ax.plot(cosKh_,eff_cosL, label='nominal')

for i in range(1,7):
  angacc = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_run2Time{i}_simul_biased.json')))
  plotshit(angacc, effreal, cosK, cosL, hphi, global_size=(effreal.shape[0],))
  effrealh = ipanema.ristra.get(effreal).reshape(N,N,N)
  eff_cosL = np.sum(effrealh,(0,2))/len(cosKh_)**2
  ax.plot(cosKh_,eff_cosL, ':',label=f'$t$ bin = {i}')
ax.set_xlabel(r'cos $\theta_{\mu}$')
ax.legend()
ax.set_ylim(0.9,1.25)
fig.savefig('angacc_timedep_cosL.pdf')

#%% hphi
fig, ax = plt.subplots()
nom = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_run2_simul_biased.json')))
plotshit(nom, effnomi, cosK, cosL, hphi, global_size=(effnomi.shape[0],))
efftheoh = ipanema.ristra.get(effnomi).reshape(N,N,N)
eff_hphi = np.sum(efftheoh,(1,2))/len(cosKh_)**2
ax.plot(cosKh_,eff_hphi, label='nominal')

for i in range(1,7):
  angacc = ipanema.ristra.allocate(np.array(ipanema.Parameters.load(f'output/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_run2Time{i}_simul_biased.json')))
  plotshit(angacc, effreal, cosK, cosL, hphi, global_size=(effreal.shape[0],))
  effrealh = ipanema.ristra.get(effreal).reshape(N,N,N)
  eff_hphi = np.sum(effrealh,(1,2))/len(cosKh_)**2
  ax.plot(cosKh_,eff_hphi, ':',label=f'$t$ bin = {i}')
ax.set_xlabel(r'cos $\phi_h$')
ax.legend()
ax.set_ylim(0.9,1.25)
fig.savefig('angacc_timedep_hphi.pdf')







#%% do cosK plot
eff_cosK = np.sum(effrealh,(1,0))/len(cosKh_)**2
plt.plot(cosKh_,eff_cosK)
eff_cosK = np.sum(efftheoh,(1,0))/len(cosKh_)**2
plt.plot(cosKh_,eff_cosK)
plt.ylim(0.9,1.25)

effnomi.shape[0]

#%% do hphi plot
eff_hphi = np.sum(effrealh,(0,2))/len(hphih_)**2
plt.plot(hphih_,eff_hphi)
eff_hphi = np.sum(efftheoh,(0,2))/len(hphih_)**2
plt.plot(hphih_,eff_hphi)
plt.ylim(0.9,1.25)



#%% do hphi plot
eff_hphi = np.sum(effrealh,(1,2))/len(hphih_)**2
plt.plot(hphih_,eff_hphi)
eff_hphi = np.sum(efftheoh,(1,2))/len(hphih_)**2
plt.plot(hphih_,eff_hphi)
plt.ylim(0.9,1.25)

plt.plot(cosLh_,np.sum(effh,(0,2))/len(cosLh_)**2)
plt.ylim(0.9,1.25)

plt.plot(np.sum(effh,(2,1)))


seffr[0]-effr[1]
effh[N/2][N/2]
ipanema.ristra.get(effh)[0][0]
plt.plot(ipanema.ristra.get(cosKh_),ipanema.ristra.get(effh)[N//2][N//2],'.')
plt.ylim(0.9,1.25)
plt.plot(ipanema.ristra.get(cosLh_),effh[N//2,:,0],'.')

effh[N//2,:,N//2].shape


cosLh[0,0,:]

plt.plot(X[0,:],np.sum(Z,1))



plt.plot(ipanema.ristra.get(cosL),ipanema.ristra.get(eff),'.')
plt.plot(ipanema.ristra.get(hphi),ipanema.ristra.get(eff),'.')


fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
X, Y, Z = get_test_data(0.05)
C = np.linspace(-5, 5, Z.size).reshape(Z.shape)
scamap = plt.cm.ScalarMappable(cmap='inferno')
fcolors = scamap.to_rgba(C)
ax.plot_surface(X, Y, Z, facecolors=fcolors, cmap='inferno')
fig.colorbar(scamap)
plt.show()



fig = plt.figure()
ax = fig.add_subplot(100, projection='3d')
ax.scatter(a, b, c, cc=cc, cmap=mplt.hot())




np.trapz(np.trapz(effh, cosKh_), cosKh_)

from multihist import *
m = Hist1d([0, 3, 1, 6, 2, 9], bins=3)

# ...or add data incrementally:
m = Hist1d(bins=100, range=(-3, 4))
m.add(np.random.normal(0, 0.5, 10**4))
m.add(np.random.normal(2, 0.2, 10**3))

# Get the data back out:
print(m.histogram, m.bin_edges)

# Access derived quantities like bin_centers, normalized_histogram, density, cumulative_density, mean, std
plt.plot(m.bin_centers, m.normalized_histogram, label="Normalized histogram", linestyle='-')
plt.plot(m.bin_centers, m.density, label="Empirical PDF", linestyle='-')
plt.plot(m.bin_centers, m.cumulative_density, label="Empirical CDF", linestyle='-')
plt.title("Estimated mean %0.2f, estimated std %0.2f" % (m.mean, m.std))
plt.legend(loc='best')
plt.show()

# Slicing and arithmetic behave just like ordinary ndarrays
print("The fourth bin has %d entries" % m[3])
m[1:4] += 4 + 2 * m[-27:-24]
print("Now it has %d entries" % m[3])

# Of course I couldn't resist adding a canned plotting function:
m.plot()
plt.show()

# Create and show a 2d histogram. Axis names are optional.
m2 = Histdd(bins=100, range=[[-5, 3], [-3, 5]], axis_names=['x', 'y'])
m2.add(np.random.normal(1, 1, 10**6), np.random.normal(1, 1, 10**6))
m2.add(np.random.normal(-2, 1, 10**6), np.random.normal(2, 1, 10**6))
m2.plot()
plt.show()

# x and y projections return Hist1d objects
m2.projection('x').plot(label='x projection')
m2.projection(1).plot(label='y projection')
plt.legend()
plt.show()



effh = ipanema.ristra.get(eff).reshape(N,N,N)
plt.plot(effh[0][0])


import numpy as np
xx = np.linspace(-5,5,100)
yy = np.linspace(-5,5,100)
x, y = np.meshgrid(yy, xx)
z = x * np.sin(y)


from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
#cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-40, 40)
ax.set_ylabel('Y')
ax.set_ylim(-40, 40)
ax.set_zlabel('Z')
ax.set_zlim(-100, 100)

plt.plot(X, np)



plt.plot(X[0,:],np.sum(Z,1))
plt.plot(Y[:,0],np.sum(Z,0))



np.trapz(effh, hphih_)

np.trapz(np.trapz(effh, hphih_), cosLh_)
plt.plot( np.trapz(np.trapz(effh, hphih_), cosLh_) )
cosKh_
plt.plot(1)

x, y = np.linspace(0, 5, 11), np.linspace(0, 7, 22)
X, Y = np.meshgrid(x, y)
z = 1 + 2*X + Y + X*Y
4*np.trapz(np.trapz(z, x), y)
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
for ml,mh in zip(badjanak.config['mHH'][:-1],badjanak.config['mHH'][1:]):
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
