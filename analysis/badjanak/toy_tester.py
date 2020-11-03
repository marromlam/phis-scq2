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

def dG5toys(output,
            G, DG, DM,
            CSP,
            ASlon, APlon, APpar, APper,
            pSlon, pPlon, pPpar, pPper,
            dSlon, dPlon, dPpar, dPper,
            lSlon, lPlon, lPpar, lPper,
            # Time limits
            tLL, tUL,
            # Time resolution
            sigma_offset, sigma_slope, sigma_curvature,
            mu,
            # Flavor tagging
            eta_os, eta_ss,
            p0_os,  p1_os, p2_os,
            p0_ss,  p1_ss, p2_ss,
            dp0_os, dp1_os, dp2_os,
            dp0_ss, dp1_ss, dp2_ss,
            # Time acceptance
            timeacc,
            # Angular acceptance
            angacc,
            # Flags
            use_fk=1, use_angacc = 0, use_timeacc = 0,
            use_timeoffset = 0, set_tagging = 0, use_timeres = 0,
            prob_max = 2.7,
            BLOCK_SIZE=256, **crap):
  """
  Look at kernel definition to see help
  The aim of this function is to be the fastest wrapper
  """
  g_size, l_size = get_sizes(output.shape[0],BLOCK_SIZE)
  badjanak.__KERNELS__.dG5toy(
    # Input and output arrays
    output,
    # Differential cross-rate parameters
    np.float64(G), np.float64(DG), np.float64(DM),
    CSP.astype(np.float64),
    ASlon.astype(np.float64), APlon.astype(np.float64), APpar.astype(np.float64), APper.astype(np.float64),
    np.float64(pSlon),                 np.float64(pPlon),                 np.float64(pPpar),                 np.float64(pPper),
    dSlon.astype(np.float64),          np.float64(dPlon),                 np.float64(dPpar),                 np.float64(dPper),
    np.float64(lSlon),                 np.float64(lPlon),                 np.float64(lPpar),                 np.float64(lPper),
    # Time range
    np.float64(tLL), np.float64(tUL),
    # Time resolution
    np.float64(sigma_offset), np.float64(sigma_slope), np.float64(sigma_curvature),
    np.float64(mu),
    # Flavor tagging
    np.float64(eta_os), np.float64(eta_ss),
    np.float64(p0_os), np.float64(p1_os), np.float64(p2_os),
    np.float64(p0_ss), np.float64(p1_ss), np.float64(p2_ss),
    np.float64(dp0_os), np.float64(dp1_os), np.float64(dp2_os),
    np.float64(dp0_ss), np.float64(dp1_ss), np.float64(dp2_ss),
    # Decay-time acceptance
    timeacc.astype(np.float64),
    # Angular acceptance
    angacc.astype(np.float64),
    # Flags
    np.int32(use_fk), np.int32(len(CSP)), np.int32(use_angacc), np.int32(use_timeacc),
    np.int32(use_timeoffset), np.int32(set_tagging), np.int32(use_timeres),
    np.float64(prob_max), np.int32(len(output)),
    global_size=g_size, local_size=l_size)
    #grid=(int(np.ceil(output.shape[0]/BLOCK_SIZE)),1,1), block=(BLOCK_SIZE,1,1))




# %% Run toy
badjanak.config['debug'] = 0
badjanak.get_kernels()

# load parameters
p = ipanema.Parameters.load('analysis/params/generator/2015/MC_Bs2JpsiPhi.json')
p = badjanak.cross_rate_parser_new(**p.valuesdict())

# add modifications
p['angacc'][0] = 1.0
p['angacc'][1] = 4.0
p['angacc'][2] = 1.0
p['angacc'][6] = 10.0
for k,v in p.items():
  print(f" {k:>20} : {v}")

# prepare output array
out = []
for ml,mh in zip(badjanak.config['x_m'][:-1],badjanak.config['x_m'][1:]):
  out += 100000*[10*[0.5*(ml+mh)]]
out = ipanema.ristra.allocate(np.float64(out))

# generate
dG5toys(out, **p, use_angacc=1)

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
  'truecosK_GenLvl':genarr[:,0],
  'truecosL_GenLvl':genarr[:,1],
  'truehphi_GenLvl':genarr[:,2],
  'truetime_GenLvl':genarr[:,3],
  'X_M':genarr[:,4],
  'sigmat':genarr[:,5],
  'B_ID':genarr[:,6],
  'B_ID_GenLvl':genarr[:,7],
  'tagOS_eta':genarr[:,8],
  'tagSS_eta':genarr[:,9],
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
