DESCRIPTION = """
    Plot angular efficienty
"""


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []

from ipanema.confidence import wrap_unc, get_confidence_bands
from ipanema import initialize, ristra, Parameters, Sample
from utils.helpers import  version_guesser, trigger_scissors
from ipanema.core.python import ndmesh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot

import argparse
initialize('cuda',1)
import badjanak

import hjson
all_knots = hjson.load(open('config.json'))['time_acceptance_knots']

'naive2knots1'.split('knots')[0][-1]




# wrapper arround ang_eff cuda kernel
#     it can project to 1,2,3 = cosK,cosK,hphi variables
def angeff_plot(angacc, cosK, cosL, hphi, project=None):
  eff = ristra.zeros_like(cosK)
  try:
    _angacc = ristra.allocate(np.array(angacc))
  except:
    _angacc = ristra.allocate(np.array([a.n for a in angacc]))
  badjanak.__KERNELS__.plot_moments(_angacc, eff, cosK, cosL, hphi, global_size=(eff.shape[0],))
  n = round(eff.shape[0]**(1/3))
  res = ristra.get(eff).reshape(n,n,n)
  if project==1:
    return np.sum(res,(1,0))
  if project==2:
    return np.sum(res,(1,2))
  if project==3:
    return np.sum(res,(2,0))
  return res



def create_toy(pars, nevts=1e6):
  out = ristra.allocate(np.float64(int(nevts)*[10*[1020.]]))
  badjanak.dG5toys(out, **badjanak.parser_rateBs(**pars.valuesdict()),
                   use_angacc=0, use_timeacc=0, use_timeres=0,
                   set_tagging=2, use_timeoffset=0,
                   seed=int(1e10*np.random.rand()) )
  genarr = ristra.get(out)
  gendic = {
    'cosK'        :  genarr[:,0],
    'cosL'        :  genarr[:,1],
    'hphi'        :  genarr[:,2],
    'time'        :  genarr[:,3],
    'gencosK'     :  genarr[:,0],
    'gencosL'     :  genarr[:,1],
    'genhphi'     :  genarr[:,2],
    'gentime'     :  genarr[:,3],
    'mHH'         :  genarr[:,4],
    'sigmat'      :  genarr[:,5],
    'idB'         :  genarr[:,6],
    'genidB'      :  genarr[:,7],
  }
  return pd.DataFrame.from_dict(gendic)




"""
p = Parameters.load('analysis/params/generator/2016/MC_Bs2JpsiPhi_dG0.json')



plt.hist(ristra.get(out)[:,1]);
plt.hist(ristra.get(out)[:,0]);
plt.hist(ristra.get(out)[:,2]);




tgc, tge = np.histogram(tmc['cosL'], bins=20, density=True);
mcc, mce = np.histogram(smc.df['cosL'], bins=tge, weights=smc.df[sw], density=True);
cbs = 0.5*(tge[1:]+tge[:-1])


plt.plot(cbs, mcc/tgc, '.')
plt.ylim(0.9,1.25)


all_knots["2"]
all_knots["2"][1-1]



smc = Sample.from_root('/scratch17/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r1.root', cuts='hlt1b')
smc.allocate(time='time')
smc.allocate(cosK='cosK')
smc.allocate(cosL='cosL')
s

#Parameters.load(f'output/params/angular_acceptance//Bs2JpsiPhi/v0r5_naive_{trigger}.json')





args = {}
args['year'] = '2016'
args['mode'] = 'MC_Bs2JpsiPhi_dG0'
args['trigger'] = 'biased'
args['nknots'] = '2'
args['angular_acceptance'] = "output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive_biased.json,output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots1_biased.json,output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots2_biased.json,output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots3_biased.json"
args['output'] = "output/figures/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots_biased_cosK.pdf,output/figures/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots_biased_cosL.pdf,output/figures/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_naive2knots_biased_hphi.pdf"
"""




if __name__ == '__main__':

  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi data sample')
  p.add_argument('--angular_acceptance', help='Bs2JpsiPhi MC generator parameters')
  p.add_argument('--output', help='Bs2JpsiPhi MC angular acceptance')
  p.add_argument('--nknots', help='Bs2JpsiPhi MC angular acceptance tex')
  p.add_argument('--mode', help='Mode to compute angular acceptance with')
  p.add_argument('--year', help='Year to compute angular acceptance with')
  p.add_argument('--version', help='Version of the tuples')
  p.add_argument('--trigger', help='Trigger to compute angular acceptance with')
  p.add_argument('--angacc', help='Trigger to compute angular acceptance with')
  p.add_argument('--timeacc', help='Trigger to compute angular acceptance with')
  args = vars(p.parse_args())

  # get list of knots
  knots = all_knots[args['nknots']]

  MODE = args['mode']
  TRIGGER = args['trigger']
  print(TRIGGER)
  years = args['year'].split(',')
  samples = args['samples'].split(',')
  gen_params = args['params'].split(',')
  gen_params = [Parameters.load(p) for p in gen_params]
  output = args['output'].split(',')
  angaccs = args['angular_acceptance'].split(',')

  # Create grid of N points in space
  N = 100 # number of points to plot
  cosK = np.linspace(-1,1,N)
  cosL = np.linspace(-1,1,N)
  hphi = np.linspace(-np.pi,+np.pi,N)
  cosKh, cosLh, hphih = ndmesh(cosK, cosL, hphi)
  cosKd = ristra.allocate( cosKh.reshape(N**3) )
  cosLd = ristra.allocate( cosLh.reshape(N**3) )
  hphid = ristra.allocate( hphih.reshape(N**3) )

  # sw switcher
  sw = 'sw'
  if 'MC_Bs2JpsiPhi' in MODE:
    sw = 'sw/gb_weights'

  if TRIGGER=='combined':
    TRIGGER = ['biased', 'unbiased']
  else:
    TRIGGER = [TRIGGER]

  labeled = False
  if len(years)>1:
    labeled= True
  if len(TRIGGER)>1:
    labeled = True

  # Run all plots
  for k, var in enumerate(['cosK', 'cosL', 'hphi']):
    if var=='cosK':
      proj = 1; bounds = (-1,1); tex = r'\mathrm{cos}\theta_K'; x=cosK
    elif var=='cosL':
      proj = 3; bounds = (-1,1); tex = r'\mathrm{cos}\theta_{\mu}'; x=cosL
    elif var=='hphi':
      proj = 2; bounds = (-np.pi,np.pi); tex = r'\phi_h \mathrm{[rad]}'; x=hphi
    # loop in years
    fig, ax = plt.subplots()
    for y, year in enumerate(years):
      # load the sample
      mc_sample = Sample.from_root(samples[y])
      # since there is only a sample per year, this is the right moment to
      # generate the toy MC
      toy_sample = create_toy(gen_params[y])

      # loop in triggers
      for trigger in TRIGGER:
        smc = mc_sample.df.query(trigger_scissors(trigger))
        tmc = toy_sample.query("time>0.3")

        # get the number of events in sample
        tlen = len(tmc[var]); slen = len(smc[var])
        # calculate the integral
        tgc, tge = np.histogram(tmc[var], bins=200, density=True);
        mcc, mce = np.histogram(smc[var], bins=tge, weights=smc.eval(sw), density=True);
        cbs = 0.5*(tge[1:]+tge[:-1])
        pint = np.trapz(mcc/tgc,cbs)

        # now create a set of points
        tgc, tge = np.histogram(tmc[var], bins=20, density=True);
        mcc, mce = np.histogram(smc[var], bins=tge, weights=smc.eval(sw), density=True);
        cbs = 0.5*(tge[1:]+tge[:-1])

        # load angular acceptance parameters for this year-category
        angacc = Parameters.load(angaccs[0])

        # first plot the baseline and its confidence band (1sigma)
        eff = angeff_plot(angacc, cosKd, cosLd, hphid, proj)
        norm = np.trapz(eff, x)
        langacc = [p.uvalue for p in angacc.values()]
        yunc = wrap_unc(lambda p: angeff_plot(p, cosKd, cosLd, hphid, proj), langacc)
        yl, yh = get_confidence_bands(yunc)
        ax.fill_between(x, pint*yl/norm, pint*yh/norm, alpha=0.2)
        ax.plot(x, pint*eff/norm, label='full range', color='C0')
        err = mcc/tgc-np.sqrt((len(mcc)*mcc)/(len(tgc)*tgc))
        ax.errorbar(cbs, mcc/tgc, yerr=[err,err], xerr=[cbs-tge[:-1],tge[1:]-cbs],
                    fmt='o', color='C0')

        # then run over all time-binned angular acceptances
        for i, angacc_ in enumerate(angaccs[1:]):
          # place time cuts in both samples
          ll, ul = knots[i:i+2]
          timecut = f"time>={ll} & time<={ul}"
          smc = mc_sample.df.query(trigger_scissors(trigger, timecut))
          tmc = toy_sample.query(timecut)

          # first create points
          # get the number of events in sample
          tlen = len(tmc[var]); slen = len(smc[var])

          # calculate the integral
          tgc, tge = np.histogram(tmc[var], bins=200, density=True);
          mcc, mce = np.histogram(smc[var], bins=tge, weights=smc.eval(sw), density=True);
          cbs = 0.5*(tge[1:]+tge[:-1])
          pint = np.trapz(mcc/tgc,cbs)
          tgc, tge = np.histogram(tmc[var], bins=20, density=True);
          mcc, mce = np.histogram(smc[var], bins=tge, weights=smc.eval(sw), density=True);
          cbs = 0.5*(tge[1:]+tge[:-1])

          # load angular acceptance parameters for this year-category
          angacc = Parameters.load(angacc_)

          # plot efficienty
          eff = angeff_plot(angacc, cosKd, cosLd, hphid, proj)
          eff /= np.trapz(eff, x)
          ax.plot(x,pint*eff, '-.',label=f'$t \in ({ll},{ul})$', color=f'C{i+1}')
          err = mcc/tgc-np.sqrt((len(mcc)*mcc)/(len(tgc)*tgc))
          ax.errorbar(cbs, mcc/tgc, yerr=[err,err],
                      xerr=[cbs-tge[:-1],tge[1:]-cbs], fmt='.', color=f'C{i+1}')
    ax.set_xlabel(f'${tex}$')
    ax.set_ylabel(f'$\\varepsilon({tex})$ [a.u.]')
    ax.legend()
    ax.set_ylim(0.9,1.25)
    #ax.set_title(f"{year} {trigger}")
    fig.savefig(output[k])
