__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']
__all__ = ['mass_fitter']


# Modules {{{

import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, plotting, Sample, splot)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors, cuts_and


# initialize ipanema3 and compile lineshapes
ipanema.initialize(os.environ['IPANEMA_BACKEND'], 1)
prog = THREAD.compile("""
#define USE_DOUBLE 1
#include <ipanema/core.c>
#include <ipanema/complex.c>
#include <ipanema/stats.c>
#include <ipanema/special.c>
#include <ipanema/lineshapes.c>
#include <exposed/kernels.ocl>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])

# }}}


# ipatia + exponential {{{

def ipatia_exponential(mass, signal, nsigBs=0, nsigBd=0, nexp=0,
                       muBs=0, sigmaBs=10, muBd=5000, sigmaBd=50,lambd=0, zeta=0, beta=0, aL=0, nL=0, aR=0, nR=0,
                       b=0, norm=1):
    # ipatia
    prog.py_ipatia(signal, mass, np.float64(muBs), np.float64(sigmaBs),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR), np.float64(nR),
                   global_size=(len(mass)))
    pdfBs = 1.0 * signal.get()
    signal = 0*signal
    prog.py_ipatia(signal, mass, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR), np.float64(nR),
                   global_size=(len(mass)))
    pdfBd = 1.0 * signal.get()
    backgr = ristra.exp(mass*b).get()
    # normalize
    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x*0
    prog.py_ipatia(_y, _x, np.float64(muBs), np.float64(sigmaBs),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)))
    nBs = np.trapz(ristra.get(_y), ristra.get(_x))
    _y = _x*0
    prog.py_ipatia(_y, _x, np.float64(muBd), np.float64(sigmaBd),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)))
    nBd = np.trapz(ristra.get(_y), ristra.get(_x))
    nbackgr = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
    # compute pdf value
    ans = norm * ( nsigBs*pdfBs/nBs + nsigBd*pdfBd/nBd + nexp*backgr/nbackgr )
    return ans

# }}}


# CB Bs2JpsiPhi mass model {{{

def cb_exponential(mass, signal, nsigBs=0, nsigBd=0, nexp=0,
                   muBs=0, sigmaBs=10, muBd=5000, sigmaBd=50, aL=0, nL=0, aR=0, nR=0,
                   b=0, norm=1):
  # compute backgrounds
  pExp = ristra.get(ristra.exp(mass*b))
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(muBs), np.float64(sigmaBs),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pBs = ristra.get(signal)
  prog.py_double_crystal_ball(signal, mass, np.float64(muBd), np.float64(sigmaBd),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pBd = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x*0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(muBs), np.float64(sigmaBs),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  nBs = np.trapz(ristra.get(_y), ristra.get(_x))
  prog.py_double_crystal_ball(_y, _x, np.float64(muBd), np.float64(sigmaBd),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  nBd = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nExp = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
  # compute pdf value
  ans = nsigBs*pBs/nBs + nsigBd*pBd/nBd + nexp*pExp/nExp
  return norm*ans

# }}}


# Bs mass fit function {{{

def mass_fitter(odf,
                mass_range=False, mass_branch='B_ConstJpsi_M_1',
                mass_weight='B_ConstJpsi_M_1/B_ConstJpsi_M_1',
                cut=False,
                figs = False, model=False, has_bd=False,
                trigger='combined', input_pars=False, sweights=False, verbose=False):

    # mass range cut
    if not mass_range:
      mass_range = ( min(odf[mass_branch]), max(odf[mass_branch]))
    mass_cut = f'B_ConstJpsi_M_1 > {mass_range[0]} & B_ConstJpsi_M_1 < {mass_range[1]}'
    # mass_cut = f'B_ConstJpsi_M_1 > 5220 & B_ConstJpsi_M_1 < 5330'

    # mass cut and trigger cut 
    current_cut = trigger_scissors(trigger, cuts_and(mass_cut, cut))
    print(current_cut)
    print(has_bd)


    # Select model and set parameters {{{
    #    Select model from command-line arguments and create corresponding set of
    #    paramters

    pars = ipanema.Parameters()
    # Create common set of Bs parameters (all models must have and use)
    pars.add(dict(name='nsigBs',    value=0.8, min=0.2,  max=1,    free=True,  latex=r'N_{B_s}'))
    pars.add(dict(name='muBs',      value=5367,  min=5200, max=5500,             latex=r'\mu_{B_s}'))
    pars.add(dict(name='sigmaBs',   value=5,    min=5,    max=100,  free=True,  latex=r'\sigma_{B_s}'))

    if input_pars:
      _pars = ipanema.Parameters.clone(input_pars)
      _pars.lock()
      _pars.remove('nsigBs', 'muBs', 'sigmaBs')
      _pars.unlock('b')
      pars = pars + _pars
    else:
      if 'ipatia' in model:
        # Hypatia tails {{{
        pars.add(dict(name='lambd',   value=-1.5,  min=-4,   max=-1.1, free=True,  latex=r'\lambda'))
        pars.add(dict(name='zeta',    value=1e-5,                      free=False, latex=r'\zeta'))
        pars.add(dict(name='beta',    value=0.0,                       free=False, latex=r'\beta'))
        pars.add(dict(name='aL',      value=1.23,  min=0.5, max=5.5,   free=True,  latex=r'a_l'))
        pars.add(dict(name='nL',      value=1.05,  min=0,   max=6,     free=True,  latex=r'n_l'))
        pars.add(dict(name='aR',      value=1.03,  min=0.5, max=5.5,   free=True,  latex=r'a_r'))
        pars.add(dict(name='nR',      value=1.02,  min=0,   max=6,     free=True,  latex=r'n_r'))
        # }}}
      elif "crystalball" in model:
        # Crystal Ball tails {{{
        pars.add(dict(name='aL',      value=1.4,  min=0.5, max=3.5,    free=True,  latex=r'a_l'))
        pars.add(dict(name='nL',      value=1,     min=1,   max=500,   free=True,  latex=r'n_l'))
        pars.add(dict(name='aR',      value=1.4,  min=0.5, max=3.5,    free=True,  latex=r'a_r'))
        pars.add(dict(name='nR',      value=1,     min=1,   max=500,   free=True,  latex=r'n_r'))
        # }}}
      # Combinatorial background
      pars.add(dict(name='b',         value=-4e-3, min=-1,  max=1,     free=True,  latex=r'b'))
      pars.add(dict(name='nexp',      formula="1-nsigBs",                          latex=r'N_{comb}'))

    if has_bd:
      # Create common set of Bd parameters
      DMsd = 5366.89 - 5279.63
      pars.add(dict(name='nsigBd',    value=0.00,  min=0.,  max=0.05,   free=True,  latex=r'N_{B_d}'))
      pars.add(dict(name='muBd',      formula=f"muBs-{DMsd}",                      latex=r'\mu_{B_d}'))
      # pars.add(dict(name='sigmaBd',   value=1,    min=5,    max=20,    free=True,  latex=r'\sigma_{B_d}'))
      pars.add(dict(name='sigmaBd', formula="sigmaBs",                           latex=r'\sigma_{B_d}'))
      # Combinatorial background
      pars.pop('nexp')
      pars.add(dict(name='nexp',     formula="1-nsigBs-nsigBd", latex=r'N_{comb}'))
    print(pars)

    # }}}


    # Chose model {{{

    if model == 'ipatia':
      pdf = ipatia_exponential 
    elif model == 'crystalball':
      pdf = cb_exponential 


    def fcn(params, data):
        p = params.valuesdict()
        prob = pdf(data.mass, data.pdf, **p)
        return -2.0 * np.log(prob) * ristra.get(data.weight)

    # }}}
    

    # Allocate the sample variables {{{

    print(f"Cut: {current_cut}")
    print(f"Mass branch: {mass_branch}")
    print(f"Mass weight: {mass_weight}")
    rd = Sample.from_pandas(odf)
    _proxy = np.float64(rd.df[mass_branch]) * 0.0
    rd.chop(current_cut)
    rd.allocate(mass=mass_branch, pdf=f'0*{mass_branch}', weight=mass_weight)
    print(f"The dataframe has the following shape: {rd.shape}")
    # print(rd)

    # }}}

    # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='nelder', verbose=verbose)
    # for name in ['nsig', 'mu', 'sigma']:
    #     pars[name].init = res.params[name].value
    # res = False
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='minuit',
                           verbose=verbose, strategy=1, tol=0.05)
    if res:
      print(res)
      fpars = ipanema.Parameters.clone(res.params)
    else:
      print("Could not fit it!. Cloning pars to res")
      fpars = ipanema.Parameters.clone(pars)
      print(fpars)

    fig, axplot, axpull = plotting.axes_plotpull()
    hdata = ipanema.histogram.hist(ristra.get(rd.mass), weights=rd.df.eval(mass_weight),
                                   bins=40, density=False)
    axplot.errorbar(hdata.bins, hdata.counts,
                    yerr=[hdata.errh, hdata.errl],
                    xerr=2*[hdata.edges[1:]-hdata.bins], fmt='.k')

    norm = hdata.norm#*(hdata.bins[1]-hdata.bins[0])
    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0*mass

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if 'nsigBd' in _p:
      _p['nsigBd'].set(value=0, min=-np.inf, max=np.inf)
    if 'nexp' in _p:
      _p['nexp'].set(value=0, min=-np.inf, max=np.inf)
    _x, _y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(),
                                              norm=norm))
    axplot.plot(_x, _y, color="C1", label=rf'$B_s^0$ {model}')

    # plot backgrounds: nsig -> 0
    if has_bd:
      _p = ipanema.Parameters.clone(fpars)
      if 'nsigBs' in _p: _p['nsigBs'].set(value=0,min=-np.inf, max=np.inf )
      if 'nexp' in _p:   _p['nexp'].set(value=0, min=-np.inf, max=np.inf)
      _x, _y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(),
                                                norm=norm))
      axplot.plot(_x, _y, '-.', color="C2", label='Bd')

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(), norm=norm))
    axplot.plot(x, y, color='C0')
    axpull.fill_between(hdata.bins,
                        ipanema.histogram.pull_pdf(x, y, hdata.bins,
                                                   hdata.counts, hdata.errl,
                                                   hdata.errh),
                        0, facecolor="C0", alpha=0.5)
    axpull.set_xlabel(r'$m(J/\psi KK)$ [MeV/$c^2$]')
    axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel(rf"Candidates")
    axplot.legend(loc="upper left")
    if figs:
      os.makedirs(figs, exist_ok=True)
      fig.savefig(os.path.join(figs, f"fit.pdf"))
      fig.savefig(f"fit.pdf")
    axplot.set_yscale('log')
    try:
      axplot.set_ylim(1e0,1.5*np.max(y))
    except:
      print('axes not scaled')
    if figs:
      fig.savefig(os.path.join(figs, f"logfit.pdf"))
      fig.savefig(f"logfit.pdf")
    plt.close()


    # compute sWeights if asked {{{

    if sweights:
        # separate paramestes in yields and shape parameters
        _yields = ipanema.Parameters.find(fpars, "nsig.*") + ["nexp"]
        _pars = list(fpars)
        [_pars.remove(_y) for _y in _yields]
        _yields = ipanema.Parameters.build(fpars, _yields)
        _pars = ipanema.Parameters.build(fpars, _pars)

        sw = splot.compute_sweights(lambda *x, **y: pdf(rd.mass, rd.pdf, *x, **y), _pars, _yields)
        for k,v in sw.items():
          _sw = np.copy(_proxy)
          _sw[list(rd.df.index)] = v * np.float64(rd.df.eval(mass_weight))
          sw[k] = _sw
        return (fpars, sw)

    # }}}

    return (fpars, False)

# }}}


# command-line interface {{{

if __name__ == '__main__':
  p = argparse.ArgumentParser(description="mass fit")
  p.add_argument('--sample')
  p.add_argument('--input-params', default=False)
  p.add_argument('--output-params')
  p.add_argument('--output-figures')
  p.add_argument('--mass-model')
  p.add_argument('--mass-weight')
  p.add_argument('--mass-bin', default=False)
  p.add_argument('--trigger')
  p.add_argument('--sweights')
  p.add_argument('--mode')
  args = vars(p.parse_args())
  
  if args['sweights']:
    sweights = True
  else:
    sweights = False

  if args['input_params']:
    input_pars = ipanema.Parameters.load(args['input_params'])
  else:
    input_pars = False
  
  branches = ['B_ConstJpsi_M_1', 'hlt1b', 'X_M']

  if args['mass_weight']:
    mass_weight = args['mass_weight']
    branches += [mass_weight]
  else:
    mass_weight = 'B_ConstJpsi_M_1/B_ConstJpsi_M_1'

  cut = False
  if "prefit" in args['output_params']:
    cut = "B_BKGCAT == 0 | B_BKGCAT == 10 | B_BKGCAT == 50"
    branches += ['B_BKGCAT']
  
  sample = Sample.from_root(args['sample'], branches=branches)
  

  mass_range=(5202, 5548)
  if args['mass_bin']:
    if 'Bd2JpsiKstar' in args['mode']:
      mass = [826, 861, 896, 931, 966]
    elif 'Bs2JpsiPhi' in args['mode']:
      mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    if args['mass_bin'] == 'all':
      mLL = mass[0]
      mUL = mass[-1]
    else:
      bin = int(args['mass_bin'][-1])
      mLL = mass[bin-1]
      mUL = mass[bin]
    if "LSB" in args['mass_bin']:
      mass_range=(5202, 5367+50)
    elif "RSB" in args['mass_bin']:
      mass_range=(5367-80, 5548)
    cut = f"({cut}) & X_M>{mLL} & X_M<{mUL}" if cut else f"X_M>{mLL} & X_M<{mUL}"
  print(f"Cut:", cut)
  # sample.chop(cut)

  pars, sw = mass_fitter(sample.df,
                     mass_range=mass_range, mass_branch='B_ConstJpsi_M_1', mass_weight=mass_weight,
                     figs = args['output_figures'], model=args['mass_model'],
                     cut=cut, sweights=sweights,
                     has_bd=True if args['mode']=='Bs2JpsiPhi' else False,
                     trigger=args['trigger'], input_pars=input_pars, verbose=False)
  pars.dump(args['output_params'])
  if sw: np.save(args['sweights'], sw)

# }}}


# vim:foldmethod=marker
