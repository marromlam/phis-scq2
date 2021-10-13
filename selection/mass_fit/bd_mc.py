import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, plotting, Sample)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors


# initialize ipanema3 and compile lineshapes
ipanema.initialize(os.environ['IPANEMA_BACKEND'], 1)
prog = THREAD.compile("""
#define USE_DOUBLE 1
#include <ipanema/core.c>
#include <ipanema/complex.c>
#include <ipanema/special.c>
#include <ipanema/lineshapes.c>
#include <exposed/kernels.ocl>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])


# ipatia + exponential {{{

def ipatia_exponential(mass, signal, nsig, nbkg,
           mu, sigma, lambd, zeta, beta, aL, nL, aR, nR,
           b, norm=1):
    # ipatia
    prog.py_ipatia(signal, mass, np.float64(mu), np.float64(sigma),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(mass)))
    backgr = ristra.exp(mass*b)
    # normalize
    _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
    _y = _x*0
    prog.py_ipatia(_y, _x, np.float64(mu), np.float64(sigma),
                   np.float64(lambd), np.float64(zeta), np.float64(beta),
                   np.float64(aL), np.float64(nL), np.float64(aR),
                   np.float64(nR), global_size=(len(_x)))
    nsignal = np.trapz(ristra.get(_y), ristra.get(_x))
    nbackgr = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
    # compute pdf value
    ans = norm * ( nsig*signal/nsignal + (1.-nsig)*backgr/nbackgr )
    return ans

# }}}


# crystal-ball + exponential {{{

def cb_exponential(mass, signal, nsig, mu, sigma, aL, nL, aR, nR, b, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass*b))
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(mu), np.float64(sigma),
                      np.float64(aL), np.float64(nL), np.float64(aR),
                      np.float64(nR), global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)*0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mu), np.float64(sigma),
                      np.float64(aL), np.float64(nL), np.float64(aR),
                      np.float64(nR), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x*b)), ristra.get(_x))
  # compute pdf value
  ans = nsig*(pcb/npb) + (1.-nsig)*(pexpo/nexpo)
  return norm*ans

# }}}


def mass_fitter(odf,
                mass_range=False, mass_branch='B_ConstJpsi_M_1',
                figs = False, label=False,
                trigger='combined', verbose=False):

    # mass range cut
    if not mass_range:
      mass_range = ( min(odf[mass_branch]), max(odf[mass_branch]))
    mass_cut = f'B_ConstJpsi_M_1 > {mass_range[0]} & B_ConstJpsi_M_1 < {mass_range[1]}'
    # mass_cut = f'B_ConstJpsi_M_1 > 5220 & B_ConstJpsi_M_1 < 5330'

    # mass cut and trigger cut 
    current_cut = trigger_scissors(trigger, mass_cut)

    MODEL = "cb_noghost"


    # Select model and set parameters {{{
    #    Select model from command-line arguments and create corresponding set of
    #    paramters
    pars = ipanema.Parameters()
    # Create common set of parameters (all models must have and use)
    pars.add(dict(name='nsig', value=0.50, min=0.2, max=1, free=True,
                  latex=r'N_{signal}'))
    pars.add(dict(name='mu', value=5280, min=5200, max=5400,
                  latex=r'\mu'))
    pars.add(dict(name='sigma', value=8, min=5, max=100, free=True,
                  latex=r'\sigma'))

    if "cb" in MODEL.split('_'):  # {{{
      # crystal ball tails
      pars.add(dict(name='aL', value=1.4, latex=r'a_l',min=-50, max=50,
                    free=True))
      pars.add(dict(name='nL', value=1, latex=r'n_l',min=-500, max=500,
                    free=True))
      pars.add(dict(name='aR', value=1.5, latex=r'a_r',min=-50, max=500,
                    free=True))
      pars.add(dict(name='nR', value=1, latex=r'n_r',min=-500, max=500,
                    free=True))
      if "argus" in MODEL.split('_'):
        pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                      latex=r'N_{part.reco.}'))
        pars.add(dict(name='c', value=20, min=-1000, max=100, free=True,
                      latex=r'c'))
        pars.add(dict(name='p', value=1, min=0.1, max=50, free=True,
                      latex=r'p'))
        pars.add(dict(name='m0', value=5155, min=5100, max=5220, free=True,
                      latex=r'm_0'))
        pdf = cb_argus
        print("Using CB + argus pdf")
      elif "physbkg" in MODEL.split('_'):
        pars.add(dict(name='nbkg', value=0.02, min=0, max=1, free=True,
                      latex=r'N_{background}'))
        pars.add(dict(name='c', value=0.001, min=-1000, max=100, free=True,
                      latex=r'c'))
        pars.add(dict(name='p', value=1, min=0.01, max=50, free=True,
                      latex=r'p'))
        pars.add(dict(name='m0', value=5175, min=5150, max=5200, free=True,
                      latex=r'm_0'))
        pdf = cb_physbkg
        print("Using CB + physbkg pdf")
      else:
        # pars.add(dict(name='nbkg', value=0.00, min=0, max=1, free=False,
        #               latex=r'N_{background}'))
        pdf = cb_exponential
      # }}}
    elif "ipatia" in MODEL.split('_'):
      # ipatia tails {{{
      pars.add(dict(name='lambd', value=-1, min=-20, max=0, free=True,
                    latex=r'\lambda'))
      pars.add(dict(name='zeta', value=0.0, latex=r'\zeta', free=False))
      pars.add(dict(name='beta', value=0.0, latex=r'\beta', free=False))
      pars.add(dict(name='aL', value=1, latex=r'a_l', free=True))
      pars.add(dict(name='nL', value=30, latex=r'n_l', free=True))
      pars.add(dict(name='aR', value=1, latex=r'a_r', free=True))
      pars.add(dict(name='nR', value=30, latex=r'n_r', free=True))
      pdf = ipatia
      # }}}

    # EXPONENCIAL Parameters {{{
    pars.add(dict(name='b', value=-0.005, min=-1, max=1, latex=r'b'))
    # pars.add(dict(name='nexp', value=0.02, min=0, max=1, free=True,
    #               formula=f"1-nsig{'-nbkg' if 'nbkg' in pars else ''}",
    #               latex=r'N_{exp}'))
    # }}}
    # print(pars)
    # }}}

    def fcn(params, data):
        p = params.valuesdict()
        prob = pdf(data.mass, data.pdf, **p)
        return -2.0 * np.log(prob) * ristra.get(data.weight)
    
    print(current_cut)
    rd = Sample.from_pandas(odf)
    rd.chop(current_cut)
    rd.allocate(mass=f'B_ConstJpsi_M_1', pdf=f'0*B_ConstJpsi_M_1', weight=f'B_ConstJpsi_M_1/B_ConstJpsi_M_1')
    # print(rd)

    # res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='nelder', verbose=verbose)
    # for name in ['nsig', 'mu', 'sigma']:
    #     pars[name].init = res.params[name].value
    res = ipanema.optimize(fcn, pars, fcn_kwgs={'data':rd}, method='minuit', verbose=verbose, strategy=2, tol=0.05)
    if res:
      # print(res)
      fpars = ipanema.Parameters.clone(res.params)
    else:
      print("could not fit it!. Cloning pars to res")
      fpars = ipanema.Parameters.clone(pars)

    fig, axplot, axpull = plotting.axes_plotpull()
    hdata = ipanema.histogram.hist(ristra.get(rd.mass), weights=None,
                                   bins=50, density=False)
    axplot.errorbar(hdata.bins, hdata.counts,
                    yerr=[hdata.errh, hdata.errl],
                    xerr=2*[hdata.edges[1:]-hdata.bins], fmt='.k')

    norm = hdata.norm*(hdata.bins[1]-hdata.bins[0])
    mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
    signal = 0*mass

    # plot signal: nbkg -> 0 and nexp -> 0
    _p = ipanema.Parameters.clone(fpars)
    if 'nbkg' in _p:
      _p['nbkg'].set(value=0)
    if 'nexp' in _p:
      _p['nexp'].set(value=0)
    _x, _y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm))
    axplot.plot(_x, _y, color="C1", label='signal')

    # plot backgrounds: nsig -> 0
    # _p = ipanema.Parameters.clone(fpars)
    # if 'nexp' in _p:
    #   _p['nexp'].set(value=_p['nexp'].value)
    # _p['nsig'].set(value=0)
    # _x, _y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(), norm=hdata.norm))
    # axplot.plot(_x, _y, '-.', color="C2", label='background')

    # plot fit with all components and data
    _p = ipanema.Parameters.clone(fpars)
    x, y = ristra.get(mass), ristra.get(pdf(mass, signal, **_p.valuesdict(),
                                            norm=hdata.norm))
    axplot.plot(x, y, color='C0')
    axpull.fill_between(hdata.bins,
                        ipanema.histogram.pull_pdf(x, y, hdata.bins,
                                                   hdata.counts, hdata.errl,
                                                   hdata.errh),
                        0, facecolor="C0", alpha=0.5)
    axpull.set_xlabel(r'$m(B_d^0)$ [MeV/$c^2$]')
    axpull.set_ylim(-3.5, 3.5)
    axpull.set_yticks([-2.5, 0, 2.5])
    axplot.set_ylabel(rf"Candidates")
    if figs:
      fig.savefig(os.path.join(figs, f"mass_{label}.pdf"))
    axplot.set_yscale('log')
    axplot.set_ylim(1e0,1.5*np.max(y))
    if figs:
      fig.savefig(os.path.join(figs, f"logmass_{label}.pdf"))
    plt.close()

    # Dump parameters to json
    number_of_events = len(rd.mass)
    for par in ['nsig']:
      _par = number_of_events * fpars[par].uvalue
      fpars[par].set(value=_par.n, stdev=_par.s, min=-np.inf, max=+np.inf)
    return fpars



#     if add_sweights:
#         sdata = SData(Name='splot_'+trig_type, Pdf=pdf, Data=ds)
#         addSWeightToTree(sdata.data('signal_'+trig_type), tree_with_sw, f'sw_{trig_type}', trigCut)
#         sufixes.push_back(f'sw_{trig_type}')
# 
#     # write parameters to dictionary
#     pars_dict.update({trig_type: [{'Name': param.GetName(),
#                                    'Value': param.getVal(),
#                                    'Error': param.getError()} for param in pdf_pars]})
# 
#     # save fit result to file
#     if fit_result_file:
#         with open(fit_result_file, 'w') as f:
#             json.dump(pars_dict, f, indent=4)
# 
#     # Store n-tuple in ROOT file.
#     if add_sweights:
#         addProductToTree(tree_with_sw, sufixes, 'sw')
#         for trig_type in types:
#             tree_with_sw.SetBranchStatus(f'sw_{trig_type}', 0)
#         output_reduced_file = TFile(output_file, 'recreate')
#         tree_with_sw_reduced = tree_with_sw.CloneTree()
#         tree_with_sw_reduced.Write(output_tree_name, TObject.kOverwrite)
#         output_reduced_file.Close()
#         print('sWeighted nTuple is saved: ', output_reduced_file)


# vim:foldmethod=marker
