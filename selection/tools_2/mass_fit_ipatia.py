__all__ = []
import os
import argparse
import json
import cppyy


import os
import ipanema
import argparse
import uproot3 as uproot
import numpy as np
from ipanema import (ristra, plotting, Sample)
import matplotlib.pyplot as plt
from utils.strings import printsec, printsubsec
from utils.helpers import trigger_scissors


TRIGGER_TYPES = {'biased': 'hlt1b==1',
                 'unbiased': 'hlt1b==0',
                 'noTrigCat': ''}

FIT_OPTS = dict(
    NumCPU=4,
    Timer=1,
    Save=True,
    Verbose=False,
    Optimize=2,
    Minimizer='Minuit2')


# initialize ipanema3 and compile lineshapes
ipanema.initialize(config.user['backend'], 1)
prog = THREAD.compile("""
#define USE_DOUBLE 1
#include <ipanema/core.c>
#include <ipanema/complex.c>
#include <ipanema/special.c>
#include <ipanema/lineshapes.c>
#include <exposed/kernels.ocl>
""", compiler_options=[f"-I{ipanema.IPANEMALIB}"])


# PDF models ------------------------------------------------------------------
#    Select pdf model to fit data. {{{


# CB + ARGUS + EXP {{{

def ipatia_exponential(mass, signal, nsig, nbkg,
                       mu, sigma, lambd, zeta, beta, aL, nL, aR, nR,
                       b, norm=1):
  # ipatia
  prog.py_ipatia(signal, mass, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(mass)))
  backgr = ristra.exp(mass * b)
  # normalize
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = _x * 0
  prog.py_ipatia(_y, _x, np.float64(mu), np.float64(sigma),
                 np.float64(lambd), np.float64(zeta), np.float64(beta),
                 np.float64(aL), np.float64(nL), np.float64(aR),
                 np.float64(nR), global_size=(len(_x)))
  nsignal = np.trapz(ristra.get(_y), ristra.get(_x))
  nbackgr = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = norm * (nsig * signal / nsignal + (1. - nsig) * backgr / nbackgr)
  return ans

# }}}


# CB + PHYSBKG + EXP {{{

def cb_physbkg(mass, signal, nsig, nbkg, nexp, mu, sigma, aL, nL, aR, nR, b, m0=990, c=1,
               p=1, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  prog.py_physbkg(signal, mass, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(mass)))
  pargus = ristra.get(signal)
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize rgus
  prog.py_physbkg(_y, _x, np.float64(m0), np.float64(c), np.float64(p),
                  global_size=(len(_x)))
  nargus = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + nexp * (pexpo / nexpo) + nbkg * (pargus / nargus)
  return norm * ans


def cb_exponential(mass, signal, nsig, mu, sigma, aL, nL, aR, nR, b, norm=1):
  # compute backgrounds
  pexpo = ristra.get(ristra.exp(mass * b))
  # get signal
  prog.py_double_crystal_ball(signal, mass, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(mass)))
  pcb = ristra.get(signal)
  # normalize arrays
  _x = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000)
  _y = ristra.linspace(ristra.min(mass), ristra.max(mass), 1000) * 0
  # normalize cb-shape
  prog.py_double_crystal_ball(_y, _x, np.float64(mu), np.float64(sigma),
                              np.float64(aL), np.float64(nL), np.float64(aR),
                              np.float64(nR), global_size=(len(_x)))
  npb = np.trapz(ristra.get(_y), ristra.get(_x))
  # normalize exp
  nexpo = np.trapz(ristra.get(ristra.exp(_x * b)), ristra.get(_x))
  # compute pdf value
  ans = nsig * (pcb / npb) + (1. - nsig) * (pexpo / nexpo)
  return norm * ans
# }}}

# }}}


def read_params(params_to_fix_file):
  with open(params_to_fix_file, 'r') as stream:
    return json.load(stream)


def ipatia_exp(data, mass, mass_range, mode, background, trig_type):

  if 'Bd' in mode:
    alpha1_value = 1.1  # 2.15
    alpha1_max = 10
    alpha2_value = 2.5  # 2.34
    alpha2_max = 10
    n1_value = 2.4
    n2_value = 3.3
    if trig_type == 'noTrigCat':
      sigma_value = 9.4
      m_sig_lambda_value = -2.1
      gamma_value = -0.0011
      s_yield = 0.82 * data.sumEntries()
      b_yield = 0.18 * data.sumEntries()
    elif trig_type == 'biased':
      sigma_value = 9.3
      m_sig_lambda_value = -2.7
      gamma_value = -0.001
      s_yield = 0.77 * data.sumEntries()
      b_yield = 0.03 * data.sumEntries()
    else:
      sigma_value = 9.1
      m_sig_lambda_value = -2.6
      gamma_value = -0.0011
      s_yield = 0.7 * data.sumEntries()
      b_yield = 0.03 * data.sumEntries()

  if 'Bu' in mode:
    sigma_value = 11.2
    m_sig_lambda_value = -2.65
    alpha1_max = 5
    alpha2_value = 2.23
    alpha2_max = 5
    n1_value = 2.83
    n2_value = 3.14
    gamma_value = -0.001
    if trig_type == 'noTrigCat':
      alpha1_value = 2.2  # 1.99
      s_yield = 0.40 * data.sumEntries()
      b_yield = 0.60 * data.sumEntries()
    elif trig_type == 'biased':
      alpha1_value = 2.2  # 1.99
      s_yield = 0.35 * data.sumEntries()
      b_yield = 0.02 * data.sumEntries()
    else:
      alpha1_value = 2.45  # 1.99
      s_yield = 0.9 * data.sumEntries()
      b_yield = 0.09 * data.sumEntries()

  # signal
  mean = RealVar(f'mean_{trig_type}', Unit='MeV', Value=5279.9, MinMax=mass_range)
  sigma = RealVar(f'sigma_{trig_type}', Unit='MeV', Value=sigma_value, MinMax=(3, 20))
  m_sig_lambda = RealVar(f'm_sig_lambda_{trig_type}', Title='B Mass resolution lambda', Value=m_sig_lambda_value, MinMax=(-6, 2))
  m_sig_zeta = RealVar(f'm_sig_zeta_{trig_type}', Title='B Mass resolution zeta', Value=0)
  m_sig_beta = RealVar(f'm_sig_beta_{trig_type}', Title='B Mass resolution beta', Value=0)
  alpha1 = RealVar(f'alpha1_{trig_type}', Value=alpha1_value, MinMax=(1, alpha1_max))
  alpha2 = RealVar(f'alpha2_{trig_type}', Value=alpha2_value, MinMax=(1, alpha2_max))
  n1 = RealVar(f'n1_{trig_type}', Value=n1_value, MinMax=(1, 5))  # 10
  n2 = RealVar(f'n2_{trig_type}', Value=n2_value, MinMax=(1, 5))  # 10
  pdf_s = Pdf(Name=f'pdf_s_{trig_type}', Type=RooIpatia2, Parameters=(mass, m_sig_lambda, m_sig_zeta, m_sig_beta, sigma, mean, alpha1, n1, alpha2, n2))

  # background
  if background:
    gamma = RealVar(f'gamma_{trig_type}', Value=gamma_value, MinMax=(-1, 1))
    pdf_b = Pdf(Name=f'pdf_b_{trig_type}', Type=RooExponential, Parameters=(mass, gamma))
    tot = data.sumEntries()
    signal = Component(f'signal_{trig_type}', (pdf_s, ), Yield=(s_yield, 0, tot))
    background = Component(f'background_{trig_type}', (pdf_b, ), Yield=(b_yield, 0, tot))
    pdf = buildPdf(Components=(signal, background), Observables=[mass], Name=f'pdf_{trig_type}')
    return pdf

  return pdf_s


def mass_fit_ipatia(input_file, input_tree_name, input_weight_name, output_file,
                    output_file_tmp, output_tree_name, mode, trigcat, mass_range, params_to_fix_file,
                    params_to_fix_list, background, output_params, output_figures, add_sweights):

  printsec("Bd MC mass fit")
  ofile = ipanema.Sample.from_root(input_file)
  mass_range = (
      min(ofile.df['B_ConstJpsi_M_1']),
      max(ofile.df['B_ConstJpsi_M_1'])
  )
  mass_cut = f'B_ConstJpsi_M_1 > {mass_range[0]} & B_ConstJpsi_M_1 < {mass_range[1]}'

  MODEL = "cb_noghost"
  # read parameters to be fixed in the fit
  params_to_fix = read_params(params_to_fix_file) if params_to_fix_file else None

  types = ['biased', 'unbiased'] if trigcat else ['noTrigCat']

  # Select model and set parameters -------------------------------------------
  #    Select model from command-line arguments and create corresponding set of
  #    paramters
  pars = ipanema.Parameters()
  # Create common set of parameters (all models must have and use)
  pars.add(dict(name='nsig', value=0.90, min=0, max=1, free=True,
                latex=r'N_{signal}'))
  pars.add(dict(name='mu', value=5280, min=5200, max=5400,
                latex=r'\mu'))
  pars.add(dict(name='sigma', value=48, min=5, max=100, free=True,
                latex=r'\sigma'))

  if "cb" in MODEL.split('_'):  # {{{
    # crystal ball tails
    pars.add(dict(name='aL', value=1, latex=r'a_l', min=-50, max=50,
                  free=True))
    pars.add(dict(name='nL', value=2, latex=r'n_l', min=-500, max=500,
                  free=True))
    pars.add(dict(name='aR', value=1, latex=r'a_r', min=-50, max=500,
                  free=True))
    pars.add(dict(name='nR', value=2, latex=r'n_r', min=-500, max=500,
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
  pars.add(dict(name='b', value=-0.0014, min=-1, max=1, latex=r'b'))
  # pars.add(dict(name='nexp', value=0.02, min=0, max=1, free=True,
  #               formula=f"1-nsig{'-nbkg' if 'nbkg' in pars else ''}",
  #               latex=r'N_{exp}'))
  # }}}
  print(pars)

  def fcn(params, data):
    p = params.valuesdict()
    prob = pdf(data.mass, data.pdf, **p)
    return -2.0 * np.log(prob) * ristra.get(data.weight)

  trig_type = 'combined'
  current_cut = trigger_scissors(trig_type, mass_cut)
  # pdf_pars = pdf.getParameters(data)
  rd = Sample.from_root(input_file)
  rd.chop(current_cut)
  rd.allocate(mass=f'B_ConstJpsi_M_1', pdf=f'0*B_ConstJpsi_M_1', weight=f'B_ConstJpsi_M_1/B_ConstJpsi_M_1')
  print(rd)
  printsubsec(f"Fitting {len(rd.mass)} events.")

  # if fix parameters in the fit
  if params_to_fix:
    cat_params = params_to_fix[trig_type]
    # if fix parameters in the fit
    for par in params_to_fix_list:
      mc_param = [p for p in cat_params if par in p['Name']][0]
      par_name = mc_param['Name']
      pdf_pars.find(par_name).setVal(mc_param['Value'])
      pdf_pars.find(par_name).setConstant(True)
      print('Setting parameter', par_name, 'to constant with value', mc_param['Value'])

  res = ipanema.optimize(fcn, pars, fcn_kwgs={'data': rd}, method='minuit', verbose=True)
  print(res)
  # result.Print('v')
  # plot_mass(ds, mode, mass, pdf, trig_type, plots_loc)
  if res:
    print(res)
    fpars = ipanema.Parameters.clone(res.params)
  else:
    print("could not fit it!. Cloning pars to res")
    fpars = ipanema.Parameters.clone(pars)

  fig, axplot, axpull = plotting.axes_plotpull()
  hdata = ipanema.histogram.hist(ristra.get(rd.mass), weights=None,
                                 bins=50, density=False)
  axplot.errorbar(hdata.bins, hdata.counts,
                  yerr=[hdata.errh, hdata.errl],
                  xerr=2 * [hdata.edges[1:] - hdata.bins], fmt='.k')

  norm = hdata.norm * (hdata.bins[1] - hdata.bins[0])
  mass = ristra.linspace(ristra.min(rd.mass), ristra.max(rd.mass), 1000)
  signal = 0 * mass

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
  os.makedirs(output_figures, exist_ok=True)
  fig.savefig(os.path.join(output_figures, "mass.pdf"))
  axplot.set_yscale('log')
  axplot.set_ylim(1e0, 1.5 * np.max(y))
  fig.savefig(os.path.join(output_figures, "logmass.pdf"))
  plt.close()

  # Dump parameters to json
  number_of_events = len(rd.mass)
  for par in ['nsig']:
    _par = number_of_events * fpars[par].uvalue
    fpars[par].set(value=_par.n, stdev=_par.s)
  fpars.dump(output_params)


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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-file', help='Path to the input file')
  parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the tree')
  parser.add_argument('--input-weight-name', default='', help='Name of the input weight if any')
  parser.add_argument('--output-file', help='Output ROOT file')
  parser.add_argument('--output-file-tmp', help='Path to the temp output file so save sweights')
  parser.add_argument('--output-tree-name', default='DecayTree', help='Name of the tree')
  parser.add_argument('--mode', help='Name of the selection in yaml')
  parser.add_argument('--trigcat', action='store_true', help='Split tuple in trigger category if specified')
  parser.add_argument('--mass-range', nargs='+', type=int, default=(5210, 5350), help='Specify mass range for the fit')
  parser.add_argument('--background', action='store_true', help='Add exponential background?')
  parser.add_argument('--params-to-fix-file', help='Yaml with dict of parameter names and values')
  parser.add_argument('--params-to-fix-list', nargs='+', help='Yaml with list of parameter names to be fixed from params-to-fix-file')
  parser.add_argument('--output-params', help='To which file to save fit result')
  parser.add_argument('--output-figures', help='Where to store plots')
  parser.add_argument('--add-sweights', action='store_true', help='Calculate and add sweights if specified')
  args = parser.parse_args()
  mass_fit_ipatia(**vars(args))
