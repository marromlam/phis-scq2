__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import uproot3 as uproot
import os
# import sys
# import json
# import math as m
import argparse
# import math
import uncertainties
import ipanema
import numpy as np
import complot
import matplotlib.pyplot as plt
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes


ipanema.initialize('cuda', 1)
prog = ipanema.compile(open('analysis/time_resolution/timeres.c').read())


def model1(time, fres, mu, sigma, fprompt, fll, fsl, fwpv, taul, taus, tau1,
           tau2, share, tLL=-4, tUL=10, prob=False, norm=1):
  """"
  Model1: 2 resolution gaussians + long-live component + wrong PV
  complete me
  """
  prog.kernel_time_fit(prob, time, fres,
                       np.float64(mu), sigma, np.float64(fprompt),
                       np.float64(fll), np.float64(fsl), np.float64(fwpv),
                       np.float64(taul), np.float64(taus),
                       np.float64(tau1), np.float64(tau2), np.float64(share),
                       np.float64(tLL), np.float64(tUL),
                       global_size=(len(time),))
  return norm * prob


def fcn(params, time, prob, norm=1):
  p = params.valuesdict()
  pbin = ''
  fres = ipanema.ristra.allocate(np.float64([1 - p[f'f{pbin}'], p[f'f{pbin}']]))
  # fres = ipanema.ristra.allocate(np.float64([p[f'f{pbin}'], 1-p[f'f{pbin}']]))
  mu = p[f'mu{pbin}']
  sigma = ipanema.ristra.allocate(np.float64(
      [p[f'sigma1{pbin}'], p[f'sigma2{pbin}']]))
  fprompt = p[f'fprompt{pbin}']
  fll = p[f'fll{pbin}']
  fsl = p[f'fsl{pbin}']
  fwpv = p[f'fwpv{pbin}']
  taul = p[f'taul{pbin}']
  taus = p[f'taus{pbin}']
  tau1 = p[f'tau1{pbin}']
  tau2 = p[f'tau2{pbin}']
  share = p[f'share{pbin}']
  model1(time=time, prob=prob, fres=fres, mu=mu, sigma=sigma,
         fprompt=fprompt, fll=fll, fsl=fsl, fwpv=fwpv, taul=taul, taus=taus,
         tau1=tau1, tau2=tau2, share=share, norm=norm)
  num = ipanema.ristra.get(prob)
  den = 1
  # normalization
  # _x = np.linspace(-4, 10, 10000)
  # _y = 0 * _x
  # _y = ipanema.ristra.allocate(_y)
  # _x = ipanema.ristra.allocate(_x)
  # model1(time=_x, prob=_y, fres=fres, mu=mu, sigma=sigma, fprompt=fprompt, fll=fll,
  #        fsl=fsl, fwpv=fwpv, taul=taul, taus=taus, tau1=tau1, tau2=tau2,
  #        share=share)
  # # print(_y)
  # den = np.trapz(ipanema.ristra.get(_y), ipanema.ristra.get(_x))
  # print(num, den)
  # print(den)
  # exit()
  return -2.0 * np.log(num / den)
  # return -2.0 * ipanema.ristra.log(prob)


def time_resolution(df, time_range, sigma_range, wpv_shape, mode,
                    weight=False, cut=False, figs=False):
  """
  Master function that creates the fit to the binned in the time error bins
  sample.
  TODO: type_simult functionality
  <> -< >-
  """
  if weight:
    print("time resolution: using the weight ", weight)

  # diego suggests not to use this if
  LONG_LIVED = True
  if mode == 'MC_Bs2JpsiPhi':
    LONG_LIVED = False
    print("LONG_LIVED was disabled")

  tLL, tUL = time_range
  sLL, sUL = sigma_range
  time = 'time'
  if mode == 'MC_Bs2JpsiPhi':
    gentime = 'B_TRUETAU_GenLvl'
  sigmat = 'sigmat'

  list_of_cuts = [
      f"{time}>{tLL} & {time}<{tUL}",
      f"{sigmat}>{sLL} & {sigmat}<{sUL}"
  ]
  if mode == 'MC_Bs2JpsiPhi':
    list_of_cuts.append(
        '~(abs(B_MC_MOTHER_ID)>=541 | abs(B_MC_GD_MOTHER_ID)>=541 | abs(B_MC_GD_GD_MOTHER_ID)>=541)'
    )
    list_of_cuts.append(f"{gentime}>{tLL} & {gentime}<{tUL}")

  cut = "(" + ") & (".join(list_of_cuts) + ")"
  print(cut)
  cdf = df.query(cut)
  if mode == 'MC_Bs2JpsiPhi':
    cdf.eval(f'dt = {time} - 1000*{gentime}', inplace=True)
    time = 'dt'
    cdf = cdf.query(f"dt>-1 & dt<1")
  print(cdf)
  # sLL, sUL = time_error_bins[0], time_error_bins[-1]
  # n_of_bins = len(time_error_bins) - 1

  # NBin = len(terr)-1

  # sigmat = RealVar(Name = 'sigmat', Observable = True, MinMax = (terr[0], terr[NBin]))

  DM = 17.74
  # delta_ms = RealVar(Name = 'delta_ms', Value = 17.74)

  # state_names = []

  # NOT dev ### if type_nPV:
  # NOT dev ###     for i in range(NBin):
  # NOT dev ###         state_names += ["b{}nPV0".format(i)]
  # NOT dev ###     for i in range(NBin):
  # NOT dev ###         state_names += ["b{}nPV1".format(i)]
  # if not type_nPV:
  #      for i in range(NBin):
  #         state_names += ["b{}".format(i)]

  # sigmatCat = Category(Name = "sigmatCat", States = state_names)

  # time = RealVar(Name = 'time', Observable = True, Value = 0., MinMax = (t[0], t[1]))

  # Data = data(data_in, tree_name, time_range, time_error_bins, time, sigmat, state_names, sigmatCat, type_nPV, weight, cut)
  #
  # LERA{'Name': 'f_b0', 'Value': 0.11176278497437758, 'Error': 0.032299735913940454}
  # LERA{'Name': 'frac1_ll_b0', 'Value': 0.3947397194675784, 'Error': 0.022004423552642215}
  # LERA{'Name': 'frac_phys_2_b0', 'Value': 0.9453752139914314, 'Error': 0.010909009312546757}
  # LERA{'Name': 'frac_prompt_b0', 'Value': 0.8225648776824983, 'Error': 0.007048963040368254}
  # LERA{'Name': 'frac_tau1_b0', 'Value': 0.8192100193447707, 'Error': 0.0}
  # LERA{'Name': 'mu_res_b0', 'Value': -0.0014090904634665863, 'Error': 0.00027745920985458133}
  # LERA{'Name': 'sigma_par1_b0', 'Value': 0.02523478232120755, 'Error': 0.00039577215491256934}
  # LERA{'Name': 'sigma_par2_b0', 'Value': 0.010145853353497401, 'Error': 0.0010451197103999571}
  # LERA{'Name': 'tau1_ll_b0', 'Value': 0.10737326512206602, 'Error': 0.01562646584393622}
  # LERA{'Name': 'tau1_wpv_b0', 'Value': 0.34871749861971346, 'Error': 0.0}
  # LERA{'Name': 'tau2_ll_b0', 'Value': 1.3058202489411865, 'Error': 0.0462092334637594}
  # LERA{'Name': 'tau2_wpv_b0', 'Value': 1.9181386272565453, 'Error': 0.0}
  # LERA{'Name': 'D_b0', 'Value': 0.8959930513214064, 'Error': 0.0036077128209776486}
  # LERA{'Name': 'sigma_eff_b0', 'Value': 0.026418447049524122, 'Error': 0.0004842659479439429}
  # LERA{'Name': 'sigma_ave_b0', 'Value': 0.01906966521207871, 'Error': 0}
  # LERA{'Name': 'num_b0', 'Value': 15097.0, 'Error': 122.86984984120393}
  # LERA{'Name': 'frac_wpv_b0_', 'Value': 0.00969235558699772, 'Error': 0.00196963374798581}
  # LERA{'Name': 'D_eff', 'Value': 0.8959930513214064, 'Error': 0.0025510382287676717}

  # combData = Data[0]
  pars = ipanema.Parameters()
  pars.add(dict(name="fprompt", value=0.99,
                min=0.1, max=1, free=True,
                latex=r"f_{\mathrm{prompt}}"))
  pars.add(dict(name="f", value=0.73,
                min=0, max=1, free=True))
  pars.add(dict(name="DM", value=17.74, free=False))
  pars.add(dict(name="mu", value=0,
                min=-10, max=10, free=True))
  pars.add(dict(name="sigmap", value=0.025,
                min=0.001, max=2))
  pars.add(dict(name="sigmapp", value=0.010,
                min=0.001, max=2, free=True))
  pars.add(dict(name="sigma1",
                formula="sigmap - sigmapp * sqrt((f)/(1-f))"))
  pars.add(dict(name="sigma2",
                formula="sigmap + sigmapp * sqrt((1-f)/(f))"))
  # pars.add(dict(name="sigma1", value=0.25, min=0.001, max=2000))
  # pars.add(dict(name="sigma2", value=0.10, min=0.001, max=2000))
  pars.add(dict(name="fphys2", value=0.5 * LONG_LIVED,
                min=0, max=1, free=LONG_LIVED,
                latex=r"f_{\mathrm{phys}}"))
  pars.add(dict(name="fsl", value=0.74 * LONG_LIVED,
                min=0, max=1, free=LONG_LIVED,
                latex=r"f_{\mathrm{short-lived}}"))
  pars.add(dict(name="fll", formula="(1-fprompt)*fphys2",
                latex=r"f_{\mathrm{long-lived}}"))
  pars.add(dict(name="taul", value=0.1,
                min=0, max=1, free=LONG_LIVED))
  pars.add(dict(name="taus", value=1.5,
                min=0.2, max=2.75, free=LONG_LIVED))
  pars.add(dict(name="fwpv",
                formula="(1-fprompt)*(1-fphys2)",
                latex=r"f_{\mathrm{WPV}}"))
  # pars.add(dict(name="fwpv", formula="(1-fprompt)"))
  # add wpv parameters
  wpv_shape.lock()
  pars = pars + wpv_shape
  # parameters for calculation
  pars.add(
      dict(name="part1", formula="(1-f) * exp(-(1/2.) * (sigma1*sigma1) * (DM*DM))"))
  pars.add(
      dict(name="part2", formula="f  * exp(-(1/2.) * (sigma2*sigma2) * (DM*DM))"))
  pars.add(dict(name="dilution", formula="part1 + part2"))
  pars.add(dict(name="sigmaeff", formula="sqrt(-2*log(part1+part2))/DM"))
  print(pars)

  # fit
  timed = ipanema.ristra.allocate(np.float64(cdf[time].values))
  sigmatd = ipanema.ristra.allocate(np.float64(cdf[sigmat].values))
  prob = 0 * timed
  res = ipanema.optimize(fcn, pars, fcn_args=(timed, prob),
                         method='minuit', tol=0.05, verbose=True)
  print(res)
  fpars = ipanema.Parameters.clone(res.params)
  for k, v in fpars.items():
    v.min = -np.inf
    v.max = +np.inf
    v.set(value=res.params[k].value, min=-np.inf, max=np.inf)

  wexp = uncertainties.wrap(np.exp)
  wlog = uncertainties.wrap(np.log)
  wsqrt = uncertainties.wrap(np.sqrt)
  pars = res.params
  f = pars['f'].uvalue
  sigma1 = pars['sigmap'].uvalue - pars['sigmapp'].uvalue * ((f) / (1 - f))**0.5
  sigma2 = pars['sigmap'].uvalue + pars['sigmapp'].uvalue * ((1 - f) / (f))**0.5
  exp1 = wexp(-(1 / 2.) * (sigma1 * pars['DM'].value)**2)
  exp2 = wexp(-(1 / 2.) * (sigma2 * pars['DM'].value)**2)
  part1 = (1 - pars['f'].uvalue) * exp1
  part2 = (0 + pars['f'].uvalue) * exp2
  dilution = part1 + part2
  sigmaeff = wsqrt(-2 * wlog(part1 + part2)) / pars['DM'].value
  pars['sigmaeff'].set(value=sigmaeff.n, stdev=sigmaeff.s)
  pars['dilution'].set(value=dilution.n, stdev=dilution.s)
  sigma_ave = np.mean(ipanema.ristra.get(sigmatd))
  pars.add(dict(name='sigmaAverage', value=sigma_ave, stdev=0))
  nevts = len(timed)
  nevts = uncertainties.ufloat(nevts, np.sqrt(nevts))
  pars.add(dict(name='nevts', value=nevts.n, stdev=nevts.s))
  print(f"Dilution:          {dilution:.2uL}")
  print(f"Sigma:             {sigmaeff:.2uL}")
  print(f"Average of sigmat: {sigma_ave}")
  print(f"Number of events:  {nevts:.2uL}")
  print("New set of parameters")
  print(pars)

  timeh = ipanema.ristra.get(timed)
  sigmath = ipanema.ristra.get(sigmatd)
  # sigmath = np.float64(cdf[sigmat].values)
  probh = 0 * timeh

  # _mass = ristra.get(rd.mass)
  # _weight = rd.df.eval(mass_weight)

  fig, axplot, axpull = complot.axes_plotpull()
  hdata = complot.hist(timeh, bins=2000, density=False)

  axplot.errorbar(hdata.bins, hdata.counts, yerr=hdata.yerr, xerr=hdata.xerr,
                  fmt=".k")

  proxy_mass = ipanema.ristra.linspace(min(timeh), max(timeh), 50000)
  proxy_prob = 0 * proxy_mass

  def pdf(params, time, prob, norm):
    lkhd = fcn(params, time, prob, norm=1)
    return norm * np.exp(-lkhd / 2)

  # plot subcomponents
  species_to_plot = ['fprompt', 'fll', 'fwpv']
  for icolor, pspecie in enumerate(species_to_plot):
    _color = f"C{icolor+1}"
    _label = rf"${fpars[pspecie].latex.split('f_{')[-1][:-1]}$"
    # print(_label)
    _p = ipanema.Parameters.clone(fpars)
    # print(_p)
    for f in _p.keys():
      # print("fraction", f)
      # _p[f].set(value=fpars[pspecie].value, min=-np.inf, max=np.inf)
      if f.startswith('f') and f != pspecie:
        # print(f"looking at {f}")
        if len(f) == 1:
          0
          # print('f as is')
        elif 'fphys' in f:
          0
          # print('fphys as is')
        else:
          _p[f].set(value=0, min=-np.inf, max=np.inf)
    total_frac = fpars['fprompt'].value + fpars['fsl'].value + fpars['fll'].value + fpars['fwpv'].value
    # print(_p)
    # print("-----------------------------------------------")
    _prob = pdf(_p, proxy_mass, proxy_prob, norm=hdata.norm)
    _prob = np.nan_to_num(_prob)
    print(_prob, ipanema.ristra.sum(prob))
    axplot.plot(ipanema.ristra.get(proxy_mass), _p[pspecie].value * ipanema.ristra.get(_prob) / total_frac,
                color=_color, linestyle='--', label=_label)

  # plot fit with all components and data
  _p = ipanema.Parameters.clone(res.params)
  _prob = pdf(_p, proxy_mass, proxy_prob, norm=hdata.norm)
  axplot.plot(ipanema.ristra.get(proxy_mass), _prob, color="C0",
              label=rf"full fit $ {sLL} < \sigma_t < {sUL}$")
  pulls = complot.compute_pdfpulls(ipanema.ristra.get(proxy_mass), ipanema.ristra.get(_prob),
                                   hdata.bins, hdata.counts, *hdata.yerr)
  axpull.fill_between(hdata.bins, pulls, 0, facecolor="C0", alpha=0.5)

  # label and save the plot
  axpull.set_xlabel(r"$t$ [ps]")
  axplot.set_ylabel(r"Candidates")
  axpull.set_ylim(-6.5, 6.5)
  axpull.set_xlim(-0.5, 0.5)
  axplot.set_xlim(-0.5, 0.5)
  axpull.set_yticks([-5, 0, 5])
  # axpull.set_yticks([-2, 0, 2])
  # axpull.hlines(3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)
  # axpull.hlines(-3, mLL, mUL, linestyles='dotted', color='k', alpha=0.2)

  # axplot.legend(loc="upper right", prop={'size': 8})
  axplot.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 16})
  if figs:
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, f"fit.pdf"))
  axplot.set_yscale("log")
  try:
    axplot.set_ylim(1e0, 1.5 * np.max(_prob))
  except:
    print("axes not scaled")
  if figs:
    v_mark = 'LHC$b$'  # watermark plots
    tag_mark = 'THIS THESIS'
    watermark(axplot, version="final", scale=10.3)
    fig.savefig(os.path.join(figs, f"logfit.pdf"))
  plt.close()

  # TODO: p2vv computes also the effective dilution. but for that we need to
  #       compute the dilution in each of the sigmat bins. so this must be
  #       done in another script
  # calculate integrals of the long-lived component
  # D_eff = np.sqrt(
  #   sum(
  #     [i*j*j for i, j in zip(eff_dilution['number'],eff_dilution['value'])]
  #   )/sum(eff_dilution['number']))
  # D_eff_err = m.sqrt(pow(1./(2.*D_eff*sum(eff_dilution['number'])),2)*(pow(D_eff,2)/sum(eff_dilution['number'])+ sum([pow(i,2)/j for i, j in zip(eff_dilution['value'], eff_dilution['number'])]) + 2*sum([pow(i*j*k,2) for i,j,k in zip(eff_dilution['value'], eff_dilution['number'] , eff_dilution['error'])]) ))
  # pdf_pars.append({'Name':'D_eff', 'Value':D_eff, 'Error':D_eff_err})

  # time_bias = [ p for p in pdf_pars if p['Name'][:2] == 'mu']
  # llDataSetFile = TFile.Open(data_out, "recreate")

  # generate long live component
  _p = ipanema.Parameters.clone(fpars)
  _p['fprompt'].set(value=0, min=-np.inf, max=np.inf)
  _p['fwpv'].set(value=0, min=-np.inf, max=np.inf)

  ll_pdf = pdf(_p, proxy_mass, proxy_prob, norm=hdata.norm)
  # ll_integrals = []
  # ll_datasets = []
  #
  # for i in range(len(time_bias)):
  #     time.setRange("r"+str(i), [ t[0], time_bias[i]['Value']])
  #
  #     if type_simult:
  #
  #         pdf = total_pdf[i].pdfList().find("longl_b{}".format(i))
  #         integral = pdf.createIntegral(RooArgSet(time._var), "r"+str(i))
  #         pdf_pars.append( {"Name":'ll_integral_b{}'.format(i), "Value": integral.getVal() , "Error" : 0.})
  #         ll_dataset = pdf.generate(time._var, 1000000)
  #         ll_dataset.SetName('b_{}'.format(i))
  #         ll_dataset.Write()

  #     else:
  #         pdf = total_pdf[i].pdfList().find("longl_b{}".format(i))
  #         integral = pdf.createIntegral(RooArgSet(time._var), "r"+str(i))
  #         pdf_pars.append( {"Name":'ll_integral_b{}'.format(i), "Value": integral.getVal() , "Error" : 0.})
  #         ll_dataset = pdf.generate(RooArgSet(time._var), 10000)
  #         ll_datasets.append(ll_dataset)
  #         ll_dataset.SetName('b_{}'.format(i))
  #         ll_ttree = ll_dataset.GetClonedTree()
  #         ll_ttree.SetName('b_{}'.format(i))
  #         ll_ttree.Write()
  #

  return pars


if __name__ == '__main__':
  DESCRIPTION = """
  Calibration of the decay time resolution.
  """
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('--in-data', help='Input prompt data.')
  # parser.add_argument('--time-range', type=float, nargs='+', help='Time range in pico seconds.')
  # parser.add_argument('--time-error-bins', nargs='+', type=float, help='Decay time error in bins')
  parser.add_argument('--in-wpv', default='None',
                      help='File to the description of wpv shape from the event mixing')
  parser.add_argument('--timeres', help='Specify resolution model type')
  parser.add_argument('--mode', help='Specify resolution model type')
  # parser.add_argument('--type-wpv', choices=['CLASSICAL', 'EXPGAUSS', 'TRIPPLE', 'RESOLGAUSS', 'RESOLGAUSSDOUBLE'], help='Specify WPV model type')
  parser.add_argument('--type-nPV', action='store_true',
                      help='Categorize fit in nPV')
  # parser.add_argument('--type-simult', action='store_true', help = 'Is fit simultaneous?' )
  # parser.add_argument('--weight', default = 'None', help='Add weight to the data')
  # parser.add_argument('--cut', default = 'None', help = 'General dataset cut')
  parser.add_argument('--out-json', help='Location to save fit parameters')
  parser.add_argument('--out-plots-out', help='Location to create plots')
  parser.add_argument(
      '--out-data', help='Dataset for the LL component sampling')
  args = vars(parser.parse_args())

  timeres_binning = [0.010, 0.021, 0.026, 0.032, 0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
  # data_in = root://eoslhcb.cern.ch//eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r5/Bs2JpsiPhi_Prompt/2018/Bs2JpsiPhi_Prompt_2018_selected_bdt_v0r5.root
  # tree_name = DecayTree
  # time_range = [-4.0, 10.0]
  # time_error_bins = [0.01, 0.021, 0.026, 0.032, 0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
  # wpv_shape = /scratch17/marcos.romero/Bs2JpsiPhi-FullRun2/p2vv/scripts/run2/output/time_resolution/wpv_shape_mixEvent/parameters_CLASSICAL_2018_bins_nominal.json
  # mode = 'Bs2JpsiPhi_Prompt'
  # type_res = DOUBLE
  # type_wpv = CLASSICAL
  # type_nPV = False
  # type_simult = False
  # weight = None
  # cut = None
  # time_res_out = /scratch17/marcos.romero/Bs2JpsiPhi-FullRun2/p2vv/scripts/run2/output/time_resolution/prompt/Bs_prompt/2018/time_resolution_cleaned_double_classical_binning_nominal_cut_None.json
  # plots_out = /scratch17/marcos.romero/Bs2JpsiPhi-FullRun2/p2vv/scripts/run2/output/time_resolution/prompt/Bs_prompt/2018/plots_classical/binning_nominal/cut_None
  # data_out = /scratch17/marcos.romero/Bs2JpsiPhi-FullRun2/p2vv/scripts/run2/output/time_resolution/prompt/Bs_prompt/2018/long_lived_binning_nominal_cut_None.root

  current_bin, total_bins = args['timeres'][6:].split('of')
  current_bin, total_bins = int(current_bin), int(total_bins)
  print(current_bin)
  branches = ['time', 'sigmat', 'B_PT']

  mode = args['mode']
  if mode == 'MC_Bs2JpsiPhi':
    branches.append('B_MC_MOTHER_ID')
    branches.append('B_MC_GD_MOTHER_ID')
    branches.append('B_MC_GD_GD_MOTHER_ID')
    branches.append('B_TRUETAU')
    branches.append('B_TRUETAU_GenLvl')
  wpv = ipanema.Parameters.load(args['in_wpv'])
  print(wpv)
  weight = False
  cut = False
  # load dataframe
  df = uproot.open(args['in_data'])
  df = df[list(df.keys())[0]].pandas.df(branches=branches)

  # set time and sigmat bins
  time_range = [-4, 10]
  sigma_range = timeres_binning[current_bin - 1:current_bin + 1]
  if total_bins == 1:
    sigma_range = [timeres_binning[0], timeres_binning[-1]]

  print(sigma_range)
  print("mode", mode)
  # exit()

  # fit time in sigmat bin
  pars = time_resolution(df, time_range, sigma_range, wpv, mode,
                         figs=args['out_plots_out'])

  # export fit results
  pars.dump(args['out_json'])
  # trapallada para que snakemake non se queixe
  os.system(
      f"cp {args['out_json']} {args['out_data'].replace('json','npy')}")
  # os.makedirs(args['out_plots_out'], exist_ok=True)
  # pars.dump(os.path.join(args['out_plots_out'], "fit.pdf"))


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
