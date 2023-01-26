# calibration
#
#


__all__ = ['calibration_fit', 'calibrate_numerical', 'calibrate_translate']
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import argparse
import json
import os
import numpy as np
import uproot3 as uproot
import ipanema
import uncertainties as unc
from uncertainties import unumpy
import complot

from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes

def calibration_fit(sigmat, sigmat_errl, sigmat_errh, sigmaeff, sigmaeff_err, average):
    # model
    def model(x, *args):
        y = args[0] + args[1]*(x-average) + args[2]*(x-average)**2
        return y

    # cost function
    def fcn(params, x, y):
        p = list(params.valuesdict().values())
        chi2 = (y - model(x, *p))**2 / sigmaeff_err**2
        # chi2 += (0.5 * (sigmat_errh+sigmat_errl) * (p[1] + 2*p[2] * x))**2
        return chi2

    # parametes object
    pars = ipanema.Parameters()
    pars.add(dict(name='p0', value=0))
    pars.add(dict(name='p1', value=0.5))
    pars.add(dict(name='p2', value=0.0, free=False))
    res = ipanema.optimize(fcn, pars, fcn_args=(sigmat, sigmaeff),
                           method='minos', verbose=False)
    print(res)
    pars_linear = res.params

    # parabolic fit
    pars.unlock()
    res = ipanema.optimize(fcn, pars, fcn_args=(sigmat, sigmaeff),
                           method='minuit', verbose=False)
    pars_parab = res.params
    print(res)
    return pars_linear, pars_parab


def calibrate_numerical(df, json_in_binned, json_in_num, sigma_edges,
                        time_range, weight=False, plots=False):
    """
    Calibration of ...
    """
    NBin = len(sigma_edges) - 1
    tLL, tUL = time_range
    sLL, sUL = sigma_edges[0], sigma_edges[-1]
    time_variable = 'time'
    sigmat_variable = 'sigmat'
    offset = 0.036

    sigmat = []
    sigmat_errh = []
    sigmat_errl = []
    sigmaeff = []
    sigmaeff_err = []
    _n = []
    for i, sLL, sUL in zip(range(NBin), sigma_edges[:-1], sigma_edges[1:]):
        list_of_cuts = [
            f"{time_variable}>{tLL} & {time_variable}<{tUL}",
            f"{sigmat_variable}>{sLL} & {sigmat_variable}<{sUL}"
        ]
        cut = "(" + ") & (".join(list_of_cuts) + ")"
        cdf = df.query(cut)
        # _x.append(0.5*(sLL + sUL))
        sigmat.append(json_in_binned[i]['sigmaAverage'].value)
        sigmat_errl.append(sigmat[-1] - sLL)
        sigmat_errh.append(sUL - sigmat[-1])
        # _y.append(json_in_binned[i]['sigmaeff'].uvalue)
        sigmaeff.append(json_in_num[f'seff{i}'].value)
        sigmaeff_err.append(json_in_num[f'seff{i}'].stdev)
        _n.append(cdf.shape[0])

    average = np.float64(offset)
    sigmat = np.array(sigmat)
    sigmat_errl = np.array(sigmat_errl)
    sigmat_errh = np.array(sigmat_errh)
    sigmaeff = np.array(sigmaeff)
    sigmaeff_err = np.array(sigmaeff_err)

    print(sigmat)
    print(sigmat)
    print(sigmaeff)
    print(sigmaeff_err)

    # ja
    # sigmat     = np.array([0.01906966521207871, 0.02406386247035076, 0.029438211538530892, 0.035146922337333, 0.040951522026415305, 0.04634658932581607, 0.0512778543000633, 0.056248487816319456, 0.061244098227330325, 0.06954302155793927])
    # sigmaeff   = np.array([0.027275688301297497, 0.030351707161947825, 0.03455437573815742, 0.0396126152017193, 0.044690117398655686, 0.04958429546465629, 0.05472683062821174, 0.059469278857943925, 0.06460382399146876, 0.07313196937004908])
    # sigmat_errl       = np.array([0.009069665212078708, 0.003063862470350759, 0.0034382115385308935, 0.0031469223373329983, 0.0029515220264153055, 0.0023465893258160717, 0.0022778543000632964, 0.002248487816319457, 0.002244098227330328, 0.005543021557939273])
    # sigmat_errh       = np.array([0.001930334787921293, 0.0019361375296492386, 0.0025617884614691083, 0.002853077662667, 0.003048477973584693, 0.0026534106741839328, 0.002722145699936701, 0.0027515121836805406, 0.0027559017726696763, 0.010456978442060727])
    # sigmaeff_err = np.array([0.0003835755128655926, 0.00015652837103854474, 8.064395789443689e-05, 6.610821393348299e-05, 6.87795395975721e-05,
    #                         9.320935741000951e-05, 0.00012931050084674092, 0.0001890288887025648, 0.0002858211980309035, 0.00033979473830205765])

    pars_linear, pars_parab = calibration_fit(sigmat, sigmat_errl, sigmat_errh,
                                              sigmaeff, sigmaeff_err, average)

    # func1 = TF1("func1", "[0]+[1]*(x-[2])", 0.001, 0.15)
    # func1.SetParNames("p0", "p1", "s_ave")
    # func1.SetParameters(0., 1., average)
    # func1.FixParameter(2, average)
    # func2 = TF1("func2", "[0]+[1]*(x-[2])+[3]*(x-[2])*(x-[2])", 0.001, 0.15)
    # func2.SetParNames("p0", "p1", "s_ave", "p2")
    # func2.SetParameters(0., 1., average, 0.)
    # func2.FixParameter(2, average)
    #
    # for i in range(NBin):
    #     for pars in pdf_pars_binned:
    #         if pars["Name"] == 'sigma_ave_b{}'.format(i):
    #             sigma_ave.append(pars['Value'])
    #
    #     for pars in num_pars:
    #         if pars["Name"] == 'sigma_eff_b{}'.format(i):
    #             sigma_eff_num.append(pars["Value"])
    #             sigma_eff_num_err.append(pars["Error"])
    #         if pars["Name"][:2] == 'mu':
    #             time_biases.append(pars["Value"])
    #             time_biases_err.append(pars["Error"])
    #         if pars["Name"] == "D_eff":
    #             D_eff = pars["Value"]
    #             D_eff_err = pars["Error"]
    #     sigma_ave_err_low.append(sigma_ave[i] - terr[i])
    #     sigma_ave_err_high.append(terr[i+1] - sigma_ave[i])
    #
    #
    #
    # h1 =  TH1D("h1", "", NBin, np.asarray(terr, 'd'))
    # total = 0.
    # for i in range(NBin):
    #     h1.SetBinContent(i+1, num_entries[i])
    #     total = total + num_entries[i]
    #
    # g2 = TGraphAsymmErrors(NBin, sigma_ave, sigma_eff_num, sigma_ave_err_low,  sigma_ave_err_high, sigma_eff_num_err, sigma_eff_num_err)
    #
    # #num1 = TF1("num1", "pol1")
    # #num2 = TF1("num2", "pol2")
    #
    # func1.SetLineColor(kBlue)
    # #func1.SetLineStyle(kDashed)
    # func1.SetLineWidth(3)
    # func2.SetLineColor(kRed)
    # #func2.SetLineStyle(kDashed)
    # func2.SetLineWidth(3)
    #
    # numresult1 = g2.Fit('func1', 'SFEX0')
    # chi2_num1  =  numresult1.Chi2()/numresult1.Ndf()
    # print(numresult1.Chi2()/numresult1.Ndf())
    # numresult2 = g2.Fit('func2', 'SFEX0')
    # chi2_num2 = numresult2.Chi2()/numresult2.Ndf()
    # print(numresult2.Chi2()/numresult2.Ndf())
    #
    #
    # # t first order poly and extract paramters
    # pic = TCanvas('reso', '', 10, 10, 700, 600)
    # pic.cd()
    # gPad.SetTickx()
    # gPad.SetTicky()
    # gStyle.SetOptStat(0)
    # h1.SetTitle("");
    # h1.GetXaxis().SetTitle("#sigma_{t} [ps]")
    # h1.GetYaxis().SetTitle("#sigma_{eff} [ps]")
    #
    # g2.SetTitle("");
    # g2.GetXaxis().SetTitle("#sigma_{t} (ps)")
    # g2.GetYaxis().SetTitle("#sigma_{eff} (ps)")
    # g2.GetYaxis().SetRangeUser(0., 0.15)
    #
    # g2.SetMarkerColor(kBlack)
    # g2.SetMarkerSize(1)
    # g2.SetMarkerStyle(8)
    # gPad.SetFrameLineColor(gPad.GetFillColor())
    # g1.GetHistogram().SetMaximum(1.)
    # g1.GetHistogram().SetMinimum(0.)
    #
    # g2.GetHistogram().SetMaximum(1.)
    # g2.GetHistogram().SetMinimum(0.)
    # h1.GetXaxis().SetRangeUser(0., 0.1)
    # h1.Scale(0.35/total)
    # h1.SetLineColor(kWhite)
    # h1.SetFillColor(17)
    # h1.SetFillStyle(1001)
    #
    # h1.DrawCopy('histo')
    # g2.Draw('Psame')
    # func1.Draw("same")
    # func2.Draw("same")
    #
    # num_rho_p0_p1 = numresult1.Correlation(0,1)
    # num_rho_f2_p0_p1 = numresult2.Correlation(0,1)
    # num_rho_f2_p0_p2 = numresult2.Correlation(0,3)
    # num_rho_f2_p1_p2 = numresult2.Correlation(1,3)
    #
    # sigma_rho = 0.
    # num_sigma_rho = 0.
    # #pars["DEff"] = {"Name": "D_eff", "Value" : D_eff, "Error" : D_eff_error }
    # pars["TimeResParametersNum1Chi2"] = chi2_num1
    # pars["TiimeResParametersNum2Chi2"] = chi2_num2
    # pars["TimeResParametersNum1"] = [{ "Name": func1.GetParName(i), "Value": func1.GetParameter(i), "Error": func1.GetParError(i) } for i in range(3) ]
    # pars["TimeResParametersNum1"].append({"Name": "rho_p0_p1_time_res", "Value": num_rho_p0_p1, "Error": num_sigma_rho})
    #
    # pars["TimeResParametersNum2"] = [{ "Name": func2.GetParName(i), "Value": func2.GetParameter(i), "Error": func2.GetParError(i) } for i in range(4) ]
    # pars["TimeResParametersNum2"].append({"Name": "rho_p0_p1_time_res", "Value": num_rho_f2_p0_p1, "Error": num_sigma_rho})
    # pars["TimeResParametersNum2"].append({"Name": "rho_p0_p2_time_res", "Value": num_rho_f2_p0_p2, "Error": num_sigma_rho})
    # pars["TimeResParametersNum2"].append({"Name": "rho_p1_p2_time_res", "Value": num_rho_f2_p1_p2, "Error": num_sigma_rho})
    #
    # ptext = TPaveText(0.65, 0.65, 0.89, 0.70, "NDC")
    # ptext.SetBorderSize(0)
    # ptext.SetFillColor(kWhite)
    # ptext.Draw()
    #
    # ptext3 = TPaveText(0.65, 0.65, 0.89, 0.75, "NDC")
    # ptext3.SetBorderSize(0)
    # ptext3.SetFillColor(kWhite)
    # ptext3.AddText("num pol1: %.2f" % (numresult1.Chi2()/numresult1.Ndf()))
    # for value in pars["TimeResParametersNum1"]:
    #     ptext3.AddText("%s : %.5f +/- %.5f" % (value["Name"], value["Value"], value["Error"]))
    #
    # #ptext3.Draw()
    # leg = TLegend(0.65, 0.45, 0.89, 0.65)
    # leg.SetBorderSize()
    # leg.AddEntry("func1", "pol1 fit num", "l")
    # leg.AddEntry("func2", "pol2 fit num", "l")
    # leg.AddEntry(g2, "dilution", "p")
    # leg.Draw("same")
    # pic.SaveAs(os.path.join(plot_out, 'resolution_plot_comparisoni_numerical_only.pdf'))
    if plots:
      # model = lambda x, p: np.sum([x**l * p[l] for l in len(p)])
      def model(x, p):
        return p[0] + p[1]*(x-average) + p[2]*(x-average)**2
      sigma_proxy = np.linspace(np.min(sigma_edges), np.max(sigma_edges), 100)
      fig, axplot = complot.axes_plot()
      # print(pars_linear.array())
      # print(pars_linear.valuesarray())
      axplot.plot(sigma_proxy, model(sigma_proxy, pars_linear.valuesarray()))
      axplot.plot(sigma_proxy, model(sigma_proxy, pars_parab.valuesarray()))
      axplot.errorbar(sigmat, sigmaeff, yerr=sigmaeff_err,
                      xerr=[sigmat_errl, sigmat_errh],
                      fmt='.', color=f'k')
      # _norm = np.trapz(d)
      # h, e = np.histogram(df[sigmat_variable], bins=sigma_edges, density=True)
      # c = 0.5 * (e[:-1] + e[1:])
      axplot2 = axplot.twinx()
      # axplot2.plot(c, h, '-')
      axplot2.hist(df[sigmat_variable], bins=sigma_edges, color='k', alpha=0.2)
      # axplot2.plot(x_arr, y_arr, '.', color='C0')
      # axplot2.plot(best_x, best_y, 'o', color='C2',
      #              label=f'Optimal cut at {best_x:.2f}')
      # axplot.set_xlabel(bdt_branch.replace('_', '-'))
      # _norm = np.trapz(h, c)
      # _norm = np.trapz(_n, c)
      # print(_norm)
      # print(_n/np.sum(_n))
      # axplot.plot(c, h, color='k', alpha=0.2)
      axplot.set_xlabel(r'$\sigma_t$ [ps]')
      axplot.set_ylabel(r'$\sigma_{eff}$ [ps]')
      axplot2.set_ylabel("Candidates")
      fig.savefig(os.path.join(plots, "linear.pdf"))
      fig.savefig(os.path.join(plots, "parabolic.pdf"))

    return pars_linear, pars_parab


def calibrate_translate(df, json_in_old, json_in_num, sigma_edges,
                        time_range, weight=False, plots=False):
    """
    Calibration of ...
    """
    NBin = len(sigma_edges) - 1
    tLL, tUL = time_range
    sLL, sUL = sigma_edges[0], sigma_edges[-1]
    time_variable = 'time'
    sigmat_variable = 'sigmat'
    offset = 0.036

    sigmat = []
    sigmat_errh = []
    sigmat_errl = []
    sigmaeff = []
    sigmaeff_err = []
    _n = []

    for i, sLL, sUL in zip(range(NBin), sigma_edges[:-1], sigma_edges[1:]):
        list_of_cuts = [
            f"{time_variable}>{tLL} & {time_variable}<{tUL}",
            f"{sigmat_variable}>{sLL} & {sigmat_variable}<{sUL}"
        ]
        cut = "(" + ") & (".join(list_of_cuts) + ")"
        cdf = df.query(cut)
        print(cdf.shape)
        sigmat.append(np.mean(cdf[sigmat_variable].values))
        # sigmat.append(json_in_binned[i]['sigmaAverage'].value)
        sigmat_errl.append(sigmat[-1] - sLL)
        sigmat_errh.append(sUL - sigmat[-1])
        # _y.append(json_in_binned[i]['sigmaeff'].uvalue)
        sigmaeff.append(json_in_num[f'seff{i}'].value)
        sigmaeff_err.append(json_in_num[f'seff{i}'].stdev)
        _n.append(cdf.shape[0])
    print("N=",_n)
    average = np.float64(offset)
    sigmat = np.array(sigmat)
    sigmat_errl = np.array(sigmat_errl)
    sigmat_errh = np.array(sigmat_errh)
    sigmaeff = np.array(sigmaeff)
    sigmaeff_err = np.array(sigmaeff_err)

    print(sigmat)
    print(sigmat)
    print(sigmaeff)
    print(sigmaeff_err)

    # copiamos e pegamos os inputs de Lera para checkear
    print("++++++++++++++++++++++++++++++++")
    print("x =", sigmat)
    print("y =", sigmaeff)
    print("uy =", sigmaeff_err)
    # sigmat = np.array([0.01906966521207871, 0.02406386247035076, 0.029438211538530892, 0.035146922337333, 0.040951522026415305, 0.04634658932581607, 0.0512778543000633, 0.056248487816319456, 0.061244098227330325, 0.06954302155793927])
    # sigmaeff = np.array([0.02764900118658738, 0.031177090280900478, 0.035376195341426823, 0.04056104474443091, 0.046127998677014156, 0.05145105445398956, 0.05659221598906638, 0.06206370481389605, 0.0682574021801124, 0.07786812535719949])
    # sigmat_errl = np.array([0.009069665212078708, 0.003063862470350759, 0.0034382115385308935, 0.0031469223373329983, 0.0029515220264153055, 0.0023465893258160717, 0.0022778543000632964, 0.002248487816319457, 0.002244098227330328, 0.005543021557939273])
    # sigmat_errh = np.array([0.001930334787921293, 0.0019361375296492386, 0.0025617884614691083, 0.002853077662667, 0.003048477973584693, 0.0026534106741839328, 0.002722145699936701, 0.0027515121836805406, 0.0027559017726696763, 0.010456978442060727])
    # sigmaeff_err = np.array([0.006956510701671419, 0.004133957190790816, 0.0024829463333260076, 0.002156767431179304, 0.002286561639815726, 0.0030467450541910445, 0.004055239328439427, 0.0055906637536084465, 0.007587856523453849, 0.007702887579720337])
    print("x =", sigmat)
    print("ux =", sigmaeff)
    print("uy =", sigmaeff_err)
    print("++++++++++++++++++++++++++++++++")


    pars_linear, pars_parab = calibration_fit(sigmat, sigmat_errl, sigmat_errh,
                                              sigmaeff, sigmaeff_err, average)
    
    if plots:
      # model = lambda x, p: np.sum([x**l * p[l] for l in len(p)])
      def model(x, p):
        return p[0] + p[1]*(x-average) + p[2]*(x-average)**2
      sigma_proxy = np.linspace(np.min(sigma_edges), np.max(sigma_edges), 100)
      fig, axplot = complot.axes_plot()
      # print(pars_linear.array())
      # print(pars_linear.valuesarray())
      axplot.plot(sigma_proxy, model(sigma_proxy, pars_linear.valuesarray()))
      axplot.plot(sigma_proxy, model(sigma_proxy, pars_parab.valuesarray()))
      axplot.errorbar(sigmat, sigmaeff, yerr=sigmaeff_err,
                      xerr=[sigmat_errl, sigmat_errh],
                      fmt='.', color=f'k')
      # _norm = np.trapz(d)
      # h, e = np.histogram(df[sigmat_variable], bins=sigma_edges, density=True)
      # c = 0.5 * (e[:-1] + e[1:])
      axplot2 = axplot.twinx()
      # axplot2.plot(c, h, '-')
      axplot2.hist(df[sigmat_variable], bins=sigma_edges, color='k', alpha=0.2)
      # axplot2.plot(x_arr, y_arr, '.', color='C0')
      # axplot2.plot(best_x, best_y, 'o', color='C2',
      #              label=f'Optimal cut at {best_x:.2f}')
      # axplot.set_xlabel(bdt_branch.replace('_', '-'))
      # _norm = np.trapz(h, c)
      # _norm = np.trapz(_n, c)
      # print(_norm)
      # print(_n/np.sum(_n))
      # axplot.plot(c, h, color='k', alpha=0.2)
      axplot.set_xlabel(r'$\sigma_t$ [ps]')
      axplot.set_ylabel(r'$\sigma_{\mathrm{eff}}$ [ps]')
      axplot2.set_ylabel("Candidates")

      v_mark = 'LHC$b$'  # watermark plots
      tag_mark = 'THIS THESIS'
      watermark(axplot, version=v_mark, tag=tag_mark, scale=1.1)
      fig.savefig(os.path.join(plots, "linear.pdf"))
      fig.savefig(os.path.join(plots, "parabolic.pdf"))


    return pars_linear, pars_parab


if __name__ == '__main__':
    DESCRIPTION = "Poly"
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--in-data', help='data to plot statistics')
    p.add_argument('--in-json-bin',
                   help='json file with pdf parameters from binned fit.')
    p.add_argument('--in-json-num',
                   help='json file for the num dillution for the cross-checks')
    p.add_argument('--out-json-linear',
                   help='Location to save poly fit parameters')
    p.add_argument('--out-json-parab',
                   help='Location to save poly fit parameters')
    p.add_argument('--out-plot', help='Location to create plot')
    p.add_argument('--mode', help='Location to create plot')
    p.add_argument('--timeres', help='Location to create plot')
    args = vars(p.parse_args())

    timeres_binning = [0.010, 0.021, 0.026, 0.032,
                       0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
    time_range = [-4, 10]

    mode = args['mode']
    branches = ['time', 'sigmat', 'B_PT']

    # main dataframe
    df = uproot.open(args['in_data'])
    df = df[list(df.keys())[0]].pandas.df(branches=branches)

    # load parameters
    p = [ipanema.Parameters.load(p) for p in args['in_json_bin'].split(',')]
    pnum = ipanema.Parameters.load(args['in_json_num'])

    # branching for Bs RD mode
    os.makedirs(args['out_plot'], exist_ok=True)
    if mode == 'Bs2JpsiPhi':
        pl, pp = calibrate_translate(df, p, pnum, timeres_binning, time_range,
                                     plots=args['out_plot'])
    else:
        pl, pp = calibrate_numerical(df, p, pnum, timeres_binning, time_range,
                                     plots=args['out_plot'])

    # save the results
    pl.dump(args['out_json_linear'])
    pp.dump(args['out_json_parab'])
    # os.makedirs(args['out_plot'], exist_ok=True)
    # pp.dump(os.path.join(args['out_plot'], "merda"))


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
