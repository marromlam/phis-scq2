# wrong_pv_component
#
#

import os
# import sys
# import json
# import math as m
import argparse
import uproot3 as uproot
import numpy as np
import complot
from utils.plot import get_range, get_var_in_latex, watermark, make_square_axes

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import ipanema
ipanema.initialize('cuda', 1)


prog = ipanema.compile("""
#define USE_DOUBLE 1
#include <exposed/kernels.ocl>

WITHIN_KERNEL
ftype wrong_pv_component(const ftype x, const ftype tau1, const ftype tau2,
                         const ftype share, const ftype xmin, const ftype xmax)
{
    ftype num = 0.0;
    ftype den = 1.0;

    ftype exp1 = exp(-fabs(x)/tau1) / (2-exp(-(xmax/tau1))-exp(xmin/tau1))/tau1;
    ftype exp2 = exp(-fabs(x)/tau2) / (2-exp(-(xmax/tau2))-exp(xmin/tau2))/tau2;
    num = exp1 + (1-share) * exp2/share;
    den = 1 + (1-share)/share;

    // num = share * exp( -fabs(x)/tau1 ) + (1-share) * exp( -fabs(x)/tau2 );
    // den = (2 - exp(-(xmax/tau1)) - exp(xmin/tau1))*share*tau1 + (2 - exp(-(xmax/tau2)) - exp(xmin/tau2))*tau2 + (-2 + exp(-(xmax/tau2)) + exp(xmin/tau2))*share*tau2;

    // num +=     exp( -fabs(x)/tau1 );
    // num += (1-share) * exp( -fabs(x)/tau2 );
    // den = (2 - exp(-(xmax/tau1)) - exp(xmin/tau1))*tau1 + ((1 + exp(xmax/tau2)*(-2 + exp(xmin/tau2)))*(-1 + share)*tau2)/exp(xmax/tau2);

    //num = exp( -fabs(x)/tau1 ) + ((1-share)/share) * exp( -fabs(x)/tau2 );
    //den = (2 - exp(-(xmax/tau1)) - exp(xmin/tau1))*tau1 + ((1 + exp(xmax/tau2)*(-2 + exp(xmin/tau2)))*(-1 + share)*tau2)/(exp(xmax/tau2)*share);


    return num/den;
}


KERNEL
void kernel_wrong_pv_component(GLOBAL_MEM ftype * prob, GLOBAL_MEM const ftype *time,
                        const ftype tau1, const ftype tau2, const ftype share,
                        const ftype tLL, const ftype tUL)
{
    const int idx = get_global_id(0);
    prob[idx] = wrong_pv_component(time[idx], tau1, tau2, share, tLL, tUL);
}


// WITHIN_KERNEL
// ftype time_fit(
//   const ftype time, 
//   const ftype mu, GLOBAL_MEM ftype *sigma,
//  const ftype fprompt, const ftype fll, const ftype fsl, const ftype fwpv,
//  const ftype taul, const ftype taus, const ftype tau1, const ftype tau2,
//  const ftype share_wpv
// )
// {
//   ftype prompt = 0; 
//  ftype long_live = 0;
//  for (int i=0; i<2; ++i) {
//     prompt += exp(rpow(-mu + time,2)/(2*rpow(sigma[i],2))) / sqrt(2*M_PI)*sigma[i];
//    long_live +=     fsl * exp((rpow(sigma[i],2) + 2*mu*taus - 2*taus*time)/(2.*rpow(taus,2))) * erfc((rpow(sigma[i],2) + taus*(mu - time))/(sqrt(2.)*sigma[i]*taus)));
//     long_live += (1-fsl) * exp((rpow(sigma[i],2) + 2*mu*taul - 2*taul*time)/(2.*rpow(taul,2))) * erfc((rpow(sigma[i],2) + taul*(mu - time))/(sqrt(2.)*sigma[i]*taul));
//  }
// 
//   const ftype wpv = wrong_pv_component(time, tau1, tau2, share_wpv, -10, 10);
// 
//   return fprompt * prompt + 0.5*fll * long_live + fwpv * wpv;
// }




""")


def wpv_component_pdf(time, tau1, tau2, share, tLL=-10, tUL=10, prob=False):
    prog.kernel_wrong_pv_component(prob, time, np.float64(tau1),
                                   np.float64(tau2), np.float64(share),
                                   np.float64(tLL), np.float64(tUL),
                                   global_size=(len(time),))
    return prob




def extract_wpv_shape(df, wpv='classical', bin_sigmat=False,
                      bin_pTB=False, with_plots=False):
    """
    Main function to get WPV shape coefficients
    """
    if bin_sigmat:
        sigma_bins = [0.01, 0.021, 0.026, 0.032, 0.038,
                      0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
    else:
        sigma_bins = [0.01, 0.08]
    if bin_pTB:
        pt_bins = [0, np.inf]
        nbins = len(sigma_bins) - 1
    else:
        pt_bins = [0, np.inf]
        nbins = len(pt_bins) - 1
    if bin_pTB and bin_sigmat:
        print('ERORR')
        exit()
    tLL, tUL = -10, 10  # get this form timeres conf file
    sLL, sUL = sigma_bins[0], sigma_bins[-1]
    pLL, pUL = pt_bins[0], pt_bins[-1]
    plots = {}
    # create set of parameters to fit
    pars = ipanema.Parameters()
    if bin_sigmat or bin_pTB:
        for i in range(1, nbins+1):
            pars.add(dict(name=f'tau1.{i}', value=0.377, min=0.03, max=4.5))
            pars.add(dict(name=f'tau2.{i}', value=1.830, min=0.0, max=6.5))
            pars.add(dict(name=f'share.{i}', value=0.830, min=0.0, max=1.0))
    else:
        pars.add(dict(name='tau1', value=0.377, min=0.03, max=4.5))
        pars.add(dict(name='tau2', value=1.830, min=0.0, max=6.5))
        pars.add(dict(name='share', value=0.830, min=0.0, max=1.0))
        """
        2  tau1_wpv     3.48717e-01   6.28252e-03   6.25804e-05  -1.03019e+00
        3  tau2_wpv     1.91814e+00   8.01100e-02   5.50369e-05  -4.22239e-01
        1  frac_tau1    8.19210e-01   9.86527e-03   2.29378e-04   6.87708e-01
        """
        nbins = 1
    for ibin in range(1, nbins+1):
        pbin = "" if nbins < 2 else f".{ibin}"
        # set variables to be used
        time = 'time'
        sigmat = 'sigmat'
        pt = 'B_PT'
        fakeVertex = 1
        # cut according to current bin
        list_of_cuts = [
            f"{time}>{tLL} & {time}<{tUL}",
            f"FakeVertex == {fakeVertex}"
        ]
        if bin_sigmat:
            sLL, sUL = sigma_bins[ibin-1], sigma_bins[ibin]
        # WARNING: sigmat cut is always applyed
        list_of_cuts.append(f"{sigmat}>{sLL} & {sigmat}<{sUL}")
        if bin_pTB:
            pLL, pUL = pt_bins[ibin-1], pt_bins[ibin]
            list_of_cuts.append(f"{pt}>{pLL} & {pt}<{pUL}")
        # cut dataframe
        cut = "(" + ") & (".join(list_of_cuts) + ")"
        print(cut)
        cdf = df.query(cut)
        print(cdf)
        # create time variable and allocate it
        time = ipanema.ristra.allocate(np.float64(cdf[time].values))
        prob = 0 * time

        def fcn(params, time, prob):
            p = params.valuesdict()
            tau1 = p[f'tau1{pbin}']
            tau2 = p[f'tau2{pbin}']
            share = p[f'share{pbin}']
            wpv_component_pdf(time=time, prob=prob, tau1=tau1, tau2=tau2,
                              share=share)
            num = ipanema.ristra.get(prob)
            den = 1
            # normalization
            # _x = np.linspace(ipanema.ristra.min(time), ipanema.ristra.max(time), 100000)
            # _y = 0 * _x
            # _x = ipanema.ristra.allocate(_x)
            # _y = ipanema.ristra.allocate(_y)
            # wpv_component_pdf(time=_x, prob=_y, tau1=tau1, tau2=tau2,
            #                   share=share)
            # den = np.trapz(ipanema.ristra.get(_y), ipanema.ristra.get(_x))
            # print(den)
            return -2 * np.log(num/den)
        # minimize
        pars.lock()
        pars.unlock(f'tau1{pbin}', f'tau2{pbin}', f'share{pbin}')
        res = ipanema.optimize(fcn, pars, fcn_args=(time, prob),
                               method='minuit', tol=0.05, verbose=False)
        print(res)
        pars = res.params
        if with_plots:
            fig, axplot, axpull = complot.axes_providers.axes_plotpull()
            histo = complot.hist(ipanema.ristra.get(time), 150)
            _x = ipanema.ristra.linspace(tLL, tUL, 200)
            _y = 0*ipanema.ristra.linspace(tLL, tUL, 200)
            _y = wpv_component_pdf(time=_x, prob=_y, **pars.valuesdict())
            __x = ipanema.ristra.get(_x)
            __y = ipanema.ristra.get(_y) * histo.norm
            axplot.plot(__x, __y)
            axplot.errorbar(histo.bins, histo.counts, yerr=histo.yerr,
                            xerr=histo.xerr, fmt='.', color='k')
            pulls = complot.compute_pdfpulls(__x, __y, histo.bins, histo.counts,
                                             *histo.yerr)
            axpull.fill_between(histo.bins, pulls, 0)
            axplot.set_ylabel('Candidates')
            axpull.set_xlabel(r'$t$ [ps]')
            v_mark = 'LHC$b$'  # watermark plots
            tag_mark = 'THIS THESIS'
            plots[f"fit{pbin}"] = fig
            axplot.set_yscale('log')
            watermark(axplot, version="final", scale=9.3)
            plots[f"logfit{pbin}"] = fig
    if with_plots:
        return pars, plots
    return pars


if __name__ == '__main__':
    DESCRIPTION = """WPV"""
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--data-in', help='Input prompt data.')
    p.add_argument('--wpv', help='Specify WPV model type')
    p.add_argument('--wpv-out', help='Location to save fit parameters')
    p.add_argument('--plots-out', help='Location to create plots')
    args = vars(p.parse_args())
    branches = ['time', 'sigmat', 'B_PT', 'FakeVertex']
    df = uproot.open(args['data_in'])
    df = df[list(df.keys())[0]].pandas.df(branches=branches)
    pars, plots = extract_wpv_shape(df, args['wpv'], with_plots=True)
    pars.dump(args['wpv_out'])
    os.makedirs(args['plots_out'], exist_ok=True)
    for kn, kp in plots.items():
        kp.savefig(os.path.join(args['plots_out'], f"{kn}.pdf"))

# vim: fdm=marker ts=2 sw=2 sts=2 sr et
