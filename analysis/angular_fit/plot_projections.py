__all__ = []


if __name__ == "__main__":

    from ipanema import wrap_unc, uncertainty_wrapper, get_confidence_bands
    from ipanema import initialize, ristra, Parameters, Sample, optimize, IPANEMALIB, ristra
    from utils.helpers import version_guesser, trigger_scissors, cuts_and
    from utils.strings import printsec, printsubsec
    from ipanema.core.python import ndmesh
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import uproot3 as uproot
    from scipy.special import lpmv
    from scipy.interpolate import interp1d, interpn
    import argparse
# initialize('opencl',1)
    from analysis import badjanak
    from scipy.special import comb
# from scipy.integrate import romb, simpson
    from ipanema import plotting, hist
    import uncertainties.unumpy as unp
    import uncertainties as unc
    from scipy import stats, special
    import os
    import hjson
    import ipanema

    import config


    import argparse


    def sample_loader(Bs2JpsiPhi=False, Bd2JpsiKstar=False,
                    MC_Bs2JpsiPhi=False, MC_Bs2JpsiPhi_dG0=False,
                    MC_Bs2JpsiKK_Swave=False, MC_Bd2JpsiKstar=False,
                    trigger=['combined'], cut=None):
        samples = {}
        if Bs2JpsiPhi:
            samples['Bs2JpsiPhi'] = Bs2JpsiPhi
        if Bd2JpsiKstar:
            samples['Bd2JpsiKstar'] = Bd2JpsiKstar
        if MC_Bs2JpsiPhi:
            samples['MC_Bs2JpsiPhi'] = MC_Bs2JpsiPhi
        if MC_Bs2JpsiPhi_dG0:
            samples['MC_Bs2JpsiPhi_dG0'] = MC_Bs2JpsiPhi_dG0
        if MC_Bs2JpsiKK_Swave:
            samples['MC_Bs2JpsiKK_Swave'] = MC_Bs2JpsiKK_Swave
        if MC_Bd2JpsiKstar:
            samples['MC_Bd2JpsiKstar'] = MC_Bd2JpsiKstar

        s = {}
        for km, vm in samples.items():
            s[km] = {}
            for vy in vm:
                if '2015' in vy:
                    ky = '2015'
                elif '2016' in vy:
                    ky = '2016'
                elif '2017' in vy:
                    ky = '2017'
                elif '2018' in vy:
                    ky = '2018'
                else:
                    ValueError("I dont get this year at all")
                s[km][ky] = {}
                for kt in trigger:
                    s[km][ky][kt] = Sample.from_root(vy,
                                                    cuts=cuts_and(trigger_scissors(kt), cut), name=f"{km}-{ky}-{kt}")
                    # print(s[km][ky][kt])

        return s


    p = argparse.ArgumentParser(description="dfdf")
    p.add_argument('--version')
    p.add_argument('--trigger')
    p.add_argument('--year')
    p.add_argument('--timeacc')
    args = vars(p.parse_args())


    VERSION = args['version']
    TRIGGER = args['trigger']
    TIMEACC = args['timeacc']
    YEARS = args['year'].split(',')


    if TRIGGER == 'combined':
        triggers = ['biased', 'unbiased']
    else:
        triggers = [TRIGGER]


# VERSION = 'v1r0p1@pTB4'
# TRIGGER = 'unbiased'
# YEAR = '2016'

# pars = Parameters.load("analysis/params/generator/2016/MC_Bs2JpsiPhi_dG0.json").valuesdict()

# there is only one set of parameters
    try:
        pars = Parameters.load(
            f"output/params/physics_params/run2/Bs2JpsiPhi/{VERSION}_run2_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_{TRIGGER}.json").valuesdict(False)
    except:
        pars = Parameters.load(
            f"output/params/physics_params/run2/Bs2JpsiPhi/{VERSION}_run2_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_unbiased.json").valuesdict(False)
# df = uproot.open("/scratch46/marcos.romero/sidecar/2016/MC_Bs2JpsiPhi_dG0/v0r5.root")['DecayTree'].pandas.df().query("hlt1b!=0 & time>0.3")

# print(pars)

    all_samples = [
        f"/scratch46/marcos.romero/sidecar14/{y}/Bs2JpsiPhi/{VERSION}.root" for y in YEARS]
    all_csp = [
        f"output/params/csp_factors/{y}/Bs2JpsiPhi/{VERSION}_vgc.json" for y in YEARS]
    all_flavor = [
        "output/params/flavor_tagging/2016/Bs2JpsiPhi/v0r5_amsrd.json" for y in YEARS]
    all_timeacc = {
        'biased': [f"output/params/time_acceptance/{y}/Bd2JpsiKstar/{VERSION}_{TIMEACC}_biased.json" for y in YEARS],
        'unbiased': [f"output/params/time_acceptance/{y}/Bd2JpsiKstar/{VERSION}_{TIMEACC}_unbiased.json" for y in YEARS]
    }
    all_angacc = {
        'biased': [f"output/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_biased.json" for y in YEARS],
        'unbiased': [f"output/params/angular_acceptance/{y}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_{TIMEACC}_amsrd_unbiased.json" for y in YEARS]
    }

# timeacc = Parameters.load("output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul3_biased.json")
# angacc = Parameters.load("output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_correctedDual_none_unbiased.json").valuesdict()

    samples = sample_loader(
        Bs2JpsiPhi=all_samples,
        trigger=triggers,
    )

# since now we have all samples in a good structure, let's attach parameters to them
    average_resolution = []
    for km, vm in samples.items():
        iy = 0
        for ky, vy in vm.items():
            _csp = Parameters.load(all_csp[iy])
            _csp = Parameters.build(_csp, _csp.find('CSP.*'))
            _tag = Parameters.load(all_flavor[iy])
            for kt, vt in vy.items():
                _ta = Parameters.load(all_timeacc[kt][iy])
                tLL = _ta['tLL'].value
                tUL = _ta['tUL'].value
                _knots = np.array(Parameters.build(_ta, _ta.find("k.*"))).tolist()
                _ta = Parameters.build(_ta, _ta.find('(a|b|c).*'))
                _aa = Parameters.load(all_angacc[kt][iy])
                _aa = Parameters.build(_aa, _aa.find('w.*'))
                # attaching
                vt.csp = _csp
                vt.timeacc = _ta
                vt.angacc = _aa
                vt.tagging = _tag
                vt.chop(f"time>={tLL} & time<{tUL}")
                print(f"Found csp     {km}::{ky}::{kt}  =  {np.array(vt.csp)}")
                print(f"Found timeacc {km}::{ky}::{kt}  =  {np.array(vt.timeacc)}")
                print(f"Found angacc  {km}::{ky}::{kt}  =  {np.array(vt.angacc)}")
                print(f"Found tagging {km}::{ky}::{kt}  =  {np.array(vt.tagging)}")
        iy += 1  # increase year number by one

# TODO: It is very important to properly configure badjanak.
#       I need to prepare some piece of code to do it automatically.
    print(_knots)
    badjanak.config['knots'] = _knots
# badjanak.config['knots'][0] = 0.5
# badjanak.config['knots'] = [0.5, 0.91, 1.35, 1.96, 3.01, 7]
    badjanak.config['debug'] = 0
    badjanak.config['fast_integral'] = 0
    badjanak.config['final_extrap'] = 0
    badjanak.get_kernels(True)
    print(samples)


# if TRIGGER == 'unbiased':
####     df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b==0 & time>0.3")
# else:
####     df = uproot.open(f"/scratch46/marcos.romero/sidecar/{YEAR}/Bs2JpsiPhi/{VERSION}.root")['DecayTree'].pandas.df().query("hlt1b!=0 & time>0.3")

#### dtime = np.array(df['time'])
#### dcosL = np.array(df['cosL'])
#### dcosK = np.array(df['cosK'])
#### dhphi = np.array(df['hphi'])
####
# timeres = Parameters.load("output/params/time_resolution/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
#### csp = Parameters.load(f"output/params/csp_factors/{YEAR}/Bs2JpsiPhi/{VERSION}_vgc.json").valuesdict()
# flavor = Parameters.load("output/params/flavor_tagging/2016/Bs2JpsiPhi/v0r5_amsrd.json").valuesdict()
# angacc = Parameters.load("output/params/angular_acceptance/2016/MC_Bs2JpsiPhi_dG0/v0r5_correctedDual_none_unbiased.json").valuesdict()
#### angacc = Parameters.load(f"output/params/angular_acceptance/{YEAR}/Bs2JpsiPhi/{VERSION}_run2Dual_vgc_amsrd_simul3_amsrd_{TRIGGER}.json").valuesdict()
# timeacc = Parameters.load("output/params/time_acceptance/2016/Bd2JpsiKstar/v0r5_simul3_biased.json")
#### timeacc = Parameters.load(f"output/params/time_acceptance/{YEAR}/Bd2JpsiKstar/{VERSION}_simul3_{TRIGGER}.json")
#### knots = np.array(Parameters.build(timeacc, timeacc.find("k.*"))).tolist()
#### timeacc = Parameters.build(timeacc, timeacc.find("(a|c|b)(A|B)?.*")).valuesdict()


    def pdf_projector(params, edges, var='time', timeacc=False, angacc=False,
                    return_center=False, avgres=0.0042):
        # create flags for acceptances
        use_timeacc = True if timeacc else False
        use_angacc = True if angacc else False

        # defaults
        cosKLL = -1
        cosKUL = 1
        cosLLL = -1
        cosLUL = 1
        hphiLL = -np.pi
        hphiUL = +np.pi
        acc = 1
        _x = 0.5 * (edges[:-1] + edges[1:])

        @np.vectorize
        def prob(pars, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL, timeLL, timeUL, avgres):
            var = np.float64([0.0]*3 + [1] + [1020.] + [0.00043] + [0.0]*4)
            var = ristra.allocate(np.ascontiguousarray(var))
            pdf = ristra.allocate(np.float64([0.0]))
            badjanak.delta_gamma5_mc(var, pdf, **pars, tLL=tLL, tUL=tUL)
            num = pdf.get()
            badjanak.delta_gamma5_mc(var, pdf, **pars, cosKLL=cosKLL, cosKUL=cosKUL,
                                    cosLLL=cosLLL, cosLUL=cosLUL, hphiLL=hphiLL,
                                    hphiUL=hphiUL, tLL=timeLL, tUL=timeUL, **timeacc)
            den = pdf.get()
            return num/den

        @np.vectorize
        def vtimeacc(x):
            if use_timeacc:
                return badjanak.bspline(np.float64([x]), [v for v in timeacc.values()])
            else:
                return 1

        @np.vectorize
        def vangacc(x, proj):
            if use_angacc:
                _x = np.linspace(-1, 1, 300)
                __x, __y, __z = ristra.ndmesh(_x, _x, np.pi*_x)
                __x = ristra.allocate(__x.reshape(len(_x)**3))
                __y = ristra.allocate(__y.reshape(len(_x)**3))
                __z = ristra.allocate(__z.reshape(len(_x)**3))
                _arr = [__x, __y, __z]
                _arr[proj-1] *= x/_arr[proj-1]
                _ans = 1 / \
                    badjanak.angular_efficiency_weights(
                        [v for v in angacc.values()], *_arr, proj)
                return np.mean(_ans)
            else:
                return 1

        if var == 'time':
            timeLL, timeUL = edges[:-1], edges[1:]
            acc = vtimeacc(_x)
        elif var == 'cosL':
            cosLLL, cosLUL = edges[:-1], edges[1:]
            acc = vangacc(_x, 1)
        elif var == 'cosK':
            cosKLL, cosKUL = edges[:-1], edges[1:]
            acc = vangacc(_x, 2)
        elif var == 'hphi':
            hphiLL, hphiUL = edges[:-1], edges[1:]
            acc = vangacc(_x, 3)
        else:
            raise ValueError(f"The pdf is not {var} dependent")

        # acc /= np.trapz(acc, _x)
        acc = 1
        _pdf = prob(params, cosKLL, cosKUL, cosLLL, cosLUL,
                    hphiLL, hphiUL, timeLL, timeUL, avgres=avgres)
        _pdf /= np.trapz(_pdf, _x)
        _pdf *= acc
        _pdf /= np.trapz(_pdf, _x)
        if return_center:
            return _pdf, _x
        return _pdf


    branch = 'time'
    wvar = 'sWeight'

    var = np.linspace(tLL, tUL, 1000)
    edges = np.linspace(tLL, tUL, 61)


    fig, axplot, axpull = plotting.axes_plotpull()


    hvar = []  # where to store counts
    hbin = []  # where to store the bining
    hyerr = []  # where to store the y errors
    hxerr = []  # where to store the x errors
    pdf_y = []
    pdf_x = []


    for km, vm in samples.items():
        for ky, vy in vm.items():
            for kt, vt in vy.items():
                _hvar = hist(vt.df[branch].values, bins=edges,
                            weights=vt.df[wvar].values)
                _pdfvar, _var = pdf_projector(pars, var, branch,
                                            # timeacc=False,
                                            timeacc=vt.timeacc.valuesdict(),
                                            return_center=True, avgres=0.042)
                # exit()
                hvar.append(_hvar.counts)
                hbin.append(_hvar.bins)
                hyerr.append([np.nan_to_num(_hvar.errl),
                            np.nan_to_num(_hvar.errh)])
                hxerr.append([edges[1:]-_hvar.bins, _hvar.bins-edges[:-1]])
                pdf_x.append(_var)
                pdf_y.append(_hvar.norm * _pdfvar)

    all_hvar = np.sum(hvar, 0)
# all_yerr = [np.sqrt(all_hvar), np.sqrt(all_hvar)]
    print(hyerr)
    all_yerr = [np.sqrt(np.sum([r[0]**2 for r in hyerr], 0)),
                np.sqrt(np.sum([r[1]**2 for r in hyerr], 0))]
    print(all_yerr)
    all_pdf_y = np.sum(pdf_y, 0)

    axplot.plot(pdf_x[0], all_pdf_y, '-')
    axplot.errorbar(hbin[0], all_hvar, yerr=all_yerr, xerr=hxerr[0], fmt='.k')
# axplot.plot(hbin[0], all_hvar, '.k')
    axplot.set_yscale('log')
    axplot.set_title(f'{VERSION}-{YEARS}-{TRIGGER}')

    axpull.fill_between(hbin[0],
                        ipanema.histogram.pull_pdf(pdf_x[0], all_pdf_y,
                                                hbin[0], all_hvar, *all_yerr),
                        0, facecolor="C0", alpha=0.5)
    plt.show()


    exit()

# without acceptances
# var = np.linspace(0.3, 15, 1000)
# pdfvar, var = pdf_projector({**pars, **csp}, var, 'time', return_center=True)
# axplot.plot(var, hvar.norm * pdfvar, label='without acceptances', color='C2')

# with acceptances
    var = np.linspace(0.3, 15, 1000)
    pdfvar, var = pdf_projector(
        pars, var, 'time', timeacc=timeacc, return_center=True)
    axplot.plot(var, hvar.norm * pdfvar,
                label=f'{VERSION}-{YEAR}-{TRIGGER}', color='C0')
    axpull.fill_between(hvar.bins,
                        ipanema.histogram.pull_pdf(var, hvar.norm*pdfvar, hvar.bins,
                                                hvar.counts, hvar.errl,
                                                hvar.errh),
                        0, facecolor="C0", alpha=0.5)

# axpull.fill_between(hdata.bins,
#                     ipanema.histogram.pull_pdf(x, y, hdata.bins,
#                                                hdata.counts, hdata.errl,
#                                                hdata.errh),
#                     0, facecolor="C0", alpha=0.5)

    axpull.set_xlabel(r'$t$ [ps]')
# axpull.set_ylim(-6.5, 6.5)
    axpull.set_yticks([-5, 0, 5])
    axplot.set_ylabel(rf"Weighted candidates")
    axplot.legend(loc="upper right")
    fig.savefig(f"{VERSION}_{YEAR}_{TRIGGER}.pdf")

    exit()

    plt.close()
    hvar = hist(dcosK, bins=60)
    plt.errorbar(hvar.bins, hvar.counts, yerr=np.sqrt(hvar.counts), fmt='.')
    var = np.linspace(-1, 1, 1000)
    pdfvar, var = pdf_projector(pars, var, 'cosK', return_center=True)
    plt.plot(var, hvar.norm * pdfvar)
    pdfvar, var = pdf_projector(
        pars, var, 'cosK', angacc=angacc, return_center=True)
    var = np.linspace(-1, 1, 1000)
    plt.plot(var, hvar.norm * pdfvar)
    plt.show()


    plt.close()
    hvar = hist(dcosL, bins=60)
    plt.errorbar(hvar.bins, hvar.counts, yerr=np.sqrt(hvar.counts), fmt='.')
    var = np.linspace(-1, 1, 1000)
    pdfvar, var = pdf_projector(pars, var, 'cosL', return_center=True)
    plt.plot(var, hvar.norm * pdfvar)
    var = np.linspace(-1, 1, 1000)
    pdfvar, var = pdf_projector(
        pars, var, 'cosL', angacc=angacc, return_center=True)
    plt.plot(var, hvar.norm * pdfvar)
    plt.show()
