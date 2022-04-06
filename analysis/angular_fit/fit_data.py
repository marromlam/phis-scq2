import argparse
import os

import numpy as np
from ipanema import Parameter, Parameters, Sample, initialize, optimize, ristra

import config
from utils.helpers import (parse_angacc, timeacc_guesser, trigger_scissors,
                           version_guesser)
from utils.plot import mode_tex
from utils.strings import cammel_case_split, cuts_and, printsec, printsubsec

__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]
__all__ = []


# Modules {{{

# import uproot

initialize(config.user["backend"], 1)
from analysis import badjanak

# get bsjpsikk and compile it with corresponding flags
badjanak.config["debug"] = 0
badjanak.config["fast_integral"] = 1
badjanak.config["debug_evt"] = 774

# import some phis-scq utils


# }}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real data fitter.")
    # Samples
    parser.add_argument("--samples", help="Bs2JpsiPhi data sample")
    # Input parameters
    parser.add_argument("--angacc-biased", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--angacc-unbiased", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--timeacc-biased", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--timeacc-unbiased", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--csp", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--time-resolution", help="Bs2JpsiPhi MC sample")
    parser.add_argument("--flavor-tagging", help="Bs2JpsiPhi MC sample")
    # Output parameters
    parser.add_argument("--params", help="Bs2JpsiPhi MC sample")
    # Configuration
    parser.add_argument("--year", help="Year of data-taking")
    parser.add_argument("--version", help="Year of data-taking")
    parser.add_argument("--fit", help="Year of data-taking")
    parser.add_argument("--angacc", help="Year of data-taking")
    parser.add_argument("--timeacc", help="Year of data-taking")
    parser.add_argument("--trigger", help="Year of data-taking")
    parser.add_argument("--blind", default=1, help="Year of data-taking")
    args = vars(parser.parse_args())

    VERSION, SHARE, EVT, MAG, FULLCUT, VAR, BIN = version_guesser(args["version"])
    YEARS = args["year"].split(",")
    TRIGGER = args["trigger"]
    MODE = "Bs2JpsiPhi"
    FIT = args["fit"]
    TIMEACC = timeacc_guesser(args["timeacc"])
    TIMEACC["use_upTime"] = TIMEACC["use_upTime"] | ("UT" in args["version"])
    TIMEACC["use_lowTime"] = TIMEACC["use_lowTime"] | ("LT" in args["version"])

    MINER = "minuit"

    if TIMEACC["use_upTime"]:
        tLL = config.general["upper_time_lower_limit"]
    else:
        tLL = config.general["time_lower_limit"]
    if TIMEACC["use_lowTime"]:
        tUL = config.general["lower_time_upper_limit"]
    else:
        tUL = config.general["time_upper_limit"]
    print(TIMEACC["use_lowTime"], TIMEACC["use_upTime"])

    # Prepare the cuts -----------------------------------------------------------
    CUT = ""
    CUT = cuts_and(CUT, f"time>={tLL} & time<={tUL}")

    for k, v in args.items():
        print(f"{k}: {v}\n")

    # %% Load samples --------------------------------------------------------------
    printsubsec(f"Loading samples")

    branches_to_load = ["hlt1b"]
    # Lists of data variables to load and build arrays
    real = ["cosK", "cosL", "hphi", "time", "mHH", "sigmat"]
    real += ["tagOSdec", "tagSSdec", "tagOSeta", "tagSSeta"]
    weight = "sWeight"
    branches_to_load += real
    branches_to_load += [weight]

    if TIMEACC["use_veloWeight"]:
        weight = f"veloWeight*{weight}"
        branches_to_load += ["veloWeight"]

    if TRIGGER == "combined":
        TRIGGER = ["biased", "unbiased"]
    else:
        TRIGGER = [TRIGGER]

    data = {}
    for i, y in enumerate(YEARS):
        print(f"Fetching elements for {y}[{i}] data sample")
        data[y] = {}
        csp = Parameters.load(args["csp"].split(",")[i])
        mass = np.array(csp.build(csp, csp.find("mKK.*")))
        csp = csp.build(csp, csp.find("CSP.*"))
        flavor = Parameters.load(args["flavor_tagging"].split(",")[i])
        resolution = Parameters.load(args["time_resolution"].split(",")[i])
        badjanak.config["mHH"] = mass.tolist()
        for t in TRIGGER:
            tc = trigger_scissors(t, CUT)
            data[y][t] = Sample.from_root(
                args["samples"].split(",")[i], branches=branches_to_load, cuts=tc
            )
            data[y][t].name = f"Bs2JpsiPhi-{y}-{t}"
            data[y][t].csp = csp
            data[y][t].flavor = flavor
            data[y][t].resolution = resolution
            print(data[y][t].df)
            # Time acceptance
            c = Parameters.load(args[f"timeacc_{t}"].split(",")[i])
            tLL, tUL = c["tLL"].value, c["tUL"].value
            knots = np.array(Parameters.build(c, c.fetch("k.*")))
            badjanak.config["knots"] = knots.tolist()
            # Angular acceptance
            data[y][t].timeacc = Parameters.build(c, c.fetch("(a|b|c).*"))
            w = Parameters.load(args[f"angacc_{t}"].split(",")[i])
            data[y][t].angacc = Parameters.build(w, w.fetch("w.*"))
            # Normalize sWeights per bin
            sw = np.zeros_like(data[y][t].df[real[0]])
            for ll, ul in zip(mass[:-1], mass[1:]):
                pos = data[y][t].df.eval(f"mHH>={ll} & mHH<{ul}")
                _sw = data[y][t].df.eval(f"{weight}*(mHH>={ll} & mHH<{ul})")
                print(_sw)
                sw = np.where(pos, _sw * (sum(_sw) / sum(_sw * _sw)), sw)
            data[y][t].df["sWeight"] = sw
            data[y][t].allocate(data=real, weight="sWeight", prob="0*time")

    # Compile the kernel
    #    so if knots change when importing parameters, the kernel is compiled
    # badjanak.config["precision"]='single'
    badjanak.get_kernels(True)

    # Prepare parameters {{{

    SWAVE = True
    if "Pwave" in FIT:
        SWAVE = False
    DGZERO = False
    if "DGzero" in FIT:
        DGZERO = True
    POLDEP = False
    if "Poldep" in FIT:
        POLDEP = True
    BLIND = bool(int(args["blind"]))
    print(BLIND)

    pars = Parameters()
    list_of_parameters = [
        # S wave fractions
        Parameter(
            name="fSlon1",
            value=SWAVE * 0.5,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{1}|",
        ),
        Parameter(
            name="fSlon2",
            value=SWAVE * 0.0,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{2}|",
        ),
        Parameter(
            name="fSlon3",
            value=SWAVE * 0.0,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{3}|",
        ),
        Parameter(
            name="fSlon4",
            value=SWAVE * 0.0,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{4}|",
        ),
        Parameter(
            name="fSlon5",
            value=SWAVE * 0.0,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{5}|",
        ),
        Parameter(
            name="fSlon6",
            value=SWAVE * 0.0,
            min=0.00,
            max=0.90,
            free=SWAVE,
            latex=r"|A_S^{6}|",
        ),
        # P wave fractions
        Parameter(
            name="fPlon", value=0.5241, min=0.4, max=0.6, free=True, latex=r"|A_0|^2"
        ),
        Parameter(
            name="fPper",
            value=0.25,
            min=0.1,
            max=0.3,
            free=True,
            latex=r"|A_{\perp}|^2",
        ),
        # Weak phases
        Parameter(
            name="pSlon",
            value=0.00,
            min=-5.0,
            max=5.0,
            free=POLDEP,
            latex=r"\phi_S - \phi_0 \, \mathrm{[rad]}",
            blindstr="BsPhisSDelFullRun2",
            blind=BLIND,
            blindscale=2.0,
            blindengine="root",
        ),
        Parameter(
            name="pPlon",
            value=0.3,
            min=-5.0,
            max=5.0,
            free=True,
            latex=r"\phi_0 \, \mathrm{[rad]}",
            blindstr="BsPhiszeroFullRun2" if POLDEP else "BsPhisFullRun2",
            blind=BLIND,
            blindscale=2.0 if POLDEP else 1.0,
            blindengine="root",
        ),
        Parameter(
            name="pPpar",
            value=0.00,
            min=-5.0,
            max=5.0,
            free=POLDEP,
            latex=r"\phi_{\parallel} - \phi_0 \, \mathrm{[rad]}",
            blindstr="BsPhisparaDelFullRun2",
            blind=BLIND,
            blindscale=2.0,
            blindengine="root",
        ),
        Parameter(
            name="pPper",
            value=0.00,
            min=-5.0,
            max=5.0,
            free=POLDEP,
            blindstr="BsPhisperpDelFullRun2",
            blind=BLIND,
            blindscale=2.0,
            blindengine="root",
            latex=r"\phi_{\perp} - \phi_0 \, \mathrm{[rad]}",
        ),
        # S wave strong phases
        Parameter(
            name="dSlon1",
            value=+2.5 * SWAVE,
            min=-0.0,
            max=+3.0,
            free=SWAVE,
            latex="\delta_S^{1} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dSlon2",
            value=+1.5 * SWAVE,
            min=-0.0,
            max=+3.0,
            free=SWAVE,
            latex="\delta_S^{2} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dSlon3",
            value=+0.0 * SWAVE,
            min=-0.0,
            max=+3.0,
            free=SWAVE,
            latex="\delta_S^{3} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dSlon4",
            value=-0.0 * SWAVE,
            min=-3.0,
            max=+0.0,
            free=SWAVE,
            latex="\delta_S^{4} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dSlon5",
            value=-1.2 * SWAVE,
            min=-3.0,
            max=+0.0,
            free=SWAVE,
            latex="\delta_S^{5} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dSlon6",
            value=-2.5 * SWAVE,
            min=-3.0,
            max=+0.0,
            free=SWAVE,
            latex="\delta_S^{6} - \delta_{\perp} \, \mathrm{[rad]}",
        ),
        # P wave strong phases
        Parameter(
            name="dPlon",
            value=0.00,
            min=-2 * 3.14 * 0,
            max=2 * 3.14,
            free=False,
            latex="\delta_0 \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dPpar",
            value=3.26,
            min=-2 * 3.14 * 0,
            max=2 * 3.14,
            free=True,
            latex="\delta_{\parallel} - \delta_0 \, \mathrm{[rad]}",
        ),
        Parameter(
            name="dPper",
            value=3.1,
            min=-2 * 3.14 * 0,
            max=2 * 3.14,
            free=True,
            latex="\delta_{\perp} - \delta_0 \, \mathrm{[rad]}",
        ),
        # lambdas
        Parameter(
            name="lSlon",
            value=1.0,
            min=0.4,
            max=1.6,
            free=POLDEP,
            latex="|\lambda_S|/|\lambda_0|",
        ),
        Parameter(
            name="lPlon", value=1.0, min=0.4, max=1.6, free=True, latex="|\lambda_0|"
        ),
        Parameter(
            name="lPpar",
            value=1.0,
            min=0.4,
            max=1.6,
            free=POLDEP,
            latex="|\lambda_{\parallel}|/|\lambda_0|",
        ),
        Parameter(
            name="lPper",
            value=1.0,
            min=0.4,
            max=1.6,
            free=POLDEP,
            latex="|\lambda_{\perp}|/|\lambda_0|",
        ),
        # lifetime parameters
        Parameter(
            name="Gd",
            value=0.65789,
            min=0.0,
            max=1.0,
            free=False,
            latex=r"\Gamma_d \, \mathrm{[ps]}^{-1}",
        ),
        Parameter(
            name="DGs",
            value=(1 - DGZERO) * 0.3,
            min=0.0,
            max=1.7,
            free=1 - DGZERO,
            latex=r"\Delta\Gamma_s \, \mathrm{[ps]}^{-1}",
            blindstr="BsDGsFullRun2",
            blind=BLIND,
            blindscale=1.0,
            blindengine="root",
        ),
        Parameter(
            name="DGsd",
            value=0.03 * 0,  # min=-0.1, max= 0.1,
            free=True,
            latex=r"\Gamma_s - \Gamma_d \, \mathrm{[ps]}^{-1}",
        ),
        Parameter(
            name="DM",
            value=17.757,
            min=15.0,
            max=20.0,
            free=True,
            latex=r"\Delta m_s \, \mathrm{[ps]}^{-1}",
        ),
        Parameter(
            "eta_os",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["eta_os"].value,
            free=False,
        ),
        Parameter(
            "eta_ss",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["eta_ss"].value,
            free=False,
        ),
        Parameter(
            "p0_os",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["p0_os"].value,
            free=True,
            min=0.0,
            max=1.0,
            latex="p^{\rm OS}_{0}",
        ),
        Parameter(
            "p1_os",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["p1_os"].value,
            free=True,
            min=0.5,
            max=1.5,
            latex="p^{\rm OS}_{1}",
        ),
        Parameter(
            "p0_ss",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["p0_ss"].value,
            free=True,
            min=0.0,
            max=2.0,
            latex="p^{\rm SS}_{0}",
        ),
        Parameter(
            "p1_ss",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["p1_ss"].value,
            free=True,
            min=0.0,
            max=2.0,
            latex="p^{\rm SS}_{1}",
        ),
        Parameter(
            "dp0_os",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_os"].value,
            free=True,
            min=-0.1,
            max=0.1,
            latex="\Delta p^{\rm OS}_{0}",
        ),
        Parameter(
            "dp1_os",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_os"].value,
            free=True,
            min=-0.1,
            max=0.1,
            latex="\Delta p^{\rm OS}_{1}",
        ),
        Parameter(
            "dp0_ss",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_ss"].value,
            free=True,
            min=-0.1,
            max=0.1,
            latex="\Delta p^{\rm SS}_{0}",
        ),
        Parameter(
            "dp1_ss",
            value=data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_ss"].value,
            free=True,
            min=-0.1,
            max=0.1,
            latex="\Delta p^{\rm SS}_{1}",
        ),
    ]
    pars.add(*list_of_parameters)

    # }}}

    # Print all ingredients of the pdf {{{

    # fit parameters
    print(pars)

    # print csp factors
    lb = [
        data[y][TRIGGER[0]].csp.__str__(["value"]).splitlines()
        for i, y in enumerate(YEARS)
    ]
    print(f"\nCSP factors\n{80*'='}")
    for l in zip(*lb):
        print(*l, sep="| ")

    # print flavor tagging parameters
    lb = [
        data[y][TRIGGER[0]].flavor.__str__(["value"]).splitlines()
        for i, y in enumerate(YEARS)
    ]
    print(f"\nFlavor tagging parameters\n{80*'='}")
    for l in zip(*lb):
        print(*l, sep="| ")

    # print time resolution
    lb = [
        data[y][TRIGGER[0]].resolution.__str__(["value"]).splitlines()
        for i, y in enumerate(YEARS)
    ]
    print(f"\nResolution parameters\n{80*'='}")
    for l in zip(*lb):
        print(*l, sep="| ")

    # print time acceptances
    for t in TRIGGER:
        lt = [
            data[y][t].timeacc.__str__(["value"]).splitlines()
            for i, y in enumerate(YEARS)
        ]
        print(f"\n{t.title()} time acceptance\n{80*'='}")
        for l in zip(*lt):
            print(*l, sep="| ")
    # print(f"\n")

    # print angular acceptance
    for t in TRIGGER:
        lt = [
            data[y][t].angacc.__str__(["value"]).splitlines()
            for i, y in enumerate(YEARS)
        ]
        print(f"\n{t.title()} angular acceptance\n{80*'='}")
        for l in zip(*lt):
            print(*l, sep="| ")
    print(f"\n")

    # }}}

    # Calculate tagging constraints - currently using one value for all years only!!!

    def taggingConstraints(data):
        corr = data[str(YEARS[0])][TRIGGER[0]].flavor.corr(["p0_os", "p1_os"])
        print(corr)
        rhoOS = corr[1, 0]
        print(rhoOS)
        # print(Parameters.load('output/params/flavor_tagging/2015/Bs2JpsiPhi/v0r5.json')['rho01_os'].value)
        corr = data[str(YEARS[0])][TRIGGER[0]].flavor.corr(["p0_ss", "p1_ss"])
        print(corr)
        # print(Parameters.load('output/params/flavor_tagging/2015/Bs2JpsiPhi/v0r5.json')['rho01_ss'].value)
        # data[str(YEARS[0])][TRIGGER[0]].flavor['rho01_ss'].value
        rhoSS = corr[1, 0]

        pOS = np.matrix(
            [
                data[str(YEARS[0])][TRIGGER[0]].flavor["p0_os"].value,
                data[str(YEARS[0])][TRIGGER[0]].flavor["p1_os"].value,
            ]
        )
        pSS = np.matrix(
            [
                data[str(YEARS[0])][TRIGGER[0]].flavor["p0_ss"].value,
                data[str(YEARS[0])][TRIGGER[0]].flavor["p1_ss"].value,
            ]
        )
        print(f"pOS, pSS = {pOS}, {pSS}")

        p0OS_err = data[str(YEARS[0])][TRIGGER[0]].flavor["p0_os"].stdev
        p1OS_err = data[str(YEARS[0])][TRIGGER[0]].flavor["p1_os"].stdev
        p0SS_err = data[str(YEARS[0])][TRIGGER[0]].flavor["p0_ss"].stdev
        p1SS_err = data[str(YEARS[0])][TRIGGER[0]].flavor["p1_ss"].stdev
        print(p0OS_err, p0OS_err)

        covOS = np.matrix(
            [
                [p0OS_err**2, p0OS_err * p1OS_err * rhoOS],
                [p0OS_err * p1OS_err * rhoOS, p1OS_err**2],
            ]
        )
        covSS = np.matrix(
            [
                [p0SS_err**2, p0SS_err * p1SS_err * rhoSS],
                [p0SS_err * p1SS_err * rhoSS, p1SS_err**2],
            ]
        )
        print(f"covOS, covSS = {covOS}, {covSS}")
        print(
            f"covOS, covSS = {data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_os','p1_os'])}, {data[str(YEARS[0])][TRIGGER[0]].flavor.cov(['p0_ss','p1_ss'])}"
        )

        print(
            np.linalg.inv(
                data[str(YEARS[0])][TRIGGER[0]].flavor.cov(["p0_os", "p1_os"])
            )
        )
        print(
            np.linalg.inv(
                data[str(YEARS[0])][TRIGGER[0]].flavor.cov(["p0_ss", "p1_ss"])
            )
        )
        covOSInv = covOS.I
        covSSInv = covSS.I

        print(covSSInv, covOSInv)
        dictOut = {
            "pOS": pOS,
            "pSS": pSS,
            "covOS": covOS,
            "covSS": covSS,
            "covOSInv": covOSInv,
            "covSSInv": covSSInv,
        }

        return dictOut

    tagConstr = taggingConstraints(data)

    # define fcn function {{{

    # @profile

    def fcn_tag_constr_data(parameters: Parameters, data: dict) -> np.ndarray:
        """
        Cost function to fit real data. Data is a dictionary of years, and each
        year should be a dictionary of trigger categories. This function loops over
        years and trigger categories.  Here we are going to unblind the parameters
        to the p.d.f., thats why we call parameters.valuesdict(blind=False), by
        default `parameters.valuesdict()` has blind=True.

        Parameters
        ----------
        parameters : `ipanema.Parameters`
        Parameters object with paramaters to be fitted.
        data : dict

        Returns
        -------
        np.ndarray
        Array containing the weighted likelihoods

        """
        pars_dict = parameters.valuesdict(blind=False)
        chi2TagConstr = 0.0

        chi2TagConstr += (
            pars_dict["dp0_os"] - data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_os"].value
        ) ** 2 / data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_os"].stdev ** 2
        chi2TagConstr += (
            pars_dict["dp1_os"] - data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_os"].value
        ) ** 2 / data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_os"].stdev ** 2
        chi2TagConstr += (
            pars_dict["dp0_ss"] - data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_ss"].value
        ) ** 2 / data[str(YEARS[0])][TRIGGER[0]].flavor["dp0_ss"].stdev ** 2
        chi2TagConstr += (
            pars_dict["dp1_ss"] - data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_ss"].value
        ) ** 2 / data[str(YEARS[0])][TRIGGER[0]].flavor["dp1_ss"].stdev ** 2

        tagcvOS = np.matrix([pars_dict["p0_os"], pars_dict["p1_os"]]) - tagConstr["pOS"]
        tagcvSS = np.matrix([pars_dict["p0_ss"], pars_dict["p1_ss"]]) - tagConstr["pSS"]

        Y_OS = np.dot(tagcvOS, tagConstr["covOSInv"])
        chi2TagConstr += np.dot(Y_OS, tagcvOS.T)
        Y_SS = np.dot(tagcvSS, tagConstr["covSSInv"])
        chi2TagConstr += np.dot(Y_SS, tagcvSS.T)

        chi2 = []
        for y, dy in data.items():
            for t, dt in dy.items():
                badjanak.delta_gamma5_data(
                    dt.data,
                    dt.prob,
                    **pars_dict,
                    **dt.timeacc.valuesdict(),
                    **dt.angacc.valuesdict(),
                    **dt.resolution.valuesdict(),
                    **dt.csp.valuesdict(),
                    # **dt.flavor.valuesdict(),
                    tLL=tLL,
                    tUL=tUL,
                    use_fk=1,
                    use_angacc=1,
                    use_timeacc=1,
                    use_timeoffset=0,
                    set_tagging=1,
                    use_timeres=1,
                    BLOCK_SIZE=256,
                )
            chi2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get())

        chi2conc = np.concatenate(chi2)
        chi2TagConstr = float(chi2TagConstr[0][0] / len(chi2conc))
        return chi2conc + chi2TagConstr  # np.concatenate(chi2)

    def fcn_data(parameters: Parameters, data: dict) -> np.ndarray:
        """
        Cost function to fit real data. Data is a dictionary of years, and each
        year should be a dictionary of trigger categories. This function loops over
        years and trigger categories.  Here we are going to unblind the parameters
        to the p.d.f., thats why we call parameters.valuesdict(blind=False), by
        default `parameters.valuesdict()` has blind=True.

        Parameters
        ----------
        parameters : `ipanema.Parameters`
        Parameters object with paramaters to be fitted.
        data : dict

        Returns
        -------
        np.ndarray
        Array containing the weighted likelihoods

        """
        pars_dict = parameters.valuesdict(blind=False)
        chi2 = []

        for y, dy in data.items():
            for dt in dy.values():
                badjanak.delta_gamma5_data(
                    dt.data,
                    dt.prob,
                    **pars_dict,
                    **dt.timeacc.valuesdict(),
                    **dt.angacc.valuesdict(),
                    **dt.resolution.valuesdict(),
                    **dt.csp.valuesdict(),
                    **dt.flavor.valuesdict(),
                    use_fk=1,
                    use_angacc=1,
                    use_timeacc=1,
                    use_timeoffset=0,
                    set_tagging=1,
                    use_timeres=1,
                    tLL=tLL,
                    tUL=tUL,
                )
                chi2.append(-2.0 * (ristra.log(dt.prob) * dt.weight).get())

        return np.concatenate(chi2)

    # cost_function = fcn_data
    cost_function = fcn_tag_constr_data

    # }}}

    # Minimization {{{

    printsubsec("Simultaneous minimization procedure")

    if MINER in ("minuit", "minos"):
        result = optimize(
            cost_function,
            method=MINER,
            params=pars,
            fcn_kwgs={"data": data},
            verbose=True,
            timeit=True,
            tol=0.1,
            strategy=1,
            policy="filter",
        )
    elif MINER in ("nelder"):
        result = optimize(
            cost_function,
            method=MINER,
            params=pars,
            fcn_kwgs={"data": data},
            verbose=False,
            timeit=True,
            tol=0.1,
            strategy=2,
            policy="omit",
        )
    elif MINER in ("bfgs", "lbfgsb"):
        result = optimize(
            cost_function,
            method="minuit",
            params=pars,
            fcn_kwgs={"data": data},
            verbose=False,
            timeit=True,
            tol=0.1,
            strategy=2,
            policy="omit",
        )
        result = optimize(
            cost_function,
            method=MINER,
            params=result.pars,
            fcn_kwgs={"data": data},
            verbose=False,
            timeit=True,
            tol=0.1,
            strategy=2,
            policy="filter",
        )
    elif MINER in ("emcee"):
        result = optimize(
            cost_function,
            method="nelder",
            params=pars,
            fcn_kwgs={"data": data},
            verbose=False,
            timeit=True,
            tol=0.1,
            strategy=2,
            policy="omit",
        )
        result = optimize(
            cost_function,
            method="emcee",
            params=result.params,
            fcn_kwgs={"data": data},
            verbose=False,
            timeit=True,
            policy="omit",
            steps=5000,
            nwalkers=1000,
            burn=2,
            workers=8,
        )
    else:
        raise ValueError(f"{MINER} is now a optimizer method from ipanema")

    print(result)

    for p in [
        "fPlon",
        "fPper",
        "dPpar",
        "dPper",
        "pPlon",
        "lPlon",
        "DGsd",
        "DGs",
        "DM",
        "dSlon1",
        "dSlon2",
        "dSlon3",
        "dSlon4",
        "dSlon5",
        "dSlon6",
        "fSlon1",
        "fSlon2",
        "fSlon3",
        "fSlon4",
        "fSlon5",
        "fSlon6",
    ]:
        if args["year"] == "2015,2016":
            # print(f"{p:>12} : {result.params[p].value:+.4f}  {result.params[p]._getval(False):+.4f}")
            print(
                f"{p:>12} : {result.params[p]._getval(False):+.4f} +/- {result.params[p].stdev:+.4f}"
            )
        else:
            print(
                f"{p:>12} : {result.params[p].value:+.4f} +/- {result.params[p].stdev:+.4f}"
            )

    # }}}

    # Dump json file {{{

    print("Dump parameters")
    result.params.dump(args["params"])

# }}}


# vim:foldmethod=marker
# TODO: update tagging constraints to work with cov/corr methods from ipanema
#       parameters
