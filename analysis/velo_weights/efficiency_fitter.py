__all__ = []



from ipanema import (Sample, Parameters, ristra, optimize)
import argparse
import numpy as np
from ipanema import uncertainty_wrapper, get_confidence_bands
import complot

DOCAz_BINS = 14
BuM = 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr'

# different docaz binning schemes {{{

# 10
docaz = [0.000e+00, 8.700e-03, 1.770e-02, 2.730e-02, 3.800e-02, 5.050e-02,
         6.660e-02, 8.990e-02, 1.317e-01, 2.402e-01, 1.000e+01]
# 5
docaz = [0.,     0.0291, 0.0651, 0.1265, 0.3032, 10.]
# 8
docaz = [0.0, 0.0179, 0.0372, 0.0598, 0.0903, 0.1386, 0.2359, 0.4714, 5.0]
# 7 bins expo
docaz = [0.0, 0.3, 0.58, 0.91, 1.35, 1.96, 3.01, 7.0]

# 14 bins francesca
docaz = [0.01909757, 0.02995095, 0.04697244, 0.07366745, 0.11553356,
         0.1811927, 0.28416671, 0.44566212, 0.69893733, 1.09615194,
         1.71910844, 2.69609871, 4.22832446, 6.63133278]

# Francesca-like binning scheme
docaz = [
    0, 0.01652674, 0.02591909, 0.04064923, 0.06375068, 0.09998097,
    0.15680137, 0.2459135, 0.38566914, 0.60484961, 0.94859302,
    1.48768999, 2.33316235, 3.65912694, 6
]

# a set of 15 equially populated bins
docaz = [
    0.00000e+00, 7.80000e-03, 1.57000e-02, 2.40000e-02, 3.30000e-02,
    4.29000e-02, 5.43000e-02, 6.81000e-02, 8.57000e-02, 1.09600e-01,
    1.43700e-01, 1.94400e-01, 2.73500e-01, 4.09000e-01, 7.00300e-01,
    # 3.37956e+01
]

docaz = np.array(docaz)
# docaz = np.array(docaz)[:-1]

# }}}


cuts = ['Jpsi_LOKI_ETA>2 & Jpsi_LOKI_ETA<4.5',
        'muplus_LOKI_ETA>2 & muplus_LOKI_ETA<4.5',
        'muminus_LOKI_ETA>2 & muminus_LOKI_ETA<4.5',
        'Kplus_LOKI_ETA>2 & Kplus_LOKI_ETA<4.5', 'abs(Bu_PVConst_PV_Z)<100'
        ]
# Bu2JpsiKplus cuts
cuts += [  # 'Bu_PVConstPVReReco_chi2/Bu_PVConstPVReReco_nDOF)<5',
    'Bu_IPCHI2_OWNPV<25',
    'Bu_MINIPCHI2<25',
    '(Bu_PVConst_chi2/Bu_PVConst_nDOF)<5',
    'Bu_MINIPCHI2NEXTBEST>50 | Bu_MINIPCHI2NEXTBEST==-1',
    'Bu_LOKI_DTF_CHI2NDOF<4',
    '(Bu_PVConst_ctau/0.299792458)>0.3',
    '(Bu_PVConst_ctau/0.299792458)<14',
    # 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr>5170',
    # 'Bu_LOKI_MASS_JpsiConstr_NoPVConstr<5400'
]
# Kplus cuts
cuts += ['Kplus_TRACK_CHI2NDOF<4', 'Kplus_PT>500',  # 'Kplus_P>10000',
         'Kplus_PIDK>0'
         ]
# muons
cuts += ['muplus_TRACK_CHI2NDOF<4', 'muminus_TRACK_CHI2NDOF<4',
         'muplus_PT>500', 'muminus_PT>500', 'muplus_PIDmu>0',
         'muminus_PIDmu>0'
         ]
# Jpsi cuts
cuts += ['Jpsi_ENDVERTEX_CHI2<16', 'Bu_LOKI_FDS>3',
         'Jpsi_M>3030 & Jpsi_M<3150']
# # trigger requirements
cuts += ["Bu_L0MuonDecision_TOS==1 | Bu_L0DiMuonDecision_TOS==1",
         "Bu_Hlt1DiMuonHighMassDecision_TOS==1",
         "Bu_Hlt2DiMuonDetachedJPsiDecision_TOS==1"]
cut = "(" + ") & (".join(cuts) + ")"

cut = f"({cut}) & ({BuM}>5170 & {BuM}<5400)"






# Command line interface {{{

if __name__ == '__main__':
    # Parse command line arguments {{{

    p = argparse.ArgumentParser(description='Get efficiency in DOCAz bin.')
    p.add_argument('--params', help='Mass fit parameters')
    p.add_argument('--params-match', help='Mass fit parameters VELO matching')
    p.add_argument('--plot', help='Plot of the mass fit')
    p.add_argument('--plot-log', help='Plot of the log mass fit')
    p.add_argument('--sample', help='Year to fit')
    p.add_argument('--eff-pars', help='Year to fit')
    p.add_argument('--year', help='Year to fit')
    p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
    p.add_argument('--version', help='Version of the tuples to use')
    p.add_argument('--trigger', help='Trigger to fit')
    p.add_argument('--mass-model', help='Different flag to ... ')
    args = vars(p.parse_args())

    # }}}

    #
    TRIGGER = args["trigger"]
    MODEL = args["mass_model"]
    SAMPLE = args["sample"]

    # Create lists for all parameters
    velo_unmatch = {k: Parameters.load(p)
                    for k, p in enumerate(args['params'].split(','))}
    velo_match = {k: Parameters.load(p)
                  for k, p in enumerate(args['params_match'].split(','))}

    # We compute the efficiency doca bins
    # eff = {k: velo_match[k]['nsig'].uvalue/velo_unmatch[k]['nsig'].uvalue
    #        for k in velo_match.keys()}
    # print("The efficiency is: ")
    # for db in range(0,len(docaz)-1):
    #   print(f"[{docaz[db]},{docaz[db+1]}) : {velo_match[db]['nsig'].uvalue} / {velo_unmatch[db]['nsig'].uvalue} = {eff[db]}")

    eff = {}
    for k in velo_match.keys():
        # eff[k] = velo_match[k]['nsig'].uvalue / velo_unmatch[k]['nsig'].uvalue
        eff[k] = velo_match[k]['nsig'].uvalue / (velo_match[k]['nsig'].uvalue + velo_unmatch[k]['nsig'].uvalue)
    print("The efficiency a/a+b is: ")
    for db in range(0, len(docaz)-1):
        print(f"[{docaz[db]},{docaz[db+1]}) : {velo_match[db]['nsig'].uvalue} / {velo_unmatch[db]['nsig'].uvalue} = {eff[db]}")

    # data
    x = []
    ux_low = []
    ux_upp = []

    print("Load sample")
    sample = Sample.from_root(SAMPLE, cuts=cut)
    print(sample.df)
    for bin in range(0, len(docaz)-1):
        doca_cut = [f"(Bu_PVConst_Kplus_DOCAz>={docaz[bin]})",
                    f"(Bu_PVConst_Kplus_DOCAz<{docaz[bin+1]})"]
        doca_cut = f"({' & '.join(doca_cut)})"
        _df = sample.df.query(doca_cut)
        x.append(np.median(_df['Bu_PVConst_Kplus_DOCAz'].values))
        ux_low.append(x[-1] - docaz[bin])
        ux_upp.append(docaz[bin+1] - x[-1])
    x = np.array(x)
    ux_low = np.array(ux_low)
    ux_upp = np.array(ux_upp)
    ux = np.array(ux_low)
    y = np.array([eff[i].n for i in eff.keys()])
    uy = np.array([eff[i].s for i in eff.keys()])

    # fcn
    def model(x, a, c): return a * (1 + c * x**2)
    # def model(x, a, b, c): return (a + b*x + c*x**2)

    def fcn(pars, x, y=None, uy=None):
        p = pars.valuesdict()
        _y = model(x, *list(p.values()))
        if y is None:
            return _y
        if uy is None:
            return (y-_y)
        # return (y-_y)**2 #/ (uy**2+0*4*p['a']**2*p['c']**2*x**2*ux**2)
        # return (y-_y)**2
        return (y-_y)**2 / uy**2
        # return (y-_y)**2 / ( uy**2 + () * ux**2 )

    pars = Parameters()
    pars.add(dict(name='a', value=0.98, free=True))
    # pars.add(dict(name='b', value=0.98, free=True))
    pars.add(dict(name='c', value=-0.00098, free=True))

    res = optimize(fcn, pars, method='minuit', fcn_kwgs={
                   "x": x, "y": y, "uy": uy})  # ,
    #               residual_reduce="chi2")
    print(res)
    res.params.dump(args['eff_pars'])

    # plot eff vs. docaz
    fig, axplot, axpull = complot.axes_providers.axes_plotpull()
    axplot.errorbar(x, y, yerr=[uy, uy], xerr=[ux_low, ux_upp], fmt='.',
                    label="+".join(MODEL.split('_')))
    # ax.set_xlim(0, max(docaz))
    _x = np.linspace(0, 6)
    _y = fcn(res.params, _x)
    axplot.plot(_x, _y)
    axpull.fill_between(x, complot.compute_pdfpulls(_x, _y, x, y, ux, uy),
                        0, facecolor="C0", alpha=0.5)
    upars = [p.uvalue for p in res.params.values()]
    _y = uncertainty_wrapper(lambda p: model(_x, *p), res.params)
    _ylow, _yupp = get_confidence_bands(_y)
    axplot.fill_between(_x, _yupp, _ylow, facecolor="C0", alpha=0.2)
    axplot.set_ylim(0.85, 1.05)
    # ax.set_ylim(0.4, 0.6)
    axpull.set_xlabel("DOCAz [mm]")
    axplot.set_ylabel("Efficiency")
    axplot.legend()
    fig.savefig(args["plot"])
    axplot.set_xscale('log')
    axpull.set_xscale('log')
    fig.savefig(args["plot_log"])

# }}}


# vim: fdm=marker
