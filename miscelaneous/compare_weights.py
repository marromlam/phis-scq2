# compare_weights
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import uproot3 as uproot
import matplotlib.pyplot as plt
import os
import pandas as pd
import complot
import numpy as np

YEARS = ['2015']
MODES = ['Bs2JpsiPhi', 'MC_Bs2JpsiPhi', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar']
MODES = ['Bs2JpsiPhi', 'MC_Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0']

BsMC = "MC_Bs2JpsiPhi"
BsG0 = "MC_Bs2JpsiPhi_dG0"
BsRD = "Bs2JpsiPhi"
BdMC = "MC_Bd2JpsiKstar"
BdRD = "Bd2JpsiKstar"

k16 = '2015'

datasets = {}
for year in YEARS:
    datasets[year] = {}
    for mode in MODES:
        _p = "/scratch46/marcos.romero/sidecar14"
        # _p1 = os.path.join(_p, str(year), mode, 'v1r0@LcosKpTB1_ready.root')
        _p2 = os.path.join(_p, str(year), mode, 'v1r0p8~mX4sigmam3_sWeight.root')
        _p1 = os.path.join(_p, str(year), mode, 'v1r0p8.root')
        datasets[year][mode] = [_p2, _p1]


dfs = {}
for y in datasets:
    dfs[y] = {}
    for m in datasets[y]:
        _dfs = []
        _dfs.append(uproot.open(datasets[y][m][0])['DecayTree'].pandas.df())
        _dfs.append(uproot.open(datasets[y][m][1])['DecayTree'].pandas.df())
        dfs[y][m] = pd.concat(_dfs, axis=1)


print(np.array(dfs[k16][BsRD].eval('sw-sWeight'))[:100])
# plot Bs RD
fig, axplot, axpull = complot.axes_providers.axes_plotpull()
# axplot.plot(dfs[k16][BsRD].eval('mB'), dfs[k16][BsRD].eval('sw'),
#             '.', label='sw by Lera')
axplot.plot(dfs[k16][BsRD].eval('mB'), dfs[k16][BsRD].eval('sWeight'),
            'rx', label='sw*wLb from scq', alpha=0.5)
axpull.plot(dfs[k16][BsRD].eval('mB'), dfs[k16][BsRD].eval('sw-sWeight'),
            'rx', label='sw*wLb from scq', alpha=0.5)
# axplot.plot(dfs[k16][BsRD].eval('sw'),
#             dfs[k16][BsRD].eval('sigBsSW'),
#             '.', label='sw from scq', alpha=0.5)
axplot.set_xlabel('B mass')
axplot.set_ylabel('labeled weight')
axplot.set_title(f'{BsRD} - {k16}')
axplot.legend()
fig.show()

exit()

# plot Bs MC
fig, axplot = complot.axes_providers.axes_plot()
axplot.plot(dfs[k16][BsMC].eval('B_ConstJpsi_M_1'),
            dfs[k16][BsMC].eval('sw'),
            '.', label='sw by Lera')
axplot.plot(dfs[k16][BsMC].eval('B_ConstJpsi_M_1'),
            dfs[k16][BsMC].eval('sWeight'),
            '.', label='sw*gb_weights from scq', alpha=0.5)
axplot.set_xlabel('B mass')
axplot.set_ylabel('sWeight')
axplot.set_title(f'{BsMC} - {k16}')
axplot.legend()
fig.show()


# plot Bs MC
fig, axplot = complot.axes_providers.axes_plot()
axplot.plot(dfs[k16][BsG0].eval('mB'),
            dfs[k16][BsG0].eval('gbweights'),
            '.', label='dG0 MC gbWeights')
axplot.plot(dfs[k16][BsMC].eval('mB'),
            dfs[k16][BsMC].eval('gbweights'),
            '.', label='signal MC gbWeights')
axplot.set_xlabel('B mass')
axplot.set_ylabel('sWeight')
axplot.set_title(f'{BsMC} - {k16}')
axplot.legend()
fig.show()


# plot Bd RD
fig, axplot = complot.axes_providers.axes_plot()
axplot.plot(dfs[k16][BdRD].eval('B_ConstJpsi_M_1'),
            dfs[k16][BdRD].eval('sw'),
            '*', label='sw by Peilian')
axplot.plot(dfs[k16][BdRD].eval('B_ConstJpsi_M_1'),
            dfs[k16][BdRD].eval('sigBdSW'),
            '.', label='sw by Santiago')
axplot.set_xlabel('B mass')
axplot.set_ylabel('sWeight')
axplot.set_title(f'{BdRD} - {k16}')
axplot.legend()
fig.show()


# plot Bd MC
fig, axplot = complot.axes_providers.axes_plot()
axplot.plot(dfs[k16][BdMC].eval('B_ConstJpsi_M_1'),
            dfs[k16][BdMC].eval('sw'),
            '*', label='sw by Peilian')
axplot.plot(dfs[k16][BdMC].eval('B_ConstJpsi_M_1'),
            dfs[k16][BdMC].eval('sigBdSW'),
            '.', label='sw by Santiago')
axplot.set_xlabel('B mass')
axplot.set_ylabel('sWeight')
axplot.set_title(f'{BdMC} - {k16}')
axplot.legend()
fig.show()


# plot Bd MC
fig, axplot = complot.axes_providers.axes_plot()
axplot.plot(dfs[k16][BdMC].eval('B_ConstJpsi_M_1'),
            dfs[k16][BdMC].eval('gb_weights'),
            '.', label='gb_Weights')
axplot.set_xlabel('B mass')
axplot.set_ylabel('sWeight')
axplot.set_title(f'{BdMC} - {k16}')
axplot.legend()
fig.show()


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
