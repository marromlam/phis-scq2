import numpy as np
from scipy.stats import chi2
from ipanema import Parameters


__all__ = ['merge_std_dg0']
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


def merge_std_dg0(std, dg0, verbose=True):
    """
    Merge two datasets
    """
    # Create w and cov arrays
    std_w = np.array([std[i].value for i in std])[1:]
    dg0_w = np.array([dg0[i].value for i in dg0])[1:]
    print(std_w)
    print(dg0_w)
    std_cov = std.cov()[1:, 1:]
    dg0_cov = dg0.cov()[1:, 1:]

    # Some matrixes
    std_covi = np.linalg.inv(std_cov)
    dg0_covi = np.linalg.inv(dg0_cov)
    cov_comb_inv = np.linalg.inv(std_cov + dg0_cov)
    cov_comb = np.linalg.inv(std_covi + dg0_covi)

    # Check p-value
    chi2_value = (std_w-dg0_w).dot(cov_comb_inv.dot(std_w-dg0_w))
    dof = len(std_w)
    prob = chi2.sf(chi2_value, dof)

    # Combine angular weights
    w = np.ones((dof+1))
    w[1:] = cov_comb.dot(std_covi.dot(std_w.T) + dg0_covi.dot(dg0_w.T))

    # Combine uncertainties
    uw = np.zeros_like(w)
    uw[1:] = np.sqrt(np.diagonal(cov_comb))

    # Build correlation matrix
    corr = np.zeros((dof+1, dof+1))
    for k in range(1, cov_comb.shape[0]):
        for j in range(1, cov_comb.shape[1]):
            corr[k, j] = cov_comb[k][j]/np.sqrt(cov_comb[k][k]*cov_comb[j][j])

    # Create parameters std_w
    out = Parameters()
    for k, wk in enumerate(std.keys()):
        correl = {f'{wk}': corr[k][j]
                  for j in range(0, len(w)) if k > 0 and j > 0}
        out.add({'name': f'{wk}', 'value': w[k], 'stdev': uw[k],
                 'free': False, 'latex': f'{std[wk]}', 'correl': correl})

    if verbose:
        print(f"{'MC':>8} | {'MC_dG0':>8} | {'Combined':>8}")
        for k, wk in enumerate(std.keys()):
            print(f"{np.array(std)[k]:+1.5f}", end=' | ')
            print(f"{np.array(dg0)[k]:+1.5f}", end=' | ')
            print(f"{out[f'{wk}'].uvalue:+1.2uP}")
    return out


# vim: fdm=marker
