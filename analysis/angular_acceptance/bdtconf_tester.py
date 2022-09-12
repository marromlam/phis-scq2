__all__ = ["bdtmesh"]
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


from ipanema import ristra, initialize
import numpy as np
import builtins
import typing


if not builtins.BACKEND:
    initialize('python')


def bdtmesh(conf:int, bdt_tests:int=100, verbose:bool =False):
    """
    This file runs different bdtconfigs to test how this configuration affect
    the final fit result. This is a very expensive job, so be aware it will
    take a lot to run.
    """
    c = int(np.round(bdt_tests**0.25))
    r = 0 if abs(bdt_tests-c**4) < abs(bdt_tests-c**3*(c+1)) else 1

    n_estimators = np.round(np.linspace(
        10, 100, (r+c) if (r+c) < 30 else 30)/10)*10
    learning_rate = np.round(
        100*np.linspace(0.05, 0.3, c if c < 10 else 10))/100
    max_depth = np.round(np.linspace(2, 10, c if c < 10 else 10))
    min_samples_leaf = np.round(np.linspace(
        3e2, 5e3, c if c < 50 else 50)/1e2)*1e2

    # mesh them
    aja = ristra.ndmesh(n_estimators, learning_rate,
                        max_depth, min_samples_leaf)
    bdt_configs = np.vstack([aja[i].ravel() for i in range(len(aja))]).T
    # -> np.vstack(map(np.ravel, aja)) ?

    # select conf
    ans = bdt_configs[conf-1]

    if verbose:
        print(f"Effective number of bdt_tests = {bdt_configs.shape[0]}")
        print(f"different n_estimators = {len(n_estimators)}")
        print(f"different learning_rate = {len(learning_rate)}")
        print(f"different max_depth = {len(max_depth)}")
        print(f"different min_samples_leaf = {len(min_samples_leaf)}")
        print("Selected bdt-config = ", end='')
        print(f"{int(ans[0])}:{ans[1]:.2f}:{int(ans[2])}:{int(ans[3])}")

    ans = {
        "n_estimators": int(ans[0]),
        "learning_rate": ans[1],
        "max_depth": int(ans[2]),
        "min_samples_leaf": int(ans[3]),
        "gb_args": {"subsample": 1}
    }
    return ans
