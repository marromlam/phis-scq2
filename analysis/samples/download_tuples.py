__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

import config
import shutil
import uproot3 as uproot
import argparse
import os
import numpy as np

# }}}


# Some config {{{

vsub_dict = {
    # used to place random cuts for MC
    "evtOdd": "(eventNumber % 2) != 0",
    "evtEven": "(eventNumber % 2) == 0",
    # magnet cuts
    "magUp": "Polarity == 1",
    "magDown": "Polarity == -1",
    "bkgcat60": "B_BKGCAT != 60",
    # lower and upper time cuts
    "LT": f"time < {config.general['lower_time_upper_limit']}",
    "UT": f"time > {config.general['upper_time_lower_limit']}",
    # runNumber to check alignment in 2018
    "g210300": "runNumber > 210300",
    "l210300": "runNumber < 210300",
    # bins of pTB
    "pTB1": "B_PT >= 0 & B_PT < 3.8e3",
    "pTB2": "B_PT >= 3.8e3 & B_PT < 6e3",
    "pTB3": "B_PT >= 6e3 & B_PT <= 9e3",
    "pTB4": "B_PT >= 9e3",
    # bins of etaB
    "etaB1": "B_ETA >= 0 & B_ETA <= 3.3",
    "etaB2": "B_ETA >= 3.3 & B_ETA <= 3.9",
    "etaB3": "B_ETA >= 3.9 & B_ETA <= 6",
    # sigmat
    "sigmat1": "sigmat >= 0 & sigmat <= 0.031",
    "sigmat2": "sigmat >= 0.031 & sigmat <= 0.042",
    "sigmat3": "sigmat >= 0.042 & sigmat <= 0.15",
    # cut the Bd angular distribution
    "LcosK": "helcosthetaK<=0.0",
    "UcosK": "helcosthetaK>0.0",
    # pXB and pYB cuts
    "pXB1": "B_PX >= 0 & B_PX < 2.7e3",
    "pXB2": "B_PX >= 2.7e3 & B_PX < 4.2e3",
    "pXB3": "B_PX >= 4.2e3 & B_PX <= 6.3e3",
    "pXB4": "B_PX >= 6.3e3",
    "pYB1": "B_PY >= 0 & B_PY < 2.7e3",
    "pYB2": "B_PY >= 2.7e3 & B_PY < 4.2e3",
    "pYB3": "B_PY >= 4.2e3 & B_PY <= 6.3e3",
    "pYB4": "B_PY >= 6.3e3",
}

# }}}


# CMDLINE interfrace {{{

if __name__ == "__main__":
    # argument parser for snakemake
    DESCRIPTION = """
    This script downloads tuples from open('config.json')->['eos'] and places
    them, properly renamed within the convention of phis-scq, in the
    open('config.json')->['path'] (the so-called sidecar folder)
    """
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--year', help='Year of the tuple.')
    p.add_argument('--mode', help='Decay mode of the tuple.')
    p.add_argument('--version', help='Version of the selection pipeline')
    p.add_argument('--weight', help='The tuple surname')
    p.add_argument('--tree', help='Input file tree name.')
    p.add_argument('--output', help='Input file tree name.')
    p.add_argument('--eos', help='Input file tree name.')
    p.add_argument('--uproot-kwargs', help='Arguments to uproot.pandas.df')
    args = vars(p.parse_args())

    # Get the flags and that stuff
    # pipeline tuple version
    v = args['version'].split("@")[0].split("~")[0].split("bdt")[0]
    V = args['version'].replace('bdt', '')  #  full version for phis-scq
    y = args['year']
    m = args['mode']
    w = args['weight']
    tree = args['tree']
    EOSPATH = "root://eoslhcb.cern.ch/" + args['eos']
    is_mc = True if 'MC' in m else False

    eos_name = f"{v}_{m}_{y}_{w}.root"
    eos_path = os.path.join(f"{EOSPATH}", v, m, y)

    loc_name = f"{v}_{w}.root"
    loc_path = os.path.dirname(os.path.abspath(args['output']))

    path = os.path.join(loc_path, str(abs(hash(f'{eos_path}_{eos_name}'))))
    print(f"Required tuple : {eos_path}/{eos_name} at {loc_path}")
    os.makedirs(path, exist_ok=False)  # create hashed temp dir
    all_files = []
    all_dfs = []
    tmp_path = f'{path}/{v}.root'

    # some version substring imply using new sWeights (pTB, etaB and sigmat)
    status = 1

    if "pTB" in V:
        sw = 'sw_pt'
        eos_file = f"{eos_path}/{eos_name}"
        status = os.system(f"xrdcp -f {eos_file} {tmp_path}")
    elif "etaB" in V:
        sw = 'sw_eta'
        eos_file = f"{eos_path}/{eos_name}"
        status = os.system(f"xrdcp -f {eos_file} {tmp_path}")
    elif "sigmat" in V:
        sw = 'sw_sigmat'
        eos_file = f"{eos_path}/{eos_name}"
        status = os.system(f"xrdcp -f {eos_file} {tmp_path}")

    # version (baseline tuples)
    if status:
        sw = 'sw'
        eos_file = f"{eos_path}/{eos_name}"
        status = os.system(f"xrdcp -f {eos_file} {tmp_path}")
    if status:
        print("These tuples are not yet avaliable.")
        print("You may need to create those tuples yourself or ask phis")
        print("team to produce them")
        shutil.rmtree(path, ignore_errors=True)
        exit()

    # }}}

    # If we reached here, then all should be fine
    print(f"\n\n{80*'='}")
    print(f"Downloaded {eos_path}")
    print(f"{80*'='}\n\n")

    # If the tuple is flagged as `ready`, then we need to prepare it fo the
    # phis-scq analysis pipeline. If it is flagged as `selected` we skip it
    # and we basically copy it in the right location
    if args['weight'] == 'ready':
        # try DVTuple if the tree does not work
        try:
            result = uproot.open(tmp_path)[tree]
        except:
            try:
                result = uproot.open(tmp_path)['DVTuple'][tree]
            except:
                result = uproot.open(tmp_path)
                print("WARNING: Automatically reading tree:",
                      list(result.keys())[0])
                result = result[list(result.keys())[0]]
        result = result.pandas.df(flatten=None)

        try:
            print("There are sWeights variables")
            if 'sw_cosK_noGBw' in list(result.keys()):
                print('Adding Peilian sWeight')
                # overwrite sw variable
                result.eval("sw = sw_cosK_noGBw", inplace=True)
            else:
                print(f"Adding standard sWeight: {sw}")
                # overwrite sw variable
                result.eval(f"sw = {sw}", inplace=True)
        except:
            if 'B_BKGCAT' in list(result.keys()):
                print("WARNING: sWeight is set to zero for B_BKGCAT==60")
                result['sw'] = np.where(result['B_BKGCAT'].values != 60, 1, 0)
            else:
                print("WARNING: sWeight variable was not found. Set sw = 1")
                result['sw'] = np.ones_like(result[result.keys()[0]])

        # place cuts according to version substring
        list_of_cuts = []
        vsub_cut = None
        for k, v in vsub_dict.items():
            if k in V:
                try:
                    noe = len(result.query(v))
                    if (k in ("g210300", "l210300")) and is_mc:
                        print("MCs are not cut in runNumber")
                    elif (k in ("g210300", "l210300")) and ("2018" != y):
                        print("Only 2018 RD is cut in runNumber")
                    elif (k in ("UcosK", "LcosK")) and 'Bd2JpsiKstar' not in m:
                        print("Cut in cosK was only planned in Bd")
                    else:
                        list_of_cuts.append(v)
                    if noe == 0:
                        print(f"ERROR: This cut leaves df empty. {v}")
                        print("       Query halted.")
                except:
                    print(f"There is no such variable for the cut {v}")
        if list_of_cuts:
            vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"

        # place the cut
        if vsub_cut:
            result = result.query(vsub_cut)
        print(f"Downloaded {eos_file}")
        print(result)
        print(f"Applied cut: {vsub_cut}")

        # write
        print(f"Starting to write {os.path.basename(args['output'])} file.")
        with uproot.recreate(args['output']) as f:
            _branches = {}
            for k, v in result.items():
                if 'int' in v.dtype.name:
                    _v = np.int32
                elif 'bool' in v.dtype.name:
                    _v = np.int32
                else:
                    _v = np.float64
                _branches[k] = _v
            mylist = list(dict.fromkeys(_branches.values()))
            f[tree] = uproot.newtree(_branches)
            f[tree].extend(result.to_dict(orient='list'))
        print('Succesfully written.')

        # delete donwloaded files
        shutil.rmtree(path, ignore_errors=True)
    else:
        shutil.copy(tmp_path, args['output'])
        shutil.rmtree(path, ignore_errors=True)

# }}}


# vim: ts=4 sw=4 sts=4 et
