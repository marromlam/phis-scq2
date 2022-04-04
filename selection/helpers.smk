# Helper functions for defining inputs and outputs for rules.
#
#

import os
from os.path import isfile, join
import yaml

# xrootd config
from snakemake.remote.XRootD import RemoteProvider as XRootDRemoteProvider
XRootD = XRootDRemoteProvider(stay_on_remote=True)

# OUTPUT_DIR = os.path.join(os.path.dirname(workflow.snakefile), 'output')
# TUPLES_DIR = os.path.join(OUTPUT_DIR, 'tuples')
# OUTPUT_DIR = "output"
# TUPLES_DIR = "/scratch46/marcos.romero/sidecar14"
#
# def output_path(path):
#     assert not path.startswith('/')
#     return os.path.join(OUTPUT_DIR, path)
#
#
# def tuples_path(path):
#     assert not path.startswith('/')
#     return os.path.join(TUPLES_DIR, path)


def eos_path(path):
    path_str = str(path[0])
    return XRootD.remote('root://eoslhcb.cern.ch/'+path_str)


def load_configuration(name):
    """Return the configuration dictionary of the named file."""
    cpath = 'data/{}.yaml'.format(name)
    with open(cpath) as f:
        c = yaml.safe_load(f)

    return c


def files_from_configuration(names):
    """Return the list of all file paths given in the configurations.

    Any file path starting with `root://` is wrapped by the XRootD provider.
    """
    # Make sure we have a list
    if isinstance(names, str):
        names = [names]

    fpaths = []
    for name in names:
        c = load_configuration(name)
        paths = [XRootD.remote(p) if p.startswith('root://') else p for p in c['paths']]
        fpaths += paths

    return fpaths


def ntuple_from_yaml(name, year, version):
    """Return the path to ntuple given in the yaml.
    """
    cpath = '{}.yaml'.format(name)
    with open(cpath) as f:
        c = yaml.safe_load(f)
    path = c['fitinputs']['ntuple'].format(eos=selconfig['eos'],version=version,year=year)
    return path
