DESCRIPTION = """
    This file runs different bdtconfigs to test how this configuration affect 
    the final fit result. This is a very expensive job, so be aware it will take
    a lot to run.
"""

from hep_ml import reweight
from utils.helpers import version_guesser, timeacc_guesser, trigger_scissors
from utils.strings import cammel_case_split, cuts_and
from utils.plot import mode_tex
import badjanak
from ipanema import ristra, Sample, Parameters, Parameter, optimize
from ipanema import initialize
import multiprocessing
import time
import threading
import logging
from hep_ml.metrics_utils import ks_2samp_weighted
from warnings import simplefilter
from timeit import default_timer as timer
from scipy.stats import chi2
from uncertainties import unumpy as unp
import uncertainties as unc
import hjson
import sys
import os
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
__author__ = ['Marcos Romero']
__email__ = ['mromerol@cern.ch']


################################################################################
# %% Modules ###################################################################


# reweighting config
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# threading

# load ipanema
initialize(os.environ['IPANEMA_BACKEND'], 1)

# get badjanak and compile it with corresponding flags
badjanak.config['fast_integral'] = 0
badjanak.config['debug'] = 0
badjanak.config['debug_evt'] = 0
badjanak.get_kernels(True)

# import some phis-scq utils

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# reweighting config
# ignore future warnings
simplefilter(action='ignore', category=FutureWarning)
bdconfig = hjson.load(open('config.json'))['angular_acceptance_bdtconfig']
reweighter = reweight.GBReweighter(**bdconfig)
#40:0.25:5:500, 500:0.1:2:1000, 30:0.3:4:500, 20:0.3:3:1000
