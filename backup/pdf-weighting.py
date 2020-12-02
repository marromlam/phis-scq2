#!/usr/bin/env python
# -*- coding: utf-8 -*-




# Modules ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os, sys
import json

# openCL stuff
cl_path = os.path.join(os.environ['PHIS_SCQ'],'opencl')
import pyopencl as cl                      # Import the OpenCL GPU computing API
import pyopencl.array as cl_array                        # Import PyOpenCL Array
context = cl.create_some_context()                      # Initialize the Context
queue   = cl.CommandQueue(context)                         # Instantiate a Queue

# get Badjanak model
sys.path.append(cl_path)
from Badjanak import *

################################################################################
################################################################################
################################################################################
################################################################################

# input parameters: NEEDS TO BE CHANGED!
input_file      = "/home3/marcos.romero/phis-scq-old/MC_Bs2JpsiPhi_2016_selected_bdt_v0r1.root"
tree_name       = 'DecayTree'
output_file     = '/home3/marcos.romero/phis-scq-old/MC_Bs2JpsiPhi_2016_selected_bdt_pdfWeight_v0r1.root'
original_params = json.load(open('/home3/marcos.romero/phis-scq-old/input/tad-2016-both-simon1.json'))
target_params   = json.load(open('/home3/marcos.romero/phis-scq-old/input/tad-2016-both-simon2.json'))

# the names of the different parameter files need a more descriptive name besides simon

################################################################################
################################################################################
################################################################################
################################################################################



import ROOT
def getHelicityAngles( Kplus_P,  Kminus_P,  muplus_P,  muminus_P ) :


    import math

    # Define some vectors that we need
    vecA = ROOT.TVector3( muplus_P .Vect().Unit() );
    vecB = ROOT.TVector3( muminus_P.Vect().Unit() );
    vecC = ROOT.TVector3( Kplus_P  .Vect().Unit() );
    vecD = ROOT.TVector3( Kminus_P .Vect().Unit() );
    phi  = ROOT.TLorentzVector(  Kplus_P +  Kminus_P);
    jpsi = ROOT.TLorentzVector( muplus_P + muminus_P);
    bs   = ROOT.TLorentzVector(    phi   +    jpsi  );

    # Normals to decay planes
    el = ROOT.TVector3( (vecA.Cross(vecB)).Unit() );
    ek = ROOT.TVector3( (vecC.Cross(vecD)).Unit() );

    # The direction of mother2 in the B frame
    ez = ROOT.TVector3( phi.Vect().Unit() );

    # Calculate phi angle
    cosPhi = (ek.Dot(el));
    sinPhi = (el.Cross(ek)).Dot(ez);
    helphi = math.atan2(sinPhi, cosPhi);

    # Calculate cosThetaMu
    JpsiBoost = jpsi.BoostVector();
    muplus_P.Boost(-JpsiBoost);
    helcosthetaL = (JpsiBoost.Unit()).Dot(muplus_P.Vect().Unit());

    # Calculate cosThetaK
    PhiBoost = phi.BoostVector();
    Kplus_P.Boost(-PhiBoost);
    helcosthetaK = (PhiBoost.Unit()).Dot(Kplus_P.Vect().Unit());

    angles = [helcosthetaK, helcosthetaL, helphi]
    return angles






################################################################################
################################################################################
################################################################################
################################################################################



# Flags
config = {
  "USE_TIME_ACC":    "0",# NO  time acceptance
  "USE_TIME_OFFSET": "0",# NO  time offset
  "USE_TIME_RES":    "0",# USE time resolution
  "USE_PERFTAG":     "1",# USE perfect tagging
  "USE_TRUETAG":     "0",# NO  true tagging
}


# Compile model and get kernels
BsJpsiKK     = Badjanak(cl_path,'cl',context,queue,**config)
getCrossRate = BsJpsiKK.getCrossRate




# pdf_weighting ----------------------------------------------------------------

def pdf_weighting(input_file, tree_name, output_file,
                  target_params, original_params):

  # Load file
  print('Loading file...')
  input_file = uproot.open(input_file)[tree_name]
  data = input_file.pandas.df()
  for key in data.keys():
    print(key)

  # Prepare host arrays
  tad_vars = ['cosThetaKRef_GenLvl','cosThetaMuRef_GenLvl','phiHelRef_GenLvl',
              'time_GenLvl', 'X_M','sigmat','B_ID_GenLvl']
  #tad_vars = ['truehelcosthetaK','truehelcosthetaL','truehelphi','B_TRUETAU', 'X_M','sigmat','B_ID_GenLvl']
  vars_h = np.ascontiguousarray(data[tad_vars].values)    # input array (matrix)
  vars_h[:,3] *= 1e3                                                # time in ps
  pdf_h  = np.zeros(vars_h.shape[0])                        # output array (pdf)

  # Allocate device_arrays
  vars_d = cl_array.to_device(queue,vars_h).astype(np.float64)
  pdf_d  = cl_array.to_device(queue,pdf_h).astype(np.float64)

  # Compute!
  print('Calc weights...')
  original_pdf_h = getCrossRate(vars_d,pdf_d,original_params,7)     # 7 mKK bins
  target_pdf_h   = getCrossRate(vars_d,pdf_d,target_params,1)        # 1 mKK bin
  np.seterr(divide='ignore', invalid='ignore')                 # remove warnings
  pdfWeight = np.nan_to_num(original_pdf_h/target_pdf_h)
  data['pdfWeight'] = pdfWeight

  # Save weights to file
  print('Writting pdfWeights to file...')
  if output_file in os.path.dirname(output_file):
    os.remove(output_file)                               # delete file if exists
  with uproot.recreate(output_file,compression=None) as f:
    f["DecayTree"] = uproot.newtree({var:'float64' for var in data})
    f["DecayTree"].extend(data.to_dict(orient='list'))
  f.close()
  print(output_file+' was succesfully written')

  return pdfWeight



pdf_weighting(input_file, tree_name, output_file, target_params, original_params)





foo = uproot.open('/home3/marcos.romero/phis-scq/MC_Bs2JpsiPhi_2016_selected_bdt_PDFWeight_v0r1.root')['DecayTree'].array('pdfWeight')
bar = uproot.open('/scratch03/marcos.romero/SideCar/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights'].array('PDFWeight')



plt.plot(foo-bar)
plt.xlabel('events')
plt.ylabel('pdfWeight_simon-pdfWeight_scq')
