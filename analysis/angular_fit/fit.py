# -*- coding: utf-8 -*-
################################################################################
#                                                                              #
#                                  FITTER                                      #
#                                                                              #
################################################################################

__author__ = ['Marcos Romero']
__email__  = ['mromerol@cern.ch']



################################################################################
# %% Modules ###################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import pandas

if __name__ == '__main__':
  from ipanema import initialize
  initialize('cuda',1)
  from ipanema import Sample, Parameters, Parameter
  from ipanema import ristra, optimize

  # get bsjpsikk and compile it with corresponding flags
  import bsjpsikk
  bsjpsikk.config['debug'] = 0
  bsjpsikk.config['debug_evt'] = 1
  bsjpsikk.config['use_time_acc'] = 1
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res'] = 1
  bsjpsikk.config['use_perftag'] = 0
  bsjpsikk.config['use_truetag'] = 0
  bsjpsikk.get_kernels()

  # input parameters
  path = '/scratch17/marcos.romero/phis_samples/v0r2'
  version = 'v0r4'
  flag = 'test'
  year = 2016
  mode = 'Bs2JpsiPhi'


  ################################################################################

  samples = [f'/scratch17/marcos.romero/phis_samples/{version}/{2016}/{mode}/{flag}.root']
  coeffs_biased = [f'output/{version}/params/time_acceptance/{year}/Bd2JpsiKstar/{flag}_baseline_biased.json']
  coeffs_unbiased = [f'output/{version}/params/time_acceptance/{year}/Bd2JpsiKstar/{flag}_baseline_unbiased.json']

  years = [2016]


  ################################################################################
  # %% Load samples ##############################################################

  # List of varaibles to allocate in device array
  arr_data = []
  arr_data += ['cosK','cosL','hphi','time']                    # angular variables
  arr_data += ['X_M','sigmat']                                   # mass and sigmat
  arr_data += ['tagOS_dec','tagSS_dec', 'tagOS_eta', 'tagSS_eta']        # tagging

  def calc_sw(sw):
    try:
      return sw * ( sum(sw)/sum(sw*sw) )
    except:
      sw = sw.values
      return sw * ( sum(sw)/sum(sw*sw) )

  # Load data
  data = {}
  for i, y in enumerate(years):
    data[f'{y}'] = {}
    data[f'{y}']['biased'] = Sample.from_root(samples[i], cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==0')
    data[f'{y}']['unbiased'] = Sample.from_root(f'{path}/{y}/{mode}/test.root', cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
    data[f'{y}']['biased'].coeffs = [1,1.485637144,2.056236952,2.117312551,2.278232224,2.289663907,2.449815939,2.235728992,2.321796692]
    data[f'{y}']['unbiased'].ang_weights = np.float64([1,1.02619,1.02592,-0.000808072,0.000768617,0.00020551,1.00647,0.000455222,0.000146616,-0.000731596])
    mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
    yb = np.zeros_like(data[f'{y}']['biased'].df['sw'])
    yu = np.zeros_like(data[f'{y}']['unbiased'].df['sw'])
    for i in range(0,6):
      yb = np.where(data[f'{y}']['biased'].df.eval(f'X_M>={mass[i]} & X_M<{mass[i+1]}'),calc_sw( data[f'{y}']['biased'].df.eval(f' (X_M>={mass[i]} & X_M<{mass[i+1]})*sw ') ),yb)
      yu = np.where(data[f'{y}']['unbiased'].df.eval(f'X_M>={mass[i]} & X_M<{mass[i+1]}'),calc_sw( data[f'{y}']['unbiased'].df.eval(f' (X_M>={mass[i]} & X_M<{mass[i+1]})*sw ') ),yu)
    data[f'{y}']['biased'].df['sWeight'] = yb
    data[f'{y}']['unbiased'].df['sWeight'] = yu
    print(yb.sum())
    print(yu.sum())
    for i in range(0,6):
      print(i)
      #print(data[f'{y}']['biased'].df.query(f'X_M>={mass[i]} & X_M<{mass[i+1]}')['sWeight'])
      x = data[f'{y}']['biased'].df.query(f'X_M>={mass[i]} & X_M<{mass[i+1]}')['sw']
      print(np.sum(x)/np.sum(x*x))
      #print(data[f'{y}']['unbiased'].df.query(f'X_M>={mass[i]} & X_M<{mass[i+1]}')['sWeight'])
      x = data[f'{y}']['unbiased'].df.query(f'X_M>={mass[i]} & X_M<{mass[i+1]}')['sw']
      print(np.sum(x)/np.sum(x*x))  #data[f'{y}']['biased'].df['sWeight'] = calc_sw( data[f'{y}']['biased'].df['sw'])
    #data[f'{y}']['unbiased'].df['sWeight'] = calc_sw( data[f'{y}']['unbiased'].df['sw'])
    data[f'{y}']['unbiased'].coeffs = np.float64([1,1.007332501,1.029596128,1.000472421,0.9823659563,0.9979454187,1.004462718,0.9834125195,0.980816698])
    data[f'{y}']['biased'].ang_weights = np.float64([1,1.0204196,1.0205028,0.0026313506,0.0031254275,-0.00032937306,1.0115991,0.00025576617,4.6120163e-06,-0.0013316976])
    data[f'{y}']['biased'].allocate(data=arr_data,weight='sWeight',lkhd='0*time')
    data[f'{y}']['unbiased'].allocate(data=arr_data,weight='sWeight',lkhd='0*time')





  # %% Prepare set of parameters -------------------------------------------------

  # Some options
  SWAVE = 1; DGZERO = 0; pars = Parameters()



  # List of parameters
  list_of_parameters = [#
  Parameter(name='fSlon1',          value=SWAVE*0.48,         min=SWAVE*0.40,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon2',          value=SWAVE*0.0590,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon3',          value=SWAVE*0.0101,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon4',          value=SWAVE*0.0103,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon5',          value=SWAVE*0.059,          min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon6',          value=SWAVE*0.1930,         min=SWAVE*0.10,   max=0.60,   free=SWAVE),
  # Parameter(name='fSlon1',          value=SWAVE*0.473156516574591,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon2',          value=SWAVE*0.037061589305578,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon3',          value=SWAVE*0.00472688623744,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon4',          value=SWAVE*0.008872200912457,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon5',          value=SWAVE*0.067568158877522,          min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon6',          value=SWAVE*0.142515903747848,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  #
  Parameter(name="fPlon",           value=0.524,                min=0.5,    max=0.6,    free=True),
  Parameter(name="fPper",           value=0.25,                 min=0.1,    max=0.3,    free=True),
  #
  Parameter(name="pSlon",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  Parameter(name="pPlon",           value= 0.07,                min=-0.5,   max=0.5,    free=True),
  Parameter(name="pPpar",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  Parameter(name="pPper",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  #
  Parameter(name='dSlon1',          value=+2.327*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon2',          value=+1.619*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon3',          value=+1.055*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon4',          value=-0.225*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  Parameter(name='dSlon5',          value=-0.48*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  Parameter(name='dSlon6',          value=-1.141*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  #
  Parameter(name="dPlon",           value=0.00,                 min=-3.0,   max=3.0,    free=False),
  Parameter(name="dPpar",           value=3.26,                 min= 3.0,   max=3.5,    free=True),
  Parameter(name="dPper",           value=3.026,                 min= 2.0,   max=3.2,    free=True),
  #
  Parameter(name="lSlon",           value=1.,                   min=0.7,    max=1.3,    free=False),
  Parameter(name="lPlon",           value=1.,                   min=0.7,    max=1.3,    free=True),
  Parameter(name="lPpar",           value=1.,                   min=0.7,    max=1.3,    free=False),
  Parameter(name="lPper",           value=1.,                   min=0.7,    max=1.3,    free=False),
  #
  Parameter(name="Gd",              value= 0.65789,             min= 0.0,   max= 1.0,   free=False),
  Parameter(name="DGs",             value= (1-DGZERO)*0.0917,   min= 0.0,   max= 0.2,   free=(1-DGZERO)),
  Parameter(name="DGsd",            value= 0.03,                min=-0.1,   max= 0.1,   free=True),
  Parameter(name="DM",              value=17.768,           min=17.0,   max=18.0,   free=True),
  # CSP parameters
  Parameter(name='CSP1',            value=0.8463*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP2',            value=0.8756*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP3',            value=0.8478*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP4',            value=0.8833*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP5',            value=0.9415*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP6',            value=0.9756*SWAVE,         min=0.0,    max=1.0,    free=False),
  # Time resolution parameters
  Parameter(name='mu',              value=0.0,                  min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_offset',    value=0.01297,              min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_slope',     value=0.8446,               min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_curvature', value=0,                    min=0.0,    max=1.0,    free=False),
  # Flavor tagging parameters
  Parameter(name='eta_os',           value=0.3602,               min=0.0,    max=1.0,    free=False),
  Parameter(name='eta_ss',           value=0.4167,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p0_os',            value=0.389,                min=0.0,    max=1.0,    free=False),
  Parameter(name='p0_ss',            value=0.4325,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p1_os',            value=0.8486,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p1_ss',            value=0.9241,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp0_os',           value=0.009 ,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp0_ss',           value=0,                    min=0.0,    max=1.0,    free=False),
  Parameter(name='dp1_os',           value=0.0143,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp1_ss',           value=0,                    min=0.0,    max=1.0,    free=False)
  ]

  # update input parameters file
  pars.add(*list_of_parameters); #pars.dump(taf_path+'/input/'+FITTING_SAMPLE)


  pars['fPlon'].value = 0.512314599353
  pars['fPper'].value = 0.247866538577
  pars['dPpar'].value = 3.08736129166
  pars['dPper'].value = 2.64132238167
  pars['pPlon'].value = -0.107958702675
  pars['DGs'].value = 0.0798542641411
  pars['lPlon'].value = 1.01646881156
  pars['DGsd'].value = -0.004509478974
  pars['DM'].value = 17.7105963073
  pars['fSlon1'].value = 0.473560718454
  pars['fSlon2'].value = 0.0368893354524
  pars['fSlon3'].value = 0.00458919864125
  pars['fSlon4'].value = 0.009023172708
  pars['fSlon5'].value = 0.0678419662382
  pars['fSlon6'].value = 0.142729498712
  pars['dSlon1'].value = 2.32715533865
  pars['dSlon2'].value = 1.61459043778
  pars['dSlon3'].value = 1.06015274293
  pars['dSlon4'].value = -0.234248923015
  pars['dSlon5'].value = -0.488969205683
  pars['dSlon6'].value = -1.14817710086
  pars['fPlon'].init = 0.512314599353
  pars['fPper'].init = 0.247866538577
  pars['dPpar'].init = 3.08736129166
  pars['dPper'].init = 2.64132238167
  pars['pPlon'].init = -0.107958702675
  pars['DGs'].init = 0.0798542641411
  pars['lPlon'].init = 1.01646881156
  pars['DGsd'].init = -0.004509478974
  pars['DM'].init = 17.7105963073
  pars['fSlon1'].init = 0.473560718454
  pars['fSlon2'].init = 0.0368893354524
  pars['fSlon3'].init = 0.00458919864125
  pars['fSlon4'].init = 0.009023172708
  pars['fSlon5'].init = 0.0678419662382
  pars['fSlon6'].init = 0.142729498712
  pars['dSlon1'].init = 2.32715533865
  pars['dSlon2'].init = 1.61459043778
  pars['dSlon3'].init = 1.06015274293
  pars['dSlon4'].init = -0.234248923015
  pars['dSlon5'].init = -0.488969205683
  pars['dSlon6'].init = -1.14817710086


  pars.dump('angular_acceptance/Bs2JpsiPhi.json')

  print(pars)





  #%% Build FCN

  #data

  #bsjpsikk.get_4cs(data[f'{2016}']['unbiased'].coeffs)
  #bsjpsikk.get_4cs(data[f'{2016}']['biased'].coeffs)




  """
  # Test pdf
  bsjpsikk.diff_cross_rate_full(
  data['2016']['unbiased'].data,
  data['2016']['unbiased'].lkhd,
  w = data['2016']['unbiased'].ang_weights,
  coeffs = data['2016']['unbiased'].coeffs,
  **pars.valuesdict()
  )

  shit = Sample.from_root(f'{path}/{year}/{mode}/test.root')
  dec = np.array(shit.df['Jpsi_Hlt1DiMuonHighMassDecision_TOS']==1)
  y = np.zeros(dec.sum())
  j = 0;  k = 0
  for i in range(0,len(head)):
    if dec[i]:
      y[j] = head[i]; j +=1

  z = data['2016']['unbiased'].lkhd.get()
  w = z-y
  plt.plot(w,'.')
  """








  """

  def prepare_if(var,number):
    prec = 1e7
    num_ceil = []; num_floor = []
    if isinstance(var,str):
      v_len = 1;
    else:
      v_len = len(var)
    for i in range(v_len):
      if v_len > 1:
        num_ceil.append(np.ceil(number[i]*prec)/prec)
        num_floor.append(np.floor(number[i]*prec)/prec)
      else:
        try:
          num_ceil.append(np.ceil(number*prec)/prec)
          num_floor.append(np.floor(number*prec)/prec)
        except:
          num_ceil.append(np.ceil(number[0]*prec)/prec)
          num_floor.append(np.floor(number[0]*prec)/prec)
    if_str = "if (  "
    for i in range(v_len):
      if_str += f"({var[i]}>={num_floor[i]:+.8f}) && ({var[i]}<={num_ceil[i]:+.8f})"
      if i != v_len-1:
        if_str += "  &&  "
    if_str += "  )"
    return if_str

  data['2016']['biased'].data[99][:4]

  prepare_if(['cosK','cosL','hphi','time'],data['2016']['biased'].data.get()[99][:4])


  fk = [+0.00042441,+0.07686372,+0.08731396,-0.00520657,-0.01107498,-0.00260614,+0.00792509,+0.04785772,+0.01126178,-0.00366797]
  print(prepare_if(['f1','f2','f3','f4'],fk[:4]))
  print(prepare_if(['f7','f8','f9','f10'],fk[-4:]))

  """





  """

  bsjpsikk.diff_cross_rate_full(
  data['2016']['biased'].data,
  data['2016']['biased'].lkhd,
  w = data['2016']['biased'].ang_weights,
  coeffs = data['2016']['biased'].coeffs,
  **pars.valuesdict()
  )

  shit = Sample.from_root(f'{path}/{year}/{mode}/test.root')
  dec = np.array(shit.df['Jpsi_Hlt1DiMuonHighMassDecision_TOS']==0)
  y = np.zeros(dec.sum())
  j = 0;  k = 0
  for i in range(0,len(head)):
    if dec[i]:
      y[j] = head[i]; j +=1

  z = data['2016']['biased'].lkhd.get()
  w = z-y
  plt.plot(w,'.')

  """





  #
  # shit = Sample.from_root(f'{path}/{year}/{mode}/test.root')
  # dec = np.array(shit.df['Jpsi_Hlt1DiMuonHighMassDecision_TOS']==1)
  # y = np.zeros(100)
  # j = 0;  k = 0
  # for i in range(0,100):
  #   if dec[i]:
  #     y[i] = data['2016']['unbiased'].lkhd.get()[j]; j +=1
  #   else:
  #     y[i] = data['2016']['biased'].lkhd.get()[k]; k +=1
  #   #print(i,dec[i],j,k)
  #
  # head[:10]
  # max(head[:100]-y[:100])
  # for i in range(0,100):
  #     if (head[:100]-y[:100])[i] > 0.076:
  #         print(i)
  #
  #
  # head[82]
  # y[82]


from iminuit import Minuit as minuit
from ipanema import optimizers
from timeit import default_timer as timer

def FIT_FOR_ANGULAR_ACCEPTANCE(data):

  from ipanema import ristra, Parameters, Parameter

  SWAVE = 1; DGZERO = 0; pars = Parameters()



  # List of parameters
  list_of_parameters = [#
  Parameter(name='fSlon1',          value=SWAVE*0.48,         min=SWAVE*0.40,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon2',          value=SWAVE*0.0590,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon3',          value=SWAVE*0.0101,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon4',          value=SWAVE*0.0103,         min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon5',          value=SWAVE*0.059,          min=SWAVE*0.00,   max=0.60,   free=SWAVE),
  Parameter(name='fSlon6',          value=SWAVE*0.1930,         min=SWAVE*0.10,   max=0.60,   free=SWAVE),
  # Parameter(name='fSlon1',          value=SWAVE*0.473156516574591,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon2',          value=SWAVE*0.037061589305578,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon3',          value=SWAVE*0.00472688623744,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon4',          value=SWAVE*0.008872200912457,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon5',          value=SWAVE*0.067568158877522,          min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  # Parameter(name='fSlon6',          value=SWAVE*0.142515903747848,         min=SWAVE*0.00,   max=0.80,   free=SWAVE),
  #
  Parameter(name="fPlon",           value=0.524,                min=0.5,    max=0.6,    free=True),
  Parameter(name="fPper",           value=0.25,                 min=0.1,    max=0.3,    free=True),
  #
  Parameter(name="pSlon",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  Parameter(name="pPlon",           value= 0.07,                min=-0.5,   max=0.5,    free=True),
  Parameter(name="pPpar",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  Parameter(name="pPper",           value= 0.00,                min=-0.5,   max=0.5,    free=False),
  #
  Parameter(name='dSlon1',          value=+2.327*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon2',          value=+1.619*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon3',          value=+1.055*SWAVE,          min=-0.0,   max=+3.0,   free=SWAVE),
  Parameter(name='dSlon4',          value=-0.225*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  Parameter(name='dSlon5',          value=-0.48*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  Parameter(name='dSlon6',          value=-1.141*SWAVE,          min=-3.0,   max=+0.0,   free=SWAVE),
  #
  Parameter(name="dPlon",           value=0.00,                 min=-3.0,   max=3.0,    free=False),
  Parameter(name="dPpar",           value=3.26,                 min= 3.0,   max=3.5,    free=True),
  Parameter(name="dPper",           value=3.026,                 min= 2.0,   max=3.2,    free=True),
  #
  Parameter(name="lSlon",           value=1.,                   min=0.7,    max=1.3,    free=False),
  Parameter(name="lPlon",           value=1.,                   min=0.7,    max=1.3,    free=True),
  Parameter(name="lPpar",           value=1.,                   min=0.7,    max=1.3,    free=False),
  Parameter(name="lPper",           value=1.,                   min=0.7,    max=1.3,    free=False),
  #
  Parameter(name="Gd",              value= 0.65789,             min= 0.0,   max= 1.0,   free=False),
  Parameter(name="DGs",             value= (1-DGZERO)*0.0917,   min= 0.0,   max= 0.2,   free=(1-DGZERO)),
  Parameter(name="DGsd",            value= 0.03,                min=-0.1,   max= 0.1,   free=True),
  Parameter(name="DM",              value=17.768,           min=17.0,   max=18.0,   free=True),
  # CSP parameters
  Parameter(name='CSP1',            value=0.8463*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP2',            value=0.8756*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP3',            value=0.8478*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP4',            value=0.8833*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP5',            value=0.9415*SWAVE,         min=0.0,    max=1.0,    free=False),
  Parameter(name='CSP6',            value=0.9756*SWAVE,         min=0.0,    max=1.0,    free=False),
  # Time resolution parameters
  Parameter(name='mu',              value=0.0,                  min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_offset',    value=0.01297,              min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_slope',     value=0.8446,               min=0.0,    max=1.0,    free=False),
  Parameter(name='sigma_curvature', value=0,                    min=0.0,    max=1.0,    free=False),
  # Flavor tagging parameters
  Parameter(name='eta_os',           value=0.3602,               min=0.0,    max=1.0,    free=False),
  Parameter(name='eta_ss',           value=0.4167,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p0_os',            value=0.389,                min=0.0,    max=1.0,    free=False),
  Parameter(name='p0_ss',            value=0.4325,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p1_os',            value=0.8486,               min=0.0,    max=1.0,    free=False),
  Parameter(name='p1_ss',            value=0.9241,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp0_os',           value=0.009 ,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp0_ss',           value=0,                    min=0.0,    max=1.0,    free=False),
  Parameter(name='dp1_os',           value=0.0143,               min=0.0,    max=1.0,    free=False),
  Parameter(name='dp1_ss',           value=0,                    min=0.0,    max=1.0,    free=False)
  ]

  # update input parameters file
  pars.add(*list_of_parameters); #pars.dump(taf_path+'/input/'+FITTING_SAMPLE)


  pars['fPlon'].value = 0.512314599353
  pars['fPper'].value = 0.247866538577
  pars['dPpar'].value = 3.08736129166
  pars['dPper'].value = 2.64132238167
  pars['pPlon'].value = -0.107958702675
  pars['DGs'].value = 0.0798542641411
  pars['lPlon'].value = 1.01646881156
  pars['DGsd'].value = -0.004509478974
  pars['DM'].value = 17.7105963073
  pars['fSlon1'].value = 0.473560718454
  pars['fSlon2'].value = 0.0368893354524
  pars['fSlon3'].value = 0.00458919864125
  pars['fSlon4'].value = 0.009023172708
  pars['fSlon5'].value = 0.0678419662382
  pars['fSlon6'].value = 0.142729498712
  pars['dSlon1'].value = 2.32715533865
  pars['dSlon2'].value = 1.61459043778
  pars['dSlon3'].value = 1.06015274293
  pars['dSlon4'].value = -0.234248923015
  pars['dSlon5'].value = -0.488969205683
  pars['dSlon6'].value = -1.14817710086
  pars['fPlon'].init = 0.512314599353
  pars['fPper'].init = 0.247866538577
  pars['dPpar'].init = 3.08736129166
  pars['dPper'].init = 2.64132238167
  pars['pPlon'].init = -0.107958702675
  pars['DGs'].init = 0.0798542641411
  pars['lPlon'].init = 1.01646881156
  pars['DGsd'].init = -0.004509478974
  pars['DM'].init = 17.7105963073
  pars['fSlon1'].init = 0.473560718454
  pars['fSlon2'].init = 0.0368893354524
  pars['fSlon3'].init = 0.00458919864125
  pars['fSlon4'].init = 0.009023172708
  pars['fSlon5'].init = 0.0678419662382
  pars['fSlon6'].init = 0.142729498712
  pars['dSlon1'].init = 2.32715533865
  pars['dSlon2'].init = 1.61459043778
  pars['dSlon3'].init = 1.06015274293
  pars['dSlon4'].init = -0.234248923015
  pars['dSlon5'].init = -0.488969205683
  pars['dSlon6'].init = -1.14817710086

  import bsjpsikk
  bsjpsikk.config['debug'] = 0
  bsjpsikk.config['debug_evt'] = 1
  bsjpsikk.config['use_time_acc'] = 1
  bsjpsikk.config['use_time_offset'] = 0
  bsjpsikk.config['use_time_res'] = 1
  bsjpsikk.config['use_perftag'] = 0
  bsjpsikk.config['use_truetag'] = 0
  bsjpsikk.get_kernels()

  #%% Define fcn
  def fcn_data(parameters, data, weight = False):
    pars_dict = parameters.valuesdict()
    likelihood = []; weights = []
    for t, dt in data.items():
    # for y, dy in data.items():
    #  for t, dt in dy.items():
        bsjpsikk.diff_cross_rate_full(dt.data, dt.lkhd,
                                    w = dt.ang_weights,
                                    coeffs = dt.coeffs,
                                    **pars_dict)
        if weight:
          likelihood.append( (-2*ristra.log(dt.lkhd)*dt.weight).get() );
          weights.append( (2*dt.weight).get());
        else:
          likelihood.append( (ristra.log(dt.lkhd)).get() );
          weights.append(np.ones_like(likelihood[-1]));
    #likelihood = np.column_stack(likelihood).ravel();
    #weights = np.column_stack(weights).squeeze();
    likelihood = np.hstack(likelihood).squeeze();
    weights = np.hstack(weights).ravel();
    return likelihood + weights


  #%% Test fcn
  # bsjpsikk.config['debug'] = 5
  # bsjpsikk.config['debug_evt'] = 1
  # bsjpsikk.get_kernels()
  # print( fcn_data(pars, data=data, weight=True).sum() )
  # print( data['2016']['biased'].weight, data['2016']['unbiased'].weight, )
  # bsjpsikk.config['debug'] = 0
  # bsjpsikk.config['debug_evt'] = 0
  # bsjpsikk.get_kernels()


  mass = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  data['biased'].df['pdf'] = data['biased'].lkhd.get()
  data['unbiased'].df['pdf'] = data['unbiased'].lkhd.get()

  a = []
  for i in range(0,12):
    a.append(np.load(f'data{i}.npy'))

  b = []
  for i in range(0,12):
    if i<6:
      b.append(np.array(data['biased'].df.query(f'X_M>={mass[i]} & X_M<{mass[i+1]}')['pdf']))
    else:
      b.append(np.array(data['unbiased'].df.query(f'X_M>={mass[i-6]} & X_M<{mass[i-6+1]}')['pdf']))

  c = []
  for i in range(0,12):
    c.append( np.amax(a[i]-b[i]))

  print(c)

  #%% Prepate fitter
  shit = optimizers.Optimizer(fcn_data, params=pars, fcn_kwgs={"data": data} )
  shit.prepare_fit()
  #shit.optimize(method='hesse')
  list_of_pars = np.copy(shit.result.param_vary).tolist()
  dict_of_conf = {k:v for k,v in shit._configure_minuit_(pars).items()}

  #%% Minuit wrapper
  def parseminuit(*fvars):
    for name, val in zip(shit.result.param_vary, fvars):
      pars[name].value = val
    return fcn_data(pars, data=data, weight=True).sum()

  # give it a call!
  print('Fit is starting...')
  crap = minuit(parseminuit, forced_parameters=list_of_pars, **dict_of_conf, print_level=-1, pedantic=False)
  start = timer()
  crap.migrad()
  crap.hesse()
  print('Fit is finished! Cross your figers and pray Simon')
  print(f"Elapsed time: {timer() - start}")

  # Update pars
  for name, val in zip(shit.result.param_vary, crap.values.values()):
    pars[name].value = val
  for name, val in zip(shit.result.param_vary, crap.errors.values()):
    pars[name].stdev = val

  freees = [par.name for par in pars.values() if par.free]

  import uncertainties as unc
  simon_pars = [
  unc.ufloat(0.51230250  , 0.00309303),
  unc.ufloat(0.24786046  , 0.00426946),
  unc.ufloat(3.0865896   , 0.0726747),
  unc.ufloat(2.641715    , 0.126472),
  unc.ufloat(-0.1077952  , 0.0409865),
  unc.ufloat(1.0163400   , 0.0159215),
  unc.ufloat(-0.00450827 , 0.00248065),
  unc.ufloat(0.07982567  , 0.00811314),
  unc.ufloat(17.7110617  , 0.0567884),
  unc.ufloat(2.329586    , 0.177657),
  unc.ufloat(1.618309    , 0.332226),
  unc.ufloat(1.041925    , 0.393521),
  unc.ufloat(-0.232999   , 0.156938),
  unc.ufloat(-0.4882829  , 0.0995754),
  unc.ufloat(-1.144196   , 0.165305),
  unc.ufloat(0.4736370   , 0.0461791),
  unc.ufloat(0.03689762  , 0.00810673),
  unc.ufloat(0.00467571  , 0.00245224),
  unc.ufloat(0.00906472  , 0.00565416),
  unc.ufloat(0.0679397   , 0.0138514),
  unc.ufloat(0.1428969   , 0.0191666),
  ]

  freees = [ 'fPlon', 'fPper', 'dPpar', 'dPper', 'pPlon', 'lPlon', 'DGsd', 'DGs', 'DM', 'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6', 'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6' ]
  pars_ = Parameters.build(pars,freees);

  #%% bleh
  print(f'             ____Marcosito____   ______Simon______   ____M-S___')
  for i, par_name in enumerate(pars_.keys()):
    print(f'{par_name:>9}  {pars_[par_name].uvalue:>9.2uP} {simon_pars[i]:>9.2uP}    {pars_[par_name].uvalue.n-simon_pars[i].n:+.2E} ')

  return pars


if __name__ == '__main__':
  print(  FIT_FOR_ANGULAR_ACCEPTANCE(pars,data)  )





  exit()


























# cp output/tuples/MC_Bd2JpsiKstar/MC_Bd2JpsiKstar_2015_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/MC_Bd2JpsiKstar/2015/MC_Bd2JpsiKstar_2015_selected_bdt_sw_v0r4.root &
# cp output/tuples/MC_Bd2JpsiKstar/MC_Bd2JpsiKstar_2016_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/MC_Bd2JpsiKstar/2016/MC_Bd2JpsiKstar_2016_selected_bdt_sw_v0r4.root &
# cp output/tuples/MC_Bd2JpsiKstar/MC_Bd2JpsiKstar_2017_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/MC_Bd2JpsiKstar/2017/MC_Bd2JpsiKstar_2017_selected_bdt_sw_v0r4.root &
# cp output/tuples/MC_Bd2JpsiKstar/MC_Bd2JpsiKstar_2018_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/MC_Bd2JpsiKstar/2018/MC_Bd2JpsiKstar_2018_selected_bdt_sw_v0r4.root &
# cp output/tuples/Bd2JpsiKstar/Bd2JpsiKstar_2015_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/Bd2JpsiKstar/2015/Bd2JpsiKstar_2015_selected_bdt_sw_v0r4.root &
# cp output/tuples/Bd2JpsiKstar/Bd2JpsiKstar_2016_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/Bd2JpsiKstar/2016/Bd2JpsiKstar_2016_selected_bdt_sw_v0r4.root &
# cp output/tuples/Bd2JpsiKstar/Bd2JpsiKstar_2017_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/Bd2JpsiKstar/2017/Bd2JpsiKstar_2017_selected_bdt_sw_v0r4.root &
# cp output/tuples/Bd2JpsiKstar/Bd2JpsiKstar_2018_selected_bdt_sw.root /eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v0r4/Bd2JpsiKstar/2018/Bd2JpsiKstar_2018_selected_bdt_sw_v0r4.root &
#











#
#
#
#
# def fcn_year_unbiased(parameters, data, weight = None, prob = None):
#   pars_dict = {**parameters.valuesdict(), **dta_params_unbiased.valuesdict()}
#   if not prob: # for ploting, mainly
#     data = cu_array.to_gpu(data)
#     prob = cu_array.to_gpu(np.zeros_like(data.get()))[0]
#     getCrossRate(data, prob, pars_dict)
#     return prob.get()
#   else:
#     getCrossRate(data, prob, pars_dict)
#     if weight is not None:
#       result = (pycuda.cumath.log(prob)*weight).get()
#     else:
#       result = (pycuda.cumath.log(prob)).get()
#     return -2*result
#
#
#
# cats['test_biased'].params
#
#
# kernel_config
#
# # Compile model and get kernels
# BsJpsiKK = Badjanak(kernel_path, **kernel_config);
# getCrossRate = BsJpsiKK.getCrossRate
#
# fcn_year_biased(cats['test_biased'].params, data=cats['test_biased'].data_h)
#
# from ipanema import histogram
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# shit = optimize(fcn_year_biased, method="minuit-hesse",
#                 params=cats['test_biased'].params,
#                 kws={ 'data': cats['test_biased'].data_d,
#                       'prob': cats['test_biased'].lkhd_d,
#                     'weight': cats['test_biased'].weight_d  },
#                 verbose=True);
#
#
# shit._minuit.migrad()
#
#
#
#
#
#
#
#
#
#
#
#
#
# #%% Fit all categories ---------------------------------------------------------
# fits = {}
# # fits['BdMC_biased'].chisqr
# # fits['BdMC_biased'].ndata
# # dir(fits['BdMC_biased'])
# # fits['BdMC_biased']._repr_html_()
#
#
# # Fit each sample
# for name, cat in zip(cats.keys(),cats.values()):
#   print('Fitting %s category...' % name)
#   if cat.params:
#     fits[name] = optimize(lkhd_single_spline, method="minuit-hesse",
#                           params=cat.params,
#                           kws={'data':cat.time_d,
#                                'prob': cat.lkhd_d,
#                                'weight': cat.weight_d});
#   print('\n')
#
# # Fit the ratio BsMC/BdMC
# if params['ratio']:
#   for trig in ['_unbiased', '_biased']:
#     fits['ratio'+trig] = optimize(lkhd_ratio_spline, method="minuit-hesse",
#                           params=params['ratio'],
#                           kws={'data':  [cats['BsMC'+trig].time_d,
#                                          cats['BdMC'+trig].time_d],
#                               'prob':   [cats['BsMC'+trig].lkhd_d,
#                                          cats['BdMC'+trig].lkhd_d],
#                               'weight': [cats['BsMC'+trig].weight_d,
#                                          cats['BdMC'+trig].weight_d]});
#
# # Full fit to get decay-time acceptance
# if params['full']:
#  for trig in ['_unbiased', '_biased']:
#    fits['full'+trig] = optimize(lkhd_full_spline, method="minuit-hesse",
#                           params=params['full'],
#                           kws={'data':  [cats['BsMC'+trig].time_d,
#                                          cats['BdMC'+trig].time_d,
#                                          cats['BdDT'+trig].time_d],
#                               'prob':   [cats['BsMC'+trig].lkhd_d,
#                                          cats['BdMC'+trig].lkhd_d,
#                                          cats['BdDT'+trig].lkhd_d],
#                               'weight': [cats['BsMC'+trig].weight_d,
#                                          cats['BdMC'+trig].weight_d,
#                                          cats['BdDT'+trig].weight_d]});
# print('Fitting complete.')
#
#
#
# #%% Plot all categories --------------------------------------------------------
# from ipanema import plotting
# for name, cat in zip(cats.keys(),cats.values()):
#   print('Plotting %s category...' % name)
#   filename = ppath+cat.name[:7]+'_'+name+'.pdf'
#   plot_single_spline(fits[name].params, cat.time_h, cat.weight_h, name = filename )
#   filename = ppath+cat.name[:7]+'_'+name+'_log.pdf'
#   plot_single_spline(fits[name].params, cat.time_h, cat.weight_h, name = filename, log= True )
# print('Plotting complete.')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def pull_hist(ref_counts, counts, counts_l, counts_h):
#   """
#   This function takes an array of ref_counts (reference histogram) and three
#   arrays of the objective histogram: counts, counts_l (counts' lower limit) and
#   counts_h (counts' higher limit). It returns the pull of counts wrt ref_counts.
#   """
#   residuals = counts - ref_counts;
#   pulls = np.where(residuals>0, residuals/counts_l, residuals/counts_h)
#   return pulls
#
#
# def hist(data, weights=None, bins=60, density = False, **kwargs):
#   """
#   This function is a wrap arround np.histogram so it behaves similarly to it.
#   Besides what np.histogram offers, this function computes the center-of-mass
#   bins ('cmbins') and the lower and upper limits for bins and counts. The result
#   is a ipo-object which has several self-explained attributes.
#   """
#
#   # Histogram data
#   counts, edges = np.histogram(data, bins = bins,
#                                weights = weights, density = False,
#                                **kwargs)
#   bincs = (edges[1:]+edges[:-1])*0.5;
#   norm = counts.sum()
#
#   # Compute the mass-center of each bin
#   cmbins = np.copy(bincs)
#   for k in range(0,len(edges)-1):
#     if counts[k] != 0:
#       cmbins[k] = np.median( data[(data>=edges[k]) & (data<=edges[k+1])] )
#
#   # Compute the error-bars
#   if weights is not None:
#     errl, errh = histogram.errors_poisson(counts)
#     errl = errl**2 + histogram.errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
#     errh = errh**2 + histogram.errors_sW2(data, weights = weights, bins = bins, **kwargs)**2
#     errl = np.sqrt(errl); errh = np.sqrt(errh)
#   else:
#     errl, errh = histogram.errors_poisson(counts)
#
#   # Normalize if asked so
#   if density:
#     counts /= norm; errl /= norm;  errh /= norm;
#
#   # Construct the ipo-object
#   result = histogram.ipo(**{**{'counts':counts,
#                      'edges':edges, 'bins':bincs, 'cmbins': cmbins,
#                      'weights': weights, 'norm': norm,
#                      'density': density, 'nob': bins,
#                      'errl': errl, 'errh': errh,
#                     },
#                   **kwargs})
#   return result
#
#
# def compare_hist(data, weights = [None, None], density = False, **kwargs):
#   """
#   This function compares to histograms in data = [ref, obj] with(/out) weights
#   It returns two hisrogram ipo-objects, obj one with pulls, and both of them
#   normalized to one.
#   """
#   ref = hist(data[0], density = False, **kwargs, weights=weights[0])
#   obj = hist(data[1], density = False, **kwargs, weights=weights[1])
#   ref_norm = 1; obj_norm = 1;
#   if norm:
#     ref_norm = 1/ref.counts.sum(); obj_norm = 1/obj.counts.sum();
#   ref.counts = ref.counts*ref_norm; ref.errl *= ref_norm; ref.errh *= ref_norm
#   obj.counts = obj.counts*obj_norm; obj.errl *= obj_norm; obj.errh *= obj_norm
#   obj.add('pulls', pull_hist(ref.counts, obj.counts, obj.errl, obj.errh))
#   return ref, obj
#
#
#
#
# import sys
# sys.path.append(os.environ['PHIS_SCQ']+'tools')
# import importlib
# importlib.import_module('phis-scq-style')
# from scipy.interpolate import interp1d
#
#
#
# def pull_pdf(x_pdf, y_pdf, x_hist, y_hist, y_l, y_h):
#   """
#   This function compares one histogram with a pdf. The pdf is given with two
#   arrays x_pdf and y_pdf, these are interpolated (and extrapolated if needed),
#   contructing a cubic spline. The histogram takes x_hist (bins), y_hist(counts),
#   y_l (counts's lower limit) and y_h (counts' upper limit). The result is a
#   pull array between the histogram and the pdf.
#   (the pdf is expected to be correctly normalized)
#   """
#   s = interp1d(x_pdf, y_pdf, kind='cubic', fill_value='extrapolate')
#   residuals = y_hist - s(x_hist);
#   pulls = np.where(residuals>0, residuals/y_l, residuals/y_h)
#   return pulls
#
#
# def plot_single_spline(parameters, data, weight, log=False, name='test.pdf'):
#   ref = histogram.hist(data, weights=weight, bins = 100)
#   fig, axplot, axpull = plotting.axes_plotpull();
#   x = np.linspace(0.3,15,200)
#   y = lkhd_single_spline(parameters, x )
#   y *= ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))
#   axplot.plot(x,y)
#   axpull.fill_between(ref.bins,
#                       pull_pdf(x,y,ref.bins,ref.counts,ref.errl,ref.errh),
#                       0, facecolor="C0")
#   axplot.errorbar(ref.bins,ref.counts,yerr=[ref.errl,ref.errh], fmt='.', color='k')
#   if log:
#     axplot.set_yscale('log')
#   axpull.set_xlabel(r'$t$ [ps]')
#   axplot.set_ylabel(r'Weighted candidates')
#   fig.savefig(name)
#   plt.close()
#
#
# plot_single_spline(fit['BdMC_biased'].params, time_h['BdMC_biased'], weight_h['BdMC_biased'])
# plot_single_spline(fit['BdMC_unbiased'].params, time_h['BdMC_unbiased'], weight_h['BdMC_unbiased'])
#
# for var in vars:
#   index = vars.index(var)
#   orig, ref = compare_hist([original_data[var].values,target_data[var].values], range=ranges[index], bins = 60, norm = True, cm = False, weights=[1/original_data['gb_weights'].values,target_data['sw'].values])
#   weig, ref = compare_hist([original_data[var].values,target_data[var].values], range=ranges[index], bins = 60, norm = True, cm = False, weights=[original_data['kinWeight'].values/original_data['gb_weights'].values,target_data['sw'].values])
#   axplot, axpull = plot_pull()
#   axplot.fill_between(ref['x'],ref['y'],0,facecolor='k',alpha=0.3,step='mid')
#   axplot.fill_between(orig['x'],orig['y'],0,facecolor='C3',alpha=0.3,step='mid')
#   axplot.errorbar(weig['x'],weig['y'], yerr = [weig['y_l'],weig['y_u']],  xerr = [weig['x_l'],weig['x_r']] , fmt='.')
#   axpull.set_xticks(axpull.get_xticks()[2:-1])
#   axplot.set_yticks(axplot.get_yticks()[1:-1])
#   axpull.fill_between(orig['x'],orig['pulls'],0, facecolor="C3")
#   axpull.fill_between(weig['x'],weig['pulls'],0, facecolor="C0")
#   axpull.set_xlabel(labels[index])
#   axplot.set_ylabel(r'Weighted candidates')
#   axplot.legend([r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ data',
#                  r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ MC',
#                  r'$B_{s}^{0} \rightarrow J\!/\!\psi \phi$ MC + kinWeights'])
#   plt.savefig(var+'.pdf')
#   plt.close()
# histogram.hist(lkhd)
#






















""" FIT RESULTS FOR BIASED TRIGGER
+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BsMC       |   1.602 ± 0.034    |
| b2_BsMC       |    2.127 ± 0.03    |
| b3_BsMC       |   2.277 ± 0.037    |
| b4_BsMC       |   2.362 ± 0.035    |
| b5_BsMC       |   2.453 ± 0.037    |
| b6_BsMC       |   2.507 ± 0.043    |
| b7_BsMC       |   2.503 ± 0.042    |
| b8_BsMC       |   2.485 ± 0.038    |
+---------------+--------------------+

+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BdMC       |    1.49 ± 0.036    |
| b2_BdMC       |   1.884 ± 0.029    |
| b3_BdMC       |   1.974 ± 0.036    |
| b4_BdMC       |   2.025 ± 0.033    |
| b5_BdMC       |   2.066 ± 0.035    |
| b6_BdMC       |   2.155 ± 0.041    |
| b7_BdMC       |    2.05 ± 0.039    |
| b8_BdMC       |   2.021 ± 0.035    |
+---------------+--------------------+

+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BdData     |   1.393 ± 0.086    |
| b2_BdData     |   1.856 ± 0.071    |
| b3_BdData     |   1.839 ± 0.084    |
| b4_BdData     |   1.988 ± 0.081    |
| b5_BdData     |   1.946 ± 0.082    |
| b6_BdData     |    0.21 ± 0.01     |
| b7_BdData     |    1.846 ± 0.09    |
| b8_BdData     |   1.892 ± 0.082    |
+---------------+--------------------+
"""






"""
trig = '_biased'
out = optimize(lkhd_single_spline, method="emcee",
               params=params['BdMC'],
               burn=300, steps=2000, thin=20, nan_policy='omit',
               kws={'data':  time_d['BdMC'+trig],
                    'prob':  lkhd_d['BdMC'+trig],
                    'weight':weight_d['BdMC'+trig]}
              );

# single spline tester
trig = '_biased'
out = optimize(lkhd_single_spline, method="minuit-hesse",
               params=params['BdMC'],
               kws={'data':  time_d['BdMC'+trig],
                    'prob':  lkhd_d['BdMC'+trig],
                    'weight':weight_d['BdMC'+trig]}
              );

# double spline tester
trig = '_biased'
out = optimize(lkhd_ratio_spline, method="minuit-hesse",
              params=params['ratio'],
              kws={'data':   [cats['BsMC'+trig].time_d,
                              cats['BdMC'+trig].time_d],
                   'prob':   [cats['BsMC'+trig].lkhd_d,
                              cats['BdMC'+trig].lkhd_d],
                   'weight': [cats['BsMC'+trig].weight_d,
                              cats['BdMC'+trig].weight_d]}
              );



# triple spline terster
trig = '_biased'
out = optimize(lkhd_full_spline, method="minuit-hesse",
               params=params['full'],
               kws={'data':   [cats['BsMC'+trig].time_d,
                               cats['BdMC'+trig].time_d,
                               cats['BdDT'+trig].time_d],
                    'prob':   [cats['BsMC'+trig].lkhd_d,
                               cats['BdMC'+trig].lkhd_d,
                               cats['BdDT'+trig].lkhd_d],
                    'weight': [cats['BsMC'+trig].weight_d,
                               cats['BdMC'+trig].weight_d,
                               cats['BdDT'+trig].weight_d]}
              );

out.params.print()


cats['BdMC'+trig]

# Optimization finished in 3.0938 minutes.

"""

"""
+---------------+--------------------+
| Fit Parameter | Value ± ParabError |
+---------------+--------------------+
| b1_BsMC       |   1,603 ± 0,039    |
| b2_BsMC       |   2,127 ± 0,035    |
| b3_BsMC       |   2,274 ± 0,042    |
| b4_BsMC       |   2,355 ± 0,040    |
| b5_BsMC       |   2,440 ± 0,042    |
| b6_BsMC       |   2,477 ± 0,048    |
| b7_BsMC       |   2,459 ± 0,046    |
| b8_BsMC       |   2,430 ± 0,043    |
| b1_BdMC       |   0,924 ± 0,033    |
| b2_BdMC       |   0,891 ± 0,019    |
| b3_BdMC       |   0,872 ± 0,023    |
| b4_BdMC       |   0,865 ± 0,020    |
| b5_BdMC       |   0,851 ± 0,020    |
| b6_BdMC       |   0,874 ± 0,024    |
| b7_BdMC       |   0,837 ± 0,022    |
| b8_BdMC       |   0,834 ± 0,020    |
| b1_BdData     |    1,51 ± 0,11     |
| b2_BdData     |    2,11 ± 0,10     |
| b3_BdData     |    2,13 ± 0,12     |
| b4_BdData     |    2,33 ± 0,12     |
| b5_BdData     |    2,32 ± 0,12     |
| b6_BdData     |    2,45 ± 0,14     |
| b7_BdData     |    2,23 ± 0,13     |
| b8_BdData     |    2,30 ± 0,12     |
+---------------+--------------------+
"""




# +---------------+--------------------+
# | Fit Parameter | Value ± ParabError |
# +---------------+--------------------+
# | b1_BsMC       |   1.555 ± 0.062    |
# | b2_BsMC       |   2.087 ± 0.061    |
# | b3_BsMC       |   2.244 ± 0.066    |
# | b4_BsMC       |   2.329 ± 0.067    |
# | b5_BsMC       |   2.385 ± 0.066    |
# | b6_BsMC       |   2.439 ± 0.086    |
# | b7_BsMC       |   2.528 ± 0.073    |
# | b8_BsMC       |   2.513 ± 0.067    |
# | r1_BdMC       |   0.951 ± 0.043    |
# | r2_BdMC       |   0.904 ± 0.029    |
# | r3_BdMC       |    0.882 ± 0.03    |
# | r4_BdMC       |   0.872 ± 0.028    |
# | r5_BdMC       |   0.869 ± 0.027    |
# | r6_BdMC       |   0.885 ± 0.034    |
# | r7_BdMC       |   0.813 ± 0.028    |
# | r8_BdMC       |   0.806 ± 0.025    |
# +---------------+--------------------+
