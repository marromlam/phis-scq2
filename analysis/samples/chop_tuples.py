__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = []


# Modules {{{

import config
# import shutil
import uproot3 as uproot
import argparse
import os
import numpy as np

# }}}


# Some config {{{

vsub_dict = {
    # used to place random cuts for MC
    "evtOdd": "(eventNumber % 2) != 0",
    "evtEven": "(eventNumber % 2) == 0",
    # magnet cuts
    "magUp": "Polarity == 1",
    "magDown": "Polarity == -1",
    # "bkgcat60": "B_BKGCAT != 60",
    "bkgcat60": "time > 0",
    "bkgcat050": "B_BKGCAT != 60 & abs(B_MC_MOTHER_ID) != 541",
    # lower and upper time cuts
    "LT": f"time < {config.general['lower_time_upper_limit']}",
    "UT": f"time > {config.general['upper_time_lower_limit']}",
    "T1": "time < 1.00",
    "T2": "time > 1.00 & time < 2.05",
    "T3": "time > 2.05",
    # runNumber to check alignment in 2018
    "g210300": "runNumber > 210300",
    "l210300": "runNumber < 210300",
    # bins of pTB
    "pTB1": "B_PT >= 0 & B_PT < 3.8e3",
    "pTB2": "B_PT >= 3.8e3 & B_PT < 6e3",
    "pTB3": "B_PT >= 6e3 & B_PT <= 9e3",
    "pTB4": "B_PT >= 9e3",
    # bins of pTJpsi
    # 6.35292846, 1073.35454412, 1749.62548721, 2512.92106449, 4412.51093472
    "pTJpsi1": "Jpsi_PT >= 0 & Jpsi_PT < 1073",
    "pTJpsi2": "Jpsi_PT >= 1073 & Jpsi_PT < 1749",
    "pTJpsi3": "Jpsi_PT >= 1749 & Jpsi_PT <= 2512",
    "pTJpsi4": "Jpsi_PT >= 2512",
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
    # "LSB": "B_ConstJpsi_M_1 < 5397",
    # "RSB": "B_ConstJpsi_M_1 > 5337",
    "LSB": "B_ConstJpsi_M_1 < 5550",
    "RSB": "B_ConstJpsi_M_1 > 5200",
    # "LSB": "B_ConstJpsi_M_1 < 5417",
    # "RSB": "B_ConstJpsi_M_1 > 5287",
    "LSBsmall": "B_ConstJpsi_M_1 < 5397 & B_ConstJpsi_M_1 > 5255",
    "RSBsmall": "B_ConstJpsi_M_1 > 5337 & B_ConstJpsi_M_1 < 5460",
    # "LbkgSB": "B_ConstJpsi_M_1 > 5417",
    # "RbkgSB": "B_ConstJpsi_M_1 < 5287",
    # tagging
    "onlyOST": "abs(OS_Combination_DEC)==1 & abs(B_SSKaonLatest_TAGDEC)==0",
    "onlySST": "abs(OS_Combination_DEC)==0 & abs(B_SSKaonLatest_TAGDEC)==1",
    "onlyOSS": "abs(OS_Combination_DEC)==1 & abs(B_SSKaonLatest_TAGDEC)==1",
    "tag": "(abs(OS_Combination_DEC)==1 & abs(B_SSKaonLatest_TAGDEC)==0) | (abs(OS_Combination_DEC)==0 & abs(B_SSKaonLatest_TAGDEC)==1) | (abs(OS_Combination_DEC)==1 & abs(B_SSKaonLatest_TAGDEC)==1) ",
    # pXB and pYB cuts
    "pXB1": "B_PX >= 0 & B_PX < 2.7e3",
    "pXB2": "B_PX >= 2.7e3 & B_PX < 4.2e3",
    "pXB3": "B_PX >= 4.2e3 & B_PX <= 6.3e3",
    "pXB4": "B_PX >= 6.3e3",
    "pYB1": "B_PY >= 0 & B_PY < 2.7e3",
    "pYB2": "B_PY >= 2.7e3 & B_PY < 4.2e3",
    "pYB3": "B_PY >= 4.2e3 & B_PY <= 6.3e3",
    "pYB4": "B_PY >= 6.3e3",
    # pid cuts
    # "pid1": 'hplus_ProbNNk_corr<0.876',
    # "pid2": 'hplus_ProbNNk_corr>0.876 & hplus_ProbNNk_corr<0.965',
    # "pid3": 'hplus_ProbNNk_corr>0.965 & hplus_ProbNNk_corr<0.996',
    # "pid4": 'hplus_ProbNNk_corr>0.996',
    # "PID1": "( (hplus_PIDK<hminus_PIDK) & ({scale}*hplus_PIDK<= 12.40) & ({scale}*hplus_PIDK> 0.00) ) | ( (hminus_PIDK<=hplus_PIDK) & ({scale}*hminus_PIDK<= 12.40) & ({scale}*hminus_PIDK>0.0) )",
    # "PID2": "( (hplus_PIDK<hminus_PIDK) & ({scale}*hplus_PIDK<= 20.25) & ({scale}*hplus_PIDK>12.40) ) | ( (hminus_PIDK<=hplus_PIDK) & ({scale}*hminus_PIDK<= 20.25) & ({scale}*hminus_PIDK>12.40) )",
    # "PID3": "( (hplus_PIDK<hminus_PIDK) & ({scale}*hplus_PIDK<= 29.78) & ({scale}*hplus_PIDK>20.25) ) | ( (hminus_PIDK<=hplus_PIDK) & ({scale}*hminus_PIDK<= 29.78) & ({scale}*hminus_PIDK>20.25) )",
    # "PID4": "( (hplus_PIDK<hminus_PIDK) & ({scale}*hplus_PIDK<=129.14) & ({scale}*hplus_PIDK>29.78) ) | ( (hminus_PIDK<=hplus_PIDK) & ({scale}*hminus_PIDK<=129.14) & ({scale}*hminus_PIDK>29.78) )",

    # "PID1": "(hminus_isMuon==0) & (hplus_isMuon==0) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=12.40) & (hplus_PIDK>0.0) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=12.40) & (hminus_PIDK>0.0) ) )",
    # "PID2": "(hminus_isMuon==0) & (hplus_isMuon==0) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=20.25) & (hplus_PIDK>12.40) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=20.25) & (hminus_PIDK>12.40) ) )",
    # "PID3": "(hminus_isMuon==0) & (hplus_isMuon==0) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=29.78) & (hplus_PIDK>20.25) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=29.78) & (hminus_PIDK>20.25) ) ) ",
    # "PID4": "(hminus_isMuon==0) & (hplus_isMuon==0) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=129.14) & (hplus_PIDK>29.78) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=129.14) & (hminus_PIDK>29.78) ) )",
    #valid for z2r1
    # "PID1": "( (B_PT < 5.e4) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=12.40) & (hplus_PIDK>0.0) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=12.40) & (hminus_PIDK>0.0) ) ) )",
    # "PID2": "( (B_PT < 5.e4) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=20.25) & (hplus_PIDK>12.40) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=20.25) & (hminus_PIDK>12.40) ) ) )",
    # "PID3": "( (B_PT < 5.e4) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=29.78) & (hplus_PIDK>20.25) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=29.78) & (hminus_PIDK>20.25) ) ) )",
    # "PID4": "( (B_PT < 5.e4) & ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=300) & (hplus_PIDK>29.78) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=300) & (hminus_PIDK>29.78) ) ) )",
    "PID1": "  ( ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=12.40) & (hplus_PIDK>0.0) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=12.40) & (hminus_PIDK>0.0) ) ) )",
    "PID2": "  ( ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=20.25) & (hplus_PIDK>12.40) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=20.25) & (hminus_PIDK>12.40) ) ) )",
    "PID3": "  ( ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=29.78) & (hplus_PIDK>20.25) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=29.78) & (hminus_PIDK>20.25) ) ) )",
    "PID4": "  ( ( ( (hplus_PIDK<hminus_PIDK) & (hplus_PIDK<=300) & (hplus_PIDK>29.78) ) | ( (hminus_PIDK<=hplus_PIDK) & (hminus_PIDK<=300) & (hminus_PIDK>29.78) ) ) )",

    "Prob1": "( ( ( (hplus_ProbNNk<hminus_ProbNNk) & (hplus_ProbNNk<=0.73) & (hplus_ProbNNk>0.0) ) | ( (hminus_ProbNNk<=hplus_ProbNNk) & (hminus_ProbNNk<=0.73) & (hminus_ProbNNk>0.0) ) ) )",
    "Prob2": "( ( ( (hplus_ProbNNk<hminus_ProbNNk) & (hplus_ProbNNk<=0.914) & (hplus_ProbNNk>0.73) ) | ( (hminus_ProbNNk<=hplus_ProbNNk) & (hminus_ProbNNk<=0.914) & (hminus_ProbNNk>0.73) ) ) )",
    "Prob3": "( ( ( (hplus_ProbNNk<hminus_ProbNNk) & (hplus_ProbNNk<=0.984) & (hplus_ProbNNk>0.914) ) | ( (hminus_ProbNNk<=hplus_ProbNNk) & (hminus_ProbNNk<=0.984) & (hminus_ProbNNk>0.914) ) ) )",
    "Prob4": "( ( ( (hplus_ProbNNk<hminus_ProbNNk) & (hplus_ProbNNk<=300) & (hplus_ProbNNk>0.984) ) | ( (hminus_ProbNNk<=hplus_ProbNNk) & (hminus_ProbNNk<=300) & (hminus_ProbNNk>0.984) ) ) )",
    # "pid1": 'hplus_ProbNNk_corr<0.876',
    # "pid2": 'hplus_ProbNNk_corr>0.876 & hplus_ProbNNk_corr<0.965',
    # "pid3": 'hplus_ProbNNk_corr>0.965 & hplus_ProbNNk_corr<0.996',
    # "pid4": 'hplus_ProbNNk_corr>0.996',

    # prymary vertex cut
    "PV1": 'nPVs==1',
    "PV2": 'nPVs==2',
    "PV3": 'nPVs>=3',
}

# }}}


# CMDLINE interfrace {{{

if __name__ == "__main__":
  # argument parser for snakemake
  p = argparse.ArgumentParser(description="Chop tuples")
  p.add_argument('--year', help='Year of the tuple.')
  p.add_argument('--mode', help='Decay mode of the tuple.')
  p.add_argument('--version', help='Version of the selection pipeline')
  p.add_argument('--weight', help='The tuple surname')
  p.add_argument('--tree', help='Input file tree name.')
  p.add_argument('--input', help='Input file tree name.')
  p.add_argument('--output', help='Input file tree name.')
  p.add_argument('--uproot-kwargs', help='Arguments to uproot.pandas.df')
  args = vars(p.parse_args())

  # Get the flags and that stuff
  # pipeline tuple version
  v = args['version'].split("@")[0].split("bdt")[0]
  V = args['version'].replace('bdt', '')  # full version for phis-scq
  y = args['year']
  m = args['mode']
  w = args['weight']
  tree = args['tree']
  is_mc = True if 'MC' in m else False
  is_gun = True if 'GUN' in m else False

  # choose sWeight varaible name {{{

  if "pTB" in V:
    sw = 'sw_pt'
  elif "etaB" in V:
    sw = 'sw_eta'
  elif "sigmat" in V:
    sw = 'sw_sigmat'
  else:
    sw = 'sw'

  # }}}

  # try DVTuple if the tree does not work {{{
  print(args['input'])
  try:
    result = uproot.open(args['input'])[tree]
    result = result.pandas.df(flatten=None)
  except:
    result = uproot.open(args['input'])['DVTuple'][tree]
    result = result.pandas.df(flatten=None)

  # }}}

  # ensure sw branch exists {{{
  list_of_branches = list(result.keys())
  print(list_of_branches)

  # add pT for Jpsi if it does not exist
  # if not 'Jpsi_PT' in list_of_branches:
  #   result.eval("Jpsi_PT = sqrt( (muplus_PX + muminus_PX)**2 + (muplus_PY + muminus_PY)**2)", inplace=True)

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
      print("sWeight is set to zero for B_BKGCAT==60")
      result['sw'] = np.where(result['B_BKGCAT'].values != 60, 1, 0)
    else:
      print("sWeight variable was not found. Set sw = 1")
      result['sw'] = np.ones_like(result[result.keys()[0]])

  # }}}

  # place cuts according to version substring {{{

  # list_of_cuts = []
  # vsub_cut = None
  # scale = 0.916 if 'MC' in m else 1.00
  # scale = 1.00
  # for k, v in vsub_dict.items():
  #   if k in V:
  #     try:
  #       if (k in ("g210300", "l210300")) and is_mc:
  #         print("MCs are not cut in runNumber")
  #       elif (k in ("g210300", "l210300")) and ("2018" != y):
  #         print("Only 2018 RD is cut in runNumber")
  #       elif (k in ("UcosK", "LcosK")) and 'Bd2JpsiKstar' not in m:
  #         print("Cut in cosK was only planned in Bd")
  #       elif (k in ("LSB", "RSB", "LSBsmall", "RSBsmall")) and 'Bs2JpsiPhi' != m:
  #         print("Cut in LSB and RSB was only planned in Bs RD")
  #       elif (k in ("PID1", "PID2", "PID3", "PID4")) and 'Bs2JpsiPhi' not in m:
  #         print("PID cut was only planned in Bs modes")
  #       elif (k in ("pid1", "pid2", "pid3", "pid4")) and 'Bs2JpsiPhi' not in m:
  #         print("PID cut was only planned in Bs modes")
  #       elif (k in ("tag") and 'Bs2JpsiPhi' not in m):
  #         print("tag cut was only planned in Bs modes")
  #       elif (k in ("bkgcat050") and 'MC_Bs2JpsiPhi' not in m):
  #         print("tag cut was only planned in Bs MC modes")
  #       else:
  #         list_of_cuts.append(v.format(scale=scale))
  #         noe = len(result.query(list_of_cuts[-1]))
  #         if noe == 0:
  #           print(f"ERROR: This cut leaves df empty. {v}")
  #           print("       Query halted.")
  #     except:
  #       print(f"There is no such variable for the cut {v}")
  # if list_of_cuts:
  #   vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"
  #
  # # place the cut
  # print("Cut to be applied")
  # print(vsub_cut)
  # if vsub_cut and not is_gun:
  #   result = result.query(vsub_cut)
  # print("Showing the final dataframe:")
  # print(result)

  # }}}

  # write {{{

    list_of_cuts = []
    vsub_cut = None
    for k, v in vsub_dict.items():
        if k in V:
            try:
                # noe = len(result.query(v))
                if (k in ("g210300", "l210300")) and is_mc:
                    print("MCs are not cut in runNumber")
                elif (k in ("g210300", "l210300")) and ("2018" != y):
                    print("Only 2018 RD is cut in runNumber")
                elif (k in ("UcosK", "LcosK")) and 'Bd2JpsiKstar' not in m:
                    print("Cut in cosK was only planned in Bd")
                elif ("PID" in k):
                    if "Bs2JpsiPhi"==m:
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    elif "MC_Bs2JpsiPhi" in m:
                      v = v.replace("PIDK", "PIDK_corr")
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    # elif "Bu2" in m:
                    #   if "PID1" in k:
                    #     v = " ( (hplus_PIDK<=14) & (hplus_PIDK>0.0) )"
                    #   elif "PID2" in k:
                    #     v = " (  (hplus_PIDK<=27) & (hplus_PIDK>14) )"
                    #   elif "PID3" in k:
                    #     v = " ( (hplus_PIDK<=43) & (hplus_PIDK>27) )"
                    #   elif "PID4" in k:
                    #     v = "( (hplus_PIDK<=400) & (hplus_PIDK>43) )"
                    #   list_of_cuts.append(v)
                    #   noe = len(result.query(v))

                    elif ("Bd2JpsiKstar"==m):
                      if "PID1" in k:
                        v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=11.80) & (hplus_PIDK>0.0) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=11.89) & (abs(hminus_PIDK)>0.0) ) )"
                      elif "PID2" in k:
                        v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=20.6) & (hplus_PIDK>11.8) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=20.6) & (abs(hminus_PIDK)>11.8) ) )"
                      elif "PID3" in k:
                        v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=31.4) & (hplus_PIDK>20.6) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=31.4) & (abs(hminus_PIDK)>20.6) ) ) "
                      elif "PID4" in k:
                        v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=400) & (hplus_PIDK>31.4) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=400) & (abs(hminus_PIDK)>31.4) ) )"
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    elif ("MC_Bd2JpsiKstar"==m):
                      if "z" in V:
                        if "PID1" in k:
                            v = "( ( (hplus_PIDK_corr<abs(hminus_PIDK_corr)) & (hplus_PIDK_corr<=11.80) & (hplus_PIDK_corr>0.0) ) | ( (abs(hminus_PIDK_corr)<=hplus_PIDK_corr) & (abs(hminus_PIDK_corr)<=11.89) & (abs(hminus_PIDK_corr)>0.0) ) )"
                        elif "PID2" in k:
                            v = "( ( (hplus_PIDK_corr<abs(hminus_PIDK_corr)) & (hplus_PIDK_corr<=20.6) & (hplus_PIDK_corr>11.8) ) | ( (abs(hminus_PIDK_corr)<=hplus_PIDK_corr) & (abs(hminus_PIDK_corr)<=20.6) & (abs(hminus_PIDK_corr)>11.8) ) )"
                        elif "PID3" in k:
                            v = "( ( (hplus_PIDK_corr<abs(hminus_PIDK_corr)) & (hplus_PIDK_corr<=31.4) & (hplus_PIDK_corr>20.6) ) | ( (abs(hminus_PIDK_corr)<=hplus_PIDK_corr) & (abs(hminus_PIDK_corr)<=31.4) & (abs(hminus_PIDK_corr)>20.6) ) ) "
                        elif "PID4" in k:
                            v = "( ( (hplus_PIDK_corr<abs(hminus_PIDK_corr)) & (hplus_PIDK_corr<=400) & (hplus_PIDK_corr>31.4) ) | ( (abs(hminus_PIDK_corr)<=hplus_PIDK_corr) & (abs(hminus_PIDK_corr)<=400) & (abs(hminus_PIDK_corr)>31.4) ) )"
                      else:
                        if "PID1" in k:
                          v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=11.80) & (hplus_PIDK>0.0) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=11.89) & (abs(hminus_PIDK)>0.0) ) )"
                        elif "PID2" in k:
                          v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=20.6) & (hplus_PIDK>11.8) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=20.6) & (abs(hminus_PIDK)>11.8) ) )"
                        elif "PID3" in k:
                          v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=31.4) & (hplus_PIDK>20.6) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=31.4) & (abs(hminus_PIDK)>20.6) ) ) "
                        elif "PID4" in k:
                          v = "( ( (hplus_PIDK<abs(hminus_PIDK)) & (hplus_PIDK<=400) & (hplus_PIDK>31.4) ) | ( (abs(hminus_PIDK)<=hplus_PIDK) & (abs(hminus_PIDK)<=400) & (abs(hminus_PIDK)>31.4) ) )"
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                elif ("Prob" in k):
                    if "Bs2JpsiPhi"==m:
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    elif "MC_Bs2JpsiPhi" in m:
                      v = v.replace("ProbNNk", "ProbNNk_corr")
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    elif ("Bd2JpsiKstar"==m):
                      if "Prob1" in k:
                        v = "( ( ( (hplus_ProbNNk<(1-hminus_ProbNNk)) & (hplus_ProbNNk<=0.86) & (hplus_ProbNNk>0.0) ) | ( ((1-hminus_ProbNNk)<=hplus_ProbNNk) & ((1-hminus_ProbNNk)<=0.86) & ((1-hminus_ProbNNk)>0.0) ) ) )"
                      elif "Prob2" in k:
                        v = "( ( ( (hplus_ProbNNk<(1-hminus_ProbNNk)) & (hplus_ProbNNk<=0.96) & (hplus_ProbNNk>0.86) ) | ( ((1-hminus_ProbNNk)<=hplus_ProbNNk) & ((1-hminus_ProbNNk)<=0.96) & ((1-hminus_ProbNNk)>0.86) ) ) )"
                      elif "Prob3" in k:
                        v = "( ( ( (hplus_ProbNNk<(1-hminus_ProbNNk)) & (hplus_ProbNNk<=0.995) & (hplus_ProbNNk>0.96) ) | ( ((1-hminus_ProbNNk)<=hplus_ProbNNk) & ((1-hminus_ProbNNk)<=0.995) & ((1-hminus_ProbNNk)>0.96) ) ) )"
                      elif "Prob4" in k:
                        v = "( ( ( (hplus_ProbNNk<(1-hminus_ProbNNk)) & (hplus_ProbNNk<=100.) & (hplus_ProbNNk>0.995) ) | ( ((1-hminus_ProbNNk)<=hplus_ProbNNk) & ((1-hminus_ProbNNk)<=100.) & ((1-hminus_ProbNNk)>0.995) ) ) )"
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    elif ("MC_Bd2JpsiKstar"==m):
                      if "Prob1" in k:
                        v = "( ( ( (hplus_ProbNNk_corr<(1-hminus_ProbNNk_corr)) & (hplus_ProbNNk_corr<=0.86) & (hplus_ProbNNk_corr>0.0) ) | ( ((1-hminus_ProbNNk_corr)<=hplus_ProbNNk_corr) & ((1-hminus_ProbNNk_corr)<=0.86) & ((1-hminus_ProbNNk_corr)>0.0) ) ) )"
                      elif "Prob2" in k:
                        v = "( ( ( (hplus_ProbNNk_corr<(1-hminus_ProbNNk_corr)) & (hplus_ProbNNk_corr<=0.96) & (hplus_ProbNNk_corr>0.86) ) | ( ((1-hminus_ProbNNk_corr)<=hplus_ProbNNk_corr) & ((1-hminus_ProbNNk_corr)<=0.96) & ((1-hminus_ProbNNk_corr)>0.86) ) ) )"
                      elif "Prob3" in k:
                        v = "( ( ( (hplus_ProbNNk_corr<(1-hminus_ProbNNk_corr)) & (hplus_ProbNNk_corr<=0.995) & (hplus_ProbNNk_corr>0.96) ) | ( ((1-hminus_ProbNNk_corr)<=hplus_ProbNNk_corr) & ((1-hminus_ProbNNk_corr)<=0.995) & ((1-hminus_ProbNNk_corr)>0.96) ) ) )"
                      elif "Prob4" in k:
                        v = "( ( ( (hplus_ProbNNk_corr<(1-hminus_ProbNNk_corr)) & (hplus_ProbNNk_corr<=100.) & (hplus_ProbNNk_corr>0.995) ) | ( ((1-hminus_ProbNNk_corr)<=hplus_ProbNNk_corr) & ((1-hminus_ProbNNk_corr)<=100.) & ((1-hminus_ProbNNk_corr)>0.995) ) ) )"
                      list_of_cuts.append(v)
                      noe = len(result.query(v))

                    else:
                      print("PID cut not expected for this mode")
                else:
                    list_of_cuts.append(v)
                if noe == 0:
                    print(f"ERROR: This cut leaves df empty. {v}")
                    print("       Query halted.")
            except:
                print(f"There is no such variable for the cut {v}")
    if list_of_cuts:
        vsub_cut = f"( {' ) & ( '.join(list_of_cuts)} )"

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

  # }}}

# }}}


# vim: ts=4 sw=4 sts=4 et
