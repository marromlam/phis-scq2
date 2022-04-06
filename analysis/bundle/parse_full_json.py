__all__ = []
import os
import hjson
from ipanema import Parameters


# Download full fit_inputs from repository -------------------------------------
#     This function will connect to gitlab and download the required file w/o
#     modificacions.

def download_fulljson(version, year, where="./tmp"):
  repo = "ssh://git@gitlab.cern.ch:7999/lhcb-b2cc/Bs2JpsiPhi-FullRun2.git"
  path = f"fitinputs/{version}/"
  os.system(f"git archive --remote={repo} --prefix={where}/{version}/ "+\
            f"HEAD:{path} fit_inputs_{year}.json | tar -x")


# Convert full fit_inputs to usable --------------------------------------------
#     Convert the full json from repository to a multiple set of jsons, each
#     one corresponding to a diffenre part of the analysis

def parse_fulljson(tmp_json_path):
  tjson = hjson.load(open(f"{tmp_json_path}",'r'))


  # Time acceptance ------------------------------------------------------------
  knots = Parameters()
  timeaccbiased = Parameters(); timeaccunbiased = Parameters()

  for i,v in enumerate(tjson["TimeAccParameter"]["KnotParameter"]):
    knots.add({"name":f"k{i}", "value":v['Value'], "latex":"k_{i}", "free":False})
  knots.add({"name":f"tLL", "value":0.3, "latex":"t_{LL}", "free":False})
  knots.add({"name":f"tUL", "value":15.0, "latex":"t_{UL}", "free":False})

  for i,v in enumerate(tjson["TimeAccParameter"]["SplineBiased"]):
    timeaccbiased.add({"name":f"c{i}", "value":v['Value'], "stdev":v['Error'],
                       "latex":"c_{i}", "free":False})

  for i,v in enumerate(tjson["TimeAccParameter"]["SplineUnbiased"]):
    timeaccunbiased.add({"name":f"c{i}", "value":v['Value'], "stdev":v['Error'],
                         "latex":"c_{i}", "free":False})

  # print(80*'-')
  # print("The following parameters were loaded for time acceptance")
  # print(knots)
  # print(timeaccbiased)
  # print(timeaccunbiased)
  # print(80*'-')
  0


  # Angular acceptance ---------------------------------------------------------
  angaccbiased = Parameters(); angaccunbiased = Parameters()

  for i,v in enumerate(tjson["AngularParameterBiased"]):
    angaccbiased.add({"name":f"w{i}", "value":v['Value'], "stdev":v['Error'],
                       "latex":"w_{i}", "free":False})

  for i,v in enumerate(tjson["AngularParameterUnbiased"]):
    angaccunbiased.add({"name":f"w{i}", "value":v['Value'], "stdev":v['Error'],
                         "latex":"w_{i}", "free":False})

  print(80*'-')
  print("The following parameters were loaded for angular acceptance")
  print(angaccbiased)
  print(angaccunbiased)
  print(80*'-')


  # Time resolution ------------------------------------------------------------
  timeres = Parameters();
  for i,v in enumerate(tjson["TimeResParameters"]):
    if v['Name'] == "p0":
      timeres.add({"name":f"sigma_offset",
                   "value":v['Value'], "stdev":v['Error'],
                   "correl": {"sigma_offset": 0, "sigma_curvature": 0},
                   "latex":"\sigma_0", "free":False})
    elif v['Name'] == "p1":
      timeres.add({"name":f"sigma_slope",
                   "value":v['Value'], "stdev":v['Error'],
                   "correl":{"sigma_offset":0, "sigma_curvature":0},
                   "latex":"\sigma_1", "free":False})
    elif v['Name'] == "p2":
      timeres.add({"name":f"sigma_curvature",
                   "value":v['Value'], "stdev":v['Error'],
                   "correl": {"sigma_offset": 0, "sigma_curvature": 0},
                   "latex":"\sigma_2", "free":False})
    elif v['Name'] == "rho_p0_p1_time_res":
      timeres.add({"name":f"rho01", "value":v['Value'], "stdev":v['Error'],
               "latex":r"\rho_{01}", "free":False})
    elif v['Name'] == "rho_p1_p2_time_res":
      timeres.add({"name":f"rho12", "value":v['Value'], "stdev":v['Error'],
               "latex":r"\rho_{12}", "free":False})
    elif v['Name'] == "rho_p0_p2_time_res":
      timeres.add({"name":f"rho02", "value":v['Value'], "stdev":v['Error'],
               "latex":r"\rho_{02}", "free":False})
  print(80*'-')
  print("The following parameters were loaded for time resolution")
  print(timeres)
  print(80*'-')


  # Time resolution ------------------------------------------------------------
  flavor = Parameters();
  for i,v in enumerate(tjson["TaggingParameter"]):
    if v['Name'] == "p0_OS":
      flavor.add({"name":f"p0_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{os}^0", "free":False})
    elif v['Name'] == "p1_OS":
      flavor.add({"name":f"p1_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{os}^1", "free":False})
    elif v['Name'] == "p2_OS":
      flavor.add({"name":f"p2_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{os}^2", "free":False})
    elif v['Name'] == "dp0_OS":
      flavor.add({"name":f"dp0_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{os}^0", "free":False})
    elif v['Name'] == "dp1_OS":
      flavor.add({"name":f"dp1_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{os}^1", "free":False})
    elif v['Name'] == "dp2_OS":
      flavor.add({"name":f"dp2_os",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{os}^2", "free":False})
    elif v['Name'] == "rho_p0_p1_OS":
      flavor.add({"name":f"rho01_os",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{os}^{01}", "free":False})
    elif v['Name'] == "rho_p1_p2_OS":
      flavor.add({"name":f"rho12_os",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{os}^{12}", "free":False})
    elif v['Name'] == "rho_p0_p2_OS":
      flavor.add({"name":f"rho02_os",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{os}^{02}", "free":False})
    elif v['Name'] == "p0_SSK":
      flavor.add({"name":f"p0_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{ss}^0", "free":False})
    elif v['Name'] == "p1_SSK":
      flavor.add({"name":f"p1_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{ss}^1", "free":False})
    elif v['Name'] == "p2_SSK":
      flavor.add({"name":f"p2_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"p_{ss}^2", "free":False})
    elif v['Name'] == "dp0_SSK":
      flavor.add({"name":f"dp0_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{ss}^0", "free":False})
    elif v['Name'] == "dp1_SSK":
      flavor.add({"name":f"dp1_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{ss}^1", "free":False})
    elif v['Name'] == "dp2_SSK":
      flavor.add({"name":f"dp2_ss",
                   "value":v['Value'], "stdev":v['Error'],
                   "latex":"\Delta p_{ss}^2", "free":False})
    elif v['Name'] == "rho_p0_p1_SSK":
      flavor.add({"name":f"rho01_ss",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{ss}^{01}", "free":False})
    elif v['Name'] == "rho_p1_p2_SSK":
      flavor.add({"name":f"rho12_ss",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{ss}^{12}", "free":False})
    elif v['Name'] == "rho_p0_p2_SSK":
      flavor.add({"name":f"rho02_ss",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":r"\rho_{ss}^{02}", "free":False})
    elif v['Name'] == "eta_bar_OS":
      flavor.add({"name":f"eta_os",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":"\\bar{\\eta}_{os}", "free":False})
    elif v['Name'] == "eta_bar_SSK":
      flavor.add({"name":f"eta_ss",
                  "value":v['Value'], "stdev":v['Error'],
                  "latex":"\\bar{\\eta}_{ss}", "free":False})
  print(80*'-')
  print("The following parameters were loaded for flavor tagging")
  print(flavor)
  print(80*'-')



  csp_factors = Parameters()
  for i, d in enumerate(tjson["CspFactors"]):
    bin = i+1
    csp_factors.add({'name':f'CSP{bin}',
                     'value':d['Value'], 'stdev':d['Error'],
                     'latex': f"C_{{SP}}^{{{bin}}}",
                     'free': False})
  for i, d in enumerate(tjson["CspFactors"]):
    bin = i+1
    if not f'mKK{bin-1}' in csp_factors:
      csp_factors.add({'name':f'mKK{bin-1}',
                       'value':d['Bin_ll'], 'stdev':0,
                       'latex':f'm_{{KK}}^{{{bin-1}}}',
                       'free': False})
    if not f'mKK{bin}' in csp_factors:
      csp_factors.add({'name':f'mKK{bin}',
                       'value':d['Bin_ul'], 'stdev':0,
                       'latex':f'm_{{KK}}^{{{bin}}}',
                       'free': False})
  print(80*'-')
  print("The following parameters were loaded for CSP factors")
  print(csp_factors)
  print(80*'-')

  return (knots+timeaccbiased, knots+timeaccunbiased, angaccbiased, 
          angaccunbiased, timeres, flavor, csp_factors)
