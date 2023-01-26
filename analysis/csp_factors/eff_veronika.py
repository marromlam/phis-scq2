import argparse
# import yaml
# from ROOT import *

# veronika needs these
from numericFunctionClass import NF  # taken from Urania
import pickle as cPickle
import ROOT
import os

# I need these
import uproot3 as uproot
import numpy as np
import matplotlib.pyplot as plt


def argument_parser():
  return parser


def create_mass_bins(nob):
  mass_bins = [990, 1008, 1016, 1020, 1024, 1032, 1050]
  return mass_bins


def epsmKK(input_file1, input_file2, input_tree_name_file1, input_tree_name_file2, output_dir, mode, year):
  """
  Get efficiency

  Parameters
  ----------
  input_file : pandas.DataFrame
    Sample from the selection pipeline
  output_file : pandas.DataFrame
    Particle gun generated Sample
  """
  SWAVE = ('Swave' in mode)
  print("SWAVE ++++++ ", SWAVE)

  mKK = "X_M"

  list_branches = [
      'hplus_TRUEP_E', 'hplus_TRUEP_X', 'hplus_TRUEP_Y', 'hplus_TRUEP_Z',
      'hminus_TRUEP_E', 'hminus_TRUEP_X', 'hminus_TRUEP_Y', 'hminus_TRUEP_Z',
      'X_TRUEP_E', 'X_TRUEP_X', 'X_TRUEP_Y', 'X_TRUEP_Z',
      'B_BKGCAT', 'B_TRUETAU',
      mKK, 'gb_weights'
  ]

  # weight = f'{mKK}/{mKK}'
  weight = f'gb_weights'

  print(mode, input_file1, input_file2)

  # t1 is the root file from the selection pipeline
  df1 = uproot.open(input_file1)[input_tree_name_file1]
  df1 = df1.pandas.df(branches=list_branches)
  # t2 is the EvtGen standalone root file
  df2 = uproot.open(input_file2)[input_tree_name_file2]
  # df2 = df2.pandas.df(branches=['MKK'])
  df2 = df2.pandas.df(branches=['X_M'])
  print(df1)

  print(mode, input_file1, input_file2)

  f1 = ROOT.TFile(input_file1)
  t1 = f1.Get(input_tree_name_file1)

  f2 = ROOT.TFile(input_file2)
  t2 = f2.Get(input_tree_name_file2)

  mkk_bins = [[990, 1008], [1008, 1016], [1016, 1020],
              [1020, 1024], [1024, 1032], [1032, 1050]]
  mkk_histos = []

  mass_knots = create_mass_bins(6)
  mLL, mUL = mass_knots[0] - 10, mass_knots[-1] + 10 + 140 * SWAVE

  NBINS_WIDE = 100 + 150 * SWAVE
  NBINS_NARROW = 200 + 300 * SWAVE
  # NBINS_WIDE = 5
  # NBINS_NARROW = 7

  ll = str(980.0)
  ul = str(1060.0 + 140.0 * SWAVE)
  # print("ll ul = ", ll, ul)
  # print("mLL mUL = ", mLL, mUL)

  if SWAVE:
    df1.eval("truemX = sqrt((hplus_TRUEP_E+hminus_TRUEP_E)*(hplus_TRUEP_E+hminus_TRUEP_E) - ((hplus_TRUEP_X+hminus_TRUEP_X)*(hplus_TRUEP_X+hminus_TRUEP_X)+(hplus_TRUEP_Y+hminus_TRUEP_Y)*(hplus_TRUEP_Y+hminus_TRUEP_Y)+(hplus_TRUEP_Z+hminus_TRUEP_Z)*(hplus_TRUEP_Z+hminus_TRUEP_Z)))", inplace=True)
    df1.eval("truthMatch = (B_BKGCAT == 0 | B_BKGCAT == 10 | (B_BKGCAT == 50 & B_TRUETAU > 0))", inplace=True)
  else:
    # Warning: Is this ok?
    df1.eval("truemX = sqrt(X_TRUEP_E*X_TRUEP_E-(X_TRUEP_X*X_TRUEP_X+X_TRUEP_Y*X_TRUEP_Y+X_TRUEP_Z*X_TRUEP_Z))", inplace=True)
    df1.eval("truthMatch = (B_BKGCAT == 0 | B_BKGCAT == 10 | (B_BKGCAT == 50 & B_TRUETAU > 0))", inplace=True)
  truthMatch = "(B_BKGCAT == 0 | B_BKGCAT == 10 | (B_BKGCAT == 50 & B_TRUETAU > 0))"
  # df2.eval("mX = 1000*MKK", inplace=True)
  df2.eval("mX = X_M", inplace=True)

  mKK_true_Swave = "sqrt((hplus_TRUEP_E+hminus_TRUEP_E)*(hplus_TRUEP_E+hminus_TRUEP_E) - ((hplus_TRUEP_X+hminus_TRUEP_X)*(hplus_TRUEP_X+hminus_TRUEP_X)+(hplus_TRUEP_Y+hminus_TRUEP_Y)*(hplus_TRUEP_Y+hminus_TRUEP_Y)+(hplus_TRUEP_Z+hminus_TRUEP_Z)*(hplus_TRUEP_Z+hminus_TRUEP_Z)))"

  mKK_true = "sqrt(X_TRUEP_E*X_TRUEP_E-(X_TRUEP_X*X_TRUEP_X+X_TRUEP_Y*X_TRUEP_Y+X_TRUEP_Z*X_TRUEP_Z))"

  truth_match = "(B_BKGCAT == 0 || B_BKGCAT == 10 || (B_BKGCAT == 50 && B_TRUETAU > 0))"  # "abs(B_TRUEID)==531"#"B_BKGCAT!=60 && abs(B_TRUEID)==531"

  truth_match_Swave = "(B_BKGCAT == 0 || B_BKGCAT == 10 || (B_BKGCAT == 50 && B_TRUETAU > 0))"  # "abs(B_TRUEID)==531"#"B_BKGCAT!=60 && abs(B_TRUEID)==531"

  mKK_true_mass_cut = mKK_true + ">" + ll + "&&" + mKK_true + "<" + ul

  mKK_true_Swave_mass_cut = mKK_true_Swave + ">" + ll + "&&" + mKK_true_Swave + "<" + ul

  # this only applies to the EvtGen standalone MC {{{

  print("veronika limits", ll, ul)
  print("marcos limits", mLL, mUL)
  if SWAVE:
    hname_str_WIDE = "hmkk_WIDE(" + str(NBINS_WIDE) + "," + ll + "," + ul + ")"
    # t2.Draw(mKK_true_Swave+" >> "+hname_str_WIDE, mKK_true_Swave_mass_cut+"&&"+truth_match_Swave)
    t2.Draw("X_M >> " + hname_str_WIDE)
    hname_str_NARROW = "hmkk_NARROW(" + str(NBINS_NARROW) + "," + ll + "," + ul + ")"
    # t2.Draw(mKK_true_Swave+" >> "+hname_str_NARROW, mKK_true_Swave_mass_cut+"&&"+truth_match_Swave)
    t2.Draw("X_M >> " + hname_str_NARROW)
  else:
    hname_str_WIDE = "hmkk_WIDE(" + str(NBINS_WIDE) + "," + ll + "," + ul + ")"
    # t2.Draw(mKK_true+" >> "+hname_str_WIDE, mKK_true_mass_cut+"&&"+truth_match)
    t2.Draw("X_M >> " + hname_str_WIDE)
    hname_str_NARROW = "hmkk_NARROW(" + str(NBINS_NARROW) + "," + ll + "," + ul + ")"
    # t2.Draw(mKK_true+" >> "+hname_str_NARROW, mKK_true_mass_cut+"&&"+truth_match)
    t2.Draw("X_M >> " + hname_str_NARROW)

  hmkk_WIDE = (ROOT.gROOT.FindObject("hmkk_WIDE"))
  hmkk_NARROW = (ROOT.gROOT.FindObject("hmkk_NARROW"))

  hwide = np.histogram(df2['mX'].values, NBINS_WIDE, range=(mLL, mUL))[0]
  hnarr = np.histogram(df2['mX'].values, NBINS_NARROW, range=(mLL, mUL))[0]
  # just to have the same hitogram as the one from ROOT::Draw
  # hwide = np.array([0.] + hwide.tolist())
  # hnarr = np.array([0.] + hnarr.tolist())
  hwide = hwide.tolist()
  hnarr = hnarr.tolist()

  # print("wide hist", np.sum(hwide))
  # for i in range(NBINS_WIDE + 1):
  #   print(f"comprobar max: {hmkk_WIDE.GetBinContent(i):>20} -- {hwide[i]}")
  # print("narrow hist")
  # for i in range(NBINS_NARROW + 1):
  #   print(f"comprobar max: {hmkk_NARROW.GetBinContent(i)} -- {hnarr[i]}")
  # exit()
  # }}} all good

  # weight = "gb_weights"
  hb = []
  for i, ll, ul in zip(range(6), mass_knots[:-1], mass_knots[1:]):
    # if(i == 0 or i == 5):
    #     _nbins = NBINS_WIDE
    # else:
    #     _nbins = NBINS_NARROW
    if ll == mass_knots[0] or ul == mass_knots[-1]:
      _nbins = NBINS_WIDE
    else:
      _nbins = NBINS_NARROW
    mass_cut = f"{mKK} > {ll} & {mKK} < {ul}"
    true_mass_cut = f"truemX > {mLL} & truemX < {mUL}"
    _weight = (f"({mass_cut}) & ({true_mass_cut}) & ({truthMatch})")
    _w = df1.eval(f"( {_weight} ) * {weight}")
    # _w = df1.eval(f"( {_weight} ) * 1.0")
    _v = df1['truemX'].values
    _c, _b = np.histogram(_v, _nbins, weights=_w, range=(mLL, mUL))
    # print("bins", _b)
    hb.append([_b, _c.tolist()])
    # hb.append([_b, [0] + _c.tolist() + [0]])
    print(hb[-1])
    # hb.append(np.histogram(df1['truemX'].values, NBINS, range=(mLL, mUL)))
    # hb.append(np.histogram(df1['truemX'].values, NBINS, range=(mLL, mUL)))

  ll = str(980.0)
  ul = str(1060.0 + 140.0 * SWAVE)
  for i in range(len(mkk_bins)):
    hname = 'epsmKK_' + str(i)
    if(i == 0 or i == 5):
      NBINS = NBINS_WIDE
    else:
      NBINS = NBINS_NARROW
    hname_str = hname + "(" + str(NBINS) + "," + ll + "," + ul + ")"
    # print(hname_str)
    if(SWAVE == 0):
      cut = mKK + " > " + str(mkk_bins[i][0]) + " && " + mKK + " < " + str(mkk_bins[i][1]) + " && " + mKK_true_mass_cut + " && " + truth_match
    else:
      cut = mKK + " > " + str(mkk_bins[i][0]) + " && " + mKK + " < " + str(mkk_bins[i][1]) + " && " + mKK_true_Swave_mass_cut + " && " + truth_match_Swave

    if(SWAVE == 0):
      t1.Draw(mKK_true + " >>" + hname_str, cut)
    else:
      t1.Draw(mKK_true_Swave + " >>" + hname_str, "gb_weights*(" + cut + ")")
      # t1.Draw(mKK_true_Swave + " >>" + hname_str, "1.*(" + cut + ")")
    mkk_histos.append(ROOT.gROOT.FindObject(hname))
    print(f"\nHISTOGRAM {i}")
    for j in range(len(mkk_histos[-1])):
      # print(f"diffs: {mkk_histos[-1].GetBinContent(j+1)} -- {hb[i][1][j]}")
      print(f"vero: {mkk_histos[-1].GetBinContent(j+1)}")
    print(f"marc: {hb[i][1]}")

  masses = []
  ratios = []
  for j in range(len(hb)):
    _ratios = []
    _masses = []
    if(j == 0 or j == 5):
      NBINS = NBINS_WIDE
      # print("NBINS WIDE=",NBINS)
      for i in range(NBINS):
        # print('gordow', max(hwide[i], 1))
        ratio = hb[j][1][i] / float(max(hwide[i], 1))
        if j != 0 and hwide[i] < mLL and SWAVE:
          # print('i,j ', i, j)
          ratio = 0.
        ratio = 0 if hwide[i] == 0 else ratio

        _ratios.append(ratio)
        # print("mass ->", hb[j][0])
        # _masses.append(hb[j][0][i])
        _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i + 1]))
        # print("bin center", _masses[-1])
        # _masses.append(0)
    else:
      NBINS = NBINS_NARROW
      for i in range(NBINS):
        ratio = hb[j][1][i] / float(max(hnarr[i], 1))
        if j != 0 and hnarr[i] < mLL and SWAVE:
          ratio = 0.
        # ratio = 0 if hnarr[i] == 0 else ratio

        _ratios.append(ratio)
        _masses.append(0.5 * (hb[j][0][i] + hb[j][0][i + 1]))
        # print("mass ->", hb[j][0])
        # print("bin center", _masses[-1])
        # _masses.append(0)
    print("\nbin", j)
    print(_masses)
    print(_ratios)
    masses.append(_masses)
    ratios.append(_ratios)
  np.save(os.path.join("for_veronika"), [masses, ratios],
          allow_pickle=True)
  # exit()

  RATIOS = ratios
  MYRATIOS = ratios
  MYMASSES = masses
  for i in masses:
    print(i)
  for i in ratios:
    print(i)

  plt.close()
  for i in range(6):
    plt.plot(masses[i], RATIOS[i])
  plt.savefig("mierda.pdf")
  #
  graphs = [ROOT.TGraph(), ROOT.TGraph(), ROOT.TGraph(), ROOT.TGraph(), ROOT.TGraph(), ROOT.TGraph()]
  ratios = [[], [], [], [], [], [], []]
  masses = [[], [], [], [], [], [], []]
  ratiosid = []
  NPOINT = 0

  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf")
  print("sadfasdfasdfsadfasdf\n\n")
  m_cut_off = 980.
  for j in range(len(mkk_histos)):
    if(j == 0 or j == 5):
      NBINS = NBINS_WIDE
      # print("NBINS WIDE=",NBINS)
      for i in range(1, NBINS + 1):
        # print(f"W {i},{j} --> ", len(MYMASSES), len(MYMASSES[j]))
        # print(f"W {i},{j} --> ", len(hb), len(hb[1][j]))
        # print(f"comprobar max: {hmkk_WIDE.GetBinContent(i)} -- {hwide[i]}")
        # print("vero", mkk_histos[j][i], "marc", hb[j][1][i])
        ratio = mkk_histos[j].GetBinContent(i) / max(hmkk_WIDE.GetBinContent(i), 1)
        # print("marcos:  ", MYMASSES[j][i], hb[j][1][i], hwide[i])
        # print("veronika:", hmkk_WIDE.GetBinCenter(i + 1), mkk_histos[j][i + 1], hmkk_WIDE.GetBinContent(i + 1))

        if(hmkk_WIDE.GetBinContent(i) == 0):
          ratio = 0

        m = hmkk_WIDE.GetBinCenter(i)
        if(j != 0 and m < m_cut_off and SWAVE):
          ratio = 0.
        # print("--> W marcos:", MYRATIOS[j][i])
        # print("--> W veroka:", ratio)

        ratios[j].append(ratio)
        masses[j].append(hmkk_WIDE.GetBinCenter(i))
    else:
      NBINS = NBINS_NARROW
      # print("NBINS NARROW =",NBINS_NARROW)
      for i in range(1, NBINS + 1):
        # print(f"N {i},{j} --> ", len(MYMASSES), len(MYMASSES[j]))
        # print(f"N {i},{j} --> ", len(hb), len(hb[1][j]))
        ratio = mkk_histos[j].GetBinContent(i) * 1. / max(hmkk_NARROW.GetBinContent(i), 1)
        # print("marcos:  ", MYMASSES[j][i], hb[j][1][i], hnarr[i])
        # print("veronika:", hmkk_WIDE.GetBinCenter(i + 1), mkk_histos[j][i + 1], hmkk_WIDE.GetBinContent(i + 1))

        if(hmkk_NARROW.GetBinContent(i) == 0):
          ratio = 0

        m = hmkk_NARROW.GetBinCenter(i)
        if(j != 0 and m < m_cut_off and SWAVE):
          ratio = 0.
        # print("--> N marcos:", MYRATIOS[j][i])
        # print("--> N veroka:", ratio)

        ratios[j].append(ratio)
        masses[j].append(hmkk_NARROW.GetBinCenter(i))
    print(f"bin{j} -- ", "marcos:  ", MYMASSES[j], MYRATIOS[j])
    print(f"bin{j} -- ", "veronika:", masses[j][-1], ratios[j])

  for r in MYRATIOS:
    print(r)
  print("------------------------------------------------------")
  for r in ratios:
    print(r)

  # exit()
  print("\n\ngraphing:")
  for j in range(len(mkk_histos)):
    NPOINT = 0
    if(j == 0 or j == 5):
      for i in range(len(ratios[0])):
        graphs[j].SetPoint(NPOINT, hmkk_WIDE.GetBinCenter(i + 1), ratios[j][i])
        print("bin:", NPOINT, hmkk_WIDE.GetBinCenter(i + 1), ratios[j][i])
        print("   :", i, MYMASSES[j][i], MYRATIOS[j][i])
        NPOINT += 1
    else:
      for i in range(len(ratios[1])):
        graphs[j].SetPoint(NPOINT, hmkk_NARROW.GetBinCenter(i + 1), ratios[j][i])
        print("bin:", NPOINT, hmkk_WIDE.GetBinCenter(i + 1), ratios[j][i])
        print("   :", i, MYMASSES[j][i], MYRATIOS[j][i])
        NPOINT += 1

  graphs[1].SetLineColor(ROOT.kGreen)
  graphs[1].SetMarkerColor(ROOT.kGreen)
  graphs[2].SetLineColor(ROOT.kRed)
  graphs[2].SetMarkerColor(ROOT.kRed)
  graphs[3].SetLineColor(ROOT.kBlue)
  graphs[3].SetMarkerColor(ROOT.kBlue)
  graphs[4].SetLineColor(ROOT.kMagenta)
  graphs[4].SetMarkerColor(ROOT.kMagenta)
  graphs[5].SetLineColor(ROOT.kCyan)
  graphs[5].SetMarkerColor(ROOT.kCyan)

  c = ROOT.TCanvas()
  graphs[0].GetXaxis().SetTitle("m_{KK} [MeV/c^{2}]")
  graphs[0].GetYaxis().SetTitle("#epsilon(m_{KK})")
  graphs[0].GetXaxis().SetLimits(float(ll) - 10, float(ul) + 10)
  max_hist = {"2015": 0.1 + SWAVE * 0.05, "2016": 0.5 - SWAVE * 0.35, "2017": 0.3 - SWAVE * 0.15, "2018": 0.3 - SWAVE * 0.15, "All": 1. - SWAVE * 0.85}
  graphs[0].GetHistogram().SetMaximum(max_hist[str(year)])
  graphs[0].Draw()
  graphs[1].Draw("LP")
  graphs[2].Draw("LP")
  graphs[3].Draw("LP")
  graphs[4].Draw("LP")
  graphs[5].Draw("LP")

  # print(len(masses), masses)
  # print(len(ratios))
  # if SWAVE: gPad.SetLogy()

  c.SaveAs(output_dir + "epsmKK_" + str(year) + SWAVE * "_SWave" + ".pdf")

  # To dump: NF with ratios and masses
  functions = []
  for i in range(len(mkk_bins)):
    functions.append(NF(MYMASSES[i], MYRATIOS[i]))
    # cPickle.dump(functions[i],open(),"w"))
    with open(output_dir + "eff_hist_" + str(mkk_bins[i][0]) + "_" + str(mkk_bins[i][1]), "wb") as output_file:
      cPickle.dump(functions[i], output_file)


# command line {{{

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--simulated-sample', help='Path to the preselected input file')
  p.add_argument('--pgun-sample', help='Path to the uncut input file')
  p.add_argument('--output-figure', help='Output directory')
  p.add_argument('--output-histos', help='Output directory')
  p.add_argument('--mode', help='Name of the selection in yaml')
  p.add_argument('--year', help='Year of the selection in yaml')
  args = vars(p.parse_args())

  input_file1 = "/scratch49/marcos.romero/sidecar/2015/MC_Bs2JpsiKK_Swave/v4r0@LcosK_sWeight.root"
  input_file2 = "/scratch49/marcos.romero/sidecar/2015/GUN_Bs2JpsiKK_Swave/v2r0_ready.root"
  # input_file2 = "/scratch49/marcos.romero/sidecar/2015/MC_Bs2JpsiKK_Swave/v4r0@LcosK_chopped.root"
  input_tree_name_file1 = "DecayTree"
  input_tree_name_file2 = "DecayTree"
  output_dir = "merda.root"
  mode = "MC_Bs2JpsiKK_Swave"
  year = 2015
  # epsmKK(**vars(args))
  epsmKK(input_file1, input_file2, input_tree_name_file1, input_tree_name_file2, output_dir, mode, year)

# }}}


# vim: fdm=marker
