import argparse
# import yaml
# from ROOT import *

from numericFunctionClass import NF
import pickle as cPickle
import ROOT

# import uproot3 as uproot

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file1', help='Path to the preselected input file')
    parser.add_argument('--input-file2', help='Path to the uncut input file')
    parser.add_argument('--input-tree-name-file1', default='DecayTree', help='Name of the tree')
    parser.add_argument('--input-tree-name-file2', default='DecayTree', help='Name of the tree')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--year', help='Year of the selection in yaml')
    return parser



def epsmKK(input_file1, input_file2, input_tree_name_file1, input_tree_name_file2, output_dir, mode, year):
    """
    Get efficiency

    Parameters
    ----------
    input_file : pandas.DataFrame
      Sample from the selection pipeline
    output_file : pandas.DataFrame
      EvtGen standalone sample
    """
    SWAVE = ('Swave' in mode)

    print(mode, input_file1, input_file2)

    # t1 is the root file from the selection pipeline 
    # df1 = uproot.open(input_file1)[input_tree_name_file1].pandas.df()
    # t2 is the EvtGen standalone root file
    # df2 = uproot.open(input_file2)[input_tree_name_file2].pandas.df()

    print(mode, input_file1, input_file2)

    f1 = ROOT.TFile(input_file1)
    t1 = f1.Get(input_tree_name_file1)

    f2 = ROOT.TFile(input_file2)
    t2 = f2.Get(input_tree_name_file2)

    mkk_bins = [[ 990 , 1008 ],[ 1008 , 1016 ],[ 1016 , 1020 ],[ 1020 , 1024 ], [ 1024 , 1032 ],[ 1032 , 1050 ]]
    mkk_histos = []

    NBINS_WIDE = 100 + 150*SWAVE
    NBINS_NARROW = 200 + 300*SWAVE

    ll = str(980.0)
    ul = str(1060.0+140.0*SWAVE)


    mKK = "X_M"
    
    mKK_true_Swave = "sqrt((hplus_TRUEP_E+hminus_TRUEP_E)*(hplus_TRUEP_E+hminus_TRUEP_E) - ((hplus_TRUEP_X+hminus_TRUEP_X)*(hplus_TRUEP_X+hminus_TRUEP_X)+(hplus_TRUEP_Y+hminus_TRUEP_Y)*(hplus_TRUEP_Y+hminus_TRUEP_Y)+(hplus_TRUEP_Z+hminus_TRUEP_Z)*(hplus_TRUEP_Z+hminus_TRUEP_Z)))"
    
    mKK_true = "sqrt(X_TRUEP_E*X_TRUEP_E-(X_TRUEP_X*X_TRUEP_X+X_TRUEP_Y*X_TRUEP_Y+X_TRUEP_Z*X_TRUEP_Z))"

    truth_match = "(B_BKGCAT == 0 || B_BKGCAT == 10 || (B_BKGCAT == 50 && B_TRUETAU > 0))"#"abs(B_TRUEID)==531"#"B_BKGCAT!=60 && abs(B_TRUEID)==531"

    truth_match_Swave = "(B_BKGCAT == 0 || B_BKGCAT == 10 || (B_BKGCAT == 50 && B_TRUETAU > 0))"#"abs(B_TRUEID)==531"#"B_BKGCAT!=60 && abs(B_TRUEID)==531"

    mKK_true_mass_cut = mKK_true+">"+ll+"&&"+mKK_true+"<"+ul

    mKK_true_Swave_mass_cut = mKK_true_Swave+">"+ll+"&&"+mKK_true_Swave+"<"+ul

    # this only applies to the EvtGen standalone MC {{{

    if SWAVE:
        hname_str_WIDE = "hmkk_WIDE("+str(NBINS_WIDE)+","+ll+","+ul+")"
        #t2.Draw(mKK_true_Swave+" >> "+hname_str_WIDE, mKK_true_Swave_mass_cut+"&&"+truth_match_Swave)
        t2.Draw("1000*MKK >> "+hname_str_WIDE)
        hname_str_NARROW = "hmkk_NARROW("+str(NBINS_NARROW)+","+ll+","+ul+")"
        #t2.Draw(mKK_true_Swave+" >> "+hname_str_NARROW, mKK_true_Swave_mass_cut+"&&"+truth_match_Swave)
        t2.Draw("1000*MKK >> "+hname_str_NARROW)
    else:
        hname_str_WIDE = "hmkk_WIDE("+str(NBINS_WIDE)+","+ll+","+ul+")"
        #t2.Draw(mKK_true+" >> "+hname_str_WIDE, mKK_true_mass_cut+"&&"+truth_match)
        t2.Draw("1000*MKK >> "+hname_str_WIDE)
        hname_str_NARROW = "hmkk_NARROW("+str(NBINS_NARROW)+","+ll+","+ul+")"
        #t2.Draw(mKK_true+" >> "+hname_str_NARROW, mKK_true_mass_cut+"&&"+truth_match)
        t2.Draw("1000*MKK >> "+hname_str_NARROW)

    hmkk_WIDE = (ROOT.gROOT.FindObject("hmkk_WIDE"))
    hmkk_NARROW = (ROOT.gROOT.FindObject("hmkk_NARROW"))

    for i in range(len(mkk_bins)):
        hname = 'epsmKK_'+str(i)
        if(i == 0 or i == 5):
            NBINS = NBINS_WIDE
        else:
            NBINS = NBINS_NARROW
        hname_str = hname+"("+str(NBINS)+","+ll+","+ul+")"
        if(SWAVE==0):
            cut = mKK + " > " + str(mkk_bins[i][0]) + " && " + mKK + " < "+str(mkk_bins[i][1]) + " && " + mKK_true_mass_cut + " && " + truth_match
        else:
            cut = mKK + " > " + str(mkk_bins[i][0]) + " && " + mKK + " < "+str(mkk_bins[i][1]) + " && " + mKK_true_Swave_mass_cut + " && " + truth_match_Swave
            
        #print("cut:",cut)

        if(SWAVE==0):        
            t1.Draw(mKK_true + " >>" + hname_str,cut)
        else:
            t1.Draw(mKK_true_Swave + " >>" + hname_str,"gb_weights*("+cut+")")        
        mkk_histos.append(ROOT.gROOT.FindObject(hname))

    graphs = [ROOT.TGraph(),ROOT.TGraph(),ROOT.TGraph(),ROOT.TGraph(),ROOT.TGraph(),ROOT.TGraph()]
    ratios = [[],[],[],[],[],[],[]]
    masses = [[],[],[],[],[],[],[]]
    ratiosid = []
    NPOINT = 0

    m_cut_off = 980.
    for j in range(len(mkk_histos)):
        if(j == 0 or j == 5):
            NBINS = NBINS_WIDE
            print("NBINS WIDE=",NBINS)
            for i in range(NBINS):
                ratio = mkk_histos[j][i]*1./max(hmkk_WIDE.GetBinContent(i),1)
                
                if(hmkk_WIDE.GetBinContent(i) == 0): ratio = 0
                
                m = hmkk_WIDE.GetBinCenter(i)
                if(j!=0 and m <m_cut_off and SWAVE):
                    ratio = 0.
                
                ratios[j].append(ratio)
                masses[j].append(hmkk_WIDE.GetBinCenter(i))
        else:
            NBINS = NBINS_NARROW
            print("NBINS NARROW =",NBINS_NARROW)
            for i in range(NBINS):
                ratio = mkk_histos[j][i]*1./max(hmkk_NARROW.GetBinContent(i),1)
                 
                if(hmkk_NARROW.GetBinContent(i) == 0): ratio = 0
                 
                m = hmkk_NARROW.GetBinCenter(i)
                if(j!=0 and m <m_cut_off and SWAVE):
                    ratio = 0.
                
                ratios[j].append(ratio)
                masses[j].append(hmkk_NARROW.GetBinCenter(i))  
                
    for j in range(len(mkk_histos)):
        NPOINT = 0   
        if(j==0 or j==5):
            for i in range(len(ratios[0])): 
                graphs[j].SetPoint(NPOINT,hmkk_WIDE.GetBinCenter(i),ratios[j][i])
                NPOINT += 1
        else:
            for i in range(len(ratios[1])):
                graphs[j].SetPoint(NPOINT,hmkk_NARROW.GetBinCenter(i),ratios[j][i])
                NPOINT += 1

    graphs[1].SetLineColor(kGreen)
    graphs[1].SetMarkerColor(kGreen)
    graphs[2].SetLineColor(kRed)
    graphs[2].SetMarkerColor(kRed)
    graphs[3].SetLineColor(kBlue)
    graphs[3].SetMarkerColor(kBlue)
    graphs[4].SetLineColor(kMagenta)
    graphs[4].SetMarkerColor(kMagenta)
    graphs[5].SetLineColor(kCyan)
    graphs[5].SetMarkerColor(kCyan)

    c = TCanvas()
    graphs[0].GetXaxis().SetTitle("m_{KK} [MeV/c^{2}]")
    graphs[0].GetYaxis().SetTitle("#epsilon(m_{KK})")
    graphs[0].GetXaxis().SetLimits(float(ll)-10,float(ul)+10)
    max_hist = {"2015": 0.1+SWAVE*0.05, "2016": 0.5-SWAVE*0.35, "2017": 0.3-SWAVE*0.15, "2018": 0.3-SWAVE*0.15, "All": 1.-SWAVE*0.85}
    graphs[0].GetHistogram().SetMaximum(max_hist[str(year)])
    graphs[0].Draw()
    graphs[1].Draw("LP")
    graphs[2].Draw("LP")
    graphs[3].Draw("LP")
    graphs[4].Draw("LP")
    graphs[5].Draw("LP")

    #if SWAVE: gPad.SetLogy()

    c.SaveAs(output_dir+"epsmKK_"+str(year)+SWAVE*"_SWave"+".pdf")

    ### To dump: NF with ratios and masses
    functions = []
    for i in range(len(mkk_bins)):
        functions.append(NF(masses[i],ratios[i]))
        cPickle.dump(functions[i],file(output_dir+"eff_hist_"+str(mkk_bins[i][0])+"_"+str(mkk_bins[i][1]),"w"))    


if __name__ == '__main__':
    # parser = argument_parser()
    # args = parser.parse_args()

    input_file1 = "/scratch46/marcos.romero/sidecar14/2016/MC_Bs2JpsiPhi_dG0/v0r5@pTB3_pdfWeight.root"
    input_file2 = "/scratch46/marcos.romero/sidecar14/2016/MC_Bs2JpsiPhi_dG0/v0r5@pTB4_pdfWeight.root"
    input_tree_name_file1 = "DecayTree"
    input_tree_name_file2 = "DecayTree"
    output_dir = "merda.root" 
    mode = "MC_Bs2JpsiPhi"
    year= 2016
    # epsmKK(**vars(args))
    epsmKK(input_file1, input_file2, input_tree_name_file1, input_tree_name_file2, output_dir, mode, year)
