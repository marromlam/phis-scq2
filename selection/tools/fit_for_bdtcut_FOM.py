import os
import json
import argparse
import yaml
import numpy 
import cppyy
import math
from array import array

import multiprocessing

from ROOT import addressof
from ROOT import kDashed, kRed, kGreen, kBlue, kBlack, kTRUE, kFALSE, gPad, TGraph, TArrow
from ROOT import TMath, TAxis, TH1, TLatex, TROOT, TSystem, TCanvas, TFile, TTree, TObject, gROOT
from ROOT import ROOT, RDataFrame, vector, gInterpreter
from ROOT import RooFit, RooAbsData, RooArgSet, RooArgList, RooAbsDataStore, RooAddModel, RooAddPdf, RooCBShape, RooConstVar, RooDataSet, RooExponential, RooFFTConvPdf, RooGaussian, RooGlobalFunc, RooPlot, RooPolynomial, RooProdPdf, RooRealVar, RooVoigtian, RooStats
from constants import *
gROOT.ProcessLine('struct mysWeight{Double_t sWValue;};')
# gROOT.ProcessLine(".x ./styles/lhcbStyle.C")
from ROOT import mysWeight

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='Name of the input file')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the input tree')
    parser.add_argument('--input-branch', default='B_ConstJpsi_M_1', help='Name of the branch to be fitted')
    parser.add_argument('--bdt-branch', default='bdtg3', help='Name of the BDT branch to be cut')
    parser.add_argument('--plot-dir', help='Location in which to save the plots')
    parser.add_argument('--output-file', default='FOM_Bs2JpsiPhi_2016.pdf', help='name of output FOM plot')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--year', required=True, help='Year of data taking')
    parser.add_argument('--bdtcut', default='-1.0', help='pre-bdtg3 cut')
    parser.add_argument('--cuts', default='1', help='pre-selection cut')
    return parser

def fit_branch(sigTuple, input_branch, plot_dir, mode, year, bdtcut):
    # ROOT.EnableImplicitMT()
    
    if('Bs' in mode):
        mean_initial = BS_MASS
        s_fr = 0.061
    elif('Bd' in mode):
        mean_initial = BD_MASS
        s_fr = 0.25
    elif('Bu' in mode):
        mean_initial = BU_MASS
        s_fr = 0.2
    else:
        print("ERROR: Invalid mode", mode)
        return
    
    #sigTuple = sigFile.Get(input_tree_name)

    numEnt = float(sigTuple.GetEntries())
    print (numEnt)
    
    nSig=86270
    if(nSig > numEnt):
        nSig = numEnt-10
    
    b_fit = [numEnt-nSig, 0., numEnt] # [expected, min, max] bkg evts
    s_fit = [nSig, 0., numEnt] # [expected, min, max] signal evts

    print("Background limits (min, mean, max): ", b_fit[1], b_fit[0], b_fit[2])
    print("Signal limits     (min, mean, max): ", s_fit[1], s_fit[0], s_fit[2])

    title = "#it{m}(#it{J}/#psi #it{K}^{+}#it{K}^{-}) [MeV/#it{c}^{2}]"

    lower_cut = math.floor(sigTuple.GetMinimum(input_branch))
    upper_cut = math.ceil(sigTuple.GetMaximum(input_branch))
    
    bs_mass = RooRealVar(input_branch, title, lower_cut, upper_cut)
    fitArgs = RooArgSet(bs_mass)
    data    = RooDataSet(input_branch, input_branch, sigTuple, fitArgs)
    
    sigma_ = 50.0
        
    # Bs mass info
    mean   = RooRealVar("bs #mu", "mean of bs", mean_initial, lower_cut, upper_cut, "")
    sigma1 = RooRealVar("#sigma 1", "width of g1", 5.25, 0., sigma_, "")
    sigma2 = RooRealVar("#sigma 2", "width of g2", 8.95, 0., sigma_, "")
    gauss1 = RooGaussian("gauss1", "gauss1", bs_mass, mean, sigma1)
    gauss2 = RooGaussian("gauss2", "gauss2", bs_mass, mean, sigma2)
    frac   = RooRealVar("frac", "frac", 0.5, 0., 1.)
    gauss  = RooAddModel("gauss", "gauss", RooArgList(gauss1, gauss2), RooArgList(frac))
    # bkg flat
    slope_bs = RooRealVar("slope bs", "slope bs", -0.01, -2.0, 2.0)
    bs_bkg = RooExponential("bs_bkg", "bs bkg", bs_mass, slope_bs)
    # number of events
    nsig = RooRealVar("nsig", "number of signal events", s_fit[0], s_fit[1], s_fit[2], "")
    nbkg = RooRealVar("nbkg", "number of backgnd events", b_fit[0], b_fit[1], b_fit[2], "")
    # signal bkg pdfs
    model = RooAddPdf("bs pdf", "bs pdf", RooArgList(gauss, bs_bkg), RooArgList(nsig, nbkg))
    
    n_CPUs = multiprocessing.cpu_count()

    fitresult = model.fitTo(data, RooFit.Save(kTRUE), RooFit.Minos(kFALSE), RooFit.NumCPU(max(n_CPUs-2,1)))

    frame1 = bs_mass.frame(RooFit.Title("Bs Mass"), RooFit.Bins(140), RooFit.Range(lower_cut, upper_cut))
    frame2 = bs_mass.frame(RooFit.Bins(140), RooFit.Title(""));

    data.plotOn(frame1, RooFit.Name("dataHist"), RooFit.MarkerSize(0.8), RooFit.DataError(RooAbsData.SumW2))
    model.plotOn(frame1, RooFit.LineColor(kBlue))
    model.plotOn(frame1, RooFit.Components("gauss"), RooFit.LineColor(6))
    model.plotOn(frame1, RooFit.Components("bs_bkg"), RooFit.LineColor(3), RooFit.LineStyle(4))

    data.plotOn(frame2, RooFit.Name("dataHist"), RooFit.MarkerSize(0.8), RooFit.DataError(RooAbsData.SumW2))
    model.plotOn(frame2, RooFit.LineColor(kBlue))
    
    name_fit = "#splitline{#splitline{#splitline{N_{sig} = " + str(round(nsig.getVal())) + " #pm " + str(round(nsig.getError())) + "}{#mu = " + str(round(mean.getVal(),2)) + " #pm " + str(round(mean.getError(), 2)) + " MeV/c^{2}}}{#splitline{#sigma_{1} = " + str(round(sigma1.getVal(), 2)) + " #pm " + str(round(sigma1.getError(), 2)) + "MeV/c^{2}}{#sigma_{2} = " + str(round(sigma2.getVal(), 2)) + " #pm " + str(round(sigma2.getError(), 2)) + " MeV/c^{2}}}}{S/B = " + str(round(nsig.getVal()/nbkg.getVal(), 3)) + " #pm " + str(round(TMath.Sqrt(pow(nsig.getError()/nbkg.getVal(), 2) + pow(nsig.getVal()/pow(nbkg.getVal(), 2) * nbkg.getError(), 2)), 3)) + "}"


    myTex4 = TLatex(0.2, 0.70, name_fit)
    myTex4.SetTextFont(132)
    myTex4.SetTextSize(0.05)
    myTex4.SetLineWidth(2)
    myTex4.SetNDC()
    frame1.addObject(myTex4)

    gROOT.SetBatch(1)

    c1_Data = TCanvas("c1_Data", "", 700, 600)
    c1_Data.Divide(1, 2, 0, 0, 0)
    c1_Data.cd(2)
    gPad.SetTopMargin(0)
    gPad.SetLeftMargin(0.15)
    gPad.SetPad(0.02, 0.02, 0.98, 0.77)
    frame1.SetTitle("")
    frame1.SetMaximum(frame1.GetMaximum()*1.4)
    frame1.Draw()

    c1_Data.cd(1)
    gPad.SetTopMargin(0)
    gPad.SetLeftMargin(0.15)
    gPad.SetPad(0.02, 0.76, 0.98, 0.97)
    hpull1   = frame2.pullHist()
    mframeh1 = bs_mass.frame(RooFit.Title("Pull distribution"))
    hpull1.SetFillColor(15)
    hpull1.SetFillStyle(3144)
    mframeh1.addPlotable(hpull1, "L3")
    mframeh1.GetYaxis().SetNdivisions(505)
    mframeh1.GetYaxis().SetLabelSize(0.20)
    mframeh1.SetMinimum(-5.0)
    mframeh1.SetMaximum(5.0)
    mframeh1.Draw()

    c1_Data.SaveAs(plot_dir+"/"+mode+"_"+year+"_"+input_branch+bdtcut+".png")

    # sweighting
    floatPars = fitresult.floatParsFinal()
    for i in range(floatPars.getSize()):
        party = RooRealVar(floatPars.at(i))
        party.setConstant(kTRUE)
        
    nsig.setConstant(kFALSE)
    nbkg.setConstant(kFALSE)
    sData = RooStats.SPlot("sData", "An SPlot", data, model, RooArgList(nsig, nbkg))
    
    print ("bdtg3> "+ bdtcut+ " sig "+ str(nsig.getVal())+" bkg "+ str(nbkg.getVal()))
    print("Check SWeights:")
    print("Yield of signal is", nsig.getVal(), ".  From sWeights it is", sData.GetYieldFromSWeight("nsig"))
    print("S/sigma(S) is ", nsig.getVal()/nsig.getError(), ".  From sWeights it is", sData.GetYieldFromSWeight("nsig"))

    newFile = TFile.Open(plot_dir+"/"+mode+"_"+year+"_"+bdtcut+"_sw.root", "RECREATE")
    outFilename=plot_dir+"/"+mode+"_"+year+"_"+bdtcut+"_sw.root"
    new_tree = sigTuple.CloneTree()
    
    sWeight_struct = mysWeight()
    sWBranch = new_tree.Branch('sw', addressof(sWeight_struct, 'sWValue'), 'sw/D')
    
    for entry in range(int(new_tree.GetEntries())):
        new_tree.GetEntry(entry)
        sWeight_struct.sWValue = sData.GetSWeight(entry,"nsig")
        sWBranch.Fill()
		                
    new_tree.AutoSave('saveself')
    #newFile.Close()

    df = RDataFrame("DecayTree", outFilename)
    df=df.Define("sw2", "sw*sw;")
    sum_sw=df.Sum("sw").GetValue()
    sum_sw2=df.Sum("sw2").GetValue()
    FOM = sum_sw*sum_sw/sum_sw2
    print("FOM: " + str(FOM))
    
    return FOM 

def fit_for_bdtcut_fom(input_file, input_tree_name, input_branch, bdt_branch, plot_dir, output_file,  mode, year, bdtcut, cuts):

    xarray=array('d',[float(bdtcut)])
    yarray=array('d',[0.0])
    sigFile = TFile.Open(input_file)
    sigTuple = sigFile.Get(input_tree_name)
    gROOT.cd()
    Neff_max=0.0
    bdtcut_val=-1.0
    nbins=int((1-float(bdtcut))/0.02)
    for i in range(1, nbins):
        cutval=float(bdtcut)+0.02*i
        cut=cuts+"&&" +bdt_branch+">"+format(cutval, '.2')
        mva_tree = sigTuple.CopyTree(cut)
        print (cut)
        
        Neff = fit_branch(mva_tree, input_branch, plot_dir, mode, year, format(cutval, '.2')) 
        xarray.append(cutval)
        yarray.append(Neff)
        if (Neff>Neff_max):
            Neff_max=Neff
            bdtcut_val=cutval
        mva_tree.Reset()
    
    num=int(len(xarray))
    gr=TGraph(num, xarray, yarray)
    canvas = TCanvas("FOM_bdtg3", "", 800, 600)
    gr.Draw()
    gr.SetMarkerStyle(20)
    gr.GetXaxis().SetTitle("bdtg3")
    gr.GetYaxis().SetTitle("(#sum_{i}w_{i})^{2}/#sum_{i}(w_{i}^{2})")
    ar=TArrow(bdtcut_val, Neff_max-30000, bdtcut_val, Neff_max-1000, 0.03, ">")
    ar.SetLineStyle(7)
    ar.SetLineColor(4)
    ar.Draw()
    print ("Best FOM: "+str(Neff_max)+ " at bdtg3> "+format(bdtcut_val, '.2'))
    with open(output_file, 'w') as f:
        f.write(f"{bdtcut_val:.2f}")
    canvas.SaveAs(f"{plot_dir}/fom.pdf")

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    fit_for_bdtcut_fom(**vars(args))
