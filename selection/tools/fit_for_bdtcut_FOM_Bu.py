import os
import json
import argparse
import yaml
import numpy 
import cppyy
import math
from array import array

import multiprocessing

from ROOT import AddressOf
from ROOT import kDashed, kRed, kGreen, kBlue, kBlack, kTRUE, kFALSE, gPad, TGraph, TArrow
from ROOT import TMath, TAxis, TH1, TLatex, TROOT, TSystem, TCanvas, TFile, TTree, TObject, gROOT
from ROOT import ROOT, RDataFrame, vector, gInterpreter
from ROOT import RooFit, RooAbsData, RooArgSet, RooArgList, RooAbsDataStore, RooAddModel, RooAddPdf, RooCBShape, RooConstVar, RooDataSet, RooExponential, RooFFTConvPdf, RooGaussian, RooGlobalFunc, RooPlot, RooPolynomial, RooProdPdf, RooRealVar, RooVoigtian, RooStats, RooIpatia2
from constants import *
gROOT.ProcessLine('struct mysWeight{Double_t sWValue;};')
gROOT.ProcessLine(".x ./styles/lhcbStyle.C")
from ROOT import mysWeight

from P2VV.RooFitWrappers import (
    RealVar,
    Pdf,
    Component,
    buildPdf,
    RooObject,
    Category,
    )



def read_params(params_to_fix_file):
    with open(params_to_fix_file, 'r') as stream:
        return json.load(stream)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='Name of the input file')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the input tree')
    parser.add_argument('--input-branch', default='B_ConstJpsi_M_1', help='Name of the branch to be fitted')
    parser.add_argument('--bdt-branch', default='bdtg3', help='Name of the BDT branch to be cut')
    parser.add_argument('--plot-dir', help='Location in which to save the plots')
    parser.add_argument('--output-file', default='FOM_Bu2JpsiKplus_2016.pdf', help='name of output FOM plot')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--year', required=True, help='Year of data taking')
    parser.add_argument('--bdtcut', default='-1.0', help='pre-bdtg3 cut')
    parser.add_argument('--cuts', default='1', help='pre-selection cut')
    parser.add_argument('--params-to-fix-file', help='Yaml with dict of parameter names and values')
    parser.add_argument('--params-to-fix-list', nargs='+', help='Yaml with list of parameter names to be fixed from params-to-fix-file')
    return parser

def fit_branch(sigTuple, input_branch, plot_dir, mode, year, bdtcut, params_to_fix_file,params_to_fix_list):
    # ROOT.EnableImplicitMT()
    
    if('Bu' in mode):

        numEnt = float(sigTuple.GetEntries())
        print (numEnt)
        
        nSig=1.5392e+06
        if(nSig > numEnt):
            nSig = numEnt-10
    
        b_fit = [numEnt-nSig, 0., numEnt] # [expected, min, max] bkg evts
        s_fit = [nSig, 0., numEnt] # [expected, min, max] signal evts

        print("Background limits (min, mean, max): ", b_fit[1], b_fit[0], b_fit[2])
        print("Signal limits     (min, mean, max): ", s_fit[1], s_fit[0], s_fit[2])
        
        title = "#it{m}(#it{J}/#psi #it{K}^{+}) [MeV/#it{c}^{2}]"
    
        lower_cut = math.floor(sigTuple.GetMinimum(input_branch))
        upper_cut = math.ceil(sigTuple.GetMaximum(input_branch))
        mass_range = (5210., 5350.)
        bs_mass = RooRealVar(input_branch, title, mass_range[0], mass_range[1])
        fitArgs = RooArgSet(bs_mass)
        data    = RooDataSet(input_branch, input_branch, sigTuple, fitArgs)
        print(data)
        
        sigma_value = 11.2
        sigma_min = 3.
        sigma_max = 20.
        m_sig_lambda_value =-2.65 
        m_sig_lambda_min = -6.
        m_sig_lambda_max = 2.
        alpha1_value = 1.99
        alpha1_min = 1.
        alpha1_max = 10.
        alpha2_value = 2.23 
        alpha2_min = 1.
        alpha2_max = 10.
        n1_value = 2.83
        n1_min = 1.
        n1_max = 10.
        n2_value = 3.14
        n2_min = 1.
        n2_max = 10.
        gamma_value = -0.001
        gamma_min = -1.
        gamma_max = 1.
        
        mean = RooRealVar('mean','mean', 5279.9, 5275, 5285)
        sigma = RooRealVar('sigma','sigma', sigma_value, sigma_min, sigma_max) 
        m_sig_lambda_noTrigCat = RooRealVar('m_sig_lambda_noTrigCat', 'B Mass resolution lambda', m_sig_lambda_value)
        m_sig_zeta = RooRealVar('m_sig_zeta_', 'B Mass resolution zeta', 0)
        m_sig_beta = RooRealVar('m_sig_beta_', 'B Mass resolution beta', 0)
        alpha1_noTrigCat = RooRealVar('alpha1_noTrigCat', 'alpha1_noTrigCat', alpha1_value)
        alpha2_noTrigCat = RooRealVar('alpha2_noTrigCat', 'alpha2_noTrigCat', alpha2_value)
        n1_noTrigCat = RooRealVar('n1_noTrigCat','n1_noTrigCat', n1_value)
        n2_noTrigCat = RooRealVar('n2_noTrigCat', 'n2_noTrigCat',n2_value)
        pdf_s = RooIpatia2('pdf_s', 'pdf_s',bs_mass, m_sig_lambda_noTrigCat, m_sig_zeta, m_sig_beta, sigma, mean, alpha1_noTrigCat, n1_noTrigCat, alpha2_noTrigCat, n2_noTrigCat)
        gamma = RooRealVar('gamma', 'gamma',gamma_value, gamma_min,gamma_max)
        pdf_b = RooExponential('pdf_b','pdf_b',bs_mass, gamma)
        nsig = RooRealVar("nsig", "number of signal events", s_fit[0], s_fit[1], s_fit[2], "")
        nbkg = RooRealVar("nbkg", "number of backgnd events", b_fit[0], b_fit[1], b_fit[2], "")
        model = RooAddPdf("bu pdf", "bu pdf", RooArgList(pdf_s, pdf_b), RooArgList(nsig, nbkg))


        pdf_pars = model.getParameters(data)
        params_to_fix = read_params(params_to_fix_file) if params_to_fix_file else None
        cat_params = params_to_fix['noTrigCat']
        for par in params_to_fix_list:
            mc_param = [p for p in cat_params if par in p['Name']][0]
            par_name=mc_param['Name']
            #print(par_name)
            #print(pdf_pars.find(par_name))
            pdf_pars.find(par_name).setVal(mc_param['Value'])
            pdf_pars.find(par_name).setConstant(True)
            print('Setting parameter', par_name, 'to constant with value', mc_param['Value'])


        n_CPUs = multiprocessing.cpu_count()

        fitresult = model.fitTo(data, RooFit.Save(kTRUE), RooFit.Minos(kFALSE), RooFit.NumCPU(max(n_CPUs-2,1)))
        fitresult.Print('v')
        frame1 = bs_mass.frame(RooFit.Title("Bs Mass"), RooFit.Bins(140), RooFit.Range(lower_cut, upper_cut))
        frame2 = bs_mass.frame(RooFit.Bins(140), RooFit.Title(""));

        data.plotOn(frame1, RooFit.Name("dataHist"), RooFit.MarkerSize(0.8), RooFit.DataError(RooAbsData.SumW2))
        model.plotOn(frame1, RooFit.LineColor(kBlue))
        model.plotOn(frame1, RooFit.Components("pdf_s"), RooFit.LineColor(6))
        model.plotOn(frame1, RooFit.Components("pdf_b"), RooFit.LineColor(3), RooFit.LineStyle(4))
        
        data.plotOn(frame2, RooFit.Name("dataHist"), RooFit.MarkerSize(0.8), RooFit.DataError(RooAbsData.SumW2))
        model.plotOn(frame2, RooFit.LineColor(kBlue))
        
        name_fit = "#splitline{#splitline{#splitline{N_{sig} = " + str(round(nsig.getVal())) + " #pm " + str(round(nsig.getError())) + "}{#mu = " + str(round(mean.getVal(),2)) + " #pm " + str(round(mean.getError(), 2)) + " MeV/c^{2}}}{#splitline{#sigma = " + str(round(sigma.getVal(), 2)) + " #pm " + str(round(sigma.getError(), 2)) + "MeV/c^{2}}{S/B = " + str(round(nsig.getVal()/nbkg.getVal(), 3)) + " #pm " + str(round(TMath.Sqrt(pow(nsig.getError()/nbkg.getVal(), 2) + pow(nsig.getVal()/pow(nbkg.getVal(), 2) * nbkg.getError(), 2)), 3)) + "}"



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
    sWBranch = new_tree.Branch('sw', AddressOf(sWeight_struct, 'sWValue'), 'sw/D')
    
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

def fit_for_bdtcut_fom(input_file, input_tree_name, input_branch, bdt_branch, plot_dir, output_file,  mode, year, bdtcut, cuts, params_to_fix_file,params_to_fix_list):

    xarray=array('d',[float(bdtcut)])
    yarray=array('d',[0.0])
    sigFile = TFile.Open(input_file)
    sigTuple = sigFile.Get(input_tree_name)
    sigTuple.SetBranchStatus("*",0)
    sigTuple.SetBranchStatus("bdtg3", 1)
    sigTuple.SetBranchStatus(input_branch, 1)
 
    gROOT.cd()
    Neff_max=0.0
    bdtcut_val=-1.0
    nbins=int((1-float(bdtcut))/0.02)
    for i in range(1, nbins):
        cutval=float(bdtcut)+0.02*i
        cut=cuts+"&&" +bdt_branch+">"+format(cutval, '.2')
        if 'Bu' in mode:
            mass_range = (5210., 5350.)
            Cut_mass='B_ConstJpsi_M_1>{0}&&B_ConstJpsi_M_1<{1}'.format(mass_range[0], mass_range[1])
            cut = cut+'&&'+Cut_mass
            mva_tree = sigTuple.CopyTree(cut)
        else:
            mva_tree = sigTuple.CopyTree(cut)
        print (cut)
        
        Neff = fit_branch(mva_tree, input_branch, plot_dir, mode, year, format(cutval, '.2'), params_to_fix_file,params_to_fix_list) 
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
    canvas.SaveAs(output_file)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    fit_for_bdtcut_fom(**vars(args))
