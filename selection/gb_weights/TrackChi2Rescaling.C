#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include "TGraph.h"
#include "TArrayD.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"

// using namespace std;

TH2D* deriveTransformationMatrix( const TH1D& h1in, const TH1D& h1out)
{
  // step 1. normalize the histograms, just in case
  //
  size_t N = h1in.GetNbinsX()+2 ;
  double sumin = h1in.Integral() ;
  TArrayD h1inarray(N) ;
  for(int ibin=0; ibin<N; ++ibin)
    h1inarray[ibin]=h1in.GetBinContent(ibin)/sumin ;
  double sumout = h1out.Integral() ;
  TArrayD h1outarray(N) ;
  for(int ibin=0; ibin<N; ++ibin)
    h1outarray[ibin]=h1out.GetBinContent(ibin)/sumout ;


  TH2D* M = new TH2D("transformation","",
                     h1in.GetNbinsX(),h1in.GetXaxis()->GetXmin(),h1in.GetXaxis()->GetXmax(),
                     h1in.GetNbinsX(),h1in.GetXaxis()->GetXmin(),h1in.GetXaxis()->GetXmax() ) ;

  // we basically do two loops in paralel to match the integrals
  sumout=sumin=0 ;
  double fleft(1) ;
  int jbin(0),ibin(0) ;
  while(ibin < N && jbin<N) {
    double c = h1outarray[ibin] ;
    sumout += c ;
    double d = h1inarray[jbin] ;
    while( sumin + d*fleft <= sumout && jbin < N) {
      sumin += fleft*d ;
      M->SetBinContent( M->GetBin(ibin,jbin), fleft ) ;
      M->SetBinError( M->GetBin(ibin,jbin), 0) ; // FIXME
      fleft = 1 ;
      ++jbin ;
      if( jbin < N )  d = h1inarray[jbin] ;
    }
    double frac = (sumout - sumin) / d ;
    M->SetBinContent( M->GetBin(ibin,jbin), frac ) ;
    M->SetBinError( M->GetBin(ibin,jbin), 0) ;   // FIXME
    sumin += frac*d ;
    fleft = 1-frac ;
    ++ibin ;
  }
  return M ;
}

/*********************************************************/
// transform the histogram h1in such that
//
//   h1out[ibin] = sum_j M_ij h1in[jbin]
//
// where M_ij is the folding matrix encoded in the histogram M.
/*********************************************************/
TH1D* transform( const TH1D& h1in,
                const TH2D& M)
{

  int N = h1in.GetNbinsX()+2 ;
  TH1D* h1out = (TH1D*)(h1in.Clone(TString(h1in.GetName()) + "transformed")) ;
  for(int ibin=0; ibin<N; ++ibin) {
    double sum=0 ;
    double err2=0 ;
    for(int jbin=0; jbin<N; ++jbin) {
      int Mbin = M.GetBin(ibin,jbin) ;
      double w  = M.GetBinContent(Mbin)  ;
      double x  = h1in.GetBinContent(jbin) ;
      sum += w * x ;
      double werr = M.GetBinError(Mbin) ;
      double xerr = h1in.GetBinError(jbin) ;
      err2 += w*w*xerr*xerr + werr*werr*x*x ;
    }
    h1out->SetBinContent(ibin, sum) ;
    h1out->SetBinError(ibin, err2>0 ? std::sqrt(err2) : 0 ) ;
  }
  return h1out ;
}

/*********************************************************/
// We now do something similar, but we derive the 'function' g(x)
// that maps one distribution into another. This is what we need if we
// want to modify the fields in our signal MC ntuple to match the data.
// The function is returned as a lookuptable.
//
// Use:
// - create the lookuptable from the two distributions mch1 and datah1 for your control channel:
//
//   auto transformationfunction = deriveTransformationFunction( mch1, datah1) ;
//
// - now for every event in your signal mc tuple replace the original value of x with
//
//   auto transformed = transformationfunction( original )
//
// If the distribution of original is mch1, then the distribution of
// transformed will match datah1, preserving all correlations
// with other variables.
/*********************************************************/

struct LookupTableFunction : std::vector<double>
{
  LookupTableFunction( size_t N, double xmin, double xmax )
    : std::vector<double>(N), x0{xmin}, dx{(xmax-xmin)/(N-1)} {}
  double x0 ;
  double dx ;
  double get( double x ) {
    const size_t N = size() ;
    const int bin = (x - x0)/dx ;
    if(bin < 0 ) { return front() ; }
    if(bin >= N-1) { return back() ; }
    return at(bin) + (x - dx*bin - x0)/dx * (at(bin+1) - at(bin) ) ;
  }
} ;

LookupTableFunction deriveTransformationFunction(  const TH1D& h1in, const TH1D& h1out )
{
  // A -> simulation -> 'in'
  // B -> data -> 'out'
  // g(x) = F_A^{-1}( F_B(x) )

  // first integrate the distributions h1in and h1out
  const size_t N = h1in.GetNbinsX()+2 ;
  const double xmin = h1in.GetXaxis()->GetXmin() ;
  const double xmax = h1in.GetXaxis()->GetXmax() ;

  TArrayD h1inarray(N) ;
  {
    double sum{0} ;
    for(int ibin=0; ibin<N; ++ibin) {
      sum += h1in.GetBinContent(ibin) ;
      h1inarray[ibin] = sum ;
    }
    for(int ibin=0; ibin<N; ++ibin) h1inarray[ibin]/= sum ;
  }
  TArrayD h1outarray(N) ;
  {
    double sum{0} ;
    for(int ibin=0; ibin<N; ++ibin) {
      sum += h1out.GetBinContent(ibin) ;
      h1outarray[ibin] = sum ;
    }
    for(int ibin=0; ibin<N; ++ibin) h1outarray[ibin]/= sum ;
  }

  // choose a large number of bins (it doesn't matter, but we don't want to be limited by bin size)
  // we make it such that x = lower edge of a bin
  LookupTableFunction result(N*10,xmin,xmax) ;
  int lastBinA{0} ;
  for(size_t i=0; i<result.size(); ++i) {
    // first compute x
    const double x = result.x0 + i*result.dx ;
    // now compute y = F_B(x)
    const int binB = h1in.GetXaxis()->FindBin( x ) ;
    double y{0} ;
    if( binB<=0 )    y = 0 ;
    else if(binB>=N) y = 1 ;
    else {
      // watch out: the bin includes the integral of this bin
      const double frac = (x - h1in.GetXaxis()->GetBinLowEdge(binB)) / h1in.GetXaxis()->GetBinWidth(binB) ;
      y = frac * h1inarray[binB] + (1-frac) * h1inarray[binB-1] ;
    }
    // now compute z = F_A^{-1}(y). this is even slower.
    double z{0} ;
    int binA = lastBinA ;
    for(; binA<N; ++binA )
      if( h1outarray[binA] >= y ) break ;
    if( binA <=0 ) z = xmin ;
    else if( binA >=N ) z = xmax ;
    else {
      const double bincontents = h1outarray[binA] - h1outarray[binA-1] ;
      const double frac = bincontents>0 ? (y - h1outarray[binA-1]) / bincontents : 1 ;
      z = h1out.GetXaxis()->GetBinLowEdge(binA) + frac * h1out.GetXaxis()->GetBinWidth(binA) ;
    }
    result[i] = z ;
    lastBinA = binA ;
  }
  return result ;
}


void TrackChi2Rescaling(std::vector<std::string> vars, std::string weightOut, std::string fileNameIn, std::string fileNameOut, std::string fileNameNew, std::string treeNameIn, std::string treeNameOut, std::string plot_dir)
{
  TFile *fileIn = new TFile(fileNameIn.c_str());
  TTree *treeIn = (TTree*)fileIn->Get(treeNameIn.c_str());

  TFile *fileOut = new TFile(fileNameOut.c_str());
  TTree *treeOut = (TTree*)fileOut->Get(treeNameOut.c_str());

  TFile *newfile = new TFile(fileNameNew.c_str(), "RECREATE");
  auto newtree = treeIn->CloneTree();
  newfile->Write();

  fileIn->Close();

  std::vector<double> varBranches(vars.size());
  std::vector<double> newvarBranches(vars.size());
//   std::vector<TBranch> newBranches(vars.size());

  for (int i=0; i<static_cast<int>(vars.size()); i++)
  {
    TCanvas *canvas = new TCanvas("canvas", "",38,102,1024,768);
    canvas->cd();

    std::string var_name = std::string(vars[i]);

    TH1D *var_IN  = new TH1D((var_name+"_IN").c_str(),  (var_name+"_IN").c_str(), 80, 0, 4.);
    TH1D *var_OUT = new TH1D((var_name+"_OUT").c_str(), (var_name+"_OUT").c_str(), 80, 0., 4.);
      
    newtree-> Draw((var_name+">>"+var_name+"_IN").c_str());
    treeOut->Draw((var_name+">>"+var_name+"_OUT").c_str(), weightOut.c_str());

    LookupTableFunction result = deriveTransformationFunction( *var_IN, *var_OUT );

    TH1D *var_RES = new TH1D((vars[i]+"_RES").c_str(), (vars[i]+"_RES").c_str(), 80, 0, 4.);

    newtree->SetBranchAddress (var_name.c_str(),&varBranches[i] );
    TBranch *branch = newtree->Branch((var_name+"_scaled").c_str(),&newvarBranches[i],(var_name+"/D").c_str());

    for (int j=0; j<newtree->GetEntries(); j++)
      {
        newtree->GetEntry(j);
        newvarBranches[i] = result.get(varBranches[i]);
        var_RES->Fill(newvarBranches[i]);
        branch->Fill();
      }

    std::cout << var_name << std::endl;
    std::cout << weightOut.c_str() << std::endl;
    std::cout << var_RES->Integral() << std::endl;

    var_RES->SetLineColor(kGreen+1);
    var_RES->Scale(1./var_RES->Integral());

    var_IN->SetLineColor(kRed+1);
    var_IN->Scale(1./var_IN->Integral());

    var_OUT->SetLineColor(kBlack);
    var_OUT->Scale(1./var_OUT->Integral());

    var_RES->Draw("E1");
    var_IN->Draw("E1 SAME");
    var_OUT->Draw("E1 SAME");

    canvas->SaveAs((plot_dir+"/"+var_name+"_rescaled.pdf").c_str());
    newtree->Write();
  }

  //TODO output root file

}

//   fileIn->Close();
//   std::vector<std::string> vars { "hplus_TRACK_CHI2NDOF", "hminus_TRACK_CHI2NDOF", "muplus_TRACK_CHI2NDOF", "muminus_TRACK_CHI2NDOF" };// TODO automate?
//     TFile *fileIn = new TFile("/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v1r0/PID/MC_Bs2JpsiPhi/MC_Bs2JpsiPhi_2018_str34_sim09g_dst_selected_pidcorrected.root");
//     TFile *fileOut = new TFile("/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/v1r0/Bs2JpsiPhi/2018/Bs2JpsiPhi_2018_selected_bdt_sw_v1r0.root");
