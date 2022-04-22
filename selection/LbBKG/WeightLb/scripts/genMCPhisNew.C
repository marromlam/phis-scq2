#include "TCut.h"
#include "RooTDalitz/RooDalitzAmplitude.h"
#include "TList.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "TSystem.h"
#include "RooDataSet.h"
#include "TFile.h"
#include "RooFitResult.h"
#include "TDatime.h"
#include "RooRealConstant.h"
#include "RooMinuit.h"
#include "RooAddition.h"
#include "TLorentzVector.h"
#include "TBranch.h"
//#include "RooTDalitz/RooSUMVar.h"

using namespace RooFit;

double xmkp;
double xcostheta;
double xcostheta1;
double xcostheta2;
double xphi1;
double xphi2;
double xcosthetaB;
double xcosthetaZ;
double xcosthetaPsi;
double xphiZ;
double xphiPsi;
double xphiMu;
double xcosthetap;
double xalphaMu;
double xmjpsip;

inline double calwxll(double mX)
{
    double mk(0.493), mpi(0.139),mp(0.938);
    double m_thresh(0.0);
    double f(0.0);
    m_thresh = mk + mp - 0.005;
    // add a steeply falling function off the threshold at 600 MeV
    f += 150.0 * exp(-40.0 * (mX-m_thresh));
    f += 900.0 * exp( -pow((mX -1.52)/0.015,2));
    f += 100.0 * exp( -pow((mX -1.6)/0.1,2));
    f += 100.0 * exp( -pow((mX -1.8)/0.2,2));
    // add a slowing falling exponential function
    f += 80.0 *exp(-0.6 * (mX-m_thresh));
    f /= 1043.1;
    return 1./f;
}

inline bool exists(const std::string& name) {
    return ( access( name.c_str(), F_OK ) != -1 );
    //  struct stat buffer;
    //  return (stat (name.c_str(), &buffer) == 0);
}

void helicityJpsiLam(TLorentzVector *Pmu1, TLorentzVector *Pmu2, TLorentzVector *PJ,
        TLorentzVector *Pproton, TLorentzVector *Pkaon)
{
    TLorentzVector PJpsi = *Pmu1 + *Pmu2;
    TLorentzVector PLam = *Pproton+ *Pkaon;
    TLorentzVector PB = PJpsi+PLam;
    TLorentzVector PJpsi_MassCon = *PJ;
    //  PJpsi_MassCon.SetVectM(PJpsi.Vect(), 3096.916);
    xmkp = PLam.Mag()/1000.;
    xmjpsip = (PJpsi_MassCon + *Pproton).Mag()/1000.;
    TVector3 PBunit = PB.Vect().Unit();
    TVector3 Pbeam(0.0,0.0,1.0);
    //  TVector3  nvec = PBunit.Cross(TVector3(0,0,1));

    TVector3 boosttoparent = -PB.BoostVector();

    //  std::cout << "before " <<std::endl;
    //  PLam.Print();
    Pmu1->Boost(boosttoparent);
    Pmu2->Boost(boosttoparent);
    Pproton->Boost(boosttoparent);
    Pkaon->Boost(boosttoparent);
    PJpsi.Boost(boosttoparent);
    PLam.Boost(boosttoparent);

    TVector3 LUnit = PLam.Vect().Unit();
    TVector3 psiUnit = PJpsi.Vect().Unit();

    xcostheta = PBunit.Dot(LUnit);
    //  phi = 0;
    //Lam rest frame
    boosttoparent = -PLam.BoostVector();
    Pkaon->Boost(boosttoparent);

    TVector3 Unit = Pkaon->Vect().Unit();
    xcostheta1 = -(psiUnit).Dot(Unit);

    TVector3 aboost = -(PB.Vect() - ((PB.Vect()).Dot(LUnit))*LUnit).Unit();
    double cosphi = aboost.Dot(Unit);
    double sinphi = (((-psiUnit).Cross(aboost)).Unit()).Dot(Unit);
    xphi1 = atan2(sinphi,cosphi);

    //Jpsi rest frame
    boosttoparent = -PJpsi.BoostVector();
    Pmu2->Boost(boosttoparent);
    Unit = Pmu2->Vect().Unit();
    xcostheta2 = (psiUnit).Dot(Unit);

    cosphi = aboost.Dot(Unit);
    sinphi = (((psiUnit).Cross(aboost)).Unit()).Dot(Unit);
    xphi2 = atan2(sinphi,cosphi);

}

void helicityZK(TLorentzVector *Pmu1, TLorentzVector *Pmu2,
        TLorentzVector *Pproton, TLorentzVector *Pkaon)
{
    TLorentzVector PJpsi = *Pmu1 + *Pmu2;
    TLorentzVector PLam = *Pproton+ *Pkaon;
    TLorentzVector PB = PJpsi+PLam;
    TLorentzVector PZ = PJpsi + *Pproton;
    TVector3 PBunit = PB.Vect().Unit();
    TVector3 Pbeam(0.0,0.0,1.0);
    //  TVector3  nvec = PBunit.Cross(TVector3(0,0,1));

    TVector3 boosttoparent = -PB.BoostVector();

    //  std::cout << "before " <<std::endl;
    //  PLam.Print();
    Pmu1->Boost(boosttoparent);
    Pmu2->Boost(boosttoparent);
    Pproton->Boost(boosttoparent);
    Pkaon->Boost(boosttoparent);
    PJpsi.Boost(boosttoparent);
    PLam.Boost(boosttoparent);
    PZ.Boost(boosttoparent);

    TVector3 LUnit = PLam.Vect().Unit();
    TVector3 psiUnit = PJpsi.Vect().Unit();
    TVector3 ZUnit = PZ.Vect().Unit();

    xcosthetaB = PBunit.Dot(ZUnit);
    TVector3 aboost = (PLam.Vect()-(PLam.Vect().Dot(PBunit))*PBunit).Unit();
    double cosphi = aboost.Dot(ZUnit);
    double sinphi = ((PBunit.Cross(aboost)).Unit()).Dot(ZUnit);
    xphiZ = atan2(sinphi,cosphi);

    //Z rest frame
    boosttoparent = -PZ.BoostVector();
    PJpsi.Boost(boosttoparent);
    Pmu1->Boost(boosttoparent);
    Pmu2->Boost(boosttoparent);
    Pproton->Boost(boosttoparent);
    Pkaon->Boost(boosttoparent);

    TVector3 Unit = PJpsi.Vect().Unit();
    xcosthetaZ = -(Pkaon->Vect().Unit()).Dot(Unit);

    aboost = -(PB.Vect() - ((PB.Vect()).Dot(ZUnit))*ZUnit).Unit();
    cosphi = aboost.Dot(Unit);
    sinphi = (((-Pkaon->Vect().Unit()).Cross(aboost)).Unit()).Dot(Unit);
    xphiPsi = atan2(sinphi,cosphi);

    //Jpsi rest frame
    psiUnit = PJpsi.Vect().Unit();
    boosttoparent = -PJpsi.BoostVector();
    Pmu2->Boost(boosttoparent);
    Unit = Pmu2->Vect().Unit();
    xcosthetaPsi = (psiUnit).Dot(Unit);
    aboost = -(-Pkaon->Vect() + ((Pkaon->Vect()).Dot(psiUnit))*(psiUnit)).Unit();
    cosphi = aboost.Dot(Unit);
    sinphi = (((psiUnit).Cross(aboost)).Unit()).Dot(Unit);
    xphiMu = atan2(sinphi,cosphi);
}


void helicityTwoFrame(TLorentzVector *Pmu1, TLorentzVector *Pmu2,
        TLorentzVector *Pproton, TLorentzVector *Pkaon)
{
    TLorentzVector PJpsi = *Pmu1 + *Pmu2;
    TLorentzVector PLam = *Pproton+ *Pkaon;
    TLorentzVector PB = PJpsi+PLam;
    TLorentzVector PZ = PJpsi + *Pproton;
    TVector3 PBunit = PB.Vect().Unit();
    TVector3 Pbeam(0.0,0.0,1.0);
    //  TVector3  nvec = PBunit.Cross(TVector3(0,0,1));

    TVector3 boosttoparent = -PB.BoostVector();

    //  std::cout << "before " <<std::endl;
    //  PLam.Print();
    Pmu1->Boost(boosttoparent);
    Pmu2->Boost(boosttoparent);
    Pproton->Boost(boosttoparent);
    Pkaon->Boost(boosttoparent);
    PJpsi.Boost(boosttoparent);
    PLam.Boost(boosttoparent);
    PZ.Boost(boosttoparent);


    TLorentzVector PKaon_p = *Pkaon;
    TLorentzVector PJpsi_p = PJpsi;

    boosttoparent = -Pproton->BoostVector();
    PKaon_p.Boost(boosttoparent);
    PJpsi_p.Boost(boosttoparent);
    /*
        boosttoparent = - PLam.BoostVector();
        TLorentzVector PKaon_p = *Pkaon;
        TLorentzVector Pproton_p = *Pproton;
        PKaon_p.Boost(boosttoparent);
        Pproton_p.Boost(boosttoparent);
        boosttoparent = - Pproton_p.BoostVector();
        PKaon_p.Boost(boosttoparent);

        boosttoparent = -PZ.BoostVector();
        TLorentzVector PJpsi_p = PJpsi;
        Pproton_p = *Pproton;
        PJpsi_p.Boost(boosttoparent);
        Pproton_p.Boost(boosttoparent);
        boosttoparent = - Pproton_p.BoostVector();
        PJpsi_p.Boost(boosttoparent);
        */
    xcosthetap = (PKaon_p.Vect().Unit()).Dot(PJpsi_p.Vect().Unit());

    /*
         TLorentzVector PKaon_Z = *Pkaon;
         TLorentzVector PJpsi_Z = PJpsi;
         TLorentzVector PKaon_L = *Pkaon;
         TLorentzVector PJpsi_L = PJpsi;
         boosttoparent = -PZ.BoostVector();
         PKaon_Z.Boost(boosttoparent);
         PJpsi_Z.Boost(boosttoparent);
         boosttoparent = -PLam.BoostVector();
         PKaon_L.Boost(boosttoparent);
         PJpsi_L.Boost(boosttoparent);

         TVector3 x0_Z = (-PKaon_Z.Vect()+(PKaon_Z.Vect().Dot(PJpsi_Z.Vect().Unit()))*(PJpsi_Z.Vect().Unit())).Unit();
         TVector3 x0_L = (-PJpsi_L.Vect()+(PJpsi_L.Vect().Dot(PKaon_L.Vect().Unit()))*(PKaon_L.Vect().Unit())).Unit();
         TVector3 z0_Z = -PJpsi_p.Vect().Unit();
         std::cout << (z0_Z.Cross(x0_Z)).Unit().Dot(x0_L) << std::endl;
         */


    //Jpsi rest frame
    boosttoparent = -PJpsi.BoostVector();
    Pmu2->Boost(boosttoparent);
    Pproton->Boost(boosttoparent);
    PLam.Boost(boosttoparent);
    TVector3 z3 = Pmu2->Vect().Unit();
    TVector3 x3_Z = -(-Pproton->Vect()+(Pproton->Vect().Dot(Pmu2->Vect().Unit()))*Pmu2->Vect().Unit()).Unit();
    TVector3 x3_L = -(-PLam.Vect()+(PLam.Vect().Dot(Pmu2->Vect().Unit()))*Pmu2->Vect().Unit()).Unit();

    double sinphi = ((z3.Cross(x3_Z)).Unit()).Dot(x3_L);
    double cosphi = x3_Z.Dot(x3_L);

    xalphaMu = atan2(sinphi,cosphi);

}

//set how many SL of Lambda to float
void SetLPar(RooArgList* argli, int maxind, bool fixfirst=false)
{
    int  spin = ((RooAbsReal&)(*argli)[argli->getSize()-2]).getVal();
    //    std::cout << spin << std::endl;
    int maxi = maxind-1;
    if(spin==1&&maxi>3) maxi = 3;
    //    std::cout << spin << std::endl;
    //  if(spin==1&&ind>3)  return;
    int mini = 0;
    if(fixfirst) mini = 1;
    for(int ind=mini; ind<=maxi; ++ind) {
        RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
        var->setError(0.1); var->setConstant(0);
        RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
        var1->setError(0.1); var1->setConstant(0);
    }
    for(int ind=maxi+1; ind<6; ++ind) {
        RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
        var->setVal(0.); var->setConstant(1);
        RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
        var1->setVal(0.); var1->setConstant(1);
    }
}

void FloatPar(RooArgList* argli, int ind)
{
    if(ind>5) return;
    int  spin = ((RooAbsReal&)(*argli)[argli->getSize()-2]).getVal();
    //    std::cout << spin << std::endl;
    if(spin==1&&ind>3)  return;
    RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
    var->setError(0.1); var->setConstant(0);
    RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
    var1->setError(0.1); var1->setConstant(0);
}

void ResetPar(RooArgList* argli)
{
    int  J = ((RooAbsReal&)(*argli)[argli->getSize()-2]).getVal();
    int ind(0);
    for(int S = abs(J-2); S<=J+2; S+=2) {
        for(int L = S - 1; L<= S + 1; L+=2) {
            RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
            var->setVal(var->getVal()*sqrt(2./((double)S+1.)));
            RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
            var1->setVal(var1->getVal()*sqrt(2./((double)S+1.)));
            ind++;
        }
    }
}


void FloatZPar(RooArgList* argli)
{
    int  spin = ((RooAbsReal&)(*argli)[argli->getSize()-2]).getVal();
    //    std::cout << spin << std::endl;
    //  if(spin==1&&ind>3)  return;
    for(int ind=0; ind<=3; ++ind) {
        RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
        var->setError(0.1); var->setConstant(0);
        RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
        var1->setError(0.1); var1->setConstant(0);
    }
    if(abs(spin)<=1) {
        int ind=3;
        RooRealVar *var = (RooRealVar*)(argli->at(2*ind));
        var->setVal(0.); var->setConstant(1);
        RooRealVar *var1 = (RooRealVar*)(argli->at(2*ind+1));
        var1->setVal(0.); var1->setConstant(1);
    }
}

int main(int argc, char **argv)
{
    TDatime BeginTime;
    std::cout << "Time(begin)  " << BeginTime.GetYear() << "." << BeginTime.GetMonth() << "." << BeginTime.GetDay() << "    " << BeginTime.GetHour() << ":" << BeginTime.GetMinute() << ":" << BeginTime.GetSecond() << std::endl;


    //#ifdef __CINT__
    //  gROOT->ProcessLineSync(".x RooDalitzAmplitude.cxx+");
    //#endif
    int ZS = 5;//atoi( argv[3] );
    int  ZP = 1;//atoi( argv[4] );
    std::cout << "Fit Z J P " << ZS << " " << ZP << std::endl;

    int NoCPU(1);

    int ZS2 = 3;//atoi( argv[1] );
    int  ZP2 = -1;//atoi( argv[2] );

    std::cout << "Fit Z2 J P " << ZS2 << " " << ZP2 << std::endl;
    //  gSystem->Load("RooIpatia2_cxx.so");


    RooRealVar mkp("mkp","m(K^{-}p)",1.4,2.6);
    RooRealVar mjpsip("mjpsip","",4.,5.2);
    //  cosk:cosmu:chi
    RooRealVar cosTheta_L("cosTheta_L","cosTheta_L",-1,1);
    RooRealVar cosTheta_Jpsi("cosTheta_Jpsi","cosTheta_Jpsi",-1,1);
    RooRealVar cosTheta_Lb("cosTheta_Lb","cosTheta_Lb",-1,1);
    RooRealVar Z_cosTheta_Lb("Z_cosTheta_Lb","Z_cosTheta_Lb",-1,1);
    RooRealVar Z_cosTheta_Z("Z_cosTheta_Z","Z_cosTheta_Z",-1,1);
    RooRealVar Z_cosTheta_Jpsi("Z_cosTheta_Jpsi","Z_cosTheta_Jpsi",-1,1);
    RooRealVar cosTheta_p("cosTheta_p","cosTheta_p",-1,1);
    RooRealVar phiK("phiK","phiK",-TMath::Pi(), TMath::Pi());
    RooRealVar phiMu("phiMu","phiMu",-TMath::Pi(), TMath::Pi());
    RooRealVar Z_phiZ("Z_phiZ","Z_phiZ",-TMath::Pi(), TMath::Pi());
    RooRealVar Z_phiJpsi("Z_phiJpsi","Z_phiJpsi",-TMath::Pi(), TMath::Pi());
    RooRealVar Z_phiMu("Z_phiMu","Z_phiMu",-TMath::Pi(), TMath::Pi());
    RooRealVar alpha_Mu("alpha_Mu","alpha_Mu",-TMath::Pi(), TMath::Pi());
    RooArgList *obs = new RooArgList(mkp,cosTheta_Lb,cosTheta_L,cosTheta_Jpsi,phiK,phiMu);
    obs->add(Z_cosTheta_Lb);
    obs->add(Z_cosTheta_Z);
    obs->add(Z_cosTheta_Jpsi);
    obs->add(Z_phiZ);
    obs->add(Z_phiJpsi);
    obs->add(Z_phiMu);
    obs->add(cosTheta_p);
    obs->add(alpha_Mu);
    obs->add(mjpsip);

    RooRealVar sw("sw","sw",0);
    RooArgList *obs1 = new RooArgList(mkp,cosTheta_Lb,cosTheta_L,cosTheta_Jpsi,phiK,phiMu);
    obs1->add(Z_cosTheta_Lb);
    obs1->add(Z_cosTheta_Z);
    obs1->add(Z_cosTheta_Jpsi);
    obs1->add(Z_phiZ);
    obs1->add(Z_phiJpsi);
    obs1->add(Z_phiMu);
    obs1->add(cosTheta_p);
    obs1->add(alpha_Mu);
    obs1->add(mjpsip);

    obs1->add(sw);
    //TFile *fdata = new TFile("dummy.root");
    TFile *fdata = new TFile(TString(argv[3]));
    TTree *datree = (TTree*)fdata->Get("h1");
    RooDataSet* datars1 = new RooDataSet("datars","",datree, *obs1);
    //  RooDataSet *datars1 = (RooDataSet*)fdata->Get("datars");
    RooDataSet *datars = (RooDataSet*)datars1->reduce("mkp<2.522584&&mjpsip>4.0351880460&&mjpsip<5.125823");
    RooRealVar *index = new RooRealVar("index","index",0);
    RooDataSet *IND = new RooDataSet("IND","", RooArgSet(*index));


    double nev = datars->numEntries();
    std::cout << "sum " << nev << std::endl;
    for(int i=0; i<nev; ++i) {
        *index = i;
        //     if(i%1000==0) std::cout << "i " << i << std::endl;
        IND->add(RooArgSet(*index));

    }
    datars->merge(IND);
    //  datars->Print("V");
    RooDataSet * data_fit = new  RooDataSet(TString(datars->GetName())+TString("new"),datars->GetTitle(),datars,*datars->get(),0,"nsig_sw");
    //  data_fit->Print("V");

    obs->add(*index);

    datars->Delete();
    datars1->Delete();
    //  exit(0);
    double w(0.);
    double w2(0.);
    for( int i = 0; i < data_fit->numEntries(); ++i ){
        data_fit->get(i);
        double nsw = data_fit->weight();
        w += nsw;
        w2 += pow(nsw,2);
    }
    double swf2 = w/w2;
    double swf  = sqrt(swf2);

    TList *listZ = new TList();
    //===Lambda(Z) 1/2-
    RooRealVar m0_Z("m0_Z","m0",4.45,4.4,4.5);
    RooRealVar width_Z("width_Z","width",0.04,0,0.1);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_Bp_Z("a_Bp_Z","",0.5,-5,5);
    RooRealVar p_Bp_Z("p_Bp_Z","",0.5,-5,5);
    RooRealVar a_L0_Z("a_L0_Z","",0.03);
    RooRealVar p_L0_Z("p_L0_Z","",0.04);
    RooRealVar a_L1_Z("a_L1_Z","",0.02);
    RooRealVar p_L1_Z("p_L1_Z","",0.01);
    RooRealVar a_L2_Z("a_L2_Z","",0.02);
    RooRealVar p_L2_Z("p_L2_Z","",0.01);

    RooArgList* L_Z = new RooArgList(a_Bp_Z,p_Bp_Z,
            a_L0_Z,p_L0_Z,
            a_L1_Z,p_L1_Z,
            a_L2_Z,p_L2_Z,"L_Z");
    L_Z->add(m0_Z);
    L_Z->add(width_Z);
    L_Z->add(RooRealConstant::value(ZS));//2 x spin
    L_Z->add(RooRealConstant::value(ZP)); //parity

    listZ->Add(L_Z);

    //===Lambda(Z) 1/2-
    RooRealVar m0_Z2("m0_Z2","m0",4.35,4.2,4.4);
    RooRealVar width_Z2("width_Z2","width",0.2,0,0.5);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_Bp_Z2("a_Bp_Z2","",0.5,-5,5);
    RooRealVar p_Bp_Z2("p_Bp_Z2","",0.5,-5,5);
    RooRealVar a_L0_Z2("a_L0_Z2","",0.5);
    RooRealVar p_L0_Z2("p_L0_Z2","",0.5);
    RooRealVar a_L1_Z2("a_L1_Z2","",0.5);
    RooRealVar p_L1_Z2("p_L1_Z2","",0.5);
    RooRealVar a_L2_Z2("a_L2_Z2","",0);
    RooRealVar p_L2_Z2("p_L2_Z2","",0);

    RooArgList* L_Z2 = new RooArgList(a_Bp_Z2,p_Bp_Z2,
            a_L0_Z2,p_L0_Z2,
            a_L1_Z2,p_L1_Z2,
            a_L2_Z2,p_L2_Z2,"L_Z2");
    L_Z2->add(m0_Z2);
    L_Z2->add(width_Z2);
    L_Z2->add(RooRealConstant::value(ZS2));//2 x spin
    L_Z2->add(RooRealConstant::value(ZP2)); //parity

    listZ->Add(L_Z2);

    TList *list = new TList();
    //===Lambda(1520) 3/2-
    RooRealVar m0_1520("m0_1520","m0",1.5195);
    RooRealVar width_1520("width_1520","width",0.0156);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1520("a_p1p1_1520","",1.);
    RooRealVar p_p1p1_1520("p_p1p1_1520","",0.);
    RooRealVar a_p100_1520("a_p100_1520","",0);
    RooRealVar p_p100_1520("p_p100_1520","",0);
    RooRealVar a_m100_1520("a_m100_1520","",0);
    RooRealVar p_m100_1520("p_m100_1520","",0);
    RooRealVar a_m1m1_1520("a_m1m1_1520","",0);
    RooRealVar p_m1m1_1520("p_m1m1_1520","",0);
    RooRealVar a_p3p1_1520("a_p3p1_1520","",0);
    RooRealVar p_p3p1_1520("p_p3p1_1520","",0);
    RooRealVar a_m3m1_1520("a_m3m1_1520","",0);
    RooRealVar p_m3m1_1520("p_m3m1_1520","",0);

    RooArgList* L_1520 = new RooArgList(a_p1p1_1520,p_p1p1_1520,
            a_p100_1520,p_p100_1520,
            a_m100_1520,p_m100_1520,
            a_m1m1_1520,p_m1m1_1520,"L_1520");
    L_1520->add(a_p3p1_1520);
    L_1520->add(p_p3p1_1520);
    L_1520->add(a_m3m1_1520);
    L_1520->add(p_m3m1_1520);
    L_1520->add(m0_1520);
    L_1520->add(width_1520);
    L_1520->add(RooRealConstant::value(3.));//2 x spin
    L_1520->add(RooRealConstant::value(-1)); //parity



    //===Lambda(1600) 1/2+
    RooRealVar m0_1600("m0_1600","m0",1.6);
    RooRealVar width_1600("width_1600","width",0.15);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1600("a_p1p1_1600","",40.39/100,-300,300);
    RooRealVar p_p1p1_1600("p_p1p1_1600","",-1.018/100,-300,300);
    RooRealVar a_p100_1600("a_p100_1600","",0);
    RooRealVar p_p100_1600("p_p100_1600","",0);
    RooRealVar a_m100_1600("a_m100_1600","",0);
    RooRealVar p_m100_1600("p_m100_1600","",0);
    RooRealVar a_m1m1_1600("a_m1m1_1600","",0);
    RooRealVar p_m1m1_1600("p_m1m1_1600","",0);
    RooRealVar a_p3p1_1600("a_p3p1_1600","",0);
    RooRealVar p_p3p1_1600("p_p3p1_1600","",0);
    RooRealVar a_m3m1_1600("a_m3m1_1600","",0);
    RooRealVar p_m3m1_1600("p_m3m1_1600","",0);

    RooArgList* L_1600 = new RooArgList(a_p1p1_1600,p_p1p1_1600,
            a_p100_1600,p_p100_1600,
            a_m100_1600,p_m100_1600,
            a_m1m1_1600,p_m1m1_1600,"L_1600");
    L_1600->add(a_p3p1_1600);
    L_1600->add(p_p3p1_1600);
    L_1600->add(a_m3m1_1600);
    L_1600->add(p_m3m1_1600);
    L_1600->add(m0_1600);
    L_1600->add(width_1600);
    L_1600->add(RooRealConstant::value(1.));//2 x spin
    L_1600->add(RooRealConstant::value(1)); //parity


    //===Lambda(1670) 1/2-
    RooRealVar m0_1670("m0_1670","m0",1.67);
    RooRealVar width_1670("width_1670","width",0.035);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1670("a_p1p1_1670","",-0.5976/100,-300,300);
    RooRealVar p_p1p1_1670("p_p1p1_1670","",1.218/100,-300,300);
    RooRealVar a_p100_1670("a_p100_1670","",0);
    RooRealVar p_p100_1670("p_p100_1670","",0);
    RooRealVar a_m100_1670("a_m100_1670","",0);
    RooRealVar p_m100_1670("p_m100_1670","",0);
    RooRealVar a_m1m1_1670("a_m1m1_1670","",0);
    RooRealVar p_m1m1_1670("p_m1m1_1670","",0);
    RooRealVar a_p3p1_1670("a_p3p1_1670","",0);
    RooRealVar p_p3p1_1670("p_p3p1_1670","",0);
    RooRealVar a_m3m1_1670("a_m3m1_1670","",0);
    RooRealVar p_m3m1_1670("p_m3m1_1670","",0);

    RooArgList* L_1670 = new RooArgList(a_p1p1_1670,p_p1p1_1670,
            a_p100_1670,p_p100_1670,
            a_m100_1670,p_m100_1670,
            a_m1m1_1670,p_m1m1_1670,"L_1670");
    L_1670->add(a_p3p1_1670);
    L_1670->add(p_p3p1_1670);
    L_1670->add(a_m3m1_1670);
    L_1670->add(p_m3m1_1670);
    L_1670->add(m0_1670);
    L_1670->add(width_1670);
    L_1670->add(RooRealConstant::value(1.));//2 x spin
    L_1670->add(RooRealConstant::value(-1)); //parity




    //===Lambda(1690) 3/2-
    RooRealVar m0_1690("m0_1690","m0",1.715);
    RooRealVar width_1690("width_1690","width",0.06);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1690("a_p1p1_1690","",42.5/100,-300,300);
    RooRealVar p_p1p1_1690("p_p1p1_1690","",2.471/100,-300,300);
    RooRealVar a_p100_1690("a_p100_1690","",0);
    RooRealVar p_p100_1690("p_p100_1690","",0);
    RooRealVar a_m100_1690("a_m100_1690","",0);
    RooRealVar p_m100_1690("p_m100_1690","",0);
    RooRealVar a_m1m1_1690("a_m1m1_1690","",0);
    RooRealVar p_m1m1_1690("p_m1m1_1690","",0);
    RooRealVar a_p3p1_1690("a_p3p1_1690","",0);
    RooRealVar p_p3p1_1690("p_p3p1_1690","",0);
    RooRealVar a_m3m1_1690("a_m3m1_1690","",0);
    RooRealVar p_m3m1_1690("p_m3m1_1690","",0);

    RooArgList* L_1690 = new RooArgList(a_p1p1_1690,p_p1p1_1690,
            a_p100_1690,p_p100_1690,
            a_m100_1690,p_m100_1690,
            a_m1m1_1690,p_m1m1_1690,"L_1690");
    L_1690->add(a_p3p1_1690);
    L_1690->add(p_p3p1_1690);
    L_1690->add(a_m3m1_1690);
    L_1690->add(p_m3m1_1690);
    L_1690->add(m0_1690);
    L_1690->add(width_1690);
    L_1690->add(RooRealConstant::value(3.));//2 x spin
    L_1690->add(RooRealConstant::value(-1)); //parity




    //===Lambda(1800) 1/2-
    RooRealVar m0_1800("m0_1800","m0",1.8);
    RooRealVar width_1800("width_1800","width",0.3);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1800("a_p1p1_1800","",0.00,-300,300);
    RooRealVar p_p1p1_1800("p_p1p1_1800","",0.00,-300,300);
    RooRealVar a_p100_1800("a_p100_1800","",0);
    RooRealVar p_p100_1800("p_p100_1800","",0);
    RooRealVar a_m100_1800("a_m100_1800","",0);
    RooRealVar p_m100_1800("p_m100_1800","",0);
    RooRealVar a_m1m1_1800("a_m1m1_1800","",0);
    RooRealVar p_m1m1_1800("p_m1m1_1800","",0);
    RooRealVar a_p3p1_1800("a_p3p1_1800","",0);
    RooRealVar p_p3p1_1800("p_p3p1_1800","",0);
    RooRealVar a_m3m1_1800("a_m3m1_1800","",0);
    RooRealVar p_m3m1_1800("p_m3m1_1800","",0);

    RooArgList* L_1800 = new RooArgList(a_p1p1_1800,p_p1p1_1800,
            a_p100_1800,p_p100_1800,
            a_m100_1800,p_m100_1800,
            a_m1m1_1800,p_m1m1_1800,"L_1800");
    L_1800->add(a_p3p1_1800);
    L_1800->add(p_p3p1_1800);
    L_1800->add(a_m3m1_1800);
    L_1800->add(p_m3m1_1800);
    L_1800->add(m0_1800);
    L_1800->add(width_1800);
    L_1800->add(RooRealConstant::value(1.));//2 x spin
    L_1800->add(RooRealConstant::value(-1)); //parity


    //===Lambda(1810) 1/2+
    RooRealVar m0_1810("m0_1810","m0",1.81);
    RooRealVar width_1810("width_1810","width",0.15);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1810("a_p1p1_1810","",-3.752/100,-300,300);
    RooRealVar p_p1p1_1810("p_p1p1_1810","",-10.28/100,-300,300);
    RooRealVar a_p100_1810("a_p100_1810","",0);
    RooRealVar p_p100_1810("p_p100_1810","",0);
    RooRealVar a_m100_1810("a_m100_1810","",0);
    RooRealVar p_m100_1810("p_m100_1810","",0);
    RooRealVar a_m1m1_1810("a_m1m1_1810","",0);
    RooRealVar p_m1m1_1810("p_m1m1_1810","",0);
    RooRealVar a_p3p1_1810("a_p3p1_1810","",0);
    RooRealVar p_p3p1_1810("p_p3p1_1810","",0);
    RooRealVar a_m3m1_1810("a_m3m1_1810","",0);
    RooRealVar p_m3m1_1810("p_m3m1_1810","",0);

    RooArgList* L_1810 = new RooArgList(a_p1p1_1810,p_p1p1_1810,
            a_p100_1810,p_p100_1810,
            a_m100_1810,p_m100_1810,
            a_m1m1_1810,p_m1m1_1810,"L_1810");
    L_1810->add(a_p3p1_1810);
    L_1810->add(p_p3p1_1810);
    L_1810->add(a_m3m1_1810);
    L_1810->add(p_m3m1_1810);
    L_1810->add(m0_1810);
    L_1810->add(width_1810);
    L_1810->add(RooRealConstant::value(1.));//2 x spin
    L_1810->add(RooRealConstant::value(1)); //parity



    //===Lambda(1820) 5/2+
    RooRealVar m0_1820("m0_1820","m0",1.82);
    RooRealVar width_1820("width_1820","width",0.08);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1820("a_p1p1_1820","",0.2,-300,300);
    RooRealVar p_p1p1_1820("p_p1p1_1820","",-10.28/100,-300,300);
    RooRealVar a_p100_1820("a_p100_1820","",0);
    RooRealVar p_p100_1820("p_p100_1820","",0);
    RooRealVar a_m100_1820("a_m100_1820","",0);
    RooRealVar p_m100_1820("p_m100_1820","",0);
    RooRealVar a_m1m1_1820("a_m1m1_1820","",0);
    RooRealVar p_m1m1_1820("p_m1m1_1820","",0);
    RooRealVar a_p3p1_1820("a_p3p1_1820","",0);
    RooRealVar p_p3p1_1820("p_p3p1_1820","",0);
    RooRealVar a_m3m1_1820("a_m3m1_1820","",0);
    RooRealVar p_m3m1_1820("p_m3m1_1820","",0);

    RooArgList* L_1820 = new RooArgList(a_p1p1_1820,p_p1p1_1820,
            a_p100_1820,p_p100_1820,
            a_m100_1820,p_m100_1820,
            a_m1m1_1820,p_m1m1_1820,"L_1820");
    L_1820->add(a_p3p1_1820);
    L_1820->add(p_p3p1_1820);
    L_1820->add(a_m3m1_1820);
    L_1820->add(p_m3m1_1820);
    L_1820->add(m0_1820);
    L_1820->add(width_1820);
    L_1820->add(RooRealConstant::value(5.));//2 x spin
    L_1820->add(RooRealConstant::value(1)); //parity

    //  list->Add(L_1820);


    //===Lambda(1830) 5/2-
    RooRealVar m0_1830("m0_1830","m0",1.83);
    RooRealVar width_1830("width_1830","width",0.095);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1830("a_p1p1_1830","",0.2,-300,300);
    RooRealVar p_p1p1_1830("p_p1p1_1830","",-10.28/100,-300,300);
    RooRealVar a_p100_1830("a_p100_1830","",0);
    RooRealVar p_p100_1830("p_p100_1830","",0);
    RooRealVar a_m100_1830("a_m100_1830","",0);
    RooRealVar p_m100_1830("p_m100_1830","",0);
    RooRealVar a_m1m1_1830("a_m1m1_1830","",0);
    RooRealVar p_m1m1_1830("p_m1m1_1830","",0);
    RooRealVar a_p3p1_1830("a_p3p1_1830","",0);
    RooRealVar p_p3p1_1830("p_p3p1_1830","",0);
    RooRealVar a_m3m1_1830("a_m3m1_1830","",0);
    RooRealVar p_m3m1_1830("p_m3m1_1830","",0);

    RooArgList* L_1830 = new RooArgList(a_p1p1_1830,p_p1p1_1830,
            a_p100_1830,p_p100_1830,
            a_m100_1830,p_m100_1830,
            a_m1m1_1830,p_m1m1_1830,"L_1830");
    L_1830->add(a_p3p1_1830);
    L_1830->add(p_p3p1_1830);
    L_1830->add(a_m3m1_1830);
    L_1830->add(p_m3m1_1830);
    L_1830->add(m0_1830);
    L_1830->add(width_1830);
    L_1830->add(RooRealConstant::value(5.));//2 x spin
    L_1830->add(RooRealConstant::value(-1)); //parity

    //  list->Add(L_1830);

    //===Lambda(1890) 3/2+
    RooRealVar m0_1890("m0_1890","m0",1.89);
    RooRealVar width_1890("width_1890","width",0.1);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1890("a_p1p1_1890","",0.2,-300,300);
    RooRealVar p_p1p1_1890("p_p1p1_1890","",-10.28/100,-300,300);
    RooRealVar a_p100_1890("a_p100_1890","",0);
    RooRealVar p_p100_1890("p_p100_1890","",0);
    RooRealVar a_m100_1890("a_m100_1890","",0);
    RooRealVar p_m100_1890("p_m100_1890","",0);
    RooRealVar a_m1m1_1890("a_m1m1_1890","",0);
    RooRealVar p_m1m1_1890("p_m1m1_1890","",0);
    RooRealVar a_p3p1_1890("a_p3p1_1890","",0);
    RooRealVar p_p3p1_1890("p_p3p1_1890","",0);
    RooRealVar a_m3m1_1890("a_m3m1_1890","",0);
    RooRealVar p_m3m1_1890("p_m3m1_1890","",0);

    RooArgList* L_1890 = new RooArgList(a_p1p1_1890,p_p1p1_1890,
            a_p100_1890,p_p100_1890,
            a_m100_1890,p_m100_1890,
            a_m1m1_1890,p_m1m1_1890,"L_1890");
    L_1890->add(a_p3p1_1890);
    L_1890->add(p_p3p1_1890);
    L_1890->add(a_m3m1_1890);
    L_1890->add(p_m3m1_1890);
    L_1890->add(m0_1890);
    L_1890->add(width_1890);
    L_1890->add(RooRealConstant::value(3.));//2 x spin
    L_1890->add(RooRealConstant::value(1)); //parity

    //  list->Add(L_1890);

    //2110 5/2+
    RooRealVar m0_2110("m0_2110","m0",2.11);
    RooRealVar width_2110("width_2110","width",0.2);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_2110("a_p1p1_2110","",1.,-300,300);
    RooRealVar p_p1p1_2110("p_p1p1_2110","",0.,-300,300);
    RooRealVar a_p100_2110("a_p100_2110","",0.);
    RooRealVar p_p100_2110("p_p100_2110","",0.);
    RooRealVar a_m100_2110("a_m100_2110","",0.);
    RooRealVar p_m100_2110("p_m100_2110","",0.);
    RooRealVar a_m1m1_2110("a_m1m1_2110","",0.);
    RooRealVar p_m1m1_2110("p_m1m1_2110","",0.);
    RooRealVar a_p3p1_2110("a_p3p1_2110","",0.);
    RooRealVar p_p3p1_2110("p_p3p1_2110","",0.);
    RooRealVar a_m3m1_2110("a_m3m1_2110","",0.);
    RooRealVar p_m3m1_2110("p_m3m1_2110","",0.);

    RooArgList* L_2110 = new RooArgList(a_p1p1_2110,p_p1p1_2110,
            a_p100_2110,p_p100_2110,
            a_m100_2110,p_m100_2110,
            a_m1m1_2110,p_m1m1_2110,"L_2110");
    L_2110->add(a_p3p1_2110);
    L_2110->add(p_p3p1_2110);
    L_2110->add(a_m3m1_2110);
    L_2110->add(p_m3m1_2110);
    L_2110->add(m0_2110);
    L_2110->add(width_2110);
    L_2110->add(RooRealConstant::value(5.));//2 x spin
    L_2110->add(RooRealConstant::value(1)); //parity

    //2100 7/2-
    RooRealVar m0_2100("m0_2100","m0",2.10);
    RooRealVar width_2100("width_2100","width",0.2);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_2100("a_p1p1_2100","",1.,-300,300);
    RooRealVar p_p1p1_2100("p_p1p1_2100","",0.,-300,300);
    RooRealVar a_p100_2100("a_p100_2100","",0.);
    RooRealVar p_p100_2100("p_p100_2100","",0.);
    RooRealVar a_m100_2100("a_m100_2100","",0.);
    RooRealVar p_m100_2100("p_m100_2100","",0.);
    RooRealVar a_m1m1_2100("a_m1m1_2100","",0.);
    RooRealVar p_m1m1_2100("p_m1m1_2100","",0.);
    RooRealVar a_p3p1_2100("a_p3p1_2100","",0.);
    RooRealVar p_p3p1_2100("p_p3p1_2100","",0.);
    RooRealVar a_m3m1_2100("a_m3m1_2100","",0.);
    RooRealVar p_m3m1_2100("p_m3m1_2100","",0.);

    RooArgList* L_2100 = new RooArgList(a_p1p1_2100,p_p1p1_2100,
            a_p100_2100,p_p100_2100,
            a_m100_2100,p_m100_2100,
            a_m1m1_2100,p_m1m1_2100,"L_2100");
    L_2100->add(a_p3p1_2100);
    L_2100->add(p_p3p1_2100);
    L_2100->add(a_m3m1_2100);
    L_2100->add(p_m3m1_2100);
    L_2100->add(m0_2100);
    L_2100->add(width_2100);
    L_2100->add(RooRealConstant::value(7.));//2 x spin
    L_2100->add(RooRealConstant::value(-1)); //parity


    //===Lambda(1405) 1/2-
    RooRealVar m0_1405("m0_1405","m0",1.4051);
    RooRealVar width_1405("width_1405","width",0.0505);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_1405("a_p1p1_1405","",0.001,-300,300);
    RooRealVar p_p1p1_1405("p_p1p1_1405","",0.,-300,300);
    RooRealVar a_p100_1405("a_p100_1405","",0);
    RooRealVar p_p100_1405("p_p100_1405","",0);
    RooRealVar a_m100_1405("a_m100_1405","",0);
    RooRealVar p_m100_1405("p_m100_1405","",0);
    RooRealVar a_m1m1_1405("a_m1m1_1405","",0);
    RooRealVar p_m1m1_1405("p_m1m1_1405","",0);
    RooRealVar a_p3p1_1405("a_p3p1_1405","",0);
    RooRealVar p_p3p1_1405("p_p3p1_1405","",0);
    RooRealVar a_m3m1_1405("a_m3m1_1405","",0);
    RooRealVar p_m3m1_1405("p_m3m1_1405","",0);

    RooArgList* L_1405 = new RooArgList(a_p1p1_1405,p_p1p1_1405,
            a_p100_1405,p_p100_1405,
            a_m100_1405,p_m100_1405,
            a_m1m1_1405,p_m1m1_1405,"L_1405");
    L_1405->add(a_p3p1_1405);
    L_1405->add(p_p3p1_1405);
    L_1405->add(a_m3m1_1405);
    L_1405->add(p_m3m1_1405);
    L_1405->add(m0_1405);
    L_1405->add(width_1405);
    L_1405->add(RooRealConstant::value(1.));//2 x spin
    L_1405->add(RooRealConstant::value(-1)); //parity


    //===Lambda(2350) 9/2+
    RooRealVar m0_2350("m0_2350","m0",2.35);
    RooRealVar width_2350("width_2350","width",0.15);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_2350("a_p1p1_2350","",0.001,-10,10);
    RooRealVar p_p1p1_2350("p_p1p1_2350","",0.,-10,10);
    RooRealVar a_p100_2350("a_p100_2350","",0);
    RooRealVar p_p100_2350("p_p100_2350","",0);
    RooRealVar a_m100_2350("a_m100_2350","",0);
    RooRealVar p_m100_2350("p_m100_2350","",0);
    RooRealVar a_m1m1_2350("a_m1m1_2350","",0);
    RooRealVar p_m1m1_2350("p_m1m1_2350","",0);
    RooRealVar a_p3p1_2350("a_p3p1_2350","",0);
    RooRealVar p_p3p1_2350("p_p3p1_2350","",0);
    RooRealVar a_m3m1_2350("a_m3m1_2350","",0);
    RooRealVar p_m3m1_2350("p_m3m1_2350","",0);

    RooArgList* L_2350 = new RooArgList(a_p1p1_2350,p_p1p1_2350,
            a_p100_2350,p_p100_2350,
            a_m100_2350,p_m100_2350,
            a_m1m1_2350,p_m1m1_2350,"L_2350");
    L_2350->add(a_p3p1_2350);
    L_2350->add(p_p3p1_2350);
    L_2350->add(a_m3m1_2350);
    L_2350->add(p_m3m1_2350);
    L_2350->add(m0_2350);
    L_2350->add(width_2350);
    L_2350->add(RooRealConstant::value(9.));//2 x spin
    L_2350->add(RooRealConstant::value(1)); //parity

    //  list->Add(L_2350);

    RooRealVar m0_2585("m0_2585","m0",2.585);
    RooRealVar width_2585("width_2585","width",0.3);
    // a &phase [1/2,1] [1/2,0] [-1/2, 0] [-1/2, -1] [3/2,1] [-3/2, -1]
    RooRealVar a_p1p1_2585("a_p1p1_2585","",1.,-300,300);
    RooRealVar p_p1p1_2585("p_p1p1_2585","",0.,-300,300);
    RooRealVar a_p100_2585("a_p100_2585","",0.);
    RooRealVar p_p100_2585("p_p100_2585","",0.);
    RooRealVar a_m100_2585("a_m100_2585","",0.);
    RooRealVar p_m100_2585("p_m100_2585","",0.);
    RooRealVar a_m1m1_2585("a_m1m1_2585","",0.);
    RooRealVar p_m1m1_2585("p_m1m1_2585","",0.);
    RooRealVar a_p3p1_2585("a_p3p1_2585","",0.);
    RooRealVar p_p3p1_2585("p_p3p1_2585","",0.);
    RooRealVar a_m3m1_2585("a_m3m1_2585","",0.);
    RooRealVar p_m3m1_2585("p_m3m1_2585","",0.);

    RooArgList* L_2585 = new RooArgList(a_p1p1_2585,p_p1p1_2585,
            a_p100_2585,p_p100_2585,
            a_m100_2585,p_m100_2585,
            a_m1m1_2585,p_m1m1_2585,"L_2585");
    L_2585->add(a_p3p1_2585);
    L_2585->add(p_p3p1_2585);
    L_2585->add(a_m3m1_2585);
    L_2585->add(p_m3m1_2585);
    L_2585->add(m0_2585);
    L_2585->add(width_2585);
    L_2585->add(RooRealConstant::value(5.));//2 x spin
    L_2585->add(RooRealConstant::value(-1)); //parity

    //  list->Add(L_2585);
    list->Add(L_1405);
    list->Add(L_1520);
    list->Add(L_1600);
    list->Add(L_1670);
    list->Add(L_1690);
    list->Add(L_1800);
    list->Add(L_1810);
    list->Add(L_1820);
    list->Add(L_1830);
    list->Add(L_1890);
    list->Add(L_2100);
    list->Add(L_2110);
    //  list->Add(L_2350);

    //  return ;
    RooDalitzAmplitude *sig = new RooDalitzAmplitude("sig","", *obs, list, listZ, TString(argv[3]),*data_fit);
    //sig->genToy("Test");
    RooArgSet* setdlz = sig->getParameters(*data_fit);
    //  RooArgSet* setdlz1 = signorm->getParameters(*data_fit);


    //   setdlz->readFromFile("result/fitz-m1.func");
    char name[10];
    if(ZP>0) {
        sprintf(name,"p%i",ZS);
    } else {
        sprintf(name,"m%i",ZS);
    }

    char nameZ2[10];
    if(ZP2>0) {
        sprintf(nameZ2,"p%i",ZS2);
    } else {
        sprintf(nameZ2,"m%i",ZS2);
    }
    //setdlz->readFromFile(TString::Format("result/fitz-%s.func",name).Data());//"fitz.func");
    //  setdlz->readFromFile(TString::Format("fit2zall-3half-%s.func",name).Data());




    //setdlz->readFromFile("func/fit2zall-m3-p5-res8.func");
    //  if(exists(TString::Format("func/fit2zall-%s-%s-res8.func",nameZ2,name).Data())) {
    //    setdlz->readFromFile(TString::Format("funcold/fit2zall-%s-%s-res8.func",nameZ2,name).Data());
    /*    a_Bp_Z2.setVal(a_Bp_Z2.getVal()/10.);
            p_Bp_Z2.setVal(p_Bp_Z2.getVal()/10.);
            a_L0_Z2.setVal(a_L0_Z2.getVal()/10.);
            p_L0_Z2.setVal(p_L0_Z2.getVal()/10.);
            a_L1_Z2.setVal(a_L1_Z2.getVal()/10.);
            p_L1_Z2.setVal(p_L1_Z2.getVal()/10.);
            a_Bp_Z.setVal(a_Bp_Z.getVal()/10.);
            p_Bp_Z.setVal(p_Bp_Z.getVal()/10.);
            a_L0_Z.setVal(a_L0_Z.getVal()/10.);
            p_L0_Z.setVal(p_L0_Z.getVal()/10.);
            a_L1_Z.setVal(a_L1_Z.getVal()/10.);
            p_L1_Z.setVal(p_L1_Z.getVal()/10.);
            */
    //  }

    SetLPar(L_1405,3);
    SetLPar(L_1520,5,true);
    SetLPar(L_1600,3);
    SetLPar(L_1670,3);
    SetLPar(L_1690,5);
    SetLPar(L_1800,5);
    SetLPar(L_1810,3);
    SetLPar(L_1820,1);
    SetLPar(L_1830,1);
    SetLPar(L_1890,3);
    SetLPar(L_2110,1);
    SetLPar(L_2100,1);
    //  SetLPar(L_2350,3);
    FloatZPar(L_Z);
    FloatZPar(L_Z2);
    //setdlz->readFromFile("params_pentaquark_paper.func");
    setdlz->readFromFile(TString(argv[4]));
    //  setdlz->readFromFile("fit2zall-1half-p1.func");
    setdlz->Print("V");
    int nres = list->GetSize();//+listZ->GetSize();
    double Dsum[20];
    sig->getInt(Dsum);
    char resname[20];
    double sum(0);
    for(int i=0; i<nres; ++i) {
        sum += Dsum[i];
    }
    for(int i=0; i<listZ->GetSize(); ++i) {
        sum += Dsum[i+nres];
    }

    TFile *flhcbmc =new TFile(TString(argv[1]),"recreate");
    TChain *ch = new TChain("DecayTree");
    ch->Add(TString(argv[2]));
    //  TCut cutB("X_M<1050&&X_M>990&&bdtg3>0.78");
    //change to loose
    /*  TCut cutB("X_M<1050&&X_M>990");
         TCut cutF("(B_LOKI_DTF_CTAUERR/0.29979245<0.15)&&(B_LOKI_DTF_CTAU/0.29979245>0.3&&B_LOKI_DTF_CTAU/0.29979245<15.)");
         TCut cutT("B_L0Global_Dec>0&&Jpsi_Hlt2DiMuonDetachedJPsiDecision_TOS>0&& (Jpsi_Hlt1DiMuonHighMassDecision_TOS>0||B_Hlt1TrackMuonDecision_TOS>0||B_Hlt1TwoTrackMVADecision_TOS>0)");
         TCut cutV("!((((hplus_ProbNNpcorr>0.7)&&(hplus_ProbNNpcorr>hminus_ProbNNpcorr)&&(abs(B_pKMuMuKplus_M[0]-5619.51)<15)) ||((hminus_ProbNNpcorr>0.7)&&(hminus_ProbNNpcorr>hplus_ProbNNpcorr)&&(abs(B_pKMuMuKminus_M[0]-5619.51)<15))))");
         TCut cutV3("!((((hplus_ProbNNkcorr<0.35|hplus_ProbNNpicorr>0.7)&&(hplus_ProbNNpicorr>hminus_ProbNNpicorr)&&(abs(B_B2KpPiJpsi_M[0]-5279.63)<15))||((hminus_ProbNNkcorr<0.35|hminus_ProbNNpicorr>0.7)&&(hminus_ProbNNpicorr>hplus_ProbNNpicorr)&&(abs(B_B2KmPiJpsi_M[0]-5279.63)<15))))");
         */
    TCut cutall("1");//cutB&&cutF&&cutT&&cutV&&cutV3);
    // TCut cutall(cutB&&cutF&&cutT);
    TTree *fChain = (TTree*)ch->CopyTree(cutall);
    Int_t           B_TRUEID;
    Int_t           Jpsi_TRUEID;
    Double_t        Jpsi_TRUEP_E;
    Double_t        Jpsi_TRUEP_X;
    Double_t        Jpsi_TRUEP_Y;
    Double_t        Jpsi_TRUEP_Z;
    Int_t           muplus_TRUEID;
    Double_t        muplus_TRUEP_E;
    Double_t        muplus_TRUEP_X;
    Double_t        muplus_TRUEP_Y;
    Double_t        muplus_TRUEP_Z;
    Int_t           muminus_TRUEID;
    Double_t        muminus_TRUEP_E;
    Double_t        muminus_TRUEP_X;
    Double_t        muminus_TRUEP_Y;
    Double_t        muminus_TRUEP_Z;
    Int_t           hplus_TRUEID;
    Double_t        hplus_TRUEP_E;
    Double_t        hplus_TRUEP_X;
    Double_t        hplus_TRUEP_Y;
    Double_t        hplus_TRUEP_Z;
    Int_t           hminus_TRUEID;
    Double_t        hminus_TRUEP_E;
    Double_t        hminus_TRUEP_X;
    Double_t        hminus_TRUEP_Y;
    Double_t        hminus_TRUEP_Z;

    TBranch        *b_B_TRUEID;   //!
    TBranch        *b_Jpsi_TRUEID;   //!
    TBranch        *b_Jpsi_TRUEP_E;   //!
    TBranch        *b_Jpsi_TRUEP_X;   //!
    TBranch        *b_Jpsi_TRUEP_Y;   //!
    TBranch        *b_Jpsi_TRUEP_Z;   //!
    TBranch        *b_muplus_TRUEID;   //!
    TBranch        *b_muplus_TRUEP_E;   //!
    TBranch        *b_muplus_TRUEP_X;   //!
    TBranch        *b_muplus_TRUEP_Y;   //!
    TBranch        *b_muplus_TRUEP_Z;   //!
    TBranch        *b_muminus_TRUEID;   //!
    TBranch        *b_muminus_TRUEP_E;   //!
    TBranch        *b_muminus_TRUEP_X;   //!
    TBranch        *b_muminus_TRUEP_Y;   //!
    TBranch        *b_muminus_TRUEP_Z;   //!
    TBranch        *b_hplus_TRUEID;   //!
    TBranch        *b_hplus_TRUEP_E;   //!
    TBranch        *b_hplus_TRUEP_X;   //!
    TBranch        *b_hplus_TRUEP_Y;   //!
    TBranch        *b_hplus_TRUEP_Z;   //!
    TBranch        *b_hminus_TRUEID;   //!
    TBranch        *b_hminus_TRUEP_E;   //!
    TBranch        *b_hminus_TRUEP_X;   //!
    TBranch        *b_hminus_TRUEP_Y;   //!
    TBranch        *b_hminus_TRUEP_Z;   //!

    //   float muplus_PIDmucorr, muminus_PIDmucorr;

    fChain->SetBranchAddress("B_TRUEID", &B_TRUEID, &b_B_TRUEID);
    fChain->SetBranchAddress("Jpsi_TRUEID", &Jpsi_TRUEID, &b_Jpsi_TRUEID);
    fChain->SetBranchAddress("Jpsi_TRUEP_E", &Jpsi_TRUEP_E, &b_Jpsi_TRUEP_E);
    fChain->SetBranchAddress("Jpsi_TRUEP_X", &Jpsi_TRUEP_X, &b_Jpsi_TRUEP_X);
    fChain->SetBranchAddress("Jpsi_TRUEP_Y", &Jpsi_TRUEP_Y, &b_Jpsi_TRUEP_Y);
    fChain->SetBranchAddress("Jpsi_TRUEP_Z", &Jpsi_TRUEP_Z, &b_Jpsi_TRUEP_Z);
    fChain->SetBranchAddress("muplus_TRUEID", &muplus_TRUEID, &b_muplus_TRUEID);
    fChain->SetBranchAddress("muplus_TRUEP_E", &muplus_TRUEP_E, &b_muplus_TRUEP_E);
    fChain->SetBranchAddress("muplus_TRUEP_X", &muplus_TRUEP_X, &b_muplus_TRUEP_X);
    fChain->SetBranchAddress("muplus_TRUEP_Y", &muplus_TRUEP_Y, &b_muplus_TRUEP_Y);
    fChain->SetBranchAddress("muplus_TRUEP_Z", &muplus_TRUEP_Z, &b_muplus_TRUEP_Z);
    fChain->SetBranchAddress("muminus_TRUEID", &muminus_TRUEID, &b_muminus_TRUEID);
    fChain->SetBranchAddress("muminus_TRUEP_E", &muminus_TRUEP_E, &b_muminus_TRUEP_E);
    fChain->SetBranchAddress("muminus_TRUEP_X", &muminus_TRUEP_X, &b_muminus_TRUEP_X);
    fChain->SetBranchAddress("muminus_TRUEP_Y", &muminus_TRUEP_Y, &b_muminus_TRUEP_Y);
    fChain->SetBranchAddress("muminus_TRUEP_Z", &muminus_TRUEP_Z, &b_muminus_TRUEP_Z);
    fChain->SetBranchAddress("hplus_TRUEID", &hplus_TRUEID, &b_hplus_TRUEID);
    fChain->SetBranchAddress("hplus_TRUEP_E", &hplus_TRUEP_E, &b_hplus_TRUEP_E);
    fChain->SetBranchAddress("hplus_TRUEP_X", &hplus_TRUEP_X, &b_hplus_TRUEP_X);
    fChain->SetBranchAddress("hplus_TRUEP_Y", &hplus_TRUEP_Y, &b_hplus_TRUEP_Y);
    fChain->SetBranchAddress("hplus_TRUEP_Z", &hplus_TRUEP_Z, &b_hplus_TRUEP_Z);
    fChain->SetBranchAddress("hminus_TRUEID", &hminus_TRUEID, &b_hminus_TRUEID);
    fChain->SetBranchAddress("hminus_TRUEP_E", &hminus_TRUEP_E, &b_hminus_TRUEP_E);
    fChain->SetBranchAddress("hminus_TRUEP_X", &hminus_TRUEP_X, &b_hminus_TRUEP_X);
    fChain->SetBranchAddress("hminus_TRUEP_Y", &hminus_TRUEP_Y, &b_hminus_TRUEP_Y);
    fChain->SetBranchAddress("hminus_TRUEP_Z", &hminus_TRUEP_Z, &b_hminus_TRUEP_Z);

    Double_t wdp;
    //hehe
    Double_t wxll;
    //  TBranch *newmp = fChain->Branch("muplus_PIDmucorr", &muplus_PIDmucorr, "muplus_PIDmucorr/F");
    //  TBranch *newmm = fChain->Branch("muminus_PIDmucorr", &muminus_PIDmucorr, "muminus_PIDmucorr/F");
    TBranch *newBranch1 = fChain->Branch("mkp", &xmkp, "mkp/D");
    TBranch *newBranch2 = fChain->Branch("mjpsip", &xmjpsip, "mjpsip/D");
    TBranch *newBranch = fChain->Branch("wdp", &wdp, "wdp/D");
    //hehe
    TBranch *newBranch3 = fChain->Branch("wxll", &wxll, "wxll/D");
    Long64_t numevt = fChain->GetEntries();
    // numevt = 100;
    for(Long64_t i=0; i<numevt; ++i) {
        fChain->GetEntry(i);
        wdp = 0.0;
        xmkp = -1.0;
        xmjpsip = -1.0;
        wxll = 0;
        if(abs(B_TRUEID)==5122)  {
            double  muplusPX, muplusPY, muplusPZ, muplusPE;
            double muminusPX, muminusPY, muminusPZ, muminusPE;
            double protonPX, protonPY, protonPZ, protonPE;
            double kaonPX, kaonPY, kaonPZ, kaonPE(-1);

            TLorentzVector p[4];
            Int_t id[4];
            p[0] = TLorentzVector(muplus_TRUEP_X,muplus_TRUEP_Y,muplus_TRUEP_Z,muplus_TRUEP_E); id[0] = muplus_TRUEID;
            p[1] = TLorentzVector(muminus_TRUEP_X,muminus_TRUEP_Y,muminus_TRUEP_Z,muminus_TRUEP_E); id[1] = muminus_TRUEID;
            p[2] = TLorentzVector(hplus_TRUEP_X,hplus_TRUEP_Y,hplus_TRUEP_Z,hplus_TRUEP_E); id[2] = hplus_TRUEID;
            p[3] = TLorentzVector(hminus_TRUEP_X,hminus_TRUEP_Y,hminus_TRUEP_Z,hminus_TRUEP_E); id[3] = hminus_TRUEID;
            Int_t matchid[4];
            Int_t match(0);
            for(int ip=0; ip<4; ++ip) {
                if(abs(id[ip])==13&&B_TRUEID*id[ip]<0) {
                    matchid[0] = ip;
                    match++;
                } else if(abs(id[ip])==13&&B_TRUEID*id[ip]>0) {
                    matchid[1] = ip;
                    match++;
                } else if(abs(id[ip])==321) {
                    matchid[2] = ip;
                    match++;
                }else if(abs(id[ip])==2212) {
                    matchid[3] = ip;
                    match++;
                }
            }
            if(match==4) {
                muplusPE = p[matchid[0]].E();
                muplusPX = p[matchid[0]].Px();
                muplusPY = p[matchid[0]].Py();
                muplusPZ = p[matchid[0]].Pz();
                muminusPE = p[matchid[1]].E();
                muminusPX = p[matchid[1]].Px();
                muminusPY = p[matchid[1]].Py();
                muminusPZ = p[matchid[1]].Pz();
                kaonPE = p[matchid[2]].E();
                kaonPX = p[matchid[2]].Px();
                kaonPY = p[matchid[2]].Py();
                kaonPZ = p[matchid[2]].Pz();
                protonPE = p[matchid[3]].E();
                protonPX = p[matchid[3]].Px();
                protonPY = p[matchid[3]].Py();
                protonPZ = p[matchid[3]].Pz();

                //      if(kaonPE>0)
                if(kaonPE>0) {
                    int ID = B_TRUEID/abs(B_TRUEID);

                    TLorentzVector *Pmu1 = new TLorentzVector( muplusPX, muplusPY, muplusPZ, muplusPE);
                    TLorentzVector *Pmu2 = new TLorentzVector( muminusPX, muminusPY, muminusPZ, muminusPE);
                    TLorentzVector *Pproton = new TLorentzVector( protonPX, protonPY, protonPZ, protonPE);
                    TLorentzVector *Pkaon = new TLorentzVector( kaonPX, kaonPY, kaonPZ, kaonPE);
                    TLorentzVector *PJ = new TLorentzVector(muplusPX+muminusPX,muplusPY+muminusPY,muplusPZ+muminusPZ,muplusPE+muminusPE);
                    if(Jpsi_TRUEID==443) PJ =  new TLorentzVector(Jpsi_TRUEP_X, Jpsi_TRUEP_Y,Jpsi_TRUEP_Z, Jpsi_TRUEP_E);
                    helicityJpsiLam(Pmu1, Pmu2, PJ, Pproton, Pkaon);
                    Pmu1 = new TLorentzVector( muplusPX, muplusPY, muplusPZ, muplusPE);
                    Pmu2 = new TLorentzVector( muminusPX, muminusPY, muminusPZ, muminusPE);
                    Pproton = new TLorentzVector( protonPX, protonPY, protonPZ, protonPE);
                    Pkaon = new TLorentzVector( kaonPX, kaonPY, kaonPZ, kaonPE);

                    helicityZK(Pmu1, Pmu2, Pproton, Pkaon);
                    Pmu1 = new TLorentzVector( muplusPX, muplusPY, muplusPZ, muplusPE);
                    Pmu2 = new TLorentzVector( muminusPX, muminusPY, muminusPZ, muminusPE);
                    Pproton = new TLorentzVector( protonPX, protonPY, protonPZ, protonPE);
                    Pkaon = new TLorentzVector( kaonPX, kaonPY, kaonPZ, kaonPE);
                    helicityTwoFrame(Pmu1, Pmu2, Pproton, Pkaon);
                    Double_t Vdlz[15];
                    Vdlz[0] = xmkp;
                    Vdlz[1] = xcostheta;
                    Vdlz[2] = xcostheta1;
                    Vdlz[3] = xcostheta2;
                    Vdlz[4] = xphi1*ID;
                    Vdlz[5] = xphi2*ID;
                    Vdlz[6] = xcosthetaB;
                    Vdlz[7] = xcosthetaZ;
                    Vdlz[8] = xcosthetaPsi;
                    Vdlz[9] = xphiZ*ID;
                    Vdlz[10] = xphiPsi*ID;
                    Vdlz[11] = xphiMu*ID;
                    Vdlz[12] = xcosthetap;
                    Vdlz[13] = xalphaMu*ID;
                    Vdlz[14] = xmjpsip;
                    wdp = sig->evaluate(Vdlz);
                    //double wxll = calwxll(xmkp)*sig->getphsp(xmkp);
                    wxll = calwxll(xmkp)*sig->getphsp(xmkp);
                    if(isnan(wxll)) wxll=0;
                    wdp = wdp * wxll;
                }
                //        std::cout << xmkp << " " << wdp << std::endl;
            }
        }

        newBranch->Fill();
        newBranch1->Fill();
        newBranch2->Fill();
        //hehe
        newBranch3->Fill();

        //newmm->Fill();
        //    newmp->Fill();
    }
    fChain->Write();
    flhcbmc->Close();
    std::cout << "AmplitudeFit() successfully completed!" << std::endl;
    TDatime FinishTime;
    std::cout << "Time(finish)  " << FinishTime.GetYear() << "." << FinishTime.GetMonth() << "." << FinishTime.GetDay() << "    " << FinishTime.GetHour() << ":" << FinishTime.GetMinute() << ":" << FinishTime.GetSecond() << std::endl;
    return 0;
}

