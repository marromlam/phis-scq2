// from https://gitlab.cern.ch/kgizdov/BsJpsiPhi/blob/master/analysis/src/lib/calculate_helicity_angles.cpp
double HelicityCosTheta(TLorentzVector Kplus_P, TLorentzVector muplus_P, TLorentzVector muminus_P){

    TLorentzVector Jpsi_P = muplus_P + muminus_P;
    TLorentzVector Bs_P = Jpsi_P + Kplus_P;

    double PP = Bs_P.M2();
    double PD = Bs_P.Dot(muplus_P);
    double PQ = Bs_P.Dot(Jpsi_P);
    double QQ = Jpsi_P.M2();
    double QD = Jpsi_P.Dot(muplus_P);
    double DD = muplus_P.M2();

    return (PD*QQ - PQ*QD)/TMath::Sqrt((PQ*PQ-PP*QQ)*(QD*QD-DD*QQ));
}