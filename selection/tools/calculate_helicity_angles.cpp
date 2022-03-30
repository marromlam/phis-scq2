std::vector<double> HelicityAngles(const TLorentzVector& Kpl_P, const TLorentzVector& Kmin_P, const TLorentzVector& mupl_P, const TLorentzVector& mumin_P)
{


    TLorentzVector Kplus_P(Kpl_P);
    TLorentzVector Kminus_P(Kmin_P);
    TLorentzVector muplus_P(mupl_P);
    TLorentzVector muminus_P(mumin_P);

    // Bs, KK, mm momenta 4 vectors
    TLorentzVector KK_P = Kplus_P + Kminus_P;
    TLorentzVector mm_P = muplus_P + muminus_P;
    TLorentzVector KKmm_P = KK_P + mm_P;

    // Unit vector along mumu direction in the KK mass r.f.
    muplus_P.Boost( - KK_P.BoostVector() );
    muminus_P.Boost( - KK_P.BoostVector() );
    TVector3 e_KK = - (muplus_P + muminus_P).Vect().Unit();

    // Boost the muons back to lab frame
    muplus_P.Boost( KK_P.BoostVector() );
    muminus_P.Boost( KK_P.BoostVector() );

    // Unit vector along KK direction in the mm mass r.f.
    Kplus_P.Boost( - mm_P.BoostVector() );
    Kminus_P.Boost( - mm_P.BoostVector() );
    TVector3 e_mm = - (Kplus_P+Kminus_P).Vect().Unit();

    // Boost the Kaons back to lab frame
    Kplus_P.Boost( mm_P.BoostVector() );
    Kminus_P.Boost( mm_P.BoostVector() );

    // Unit vector along KK direction in the mm mass r.f.
    Kplus_P.Boost( - KKmm_P.BoostVector() );
    Kminus_P.Boost( - KKmm_P.BoostVector() );
    muplus_P.Boost( - KKmm_P.BoostVector() );
    muminus_P.Boost( - KKmm_P.BoostVector() );
    TVector3 e_KKmm = (muplus_P + muminus_P).Vect().Unit();

    // Perpendicular vectors to KK and mm planes in the KKmmm r.f.
    TVector3 eta_KK = ( Kplus_P.Vect().Cross( Kminus_P.Vect()) ).Unit();
    TVector3 eta_mm = ( muplus_P.Vect().Cross( muminus_P.Vect()) ).Unit();

    Kplus_P.Boost( KKmm_P.BoostVector() );
    Kminus_P.Boost( KKmm_P.BoostVector() );
    muplus_P.Boost( KKmm_P.BoostVector() );
    muminus_P.Boost( KKmm_P.BoostVector() );

    // Helicity angles.
    Kplus_P.Boost( - KK_P.BoostVector() );
    muplus_P.Boost( - mm_P.BoostVector() );

   std::vector<double> angles;

    angles.push_back(( Kplus_P.Vect().Unit()  ).Dot(e_KK));
    angles.push_back(( muplus_P.Vect().Unit() ).Dot(e_mm));

    if (eta_KK.Cross(eta_mm).Dot(e_KKmm) > 0)
    {
        angles.push_back(TMath::ACos(eta_KK.Dot(eta_mm)));
    }
    else
    {
        angles.push_back(-1.*TMath::ACos(eta_KK.Dot(eta_mm)));
   }

    return angles;
}