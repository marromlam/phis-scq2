////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay rate Bs -> mumuKK                        //
//                                                                            //
//   Created: 2019-11-18                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file contains the following __kernels:                               //
//    * pyDiffRate: Computes Bs2MuMuKK pdf looping over the events. Now it    //
//                  handles a binned X_M fit without splitting beforehand the //
//                  data, it launches a thread per mass bin.                  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

// Debugging 0 [0,1,2,3,>3]
__constant int DEBUG = 4;

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>




// mKK mass bins
#define nknots 7
#define sigma_t 0.15
#define USE_TIME_ACC 0

__constant double X_M[7] = {990, 1008, 1016, 1020, 1024, 1032, 1050};
__constant double pdfweights[10] = {1,1,1,0,0,0,1,0,0,0};
__constant double knots[7] =  { 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 };

// Include disciplines
#include "/home3/marcos.romero/phis-scq/opencl/DecayTimeAcceptance.cl"
#include "/home3/marcos.romero/phis-scq/opencl/TimeAngularDistribution.cl"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__global
double getDiffRate( double *data, double G, double DG, double DM, double CSP,
                    double APlon, double ASlon, double APpar, double APper,
                    double phisPlon, double phisSlon, double phisPpar, double phisPper,
                    double deltaSlon, double deltaPlon, double deltaPpar, double deltaPper,
                    double lPlon, double lSlon, double lPpar, double lPper,
                    double tLL, double tUL,
                    double *coeffs,
                    bool USE_FK)
{
  if ((DEBUG > 3) && ( get_global_id(0) == 0) )
  {
    printf("*USE_FK       :  %d\n", USE_FK);
    printf("*USE_TIME_ACC :  %d\n", USE_TIME_ACC);
    //printf("CSP           : %+lf\n", CSP);
    //printf("ASlon         : %+lf\n", ASlon);
    printf("G             : %+lf\n", G);
    printf("DG            : %+lf\n", DG);
    printf("CSP           : %+lf\n", CSP);
    printf("ASlon         : %+lf\n", ASlon);
    printf("APlon         : %+lf\n", APlon);
    printf("APpar         : %+lf\n", APpar);
    printf("APper         : %+lf\n", APper);
    printf("phisSlon      : %+lf\n", phisSlon);
    printf("phisPlon      : %+lf\n", phisPlon);
    printf("phisPpar      : %+lf\n", phisPpar);
    printf("phisPper      : %+lf\n", phisPper);
    printf("deltaSlon     : %+lf\n", deltaSlon);
    printf("deltaPlon     : %+lf\n", deltaPlon);
    printf("deltaPper     : %+lf\n", deltaPper);
    printf("deltaPpar     : %+lf\n", deltaPpar);
    printf("lSlon         : %+lf\n", lSlon);
    printf("lPlon         : %+lf\n", lPlon);
    printf("lPper         : %+lf\n", lPper);
    printf("lPpar         : %+lf\n", lPpar);
    printf("tLL           : %+lf\n", tLL);
    printf("tUL           : %+lf\n", tUL);
    printf("COEFFS        : %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
    printf("                %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  }

  // variables
  double normweights[10] = {1,1,1,0,0,0,1,0,0,0};
  double cosK = data[0];
  double cosL = data[1];
  double hphi = data[2];
  double time = data[3];
  if ((time>tUL) || (time<tLL)) {return 0;}
  // add when needed:
  // double sigma_t 		= data[4];
  // double q_OS 			= data[5];
  // double qSlonSK 		= data[6];
  // double eta_OS 		= data[7];
  // double etaSlonSK 	= data[8];
  // int year 					= data[9];




/*
  NOT YET IMPLEMENTED STUFF

  double sigma_t =  delta(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);

  double sigma_t_1 = delta_1(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);
  double sigma_t_2 = delta_2(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);

  double omega_OS = omega(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
  double omega_bar_OS = omega_bar(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
  double omegaSlonSK = omega(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);
  double omega_barSlonSK = omega_bar(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);

  double taggingPparrs_OS[3] = {omega_OS, omega_bar_OS, q_OS};
  double taggingPparrsSlonSK[3] = {omegaSlonSK, omega_barSlonSK, qSlonSK};

  fix_taggingPparrs(taggingPparrs_OS);
  fix_taggingPparrs(taggingPparrsSlonSK);

  omega_OS = taggingPparrs_OS[0];
  omega_bar_OS = taggingPparrs_OS[1];
  omegaSlonSK = taggingPparrsSlonSK[0];
  omega_barSlonSK = taggingPparrsSlonSK[1];

  if((taggingPparrs_OS[0] == 0.5 || taggingPparrs_OS[1] == 0.5) && (taggingPparrs_OS[0] != taggingPparrs_OS[1]))
  printf("OS tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrs_OS[0], taggingPparrs_OS[1], taggingPparrs_OS[2]);
  else
  q_OS = taggingPparrs_OS[2];

  if((taggingPparrsSlonSK[0] == 0.5 || taggingPparrsSlonSK[1] == 0.5) && (taggingPparrsSlonSK[0] != taggingPparrsSlonSK[1]))
  printf("SSK tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrsSlonSK[0], taggingPparrsSlonSK[1], taggingPparrsSlonSK[2]);
  else
  qSlonSK = taggingPparrsSlonSK[2];

*/




  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set sigma_t = 0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  cdouble_t exp_p, exp_m, exp_i;
  double t_offset = 0.0;//delta(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);

  exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., sigma_t);
  exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., sigma_t);
  exp_i = getExponentialConvolution(time-t_offset,          G, DM, sigma_t);

  //double ta = pyopencl::real(0.5*(exp_m + exp_p)); // cosh = (exp_m + exp_p)/2
  //double tb = pyopencl::real(0.5*(exp_m - exp_p)); // sinh = (exp_m - exp_p)/2
  //double tc = pyopencl::real(exp_i);                    // exp_i = cos + I*sin
  //double td = pyopencl::imag(exp_i);                    // exp_i = cos + I*sin
  double ta = 0.5*cdouble_add(exp_m,exp_p).real;
  double tb = 0.5*cdouble_add(exp_m,cdouble_mul(cdouble_new(-1,0),exp_p)).real;
  double tc = exp_i.real;
  double td = exp_i.imag;



  // Flavor tagging ------------------------------------------------------------
  double omegaOSB = 0; double omegaOSBbar = 0;
  double omegaSSB = 0; double omegaSSBbar = 0;
  double tagOS = 0; double tagSS = 0;

  bool useTrueTag = 1;
  if (useTrueTag)
  {
    tagOS = 0.5;
  }

  //   tagOS = meas->tag_decision;
  //   tagSS = meas->tag_decision_ss;
  //
  //   double meas_omega    = meas->tag_omega;
  //   double meas_omega_ss = meas->tag_omega_ss;
  //
  //   double ma(0.99),mi(0);
  //
  //   omega_os_B =    std::max(std::min(params->tag_p0()    + params->tag_deltap0()/2.0    +(params->tag_p1()    + params->tag_deltap1()/2.0)    * (meas_omega - params->tag_eta()),ma),mi);
  //   omega_os_Bbar = std::max(std::min(params->tag_p0()    - params->tag_deltap0()/2.0    +(params->tag_p1()    - params->tag_deltap1()/2.0)    * (meas_omega - params->tag_eta()),ma),mi);
  //
  //   omega_ss_B =    std::max(std::min(params->tag_ss_p0() + params->tag_ss_deltap0()/2.0 +(params->tag_ss_p1() + params->tag_ss_deltap1()/2.0) * (meas_omega_ss - params->tag_ss_eta()),ma),mi);
  //   omega_ss_Bbar = std::max(std::min(params->tag_ss_p0() - params->tag_ss_deltap0()/2.0 +(params->tag_ss_p1() - params->tag_ss_deltap1()/2.0) * (meas_omega_ss - params->tag_ss_eta()),ma),mi);
  //
  //   if((1.0 + tagOS * (1-2*omega_os_B))*(1.0 + tagSS * (1-2*omega_ss_B)) == 0 && (1.0 - tagOS * (1-2*omega_os_Bbar))*(1.0 - tagSS * (1-2*omega_ss_Bbar)) == 0){
  //     omega_ss_Bbar=0.5;
  //     omega_ss_B=0.5;
  //     omega_os_Bbar=0.5;
  //     omega_os_B=0.5;
  //   }
  // }

  // Decay-time acceptance -----------------------------------------------------
  //     To get rid of decay-time acceptance set USE_TIME_ACC to False.
  double dta = 1.0;
  if (USE_TIME_ACC)
  {
    dta = calcTimeAcceptance(time, coeffs, tLL, tUL);
  }



  // Compute per event pdf -----------------------------------------------------
  double vnk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vak[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vbk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vck[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vdk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  double nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
  double pdfB = 0.0; double pdfBbar = 0.0;

  for(int k = 1; k <= 10; k++)
  {
    nk = getN(APlon,ASlon,APpar,APper,CSP,k);
    if (USE_FK)
    {
      fk = 9./(16.*M_PI)*getF(cosK,cosL,hphi,k);
    }
    else
    {
      fk = normweights[k-1];
    }

    ak = getA(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getB(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getC(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getD(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);

    hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
    hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);

    pdfB    += nk*hk_B*fk;
    pdfBbar += nk*hk_Bbar*fk;

    vnk[k-1] = 1.*nk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
  }



  // Compute pdf integral ------------------------------------------------------
  double intBBar[2] = {0.,0.};
  if (sigma_t == 0 && USE_TIME_ACC == 0)
  {
    // Here we can use the simplest 4xPi integral of the pdf since there are no
    // resolution effects
    integralSimple(intBBar,
                   vnk, vak, vbk, vck, vdk, normweights, G, DG, DM, tLL, tUL);
  }
  else
  {
    // This integral works for all decay times, remember sigma_t != 0.
    double foo[7] =  { 0.30, 0.58, 0.91, 1.35, 1.96, 3.01, 7.00 };
    integralFullSpline(intBBar,
                       vnk, vak, vbk, vck, vdk,
                       normweights,  G, DG,  DM,
                       sigma_t,
                       tLL,
                       t_offset,
                       7,
                       foo, coeffs);
  }
  double intB = intBBar[0]; double intBbar = intBBar[1];



  // Cooking the output --------------------------------------------------------
  double num = 1.0; double den = 1.0;
  num = dta*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
        );
  den = 1.0*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * intB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * intBbar
        );



  // DEBUG ! -------------------------------------------------------------------
  if (DEBUG >= 1 && ( get_global_id(0) == 0))
  {
    printf("INPUT     : cosK=%+lf\tcosL=%+lf\thphi=%+lf\ttime=%+lf\n",
           cosK,cosL,hphi,time);
    printf("RESULT    : pdf=%+0.8lf\tipdf=%+0.3lf\tpdf/ipdf=%+lf\n",
           num,den,num/den);
    if (DEBUG >= 2)
    {
      printf("RESULT    : pdfB=%+lf\tpdBbar=%+lf\tipdfB=%+lf\tipdfBbar=%+lf\n",
             pdfB,pdfBbar,intB,intBbar);
      printf("RESULT    : dta=%+lf\tpdBbar=%+lf\tipdfB=%+lf\tipdfBbar=%+lf\n",
             dta,pdfBbar,intB,intBbar);
      if (DEBUG >= 3)
      {
        printf("TIME ACC  : ta=%.8lf\ttb=%.8lf\ttc=%.8lf\ttd=%.8lf\n",
               ta,tb,tc,td);
        if (DEBUG >= 4)
        {
          // for(int k = 0; k < 10; k++)
          // {
          //   printf("--> %.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", vnk[k], vak[k], vbk[k], vck[k], vdk[k],
          //                    normweights[k]);
          // }
        }
      }
    }
  }


  // That's all folks!
  return num/den;
}











































/*





__global
void ct_part(double params, double *time, double sigma_1, double f_res_1,
             double sigma_2, double f_res_2, double sigma_3, double mu,
             double Y_light, double Y_heavy, double Y_one, double Y_two);
{
  double a = 1+2;
}








__global
double getAlgo(double *data, double G, double DG, double DM, double CSP,
                    double APlon, double ASlon, double APpar, double APper,
                    double phisPlon, double phisSlon, double phisPpar, double phisPper,
                    double deltaSlon, double deltaPlon, double deltaPpar, double deltaPper,
                    double lPlon, double lSlon, double lPpar, double lPper, double tLL, double tUL, bool USE_FK)
{
  // variables
  double normweights[10] = {1,1,1,0,0,0,1,0,0,0};
  double cosK = data[0];
  double cosL = data[1];
  double hphi = data[2];
  double time = data[3];




  bool NORMALIZE              = 1;
  bool TIME_DEP_ANGACC_NORMS  = 0;
  bool ONE_ANGLE              = 0;
  bool ONLY_EVEN              = 0;
  bool ONLY_ODD               = 0;
  bool ONLY_SWAVE             = 0;
  bool HAS_1D_EFF             = 0;
  bool FITBD                  = 0;
  bool FIT_1D_EXP_TIME        = 0;
  bool USE_TRUEANGLES         = 1;
  bool INTEGRATE_ANGLES       = 0;
  bool USE_PER_EVENT_RES      = 0;
  bool USE_TIME_RES           = 0;
  bool USE_TRUETIME           = 1;
  bool USE_TRUTH              = 0;
  bool SEP_TAG_PARAMS         = 0;
  bool COMBINE_TAGGER         = 0;
  bool PER_EVENT_DILUTION     = 0;
  bool PERFECT_TAGGING        = 1;
  bool USE_OVL_TAGS           = 0;
  bool FIT_TAG_EFF            = 0;
  bool NORM_SEPARATELY        = 1;
  bool COS_DEL_PAR            = 1;

  // Normalization
  double norm = 1.0;
  if (NORMALIZE)
  {
    if (USE_PER_EVENT_RES || PER_EVENT_DILUTION || TIME_DEP_ANGACC_NORMS)
    {
      if (NORM_SEPARATELY)
        printf("norm=%lf\n", norm);
        //norm = calc_normalization(params, meas,true);
      else
        printf("norm=%lf\n", norm);
        //norm = calc_normalization(params, meas);
    }
    else if (NORM_SEPARATELY)
    {
      norm = normalization(meas); // this is the one
    }
    else//just give back what has been cached
    {
      norm = normalization();
    }
  }
  printf("norm=%lf\n", norm);
  if (norm < 0 ) { printf("Shit!\n"); return; }



  // Efficiency
  double eff = 1.0;
  //should actually never use efficiencies (only for plotting bs)
  //not true if bkg is present!! TODO

  if (use_efficiencies)
    {
      if (use_angular_acc && use_time_acc)
        {
          eff *= fx.eff(meas->ct(), meas->cos_theta(), meas->phi(), meas->cos_psi());
        }
      else if (use_angular_acc)
        {
          eff *= fx.eff(meas->cos_theta(), meas->phi(), meas->cos_psi());

        }
      else if (use_time_acc)
        {
          eff *= fx.eff(meas->ct());
        }
    }
  std::cout << "eff=" << eff << std::endl;



  // Get angular f coeffs
  double f1  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 1);
  double f2  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 2);
  double f3  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 3);
  double f4  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 4);
  double f5  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 5);
  double f6  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 6);
  double f7  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 7);
  double f8  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 8);
  double f9  = 9./(16.*M_PI)*getF(cosK,cosL,hphi, 9);
  double f10 = 9./(16.*M_PI)*getF(cosK,cosL,hphi,10);


  // Angular part ¿?
  double ASlon2 = ASlon*ASlon;
  double APpar2 = APpar*APpar;
  double APper2 = APper*APper;
  double APlon2 = APlon*APlon;

  double Cs = params->Cs(); // ?
  double Ss = params->Ss();
  double Ds = params->Ds();

  if (ONE_ANGLE)
  {
    APper2 = 1.0 - APlon2;
    APpar2 = 0.0;
    ASlon2 = 0.0;
  }

  //the cp-even part
  if (ONLY_EVEN)
  {
    APper2 = 0.0; APper = 0.0;
    ASlon2 = 0.0; ASlon = 0.0; //"s-wave is cp-odd"
  }
  //the cp-odd part
  if (ONLY_ODD)
  {
    APlon2 = 0.0; APlon = 0.0;
    APpar2 = 0.0; APpar = 0.0;
  }
  //the pure swave part
  if (ONLY_SWAVE)
  {
    APlon = 0.0;
    APpar = 0.0;
    APper = 0.0;
  }




  double a7  = 1.;

  double a8  =  0.5 * (cos(deltaSlon - deltaPpar) - lSlon*lPpar*cos(deltaSlon - deltaPpar - phisSlon + phisPpar));
  double a9  = -0.5 * (sin(deltaSlon - deltaPper) + lSlon*lPper*sin(deltaSlon - deltaPper - phisSlon + phisPper));
  double a10 =  0.5 * (cos(deltaSlon            ) - lSlon*lPlon*cos(deltaSlon             - phisSlon + phisPlon));

  double b7  =  Ss;

  double b8  =  0.5 * ( lSlon * cos(deltaSlon - deltaPpar - phisSlon) - lPpar * cos(deltaPpar - deltaSlon - phisPpar));
  double b9  = -0.5 * ( lSlon * sin(deltaSlon - deltaPper - phisSlon) - lPper * sin(deltaPper - deltaSlon - phisPper));
  double b10 =  0.5 * ( lSlon * cos(deltaSlon             - phisSlon) - lPlon * cos(          - deltaSlon - phisPlon));

  double c7  = Cs;

  double c8  =  0.5 * (cos(deltaSlon - deltaPpar) + lSlon*lPpar*cos(deltaSlon - deltaPpar - phisSlon + phisPpar));
  double c9  = -0.5 * (sin(deltaSlon - deltaPper) - lSlon*lPper*sin(deltaSlon - deltaPper - phisSlon + phisPper));
  double c10 =  0.5 * (cos(deltaSlon            ) + lSlon*lPlon*cos(deltaSlon             - phisSlon + phisPlon));

  double d7  = Ss;

  double d8  =  0.5 * ( lSlon * sin(deltaSlon - deltaPpar - phisSlon) - lPpar * sin(deltaPpar - deltaSlon - phisPpar));
  double d9  = -0.5 * (-lSlon * cos(deltaSlon - deltaPper - phisSlon) + lPper * cos(deltaPper - deltaSlon - phisPper));
  double d10 =  0.5 * ( lSlon * sin(deltaSlon             - phisSlon) - lPlon * sin(          - deltaSlon - phisPlon));

  double Z_light_s = 0, Z_heavy_s = 0, Z_one_s = 0, Z_two_s = 0;
  if (COS_DEL_PAR)
  {
    //are factors 0.5 missing?->added, see also s-wave note, needed to get correct ratio with the osc. terms
    //needed minus change due to note jhep 0909:074 sign error, the amplitude noted was in fact Im(A_perp^* A_s) and not Im(A_perp A_s^*)
    //note: minus sign in front of f9 was incorrect, no minus sign found from arxiv:1008.4283v2
    Z_light_s =
              ASlon * ASlon * f7  * 0.5 * (1+lSlon*lSlon) * (a7-b7)/2
      + CSP * APpar * ASlon * f8  * (a8-b8)/2
      + CSP * APper * ASlon * f9  * (a9-b9)/2
      + CSP * APlon * ASlon * f10 * (a10-b10)/2
      ;


    Z_heavy_s =
              ASlon * ASlon * f7  * 0.5 * (1+lSlon*lSlon) * (a7+b7)/2
      + CSP * APpar * ASlon * f8  * (a8+b8)/2
      + CSP * APper * ASlon * f9  * (a9+b9)/2
      + CSP * APlon * ASlon * f10 * (a10+b10)/2
      ;


    Z_one_s =
      +       ASlon * ASlon * f7  * 0.5 * (1+lSlon*lSlon) * d7
      + CSP * APpar * ASlon * f8  * d8
      + CSP * APper * ASlon * f9  * d9
      + CSP * APlon * ASlon * f10 * d10
      ;


    Z_two_s =
      +       ASlon * ASlon * f7  * 0.5 * (1+lSlon*lSlon) * c7
      + CSP * APpar * ASlon * f8  * c8
      + CSP * APper * ASlon * f9  * c9
      + CSP * APlon * ASlon * f10 * c10
      ;
  }
  else
  {

    double Z_light_s =
      +0.5*f7*ASlon2*(1.0-D)
      //+0.5*f9*sqrt( aperp2*as2 )*sin(dperp-ds)*(1-cos_phi_s);
      +0.5*f9*sqrt( aperp2*as2 )*sin_del_perp_s*(1-D);
      //-0.5*f9*sqrt( aperp2*as2 )*sin_del_perp_s*(1-cos_phi_s);

    double Z_heavy_s =
      +0.5*f7*ASlon2*(1.0+D)
      //+0.5*f9*sqrt( aperp2*as2 )*sin(dperp-ds)*(1+cos_phi_s);
      +0.5*f9*sqrt( aperp2*as2 )*sin_del_perp_s*(1+D);
    //-0.5*f9*sqrt( aperp2*as2 )*sin_del_perp_s*(1+cos_phi_s);

    double Z_one_s = 0.0; Z_two_s = 0.0

  }
  printf("Z_light_s=%lf, Z_heavy_s=%lf\n",Z_light_s,Z_heavy_s);
  printf("Z_one_s=%lf, Z_two_s=%lf\n",Z_one_s,Z_two_s);


  // Time dependent part -------------------------------------------------------
  double Y_light = 0, Y_heavy = 0, Y_one = 0, Y_two = 0;
  double f_res_1 = 1.0;
  double f_res_2 = 1.0;
  double sigma_1 = 1.0;
  double sigma_2 = 1.0;
  double sigma_3 = 1.0;
  double mu      = 0.0;


  if (USE_PER_EVENT_RES)
  {
    //f_res_1 = params->ct_res_1_sig();
    //f_res_2 = params->ct_res_2_sig();
    //sigma_1 = meas->sigma_ct()*params->s_sigma_ct_1();
    //sigma_2 = meas->sigma_ct()*params->s_sigma_ct_2();
    //sigma_3 = meas->sigma_ct()*params->s_sigma_ct_3();

    calc_per_event_res_params(params, meas, sigma_1, sigma_2, sigma_3, f_res_1, f_res_2, mu );

  }
  else if (USE_TIME_RES)
  {
    f_res_1 = params->ct_res_1_sig();
    f_res_2 = params->ct_res_2_sig();
    sigma_1 = params->s_ct_sigma() * params->ct_sigma_1_sig();
    sigma_2 = params->s_ct_sigma() * params->ct_sigma_2_sig();
    sigma_3 = params->s_ct_sigma() * params->ct_sigma_3_sig();
    mu = params->ct_sigma_mu();
  }
  else
  {
    f_res_1 = 1.;
    f_res_2 = 1.;
    sigma_1 = 0.;
    sigma_2 = 0.;
    sigma_3 = 0.;
    mu = 0;
  }

  if (USE_TRUETIME && !USE_TRUTH)
  {
    ct_part(params, meas->true_ct(), sigma_1,
          f_res_1, sigma_2,
          f_res_2, sigma_3, mu,
          Y_light, Y_heavy,
          Y_one, Y_two);
  }
  else
  {
    ct_part(params, meas->ct(), sigma_1,
          f_res_1, sigma_2,
          f_res_2, sigma_3, mu,
          Y_light, Y_heavy,
          Y_one, Y_two);
  }

  if (Y_light<0 || Y_heavy<0)
  {
    printf("time dependent part negative/nan: %lf,%lf\n", Y_light, Y_heavy);
  }

  double result = 0.0;
  double prob_Bs, prob_Bsbar = 0.0;






  return 1.0;

}








































__kernel
void pyAlgo(__global const double *data, __global double *lkhd,
                double G, double DG, double DM,
                __global const double * CSP,
                __global const double * ASlon,
                double APlon, double APpar, double APper, double phisPlon,
                double phisSlon, double phisPpar, double phisPper,
                double deltaSlon, double deltaPlon, double deltaPpar,
                double deltaPper,
                double lPlon, double lSlon, double lPpar, double lPper,
                double tLL, double tUL,
                int Nevt)
{
  int evt = get_global_id(0);
  int bin = get_global_id(1);
  if (evt >= Nevt) { return; }

  double mass = data[evt*5+4];
  if (get_local_size(1) > 1)                   // if fitting binned X_M spectrum
  {
    if ((mass >= X_M[bin]) && (mass < X_M[bin+1]))
    {
      double data4[4] = {data[evt*5+0],data[evt*5+1],data[evt*5+2],data[evt*5+3]};
      lkhd[evt] = getAlgo(data4,
                              G, DG, DM, CSP[bin], APlon, ASlon[bin], APpar, APper, phisPlon,
                              phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                              deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper, tLL, tUL, 1);
    }
  }
  else
  {
    double data4[4] = {data[evt*5+0],data[evt*5+1],data[evt*5+2],data[evt*5+3]};
    lkhd[evt] = getAlgo(data4,
                            G, DG, DM, CSP[0], APlon, ASlon[0], APpar, APper, phisPlon,
                            phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                            deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper, tLL, tUL, 1);
  }

  //printf("%lf\n", data[evt]);

  // DEBUG ! -------------------------------------------------------------------
  //printf("CSP: %lf\n", CSP[bin]);
  //printf("%d,%d,%d,%d\n",get_global_size(0),get_global_size(1),get_local_size(0),get_local_size(1));


}


*/










////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyDiffRate //////////////////////////////////////////////////////////

__kernel
void pyDiffRate(__constant double *data, __global double *lkhd,
                double G, double DG, double DM,
                __global const double * CSP,
                __global const double * ASlon,
                double APlon, __global const double * APpar, double APper,
                double phisSlon,
                double phisPlon, double phisPpar, double phisPper,
                __global const double * deltaSlon,
                double deltaPlon, double deltaPpar, double deltaPper,
                double lPlon,
                double lSlon, double lPpar, double lPper,
                double tLL, double tUL,
                __constant double *coeffs,
                int Nevt)
{
  int evt = get_global_id(0);
  int bin = get_global_id(1);
  if (evt >= Nevt) { return; }

  double shit[28];                                // check why this is mandatory
  for (int index =0; index < 28; index++)
  {
    shit[index] = coeffs[index];
  }

  double mass = data[evt*5+4];
  if (get_local_size(1) > 1)                   // if fitting binned X_M spectrum
  {
    if ((mass >= X_M[bin]) && (mass < X_M[bin+1]))
    {
      double data4[4] = {data[evt*5+0],data[evt*5+1],data[evt*5+2],data[evt*5+3]};
      lkhd[evt] = getDiffRate(data4,
                              G, DG, DM, CSP[bin],
                              APlon, ASlon[bin], APpar[bin], APper,
                              phisPlon, phisSlon, phisPpar, phisPper,
                              deltaSlon[bin], deltaPlon, deltaPpar, deltaPper,
                              lPlon, lSlon, lPpar, lPper,
                              tLL, tUL,
                              shit, 1);
    }
  }
  else
  {
    double data4[4] = {data[evt*5+0],data[evt*5+1],data[evt*5+2],data[evt*5+3]};
    lkhd[evt] = getDiffRate(data4,
                            G, DG, DM, CSP[0],
                            APlon, ASlon[0], APpar[0], APper,
                            phisPlon, phisSlon, phisPpar, phisPper,
                            deltaSlon[0], deltaPlon, deltaPpar, deltaPper,
                            lPlon, lSlon, lPpar, lPper,
                            tLL, tUL,
                            shit, 1);
  }

  //printf("%lf\n", data[evt]);
  //printf("CSP: %lf\n", CSP[bin]);
  //printf("%d,%d,%d,%d\n",get_global_size(0),get_global_size(1),get_local_size(0),get_local_size(1));
}




////////////////////////////////////////////////////////////////////////////////
