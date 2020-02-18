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
//                  data --it launches a thread per mass bin.                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

// Include disciplines
#include "DecayTimeAcceptance.cl"
#include "TimeAngularDistribution.cl"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



//__global
double getDiffRate( double *data, double G, double DG, double DM, double CSP,
                    double ASlon, double APlon, double APpar, double APper,
                    double phisSlon, double phisPlon, double phisPpar, double phisPper,
                    double deltaSlon, double deltaPlon, double deltaPpar, double deltaPper,
                    double lSlon, double lPlon, double lPpar, double lPper,
                    double tLL, double tUL,
                    double *coeffs,
                    bool USE_FK)
{
  if ((DEBUG > 3) && ( get_global_id(0) < DEBUG_EVT) )
  {
    printf("*USE_FK            : %d\n", USE_FK);
    printf("*USE_TIME_ACC      : %d\n", USE_TIME_ACC);
    printf("*USE_TIME_OFFSET   : %d\n", USE_TIME_OFFSET);
    printf("*USE_TIME_RES      : %d\n", USE_TIME_RES);
    printf("*USE_PERFTAG       : %d\n", USE_PERFTAG);
    printf("*USE_TRUETAG       : %d\n", USE_TRUETAG);
    printf("G                  : %+lf\n", G);
    printf("DG                 : %+lf\n", DG);
    printf("DM                 : %+lf\n", DM);
    printf("CSP                : %+lf\n", CSP);
    printf("ASlon              : %+lf\n", ASlon);
    printf("APlon              : %+lf\n", APlon);
    printf("APpar              : %+lf\n", APpar);
    printf("APper              : %+lf\n", APper);
    printf("phisSlon           : %+lf\n", phisSlon);
    printf("phisPlon           : %+lf\n", phisPlon);
    printf("phisPpar           : %+lf\n", phisPpar);
    printf("phisPper           : %+lf\n", phisPper);
    printf("deltaSlon          : %+lf\n", deltaSlon);
    printf("deltaPlon          : %+lf\n", deltaPlon);
    printf("deltaPper          : %+lf\n", deltaPper);
    printf("deltaPpar          : %+lf\n", deltaPpar);
    printf("lSlon              : %+lf\n", lSlon);
    printf("lPlon              : %+lf\n", lPlon);
    printf("lPper              : %+lf\n", lPper);
    printf("lPpar              : %+lf\n", lPpar);
    printf("tLL                : %+lf\n", tLL);
    printf("tUL                : %+lf\n", tUL);
    printf("COEFFS             : %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
    printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
            coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  }

  double normweights[10] = {1,1,1,0,0,0,1,0,0,0};

  // Variables -----------------------------------------------------------------
  //     Make sure that the input it's in this order.
  //     lalala
  double cosK       = data[0];                      // Time-angular distribution
  double cosL       = data[1];
  double hphi       = data[2];
  double time       = data[3];

  double sigma_t    = data[4];                                // Time resolution

  double qOS        = data[5];                                        // Tagging
  double qSS        = data[5];

  // double eta_OS 		= data[7];
  // double etaSlonSK 	= data[8];
  // int year 					= data[9];

  // if ((time>=tUL) || (time<=tLL))
  // {
  //   printf("WARNING            : Event with time not within [tLL,tUL].\n");
  //   return 0;
  // }



  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set sigma_t = 0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  cdouble_t exp_p, exp_m, exp_i;
  double t_offset = 0.0; double delta_t = 0.0;
  double sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;
  double sigma_t_a = 0, sigma_t_b = 0, sigma_t_c = 0;

  if (USE_TIME_OFFSET)
  {
    t_offset = getTimeCal(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  }
  if (USE_TIME_RES)
  {
    delta_t  = getTimeCal(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
  }
  //printf("delta_t=%lf,\tt_offset=%lf\n",delta_t,t_offset);
  exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., delta_t);
  exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., delta_t);
  exp_i = getExponentialConvolution(time-t_offset,          G, DM, delta_t);

  //double ta = pyopencl::real(0.5*(exp_m + exp_p)); // cosh = (exp_m + exp_p)/2
  //double tb = pyopencl::real(0.5*(exp_m - exp_p)); // sinh = (exp_m - exp_p)/2
  //double tc = pyopencl::real(exp_i);                    // exp_i = cos + I*sin
  //double td = pyopencl::imag(exp_i);                    // exp_i = cos + I*sin
  double ta = 0.5*cdouble_add(exp_m,exp_p).real;
  double tb = 0.5*cdouble_add(exp_m,cdouble_mul(cdouble_new(-1,0),exp_p)).real;
  double tc = exp_i.real;
  double td = exp_i.imag;



  // Flavor tagging ------------------------------------------------------------
  double omegaOSB = 0; double omegaOSBbar = 0; double tagOS = 0;
  double omegaSSB = 0; double omegaSSBbar = 0; double tagSS = 0;

  if (USE_TRUETAG)
  {
    tagOS = 0.0;
    tagSS = 0.0;
  }
  else if (USE_PERFTAG)
  {
    tagOS = qOS/531;
    tagSS = qSS/531;
    if ((tagOS == 0)|(tagSS == 0))
    {
      printf("This events is not tagged!\n");
    }
  }



  // Decay-time acceptance -----------------------------------------------------
  //     To get rid of decay-time acceptance set USE_TIME_ACC to False. If True
  //     then calcTimeAcceptance locates the time bin of the event and returns
  //     the value of the cubic spline.
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
      fk = (9/(16*M_PI))*getF(cosK,cosL,hphi,k);
    }
    else
    {
      fk = normweights[k-1]; // these are 0s or 1s
    }

    ak = getA(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getB(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getC(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getD(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);

    hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
    hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);

    pdfB    += nk*fk*hk_B; pdfBbar += nk*fk*hk_Bbar;

    vnk[k-1] = 1.*nk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
  }



  // Compute pdf integral ------------------------------------------------------
  double intBBar[2] = {0.,0.};
  if ( (delta_t == 0) & (USE_TIME_ACC == 0) )
  {
    // Here we can use the simplest 4xPi integral of the pdf since there are no
    // resolution effects
    integralSimple(intBBar,
                   vnk, vak, vbk, vck, vdk, normweights, G, DG, DM, tLL, tUL);
  }
  else
  {
    // This integral works for all decay times, remember sigma_t != 0.
    double knots[NKNOTS];
    for (int i=0; i<NKNOTS; i++) {knots[i] = KNOTS[i];}   // why is this needed?
    integralFullSpline(intBBar,
                       vnk, vak, vbk, vck, vdk,
                       normweights,  G, DG,  DM,
                       sigma_t,
                       tLL,
                       t_offset,
                       NKNOTS, knots,
                       coeffs);
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
  if ((DEBUG >= 1) && ( get_global_id(0) < DEBUG_EVT))
  {
    printf("INPUT              : cosK=%+lf\tcosL=%+lf\thphi=%+lf\ttime=%+lf\tq=%+lf\n",
           cosK,cosL,hphi,time,tagOS);
    printf("RESULT             : pdf=%.8lf\tipdf=%.8lf\tpdf/ipdf=%.15lf\n",
           num,den,num/den);
    if (DEBUG >= 2)
    {
      printf("RESULT             : pdfB=%+lf\tpdBbar=%+lf\tipdfB=%+lf\tipdfBbar=%+lf\n",
             pdfB,pdfBbar,intB,intBbar);
      printf("RESULT             : dta=%+lf\tpdBbar=%+lf\tipdfB=%+lf\tipdfBbar=%+lf\n",
             dta,pdfBbar,intB,intBbar);
      if (DEBUG >= 3)
      {
        printf("TIME ACC           : ta=%.8lf\ttb=%.8lf\ttc=%.8lf\ttd=%.8lf\n",
               ta,tb,tc,td);
        if (DEBUG >= 4)
        {
          for(int k = 0; k < 10; k++)
          {
            printf("ANGULAR PART   (%d) : %+lf\t%+lf\t%+lf\t%+lf\t%+lf\t%+lf\n",
                   k,vnk[k], vak[k], vbk[k], vck[k], vdk[k],
                   (9/(16*M_PI))*getF(cosK,cosL,hphi,k+1));
          }
        }
      }
    }
  }


  // That's all folks!
  return num/den;
}
