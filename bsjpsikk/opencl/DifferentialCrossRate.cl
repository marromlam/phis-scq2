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
                    double pSlon, double pPlon, double pPpar, double pPper,
                    double dSlon, double dPlon, double dPpar, double dPper,
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
    printf("pSlon           : %+lf\n", pSlon);
    printf("pPlon           : %+lf\n", pPlon);
    printf("pPpar           : %+lf\n", pPpar);
    printf("pPper           : %+lf\n", pPper);
    printf("dSlon          : %+lf\n", dSlon);
    printf("dPlon          : %+lf\n", dPlon);
    printf("dPper          : %+lf\n", dPper);
    printf("dPpar          : %+lf\n", dPpar);
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
  double t_offset = 0.0; double d_t = 0.0;
  double sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;
  double sigma_t_a = 0, sigma_t_b = 0, sigma_t_c = 0;

  if (USE_TIME_OFFSET)
  {
    t_offset = getTimeCal(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  }
  if (USE_TIME_RES)
  {
    d_t  = getTimeCal(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
  }
  //printf("d_t=%lf,\tt_offset=%lf\n",d_t,t_offset);
  exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., d_t);
  exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., d_t);
  exp_i = getExponentialConvolution(time-t_offset,          G, DM, d_t);

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
    tagOS = qOS/fabs(qOS);
    tagSS = qSS/fabs(qSS);
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

    ak = getA(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getB(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getC(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getD(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);

    //hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
    //hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);

    if (DM != 0)
    {
      hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
      hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);
    }
    else
    {
      // this is the Bd2JpsiKstar p.d.f
      hk_B = ak*ta + ck*tc;
      if ( (k==4) || (k==6)  || (k==9) )
      {
        hk_Bbar = tagOS*ak*ta + tagOS*ck*tc;
      }
      else
      {
        hk_Bbar = ak*ta + ck*tc;
      }
    }
    pdfB    += nk*fk*hk_B; pdfBbar += nk*fk*hk_Bbar;
    vnk[k-1] = nk; vak[k-1] = ak; vbk[k-1] = bk; vck[k-1] = ck; vdk[k-1] = dk;

//     if ( get_global_id(0) <5)
//     {
//     printf("%lf\t", k, tagOS == 1 ? nk*fk*hk_B/(    (9/(16*M_PI))  * exp(-G*time)  ) : nk*fk*hk_Bbar/(    (9/(16*M_PI))  * exp(-G*time)  ) );
//     if (k==10){
//     printf("\n%lf,\t%lf,\t%lf,\t%lf,\t%lf,\t%lf,\t%lf,\t%lf,\t%lf,\t%lf\t\t[%lf]\n",
//       APlon*APlon                              * ( cosK*cosK*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)),
//       APpar*APpar                              * ( 0.5*sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*(1.-cos(hphi)*cos(hphi)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)) ),
//       APper*APper                              * ( 0.5*sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*(1.-sin(hphi)*sin(hphi)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)) ),
// tagOS*APpar*APper*sin(dPper-dPpar)     * ( sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)*sin(hphi)*cos(hphi)),
//       APlon*APpar*cos(dPpar)               * ( sqrt(2.)*sqrt(1.-cosK*cosK)*cosK*sqrt(1.-cosL*cosL)*cosL*cos(hphi) ),
// tagOS*APlon*APper*sin(dPper)               * (-sqrt(2.)*sqrt(1.-cosK*cosK)*cosK*sqrt(1.-cosL*cosL)*cosL*sin(hphi) ),
//       ASlon*ASlon                              * ( sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)/3. ),
//       APpar*ASlon*CSP*cos(dPpar-dSlon) * ( 2.*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*cosL*cos(hphi)/sqrt(6.) ),
// tagOS*APper*ASlon*CSP*sin(dPper-dSlon) * (-2.*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*cosL*sin(hphi)/sqrt(6.) ),
//       APlon*ASlon*CSP*cos(dSlon)           * ( 2.*cosK*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)/sqrt(3.) ) ,tagOS
//     );
//
//                                                     }
//     }
  }








  // Compute pdf integral ------------------------------------------------------
  double intBBar[2] = {0.,0.};
  if ( (d_t == 0) & (USE_TIME_ACC == 0) )
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
//   if ( get_global_id(0) == 0)
//   {
//     printf("pdfB = %lf\n",  dta*(
//           (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
//           (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
//           ) );
//     printf("simon = %lf\n", (    (9/(16*M_PI))  * exp(-G*time)  )* (
//       APlon*APlon                              * ( cosK*cosK*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)) +
//       APpar*APpar                              * ( 0.5*sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*(1.-cos(hphi)*cos(hphi)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)) ) +
//       APper*APper                              * ( 0.5*sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*(1.-sin(hphi)*sin(hphi)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)) ) +
// tagOS*APpar*APper*sin(dPper-dPpar)     * ( sqrt(1.-cosK*cosK)*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)*sin(hphi)*cos(hphi)) +
//       APlon*APpar*cos(dPpar)               * ( sqrt(2.)*sqrt(1.-cosK*cosK)*cosK*sqrt(1.-cosL*cosL)*cosL*cos(hphi) ) +
// tagOS*APlon*APper*sin(dPper)               * (-sqrt(2.)*sqrt(1.-cosK*cosK)*cosK*sqrt(1.-cosL*cosL)*cosL*sin(hphi) ) +
//       ASlon*ASlon                              * ( sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)/3. ) +
//       APpar*ASlon*CSP*cos(dPpar-dSlon) * ( 2.*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*cosL*cos(hphi)/sqrt(6.) ) +
// tagOS*APper*ASlon*CSP*sin(dPper-dSlon) * (-2.*sqrt(1.-cosK*cosK)*sqrt(1.-cosL*cosL)*cosL*sin(hphi)/sqrt(6.) ) +
//       APlon*ASlon*CSP*cos(dSlon)           * ( 2.*cosK*sqrt(1.-cosL*cosL)*sqrt(1.-cosL*cosL)/sqrt(3.) )  )
//     );
//     printf("integrals = %lf\t %lf\n", intB, (1/G)*(exp(-G*tLL)-exp(-G*tUL))*(APlon*APlon+APpar*APpar+APper*APper+ASlon*ASlon) );
//   }
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
      printf("RESULT             : dta=%+lf\n",
             dta);
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
