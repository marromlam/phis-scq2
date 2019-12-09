////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//   Created: 2019-01-25                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of p-scq packages, Santiago's framework for the       //
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

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>

// Include disciplines
#include "DecayTimeAcceptance.cu"
#include "TimeAngularDistribution.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__device__
double getDiffRate( double *data,
                    double G, double DG, double DM, double CSP,
                    double ASlon, double APlon, double APpar, double APper,
                    double pSlon, double pPlon, double pPpar, double pPper,
                    double dSlon, double dPlon, double dPpar, double dPper,
                    double lSlon, double lPlon, double lPpar, double lPper,
                    double tLL, double tUL,
                    double *coeffs,
                    bool USE_FK)
{
  if ((DEBUG > 3) && ( threadIdx.x + blockDim.x * blockIdx.x < DEBUG_EVT) )
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
    printf("pSlon              : %+lf\n", pSlon);
    printf("pPlon              : %+lf\n", pPlon);
    printf("pPpar              : %+lf\n", pPpar);
    printf("pPper              : %+lf\n", pPper);
    printf("dSlon              : %+lf\n", dSlon);
    printf("dPlon              : %+lf\n", dPlon);
    printf("dPper              : %+lf\n", dPper);
    printf("dPpar              : %+lf\n", dPpar);
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

  if ((time>=tUL) || (time<=tLL))
  {
    printf("WARNING            : Event with time not within [tLL,tUL].\n");
  }



  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set sigma_t = 0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  pycuda::complex<double> exp_p, exp_m, exp_i;
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

  exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., delta_t);
  exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., delta_t);
  exp_i = getExponentialConvolution(time-t_offset,          G, DM, delta_t);

  double ta = pycuda::real(0.5*(exp_m + exp_p));     // cosh = (exp_m + exp_p)/2
  double tb = pycuda::real(0.5*(exp_m - exp_p));     // sinh = (exp_m - exp_p)/2
  double tc = pycuda::real(exp_i);                        // exp_i = cos + I*sin
  double td = pycuda::imag(exp_i);                        // exp_i = cos + I*sin



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
      printf("This event is not tagged!\n");
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
  double vfk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
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

    hk_B    = (ak*ta + bk*tb + ck*tc + dk*td); pdfB    += nk*fk*hk_B;
    hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td); pdfBbar += nk*fk*hk_Bbar;

    vnk[k-1] = 1.*nk; vfk[k-1] = 1.*fk;
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
    // This integral works for all decay times, remember delta_t != 0.
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
  if ((DEBUG >= 1) && ( threadIdx.x + blockDim.x * blockIdx.x < DEBUG_EVT))
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
                   k,vnk[k], vak[k], vbk[k], vck[k], vdk[k], vfk[k]);
          }
        }
      }
    }
  }


  // That's all folks!
  return num/den;
}






































































/*





__device__
void intgTimeAcceptance(double time_terms[4], double sigma, double gamma,
                            double dgamma, double dm, double *knots,
                            double *coeffs, int n, double t0)
{
  // Add tUL to knots list
  knots[7] = 15; n += 1;
  int const N = 7+1;
  double x[N];

  double aux1 = 1./(sqrt(2.)*sigma);

  for(int i = 0; i < spl_bins+1; i++)
  {
    x[i] = (knots[i] - t0)*aux1;
  }

  // Fill S matrix                (TODO speed to be gained here - S is constant)
  double S[spl_bins][4][4];
  for (int bin=0; bin < spl_bins; ++bin)
  {
    for (int i=0; i<4; ++i)
    {
      for (int j=0; j<4; ++j)
      {
        if(i+j < 4)
        {
          S[bin][i][j] = getCoeff(coeffs,bin,i+j)
                         *Factorial(i+j)/Factorial(j)/Factorial(i)/pow(2.0,i+j);
        }
        else
        {
          S[bin][i][j] = 0.;
        }
      }
    }
  }


  pycuda::complex<double> z_sinh, K_sinh[4], M_sinh[spl_bins+1][4];
  pycuda::complex<double> z_cosh, K_cosh[4], M_cosh[spl_bins+1][4];
  pycuda::complex<double> z_trig, K_trig[4], M_trig[spl_bins+1][4];

  z_cosh = sigma*pycuda::complex<double>(gamma-0.5*dgamma,0.)/sqrt(2.);
  z_sinh = sigma*pycuda::complex<double>(gamma+0.5*dgamma,0.)/sqrt(2.);
  z_trig = sigma*pycuda::complex<double>(gamma,-dm)/sqrt(2.);

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_cosh[j] = Kn(z_cosh,j);
    K_sinh[j] = Kn(z_sinh,j);
    K_trig[j] = Kn(z_trig,j);
  }

  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < spl_bins+1; ++bin)
    {
      M_sinh[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma-0.5*dgamma,0.);
      M_cosh[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma+0.5*dgamma,0.);
      M_trig[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma,dm);
    }
  }

  // Fill the delta factors to multiply by the integrals
  double sigma_fact[4];
  for (int i=0; i<4; ++i)
  {
    sigma_fact[i] = pow(sigma*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, sinh, cos, sin terms
  double int_sinh = 0; double int_cosh = 0;
  pycuda::complex<double> int_trig = pycuda::complex<double>(0.,0.);

  for (int bin=0; bin < spl_bins; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        int_sinh += pycuda::real(S[bin][j][k]*(M_sinh[bin+1][j]-M_sinh[bin][j])
                    *K_cosh[k])*sigma_fact[j+k];

        int_cosh += pycuda::real(S[bin][j][k]*(M_cosh[bin+1][j]-M_cosh[bin][j])
                    *K_sinh[k])*sigma_fact[j+k];

        int_trig += S[bin][j][k]*(M_trig[bin+1][j] - M_trig[bin][j])
                    *K_trig[k]*sigma_fact[j+k];
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = 0.5*(int_sinh + int_cosh);
  time_terms[1] = 0.5*(int_sinh - int_cosh);
  time_terms[2] = pycuda::real(int_trig);
  time_terms[3] = pycuda::imag(int_trig);

}



*/
/*
//This integral works for all decay times.
__device__
void integral4pitime_full_spline( double integral[2], double vNk[10], double vak[10],double vbk[10],
                                  double vck[10],double vdk[10], double *normweights, double G, double DG, double DM,
                                  double delta_t, double t_ll, double t_offset, int spline_Nknots, double *spline_knots, double *spline_coeffs)
{
    double time_terms[4] = {0., 0., 0., 0.};
    intgTimeAcceptance(time_terms, delta_t, G, DG, DM, spline_knots, spline_coeffs, spline_Nknots, t_offset) ;

    double int_ta = time_terms[0];
    double int_tb = time_terms[1];
    double int_tc = time_terms[2];
    double int_td = time_terms[3];

    for(int k=0; k<10; k++)
    {
        integral[0] += vNk[k]*normweights[k]*(vak[k]*int_ta + vbk[k]*int_tb + vck[k]*int_tc + vdk[k]*int_td);
        integral[1] += vNk[k]*normweights[k]*(vak[k]*int_ta + vbk[k]*int_tb - vck[k]*int_tc - vdk[k]*int_td);
    }
}
*/

/*
__device__ double IntegralTimeA(double t_0, double t_1, double G,double DG)
{
    return (2*(DG*sinh(.5*DG*t_0) + 2*G*cosh(.5*DG*t_0))*exp(G*t_1) - 2*(DG*sinh(.5*DG*t_1) + 2*G*cosh(.5*DG*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(-pow(DG, 2) + 4 *pow(G, 2));
}
__device__ double IntegralTimeB(double t_0, double t_1,double G,double DG)
{
    return (2*(DG*cosh(.5*DG*t_0) + 2*G*sinh(.5*DG*t_0))*exp(G*t_1) - 2*(DG*cosh(.5*DG*t_1) + 2*G*sinh(.5*DG*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(-pow(DG, 2) + 4*pow(G, 2));
}
__device__ double IntegralTimeC(double t_0, double t_1,double G,double DM)
{
    return ((-DM*sin(DM*t_0) + G*cos(DM*t_0))*exp(G*t_1) + (DM*sin(DM*t_1) - G*cos(DM*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(pow(DM, 2) + pow(G, 2));
}
__device__ double IntegralTimeD(double t_0, double t_1,double G,double DM)
{
    return ((DM*cos(DM*t_0) + G*sin(DM*t_0))*exp(G*t_1) - (DM*cos(DM*t_1) + G*sin(DM*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(pow(DM, 2) + pow(G, 2));
}

__device__ void Integral4PiTime(double result[2],
                                double vnk[10],
                                double vak[10],
                                double vbk[10],
                                double vck[10],
                                double vdk[10],
                                double *normweights,
                                double Gamma, double DeltaGamma, double DeltaM,
                                double tLL, double tUL, double TimeOffset)
{
  double IntTimeA = IntegralTimeA(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeB = IntegralTimeB(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeC = IntegralTimeC(tLL, tUL, Gamma, DeltaM);
  double IntTimeD = IntegralTimeD(tLL, tUL, Gamma, DeltaM);

  for(int k=0; k<10 ; k++)
  {
    result[0] += vnk[k]*normweights[k]*(vak[k]*IntTimeA +
                                        vbk[k]*IntTimeB +
                                        vck[k]*IntTimeC +
                                        vdk[k]*IntTimeD);
    result[1] += vnk[k]*normweights[k]*(vak[k]*IntTimeA +
                                        vbk[k]*IntTimeB -
                                        vck[k]*IntTimeC -
                                        vdk[k]*IntTimeD);
  }
  // printf("   INTEGRALS: %.4lf\t%.4lf\n",result[0],result[1] );
  // printf("       RANGE: %.4lf\t%.4lf\n",tLL,tUL );
  // printf("   INTEGRALS: %.4lf\t%.4lf\t%.4lf\t%.4lf\n",result[0],IntTimeA,IntTimeB,IntTimeC,IntTimeD );
}
*/



// __device__
// double getDiffRate( double *data, double G, double DG, double DM, double CSP,
//                     double APlon, double ASlon, double APpar, double APper,
//                     double pPlon, double pSlon, double pPpar, double pPper,
//                     double dSlon, double dPlon, double dPpar, double dPper,
//                     double lPlon, double lSlon, double lPpar, double lPper,
//                     double tLL, double tUL,
//                     double *coeffs,
//                     bool USE_FK)
// {
//   if ((DEBUG > 3) && ( threadIdx.x + blockDim.x * blockIdx.x < DEBUG_EVT) )
//   {
//     printf("*USE_FK            : %d\n", USE_FK);
//     printf("*USE_TIME_ACC      : %d\n", USE_TIME_ACC);
//     printf("*USE_TIME_OFFSET   : %d\n", USE_TIME_OFFSET);
//     printf("*USE_TIME_RES      : %d\n", USE_TIME_RES);
//     printf("*USE_PERFTAG       : %d\n", USE_PERFTAG);
//     printf("*USE_TRUETAG       : %d\n", USE_TRUETAG);
//     printf("G                  : %+lf\n", G);
//     printf("DG                 : %+lf\n", DG);
//     printf("DM                 : %+lf\n", DM);
//     printf("CSP                : %+lf\n", CSP);
//     printf("ASlon              : %+lf\n", ASlon);
//     printf("APlon              : %+lf\n", APlon);
//     printf("APpar              : %+lf\n", APpar);
//     printf("APper              : %+lf\n", APper);
//     printf("pSlon           : %+lf\n", pSlon);
//     printf("pPlon           : %+lf\n", pPlon);
//     printf("pPpar           : %+lf\n", pPpar);
//     printf("pPper           : %+lf\n", pPper);
//     printf("dSlon          : %+lf\n", dSlon);
//     printf("dPlon          : %+lf\n", dPlon);
//     printf("dPper          : %+lf\n", dPper);
//     printf("dPpar          : %+lf\n", dPpar);
//     printf("lSlon              : %+lf\n", lSlon);
//     printf("lPlon              : %+lf\n", lPlon);
//     printf("lPper              : %+lf\n", lPper);
//     printf("lPpar              : %+lf\n", lPpar);
//     printf("tLL                : %+lf\n", tLL);
//     printf("tUL                : %+lf\n", tUL);
//     printf("COEFFS             : %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
//     printf("                     %+lf\t%+lf\t%+lf\t%+lf\n",
//             coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
//   }
//
//
//   double normweights[10] = {1,1,1,0,0,0,1,0,0,0};
//
//   // Variables -----------------------------------------------------------------
//   //     Make sure that the input it's in this order.
//   //     lalala
//   double cosK       = data[0];                      // Time-angular distribution
//   double cosL       = data[1];
//   double hphi       = data[2];
//   double time       = data[3];
//
//   double sigma_t    = data[4];                                // Time resolution
//
//   double qOS        = data[5];                                        // Tagging
//   double qSS        = data[5];
//
//
//   // double eta_OS 		= data[7];
//   // double etaSlonSK 	= data[8];
//   // int year 					= data[9];
//
//
//
//
// /*
//   double delta_t =  delta(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
//
//   double delta_t_1 = delta_1(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);
//   double delta_t_2 = delta_2(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);
//
//   double omega_OS = omega(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
//   double omega_bar_OS = omega_bar(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
//   double omegaSlonSK = omega(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);
//   double omega_barSlonSK = omega_bar(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);
//
//   double taggingPparrs_OS[3] = {omega_OS, omega_bar_OS, q_OS};
//   double taggingPparrsSlonSK[3] = {omegaSlonSK, omega_barSlonSK, qSlonSK};
//
//   fix_taggingPparrs(taggingPparrs_OS);
//   fix_taggingPparrs(taggingPparrsSlonSK);
//
//   omega_OS = taggingPparrs_OS[0];
//   omega_bar_OS = taggingPparrs_OS[1];
//   omegaSlonSK = taggingPparrsSlonSK[0];
//   omega_barSlonSK = taggingPparrsSlonSK[1];
//
//   if((taggingPparrs_OS[0] == 0.5 || taggingPparrs_OS[1] == 0.5) && (taggingPparrs_OS[0] != taggingPparrs_OS[1]))
//   printf("OS tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrs_OS[0], taggingPparrs_OS[1], taggingPparrs_OS[2]);
//   else
//   q_OS = taggingPparrs_OS[2];
//
//   if((taggingPparrsSlonSK[0] == 0.5 || taggingPparrsSlonSK[1] == 0.5) && (taggingPparrsSlonSK[0] != taggingPparrsSlonSK[1]))
//   printf("SSK tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrsSlonSK[0], taggingPparrsSlonSK[1], taggingPparrsSlonSK[2]);
//   else
//   qSlonSK = taggingPparrsSlonSK[2];
//
// */
//
//
//
//
//   // Time resolution -----------------------------------------------------------
//   //     In order to remove the effects of conv, set delta_t = 0, so in this way
//   //     you are running the first branch of getExponentialConvolution.
//   pycuda::complex<double> exp_p, exp_m, exp_i;
//   double t_offset = 0.0; double delta_t = 0.0;
//   double sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;
//   double sigma_t_a = 0, sigma_t_b = 0, sigma_t_c = 0;
//
//   if (USE_TIME_OFFSET)
//   {
//     t_offset = getTimeCal(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
//   }
//   if (USE_TIME_RES)
//   {
//     delta_t  = getTimeCal(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
//   }
//   //printf("delta_t=%lf,\tt_offset=%lf\n",delta_t,t_offset);
//   exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., delta_t);
//   exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., delta_t);
//   exp_i = getExponentialConvolution(time-t_offset,          G, DM, delta_t);
//
//   double ta = pycuda::real(0.5*(exp_m + exp_p));     // cosh = (exp_m + exp_p)/2
//   double tb = pycuda::real(0.5*(exp_m - exp_p));     // sinh = (exp_m - exp_p)/2
//   double tc = pycuda::real(exp_i);                        // exp_i = cos + I*sin
//   double td = pycuda::imag(exp_i);                        // exp_i = cos + I*sin
//
//
//
//   // Flavor tagging ------------------------------------------------------------
//   double omegaOSB = 0; double omegaOSBbar = 0; double tagOS = 0;
//   double omegaSSB = 0; double omegaSSBbar = 0; double tagSS = 0;
//
//   if (USE_TRUETAG)
//   {
//     tagOS = 0.0;
//     tagSS = 0.0;
//   }
//   else if (USE_PERFTAG)
//   {
//     tagOS = qOS/531;
//     tagSS = qSS/531;
//     if ((tagOS == 0)|(tagSS == 0))
//     {
//       printf("This events is not tagged!\n");
//     }
//   }
//
//
//
//   // Decay-time acceptance -----------------------------------------------------
//   //     To get rid of decay-time acceptance set USE_TIME_ACC to False. If True
//   //     then calcTimeAcceptance locates the time bin of the event and returns
//   //     the value of the cubic spline.
//   double dta = 1.0;
//   if (USE_TIME_ACC)
//   {
//     dta = calcTimeAcceptance(time, coeffs, tLL, tUL);
//   }
//
//
//
//   // Compute per event pdf -----------------------------------------------------
//   double vnk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
//   double vak[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
//   double vbk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
//   double vck[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
//   double vdk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
//
//   double nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
//   double pdfB = 0.0; double pdfBbar = 0.0;
//
//   for(int k = 1; k <= 10; k++)
//   {
//     nk = getN(APlon,ASlon,APpar,APper,CSP,k);
//     if (USE_FK)
//     {
//       fk = (9/(16*M_PI))*getF(cosK,cosL,hphi,k);
//     }
//     else
//     {
//       fk = normweights[k-1]; // these are 0s or 1s
//     }
//
//     ak = getA(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
//     bk = getB(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
//     ck = getC(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
//     dk = getD(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
//
//     hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
//     hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);
//
//     pdfB    += nk*fk*hk_B; pdfBbar += nk*fk*hk_Bbar;
//
//     vnk[k-1] = 1.*nk;
//     vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
//   }
//
//
//
//   // Compute pdf integral ------------------------------------------------------
//   double intBBar[2] = {0.,0.};
//   if ( (delta_t == 0) & (USE_TIME_ACC == 0) )
//   {
//     // Here we can use the simplest 4xPi integral of the pdf since there are no
//     // resolution effects
//     integralSimple(intBBar,
//                    vnk, vak, vbk, vck, vdk, normweights, G, DG, DM, tLL, tUL);
//   }
//   else
//   {
//     // This integral works for all decay times, remember sigma_t != 0.
//     double knots[NKNOTS];
//     for (int i=0; i<NKNOTS; i++) {knots[i] = KNOTS[i];}   // why is this needed?
//     integralFullSpline(intBBar,
//                        vnk, vak, vbk, vck, vdk,
//                        normweights,  G, DG,  DM,
//                        sigma_t,
//                        tLL,
//                        t_offset,
//                        NKNOTS, knots,
//                        coeffs);
//   }
//   double intB = intBBar[0]; double intBbar = intBBar[1];
//
//
//
//   // Cooking the output --------------------------------------------------------
//   double num = 1.0; double den = 1.0;
//   num = dta*(
//         (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
//         (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
//         );
//   den = 1.0*(
//         (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * intB +
//         (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * intBbar
//         );
//
//   // DEBUG ! -------------------------------------------------------------------
//   //printf("t=%+0.3lf\tcosK=%+0.3lf\tcosL=%+0.3lf\thphi=%+0.3lf\tpdf=%+0.3lf\tipdf=%+0.3lf\t --> pdf/ipdf=%+lf\n", time,cosK,cosL,hphi, num,den,num/den);
//   //printf("USE_FK=%d\tt=%+lf\tpdf=%+lf\tipdf=%+lf\t --> pdf/ipdf=%+lf\n", USE_FK, time, num,den,num/den);
//   // for(int k = 0; k < 10; k++)
//   // {
//   //   printf("--> %.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", vNk[k], vak[k], vbk[k], vck[k], vdk[k],
//   //                    normweights[k]);
//   // }
//   return num/den;
// }
