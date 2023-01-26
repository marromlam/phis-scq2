////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//  Created: 2019-01-25                                                       //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
// #include <thrust/complex.h>
#include <pycuda-complex.hpp>
#include <curand.h>
#include <curand_kernel.h>
//#include "/scratch15/diego/gitcrap4/cuda/tag_gen.c"
#include "/home3/marcos.romero/JpsiKKAna/cuda/somefunctions.c"
#define errf_const 1.12837916709551
#define xLim 5.33
#define yLim 4.29
#define sigma_t_threshold 5
#define time_acc_bins 40
#define spl_bins 7

extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
{





////////////////////////////////////////////////////////////////////////////////
// Time evolution functions and integrals //////////////////////////////////////

__device__ double TimeA (double t, double Gamma, double DeltaGamma)
{
  return exp(-Gamma*t)*cosh(DeltaGamma*t/2);
}
__device__ double TimeB (double t, double Gamma, double DeltaGamma)
{
  return exp(-Gamma*t)*sinh(DeltaGamma*t/2);
}
__device__ double TimeC (double t, double Gamma, double DeltaM)
{
  return exp(-Gamma*t)*cos(DeltaM*t);
}
__device__ double TimeD (double t, double Gamma, double DeltaM)
{
  return exp(-Gamma*t)*sin(DeltaM*t);
}

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

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Coefficients ////////////////////////////////////////////////////////////////

__device__ double NKTerm( int k,
                          double ASlon,
                          double APper, double APlon, double APpar,
                          double ADper, double ADlon, double ADpar,
                          double CSP, double CSD, double CPD )
{
  double NK;
  switch(k) {
    case 1:   NK = pow(ASlon,2);
              break;
    case 2:   NK = APper*ASlon*CSP;
              break;
    case 3:   NK = pow(APper,2);
              break;
    case 4:   NK = APlon*ASlon*CSP;
              break;
    case 5:   NK = APlon*APper;
              break;
    case 6:   NK = pow(APlon,2);
              break;
    case 7:   NK = APpar*ASlon*CSP;
              break;
    case 8:   NK = APpar*APper;
              break;
    case 9:   NK = APlon*APpar;
              break;
    case 10:  NK = pow(APpar,2);
              break;
    case 11:  NK = ADper*ASlon*CSD;
              break;
    case 12:  NK = ADper*APper*CPD;
              break;
    case 13:  NK = ADper*APlon*CPD;
              break;
    case 14:  NK = ADper*APpar*CPD;
              break;
    case 15:  NK = pow(ADper,2);
              break;
    case 16:  NK = ADlon*ASlon*CSD;
              break;
    case 17:  NK = ADlon*APper*CPD;
              break;
    case 18:  NK = ADlon*APlon*CPD;
              break;
    case 19:  NK = ADlon*APpar*CPD;
              break;
    case 20:  NK = ADlon*ADper;
              break;
    case 21:  NK = pow(ADlon,2);
              break;
    case 22:  NK = ADpar*ASlon*CSD;
              break;
    case 23:  NK = ADpar*APper*CPD;
              break;
    case 24:  NK = ADpar*APlon*CPD;
              break;
    case 25:  NK = ADpar*APpar*CPD;
              break;
    case 26:  NK = ADpar*ADper;
              break;
    case 27:  NK = ADlon*ADpar;
              break;
    case 28:  NK = pow(ADpar,2);
              break;
    default:  printf("Wrong k index in NK. %d\\n", k); return 0.;
  }
  return NK;
}


__device__ double FKTerm( int k, double cosL, double cosK, double HelPhi )
{
  double sinK = sqrt(1. - cosK*cosK);
  double sinL = sqrt(1. - cosL*cosL);
  double sinP = sin(+HelPhi);
  double cosP = cos(+HelPhi);

  double FK;
  switch(k) {
    case 1:   FK = pow(sinL,2)/3.;
              break;
    case 2:   FK = sqrt(0.6666666666666666)*cosL*sinP*sinK*sinL;
              break;
    case 3:   FK = ((pow(cosP,2) + pow(cosL,2)*pow(sinP,2))*pow(sinK,2))/2.;
              break;
    case 4:   FK = (2*cosK*pow(sinL,2))/sqrt(3.);
              break;
    case 5:   FK = -(sqrt(2.)*cosK*cosL*sinP*sinK*sinL);
              break;
    case 6:   FK = pow(cosK,2)*pow(sinL,2);
              break;
    case 7:   FK = sqrt(0.6666666666666666)*cosP*cosL*sinK*sinL;
              break;
    case 8:   FK = cosP*sinP*pow(sinK,2)*pow(sinL,2);
              break;
    case 9:   FK = sqrt(2.)*cosP*cosK*cosL*sinK*sinL;
              break;
    case 10:  FK = ((pow(cosP,2)*pow(cosL,2) + pow(sinP,2))*pow(sinK,2))/2.;
              break;
    case 11:  FK = sqrt(3.3333333333333335)*cosK*cosL*sinP*sinK*sinL;
              break;
    case 12:  FK = sqrt(5.)*cosK*(pow(cosP,2) + pow(cosL,2)*pow(sinP,2))*pow(sinK,2);
              break;
    case 13:  FK = sqrt(10.)*pow(cosK,2)*cosL*sinP*sinK*sinL;
              break;
    case 14:  FK = -(sqrt(5.)*cosP*cosK*sinP*pow(sinK,2)*pow(sinL,2));
              break;
    case 15:  FK = (5*pow(cosK,2)*(pow(cosP,2) + pow(cosL,2)*pow(sinP,2))*pow(sinK,2))/2.;
              break;
    case 16:  FK = (sqrt(5.)*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*pow(sinL,2))/6.;
              break;
    case 17:  FK = -(sqrt(0.8333333333333334)*cosL*sinP*sinK*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*sinL)/2.;
              break;
    case 18:  FK = (sqrt(1.6666666666666667)*cosK*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*pow(sinL,2))/2.;
              break;
    case 19:  FK = (sqrt(0.8333333333333334)*cosP*cosL*sinK*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*sinL)/2.;
              break;
    case 20:  FK = (-5*cosK*cosL*sinP*sinK*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*sinL)/(2.*sqrt(6.));
              break;
    case 21:  FK = (5*pow(1 + 3*(pow(cosK,2) - pow(sinK,2)),2)*pow(sinL,2))/48.;
              break;
    case 22:  FK = sqrt(3.3333333333333335)*cosP*cosK*cosL*sinK*sinL;
              break;
    case 23:  FK = sqrt(5.)*cosP*cosK*sinP*pow(sinK,2)*pow(sinL,2);
              break;
    case 24:  FK = sqrt(10.)*cosP*pow(cosK,2)*cosL*sinK*sinL;
              break;
    case 25:  FK = sqrt(5.)*cosK*(pow(cosP,2)*pow(cosL,2) + pow(sinP,2))*pow(sinK,2);
              break;
    case 26:  FK = 5*cosP*pow(cosK,2)*sinP*pow(sinK,2)*pow(sinL,2);
              break;
    case 27:  FK = (5*cosP*cosK*cosL*sinK*(1 + 3*(pow(cosK,2) - pow(sinK,2)))*sinL)/(2.*sqrt(6.));
              break;
    case 28:  FK = (5*pow(cosK,2)*(pow(cosP,2)*pow(cosL,2) + pow(sinP,2))*pow(sinK,2))/2.;
              break;
    default:  printf("Wrong k index in FK. %d\\n", k); return 0.;
  }
  return FK;
}


__device__ double AKTerm( int k,
                          double lamSlon, double dSlon, double phisSlon,
                          double lamPper, double dPper, double phisPper,
                          double lamPlon, double dPlon, double phisPlon,
                          double lamPpar, double dPpar, double phisPpar,
                          double lamDper, double dDper, double phisDper,
                          double lamDlon, double dDlon, double phisDlon,
                          double lamDpar, double dDpar, double phisDpar )
{
  double AK;
  switch(k) {
    case 1:   AK = (1 + pow(lamSlon,2))/2.;
              break;
    case 2:   AK = (-sin(dPper - dSlon) - lamPper*lamSlon*sin(dPper - dSlon - phisPper + phisSlon))/2.;
              break;
    case 3:   AK = (1 + pow(lamPper,2))/2.;
              break;
    case 4:   AK = (cos(dPlon - dSlon) - lamPlon*lamSlon*cos(dPlon - dSlon - phisPlon + phisSlon))/2.;
              break;
    case 5:   AK = (-sin(dPlon - dPper) + lamPlon*lamPper*sin(dPlon - dPper - phisPlon + phisPper))/2.;
              break;
    case 6:   AK = (1 + pow(lamPlon,2))/2.;
              break;
    case 7:   AK = (cos(dPpar - dSlon) - lamPpar*lamSlon*cos(dPpar - dSlon - phisPpar + phisSlon))/2.;
              break;
    case 8:   AK = (-sin(dPpar - dPper) + lamPpar*lamPper*sin(dPpar - dPper - phisPpar + phisPper))/2.;
              break;
    case 9:   AK = (cos(dPlon - dPpar) + lamPlon*lamPpar*cos(dPlon - dPpar - phisPlon + phisPpar))/2.;
              break;
    case 10:  AK = (1 + pow(lamPpar,2))/2.;
              break;
    case 11:  AK = (-sin(dDper - dSlon) + lamDper*lamSlon*sin(dDper - dSlon - phisDper + phisSlon))/2.;
              break;
    case 12:  AK = (cos(dDper - dPper) - lamDper*lamPper*cos(dDper - dPper - phisDper + phisPper))/2.;
              break;
    case 13:  AK = (-sin(dDper - dPlon) - lamDper*lamPlon*sin(dDper - dPlon - phisDper + phisPlon))/2.;
              break;
    case 14:  AK = (-sin(dDper - dPpar) - lamDper*lamPpar*sin(dDper - dPpar - phisDper + phisPpar))/2.;
              break;
    case 15:  AK = (1 + pow(lamDper,2))/2.;
              break;
    case 16:  AK = (cos(dDlon - dSlon) + lamDlon*lamSlon*cos(dDlon - dSlon - phisDlon + phisSlon))/2.;
              break;
    case 17:  AK = (-sin(dDlon - dPper) - lamDlon*lamPper*sin(dDlon - dPper - phisDlon + phisPper))/2.;
              break;
    case 18:  AK = (cos(dDlon - dPlon) - lamDlon*lamPlon*cos(dDlon - dPlon - phisDlon + phisPlon))/2.;
              break;
    case 19:  AK = (cos(dDlon - dPpar) - lamDlon*lamPpar*cos(dDlon - dPpar - phisDlon + phisPpar))/2.;
              break;
    case 20:  AK = (-sin(dDlon - dDper) + lamDlon*lamDper*sin(dDlon - dDper - phisDlon + phisDper))/2.;
              break;
    case 21:  AK = (1 + pow(lamDlon,2))/2.;
              break;
    case 22:  AK = (cos(dDpar - dSlon) + lamDpar*lamSlon*cos(dDpar - dSlon - phisDpar + phisSlon))/2.;
              break;
    case 23:  AK = (-sin(dDpar - dPper) - lamDpar*lamPper*sin(dDpar - dPper - phisDpar + phisPper))/2.;
              break;
    case 24:  AK = (cos(dDpar - dPlon) - lamDpar*lamPlon*cos(dDpar - dPlon - phisDpar + phisPlon))/2.;
              break;
    case 25:  AK = (cos(dDpar - dPpar) - lamDpar*lamPpar*cos(dDpar - dPpar - phisDpar + phisPpar))/2.;
              break;
    case 26:  AK = (-sin(dDpar - dDper) + lamDpar*lamDper*sin(dDpar - dDper - phisDpar + phisDper))/2.;
              break;
    case 27:  AK = (cos(dDlon - dDpar) + lamDlon*lamDpar*cos(dDlon - dDpar - phisDlon + phisDpar))/2.;
              break;
    case 28:  AK = (1 + pow(lamDpar,2))/2.;
              break;
    default:  printf("Wrong k index in AK. %d\\n", k); return 0.;
  }
  return AK;
}


__device__ double BKTerm( int k,
                          double lamSlon, double dSlon, double phisSlon,
                          double lamPper, double dPper, double phisPper,
                          double lamPlon, double dPlon, double phisPlon,
                          double lamPpar, double dPpar, double phisPpar,
                          double lamDper, double dDper, double phisDper,
                          double lamDlon, double dDlon, double phisDlon,
                          double lamDpar, double dDpar, double phisDpar )
{
  double BK;
  switch(k) {
    case 1:   BK = lamSlon*cos(phisSlon);
              break;
    case 2:   BK = (-(lamPper*sin(dPper - dSlon - phisPper)) - lamSlon*sin(dPper - dSlon + phisSlon))/2.;
              break;
    case 3:   BK = lamPper*cos(phisPper);
              break;
    case 4:   BK = (-(lamPlon*cos(dPlon - dSlon - phisPlon)) + lamSlon*cos(dPlon - dSlon + phisSlon))/2.;
              break;
    case 5:   BK = (lamPlon*sin(dPlon - dPper - phisPlon) - lamPper*sin(dPlon - dPper + phisPper))/2.;
              break;
    case 6:   BK = -(lamPlon*cos(phisPlon));
              break;
    case 7:   BK = (-(lamPpar*cos(dPpar - dSlon - phisPpar)) + lamSlon*cos(dPpar - dSlon + phisSlon))/2.;
              break;
    case 8:   BK = (lamPpar*sin(dPpar - dPper - phisPpar) - lamPper*sin(dPpar - dPper + phisPper))/2.;
              break;
    case 9:   BK = (-(lamPlon*cos(dPlon - dPpar - phisPlon)) - lamPpar*cos(dPlon - dPpar + phisPpar))/2.;
              break;
    case 10:  BK = -(lamPpar*cos(phisPpar));
              break;
    case 11:  BK = (lamDper*sin(dDper - dSlon - phisDper) - lamSlon*sin(dDper - dSlon + phisSlon))/2.;
              break;
    case 12:  BK = (-(lamDper*cos(dDper - dPper - phisDper)) + lamPper*cos(dDper - dPper + phisPper))/2.;
              break;
    case 13:  BK = (lamDper*sin(dDper - dPlon - phisDper) + lamPlon*sin(dDper - dPlon + phisPlon))/2.;
              break;
    case 14:  BK = (lamDper*sin(dDper - dPpar - phisDper) + lamPpar*sin(dDper - dPpar + phisPpar))/2.;
              break;
    case 15:  BK = -(lamDper*cos(phisDper));
              break;
    case 16:  BK = (lamDlon*cos(dDlon - dSlon - phisDlon) + lamSlon*cos(dDlon - dSlon + phisSlon))/2.;
              break;
    case 17:  BK = (-(lamDlon*sin(dDlon - dPper - phisDlon)) - lamPper*sin(dDlon - dPper + phisPper))/2.;
              break;
    case 18:  BK = (lamDlon*cos(dDlon - dPlon - phisDlon) - lamPlon*cos(dDlon - dPlon + phisPlon))/2.;
              break;
    case 19:  BK = (lamDlon*cos(dDlon - dPpar - phisDlon) - lamPpar*cos(dDlon - dPpar + phisPpar))/2.;
              break;
    case 20:  BK = (-(lamDlon*sin(dDlon - dDper - phisDlon)) + lamDper*sin(dDlon - dDper + phisDper))/2.;
              break;
    case 21:  BK = lamDlon*cos(phisDlon);
              break;
    case 22:  BK = (lamDpar*cos(dDpar - dSlon - phisDpar) + lamSlon*cos(dDpar - dSlon + phisSlon))/2.;
              break;
    case 23:  BK = (-(lamDpar*sin(dDpar - dPper - phisDpar)) - lamPper*sin(dDpar - dPper + phisPper))/2.;
              break;
    case 24:  BK = (lamDpar*cos(dDpar - dPlon - phisDpar) - lamPlon*cos(dDpar - dPlon + phisPlon))/2.;
              break;
    case 25:  BK = (lamDpar*cos(dDpar - dPpar - phisDpar) - lamPpar*cos(dDpar - dPpar + phisPpar))/2.;
              break;
    case 26:  BK = (-(lamDpar*sin(dDpar - dDper - phisDpar)) + lamDper*sin(dDpar - dDper + phisDper))/2.;
              break;
    case 27:  BK = (lamDlon*cos(dDlon - dDpar - phisDlon) + lamDpar*cos(dDlon - dDpar + phisDpar))/2.;
              break;
    case 28:  BK = lamDpar*cos(phisDpar);
              break;
    default:  printf("Wrong k index in BK. %d\\n", k); return 0.;
  }
  return BK;
}


__device__ double CKTerm( int k,
                          double lamSlon, double dSlon, double phisSlon,
                          double lamPper, double dPper, double phisPper,
                          double lamPlon, double dPlon, double phisPlon,
                          double lamPpar, double dPpar, double phisPpar,
                          double lamDper, double dDper, double phisDper,
                          double lamDlon, double dDlon, double phisDlon,
                          double lamDpar, double dDpar, double phisDpar )
{
  double CK;
  switch(k) {
    case 1:   CK = (1 - pow(lamSlon,2))/2.;

              break;
    case 2:   CK = (-sin(dPper - dSlon) + lamPper*lamSlon*sin(dPper - dSlon - phisPper + phisSlon))/2.;

              break;
    case 3:   CK = (1 - pow(lamPper,2))/2.;

              break;
    case 4:   CK = (cos(dPlon - dSlon) + lamPlon*lamSlon*cos(dPlon - dSlon - phisPlon + phisSlon))/2.;
              break;
    case 5:   CK = (-sin(dPlon - dPper) - lamPlon*lamPper*sin(dPlon - dPper - phisPlon + phisPper))/2.;
              break;
    case 6:   CK = (1 - pow(lamPlon,2))/2.;
              break;
    case 7:   CK = (cos(dPpar - dSlon) + lamPpar*lamSlon*cos(dPpar - dSlon - phisPpar + phisSlon))/2.;
              break;
    case 8:   CK = (-sin(dPpar - dPper) - lamPpar*lamPper*sin(dPpar - dPper - phisPpar + phisPper))/2.;
              break;
    case 9:   CK = (cos(dPlon - dPpar) - lamPlon*lamPpar*cos(dPlon - dPpar - phisPlon + phisPpar))/2.;
              break;
    case 10:  CK = (1 - pow(lamPpar,2))/2.;
              break;
    case 11:  CK = (-sin(dDper - dSlon) - lamDper*lamSlon*sin(dDper - dSlon - phisDper + phisSlon))/2.;
              break;
    case 12:  CK = (cos(dDper - dPper) + lamDper*lamPper*cos(dDper - dPper - phisDper + phisPper))/2.;
              break;
    case 13:  CK = (-sin(dDper - dPlon) + lamDper*lamPlon*sin(dDper - dPlon - phisDper + phisPlon))/2.;
              break;
    case 14:  CK = (-sin(dDper - dPpar) + lamDper*lamPpar*sin(dDper - dPpar - phisDper + phisPpar))/2.;
              break;
    case 15:  CK = (1 - pow(lamDper,2))/2.;
              break;
    case 16:  CK = (cos(dDlon - dSlon) - lamDlon*lamSlon*cos(dDlon - dSlon - phisDlon + phisSlon))/2.;
              break;
    case 17:  CK = (-sin(dDlon - dPper) + lamDlon*lamPper*sin(dDlon - dPper - phisDlon + phisPper))/2.;
              break;
    case 18:  CK = (cos(dDlon - dPlon) + lamDlon*lamPlon*cos(dDlon - dPlon - phisDlon + phisPlon))/2.;
              break;
    case 19:  CK = (cos(dDlon - dPpar) + lamDlon*lamPpar*cos(dDlon - dPpar - phisDlon + phisPpar))/2.;
              break;
    case 20:  CK = (-sin(dDlon - dDper) - lamDlon*lamDper*sin(dDlon - dDper - phisDlon + phisDper))/2.;
              break;
    case 21:  CK = (1 - pow(lamDlon,2))/2.;
              break;
    case 22:  CK = (cos(dDpar - dSlon) - lamDpar*lamSlon*cos(dDpar - dSlon - phisDpar + phisSlon))/2.;
              break;
    case 23:  CK = (-sin(dDpar - dPper) + lamDpar*lamPper*sin(dDpar - dPper - phisDpar + phisPper))/2.;
              break;
    case 24:  CK = (cos(dDpar - dPlon) + lamDpar*lamPlon*cos(dDpar - dPlon - phisDpar + phisPlon))/2.;
              break;
    case 25:  CK = (cos(dDpar - dPpar) + lamDpar*lamPpar*cos(dDpar - dPpar - phisDpar + phisPpar))/2.;
              break;
    case 26:  CK = (-sin(dDpar - dDper) - lamDpar*lamDper*sin(dDpar - dDper - phisDpar + phisDper))/2.;
              break;
    case 27:  CK = (cos(dDlon - dDpar) - lamDlon*lamDpar*cos(dDlon - dDpar - phisDlon + phisDpar))/2.;
              break;
    case 28:  CK = (1 - pow(lamDpar,2))/2.;
              break;
    default:  printf("Wrong k index in CK. %d\\n", k); return 0.;
  }
  return CK;
}


__device__ double DKTerm( int k,
                          double lamSlon, double dSlon, double phisSlon,
                          double lamPper, double dPper, double phisPper,
                          double lamPlon, double dPlon, double phisPlon,
                          double lamPpar, double dPpar, double phisPpar,
                          double lamDper, double dDper, double phisDper,
                          double lamDlon, double dDlon, double phisDlon,
                          double lamDpar, double dDpar, double phisDpar )
{
  double DK;
  switch(k) {
    case 1:   DK = -(lamSlon*sin(phisSlon));
              break;
    case 2:   DK = (lamPper*cos(dPper - dSlon - phisPper) - lamSlon*cos(dPper - dSlon + phisSlon))/2.;
              break;
    case 3:   DK = -(lamPper*sin(phisPper));
              break;
    case 4:   DK = (-(lamPlon*sin(dPlon - dSlon - phisPlon)) - lamSlon*sin(dPlon - dSlon + phisSlon))/2.;
              break;
    case 5:   DK = (-(lamPlon*cos(dPlon - dPper - phisPlon)) - lamPper*cos(dPlon - dPper + phisPper))/2.;
              break;
    case 6:   DK = lamPlon*sin(phisPlon);
              break;
    case 7:   DK = (-(lamPpar*sin(dPpar - dSlon - phisPpar)) - lamSlon*sin(dPpar - dSlon + phisSlon))/2.;
              break;
    case 8:   DK = (-(lamPpar*cos(dPpar - dPper - phisPpar)) - lamPper*cos(dPpar - dPper + phisPper))/2.;
              break;
    case 9:   DK = (-(lamPlon*sin(dPlon - dPpar - phisPlon)) + lamPpar*sin(dPlon - dPpar + phisPpar))/2.;
              break;
    case 10:  DK = lamPpar*sin(phisPpar);
              break;
    case 11:  DK = (-(lamDper*cos(dDper - dSlon - phisDper)) - lamSlon*cos(dDper - dSlon + phisSlon))/2.;
              break;
    case 12:  DK = (-(lamDper*sin(dDper - dPper - phisDper)) - lamPper*sin(dDper - dPper + phisPper))/2.;
              break;
    case 13:  DK = (-(lamDper*cos(dDper - dPlon - phisDper)) + lamPlon*cos(dDper - dPlon + phisPlon))/2.;
              break;
    case 14:  DK = (-(lamDper*cos(dDper - dPpar - phisDper)) + lamPpar*cos(dDper - dPpar + phisPpar))/2.;
              break;
    case 15:  DK = lamDper*sin(phisDper);
              break;
    case 16:  DK = (lamDlon*sin(dDlon - dSlon - phisDlon) - lamSlon*sin(dDlon - dSlon + phisSlon))/2.;
              break;
    case 17:  DK = (lamDlon*cos(dDlon - dPper - phisDlon) - lamPper*cos(dDlon - dPper + phisPper))/2.;
              break;
    case 18:  DK = (lamDlon*sin(dDlon - dPlon - phisDlon) + lamPlon*sin(dDlon - dPlon + phisPlon))/2.;
              break;
    case 19:  DK = (lamDlon*sin(dDlon - dPpar - phisDlon) + lamPpar*sin(dDlon - dPpar + phisPpar))/2.;
              break;
    case 20:  DK = (lamDlon*cos(dDlon - dDper - phisDlon) + lamDper*cos(dDlon - dDper + phisDper))/2.;
              break;
    case 21:  DK = -(lamDlon*sin(phisDlon));
              break;
    case 22:  DK = (lamDpar*sin(dDpar - dSlon - phisDpar) - lamSlon*sin(dDpar - dSlon + phisSlon))/2.;
              break;
    case 23:  DK = (lamDpar*cos(dDpar - dPper - phisDpar) - lamPper*cos(dDpar - dPper + phisPper))/2.;
              break;
    case 24:  DK = (lamDpar*sin(dDpar - dPlon - phisDpar) + lamPlon*sin(dDpar - dPlon + phisPlon))/2.;
              break;
    case 25:  DK = (lamDpar*sin(dDpar - dPpar - phisDpar) + lamPpar*sin(dDpar - dPpar + phisPpar))/2.;
              break;
    case 26:  DK = (lamDpar*cos(dDpar - dDper - phisDpar) + lamDper*cos(dDpar - dDper + phisDper))/2.;
              break;
    case 27:  DK = (lamDlon*sin(dDlon - dDpar - phisDlon) - lamDpar*sin(dDlon - dDpar + phisDpar))/2.;
              break;
    case 28:  DK = -(lamDpar*sin(phisDpar));
              break;
    default:  printf("Wrong k index in DK. %d\\n", k); return 0.;
  }
  return DK;
}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Integral 4Pi time ///////////////////////////////////////////////////////////

__device__ void Integral4PiTime(double result[2],
                                double vnk[28],
                                double vak[28],
                                double vbk[28],
                                double vck[28],
                                double vdk[28],
                                double *normweights,
                                double Gamma, double DeltaGamma, double DeltaM,
                                double tLL, double tUL, double TimeOffset)
{
  double IntTimeA = IntegralTimeA(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeB = IntegralTimeB(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeC = IntegralTimeC(tLL, tUL, Gamma, DeltaM);
  double IntTimeD = IntegralTimeD(tLL, tUL, Gamma, DeltaM);

  for(int k=0; k<28 ; k++)
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
}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Cross section ///////////////////////////////////////////////////////////////

__device__
double DiffRate(double data[10],
                double ASlon,
                double APper, double APlon, double APpar,
                double ADper, double ADlon, double ADpar,
                double CSP, double CSD, double CPD,
                double phisSlon,
                double phisPper, double phisPlon, double phisPpar,
                double phisDper, double phisDlon, double phisDpar,
                double dSlon,
                double dPper, double dPpar,
                double dDper, double dDlon, double dDpar,
                double lamSlon,
                double lamPper, double lamPlon, double lamPpar,
                double lamDper, double lamDlon, double lamDpar,
                double Gamma, double DeltaGamma, double DeltaM,
                double p0_OS, double dp0_OS, double p1_OS, 
                double dp1_OS, double p2_OS, double dp2_OS, 
                double eta_bar_OS,
                double p0_SSK, double dp0_SSK, 
                double p1_SSK, double dp1_SSK, 
                double eta_bar_SSK,
                double sigma_t_a, double sigma_t_b, double sigma_t_c,
                double sigma_t_mu_a, double sigma_t_mu_b, double sigma_t_mu_c,
                double tLL, double tUL,
                double *normweights)
{

  double HelcosK       = data[0];
  double HelcosL       = data[1];
  double HelPhi        = data[2];                                 // ATTENTION
  double t             = data[3];
  double sigma_t       = data[4]; //printf("%lf\n", sigma_t);
  double q_OS          = data[5];
  double q_SSK         = data[6];
  double eta_OS        = data[7];
  double eta_SSK       = data[8];
  int    year          = data[9];



  //printf("%lf,%lf,%lf,%lf,%lf,%lf,%d,\n", HelcosK, HelcosL, HelPhi, t, q_OS, q_SSK, year);
  //printf("%lf, %lf, %lf, %lf, %lf, %lf\n", Gamma, DeltaGamma, q_OS, DeltaM, phisSlon, ASlon);
  double dPlon         = 0.;                                    // by convention
  double pdfB          = 0.;                                    // pdf is scalar
  double pdfBbar       = 0.;                                    // pdf is scalar

  //if (Gamma < 0.1){ printf("Fucking shit!\n"); }

  /*
  printf("A    = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", ASlon, APper, APlon,  APpar, ADper,  ADlon,  ADpar);
  printf("phis = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", phisSlon, phisPper,  phisPlon,  phisPpar, phisDper,  phisDlon,  phisDpar);
  printf("lamb = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", lamSlon, lamPper,  lamPlon,  lamPpar, lamDper,  lamDlon,  lamDpar);
  printf("sigma, q, eta, year = %lf, %lf, %lf, %lf, %lf, %d \n", sigma_t, q_OS, q_SSK, eta_OS, eta_SSK, year);
  */
  
  // Hardcoded spline parameters
  double spline_knots[8] = {0.3, 0.58, 0.91, 1.35, 1.96, 3.01, 7.0, 15.0};
  double spline_coeffs[28] = {1.00460352919722, -0.155198055438259, 0.631202729680957, -0.550087342112342, 0.821112671499487, 0.793892587825892, -1.00516044836065, 0.390351265957555,  1.11610511304869, -0.178609966731930, 0.0635236775270647, -0.00110812081450162,  1.20868258316517, -0.384337678101882, 0.215914574838140, -0.0387355028666187,  0.896135736285030, 0.0940503528371088, -0.0281609511511421, 0.00277394032883751,   0.958414194315690, 0.0319787999826305, -0.00753917279417193, 0.000490243943791026,  0.991526315789474, -0.00150375939849623, 0.0, 0.0};
  int spline_Nknots = 8;
  double moments[10] = {1.0249967, 0.033444123, -0.00029692872, -9.3947863e-05, -0.00012079011, -0.0010327956, -0.0033601786, -0.0007128794, 0.00074387415, -0.0251};

  // Time deltas
  double delta_t    = delta(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
  //double mu_t       = delta(delta_t,  1.13019e+01,  -8.33668e-01,  8.72467e-03);




  // Decay time resolution HARDCODED!
  //double sigma_t = 0.04554;

  //double delta_t = sigma_t;
  //double delta_t =  delta(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);
  //double timeOffset = delta_t;


  // Tagging functions ---------------------------------------------------------

  double omega_OS      = omega(    eta_OS,  p0_OS,  dp0_OS,  p1_OS,  dp1_OS, p2_OS, dp2_OS, eta_bar_OS );
  double omega_bar_OS  = omega_bar(eta_OS,  p0_OS,  dp0_OS,  p1_OS,  dp1_OS, p2_OS, dp2_OS, eta_bar_OS );
  double omega_SSK     = omega(    eta_SSK, p0_SSK, dp0_SSK, p1_SSK, dp1_SSK,   0.,     0., eta_bar_SSK);
  double omega_bar_SSK = omega_bar(eta_SSK, p0_SSK, dp0_SSK, p1_SSK, dp1_SSK,   0.,     0., eta_bar_SSK);

  double tagging_pars_OS[3]  = {omega_OS, omega_bar_OS, q_OS};
  double tagging_pars_SSK[3] = {omega_SSK, omega_bar_SSK, q_SSK};

  fix_tagging_pars(tagging_pars_OS);
  fix_tagging_pars(tagging_pars_SSK);

  omega_OS      = tagging_pars_OS[0];
  omega_bar_OS  = tagging_pars_OS[1];
  omega_SSK     = tagging_pars_SSK[0];
  omega_bar_SSK = tagging_pars_SSK[1];

  if((tagging_pars_OS[0] == 0.5 || tagging_pars_OS[1] == 0.5) && (tagging_pars_OS[0] != tagging_pars_OS[1]))
      printf("OS tag mismatch!!! Check code %lf vs %lf and %lf \n", tagging_pars_OS[0], tagging_pars_OS[1], tagging_pars_OS[2]);
  else
      q_OS = tagging_pars_OS[2];

  if((tagging_pars_SSK[0] == 0.5 || tagging_pars_SSK[1] == 0.5) && (tagging_pars_SSK[0] != tagging_pars_SSK[1]))
      printf("SSK tag mismatch!!! Check code %lf vs %lf and %lf \n", tagging_pars_SSK[0], tagging_pars_SSK[1], tagging_pars_SSK[2]);
  else
      q_SSK = tagging_pars_SSK[2];



  // Time functions ------------------------------------------------------------
  double timeOffset              = delta(delta_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  pycuda::complex<double> hyperP = conv_exp_VC(t+timeOffset, Gamma+0.5*DeltaGamma, 0., delta_t);
  pycuda::complex<double> hyperM = conv_exp_VC(t+timeOffset, Gamma-0.5*DeltaGamma, 0., delta_t);
  pycuda::complex<double> trigo  = conv_exp_VC(t+timeOffset,        Gamma, DeltaM, delta_t);
  double ta = pycuda::real(0.5*(hyperM + hyperP));  
  double tb = pycuda::real(0.5*(hyperM - hyperP));  
  double tc = pycuda::real(trigo);                     
  double td = pycuda::imag(trigo);                     

  double vnk[28] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vak[28] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vbk[28] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vck[28] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vdk[28] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};


  // Get PDF -------------------------------------------------------------------
  for(int k = 1; k <= 10; k++)
  {

    double nk = NKTerm( k, ASlon, APper, APlon, APpar, ADper, ADlon, ADpar, CSP, CSD, CPD );

    double fk = FKTerm( k, HelcosL, HelcosK, HelPhi );

    double ak = AKTerm( k,
                        lamSlon,  dSlon,  phisSlon,
                        lamPper,  dPper,  phisPper,
                        lamPlon,  dPlon,  phisPlon,
                        lamPpar,  dPpar,  phisPpar,
                        lamDper,  dDper,  phisDper,
                        lamDlon,  dDlon,  phisDlon,
                        lamDpar,  dDpar,  phisDpar );
    double bk = BKTerm( k,
                        lamSlon,  dSlon,  phisSlon,
                        lamPper,  dPper,  phisPper,
                        lamPlon,  dPlon,  phisPlon,
                        lamPpar,  dPpar,  phisPpar,
                        lamDper,  dDper,  phisDper,
                        lamDlon,  dDlon,  phisDlon,
                        lamDpar,  dDpar,  phisDpar );
    double ck = CKTerm( k,
                        lamSlon,  dSlon,  phisSlon,
                        lamPper,  dPper,  phisPper,
                        lamPlon,  dPlon,  phisPlon,
                        lamPpar,  dPpar,  phisPpar,
                        lamDper,  dDper,  phisDper,
                        lamDlon,  dDlon,  phisDlon,
                        lamDpar,  dDpar,  phisDpar );
     double dk = DKTerm( k,
                        lamSlon,  dSlon,  phisSlon,
                        lamPper,  dPper,  phisPper,
                        lamPlon,  dPlon,  phisPlon,
                        lamPpar,  dPpar,  phisPpar,
                        lamDper,  dDper,  phisDper,
                        lamDlon,  dDlon,  phisDlon,
                        lamDpar,  dDpar,  phisDpar );

    double hkB    = 3./(4.*M_PI)*(ak*ta + bk*tb + ck*tc + dk*td);
    double hkBbar = 3./(4.*M_PI)*(ak*ta + bk*tb - ck*tc - dk*td);

    pdfB += nk*fk*hkB; pdfBbar += nk*fk*hkBbar;

    vnk[k-1] = 1.*nk; 
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;

  }
  //printf("%lf,%lf\n", pdfB, pdfBbar);


  // Get p.d.f. integral -------------------------------------------------------

/*
  double integral[2] = {0.,0.};

  double time_acc = calculate_time_acc(t, spline_knots, spline_coeffs, spline_Nknots);
  double ang_acc = ang_eff(HelcosK, HelcosL, HelPhi, moments);
  
  // Integral time acceptance with a spline
  integral4pitime_spline     (integral, vnk, vak, vbk, vck, vdk, normweights, Gamma, DeltaGamma,DeltaM, delta_t, tLL, timeOffset, spline_Nknots, spline_knots, spline_coeffs);
  //integral4pitime_full_spline(integral, vnk, vak, vbk, vck, vdk, normweights, Gamma, DeltaGamma,DeltaM, delta_t, tLL, timeOffset, spline_Nknots, spline_knots, spline_coeffs);

  double intB    = integral[0];
  double intBbar = integral[1];


*/
  double Int4PiTime[2] = {0.,0.};
  double time_acc = 1+0*calculate_time_acc(t, spline_knots, spline_coeffs, spline_Nknots);
  double ang_acc = 1+0*ang_eff(HelcosK, HelcosL, HelPhi, moments);
  Integral4PiTime(Int4PiTime, vnk, vak, vbk, vck, vdk,
                  normweights,
                  Gamma, DeltaGamma, DeltaM, tLL, 15., 0.);
  double intB    = Int4PiTime[0];
  double intBbar = Int4PiTime[1];

  

  // Total PDF -----------------------------------------------------------------
  double num =  ((1. + q_OS *(1.-2.*omega_OS)     )*
                 (1. + q_SSK*(1.-2.*omega_SSK)    )*pdfB
                +(1. - q_OS *(1.-2.*omega_bar_OS) )*
                 (1. - q_SSK*(1.-2.*omega_bar_SSK))*pdfBbar);

  double den =  ((1. + q_OS *(1.-2.*omega_OS)     )*
                 (1. + q_SSK*(1.-2.*omega_SSK)    )*intB
                +(1. - q_OS *(1.-2.*omega_bar_OS) )*
                 (1. - q_SSK*(1.-2.*omega_bar_SSK))*intBbar);
  //printf("%lf,%lf,%lf\n", num, den,num/den);
  //printf("%lf,%lf\n", q_OS, q_SSK);
  //printf("%lf\n", phisSlon); ang_acc*time_acc*
  return num/den;

}

////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////
// Cross section ///////////////////////////////////////////////////////////////

__global__
void getDiffRate(double *data, double *result,
                 double ASlon,
                 double APper, double APlon, double APpar,
                 double ADper, double ADlon, double ADpar,
                 double CSP, double CSD, double CPD,
                 double phisSlon,
                 double phisPper, double phisPlon, double phisPpar,
                 double phisDper, double phisDlon, double phisDpar,
                 double dSlon,
                 double dPper, double dPpar,
                 double dDper, double dDlon, double dDpar,
                 double lamSlon,
                 double lamPper, double lamPlon, double lamPpar,
                 double lamDper, double lamDlon, double lamDpar,
                 double Gamma, double DeltaGamma, double DeltaM,
                 double tLL, double tUL,
                 double *normweights,
                 int Nevt)
{

  int row = threadIdx.x + blockDim.x * blockIdx.x;               // ntuple entry
  if (row >= 1+1*(Nevt-1)) { return; }

  // Map input info ------------------------------------------------------------
  int i0 = row*13;       // general rule for cuda matrices : index = col + row*N
  int idx        =  0 + i0;
  int idy        =  1 + i0;
  int idz        =  2 + i0;
  int idt        =  3 + i0;
  int idsigma_t  =  4 + i0;
  int idq_OS     =  5 + i0;
  int idq_SSK    =  6 + i0;
  int ideta_OS   =  7 + i0;
  int ideta_SSK  =  8 + i0;
  int iyear      = 12 + i0;

  double HelcosK       = data[idx];
  double HelcosL       = data[idy];
  double HelPhi        = data[idz];                                 // ATTENTION
  double t             = data[idt];
  double sigma_t       = data[idsigma_t];
  double q_OS          = data[idq_OS];
  double q_SSK         = data[idq_SSK];
  double eta_OS        = data[ideta_OS];
  double eta_SSK       = data[ideta_SSK];
  int    year          = data[iyear];

  double input[10] = {HelcosK, HelcosL, HelPhi, t, sigma_t, q_OS, q_SSK, eta_OS, eta_SSK, year};
  double pdf   = 0.0;

  /*if (Gamma < 0.1){
    Gamma += 0.65789;   // Use Gamma or Gs-Gd+0.65789 to fit, WARNING HARDCODED
  }*/

  // HARDCODDED STUFF --- take a look
  double sigma_t_mu_a = 0;
  double sigma_t_mu_b = 0;
  double sigma_t_mu_c = 0;
  double sigma_t_a = 0;
  double sigma_t_b = 0.8721;
  double sigma_t_c = 0.01225;

  // Hardcoded tagging parameters
  double p0_OS = 0.39; double dp0_OS = 0.009; 
  double p1_OS = 0.85; double dp1_OS = 0.014; 
  double p2_OS = 0; double dp2_OS = 0; 
  double eta_bar_OS = 0.379;
  double p0_SSK = 0.43; double dp0_SSK = 0.0; 
  double p1_SSK = 0.92; double dp1_SSK = 0; 
  double eta_bar_SSK = 0.4269;

  /*
  printf("A    = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", ASlon, APper, APlon,  APpar, ADper,  ADlon,  ADpar);
  printf("phis = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", phisSlon, phisPper,  phisPlon,  phisPpar, phisDper,  phisDlon,  phisDpar);
  printf("lamb = %lf, %lf, %lf, %lf, %lf, %lf, %lf \n", lamSlon, lamPper,  lamPlon,  lamPpar, lamDper,  lamDlon,  lamDpar);
  printf("sigma, q, eta, year = %lf, %lf, %lf, %lf, %lf, %d \n", sigma_t, q_OS, q_SSK, eta_OS, eta_SSK, year);
  */
  

  pdf = DiffRate( input,
            ASlon, APper, APlon,  APpar, ADper,  ADlon,  ADpar, CSP,  CSD,  CPD,
            phisSlon, phisPper,  phisPlon,  phisPpar, phisDper,  phisDlon,  phisDpar,
            dSlon, dPper,  dPpar, dDper,  dDlon,  dDpar,
            lamSlon, lamPper,  lamPlon,  lamPpar, lamDper,  lamDlon,  lamDpar,
            Gamma,  DeltaGamma,  DeltaM,
            p0_OS, dp0_OS, p1_OS, 
            dp1_OS, p2_OS, dp2_OS, 
            eta_bar_OS,
            p0_SSK, dp0_SSK, 
            p1_SSK, dp1_SSK, 
            eta_bar_SSK,
            sigma_t_a, sigma_t_b, sigma_t_c,
            sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c,
            tLL, tUL, normweights);

  result[row] = pdf;

}

////////////////////////////////////////////////////////////////////////////////




/*





(double *data,
 double *out,
 double CSP,
 double A_0_abs, double A_S_abs, double A_pa_abs, double A_pe_abs,
 double phis_0, double phis_S, double phis_pa, double phis_pe,
 double delta_S, double delta_pa, double delta_pe,
 double l_0_abs, double l_S_abs, double l_pa_abs, double l_pe_abs,
 double G, double DG, double DM,
 double p0_OS, double dp0_OS, double p1_OS, double dp1_OS, double p2_OS, double dp2_OS, double eta_bar_OS,
 double p0_SSK, double dp0_SSK, double p1_SSK, double dp1_SSK, double eta_bar_SSK,
 double sigma_t_a, double sigma_t_b, double sigma_t_c, double t_ll,
 double sigma_t_mu_a, double sigma_t_mu_b, double sigma_t_mu_c,
 int spline_Nknots,
 double *normweights,
 double *spline_knots,
 double *spline_coeffs,
 int Nevt)

*/



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
}
