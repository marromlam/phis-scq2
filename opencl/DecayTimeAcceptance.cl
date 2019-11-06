////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                  CUDA B -> Jpsi K K Decay Time Acceptance                  //
//                                                                            //
//   Author: Marcos Romero                                                    //
//  Created: 2019-05-28                                                       //
//                                                                            //
//                                                                            //
//   __global__: getAcceptanceSingle, getAcceptanceDouble, getSpline          //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>
#define PI 3.141592653589793

// Spline parameters
__device__ const int nknots = 7;
__device__ const int degree = 3;
__device__ const double knots[nknots] = {0.30,0.58,0.91,1.35,1.96,3.01,7.00};

// Time range
__device__ const double tLL = 0.30;
__device__ const double tUL = 15.0;

extern "C"

////////////////////////////////////////////////////////////////////////////////


{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////
// Useful functions ////////////////////////////////////////////////////////////

__device__
int getTimeBin(double t)
{
  int _i = 0;
  int _n = spl_bins-1;
  //printf("%d\n", _n);
  while(_i <= _n )
  {
    if( t < knots[_i] ) {break;}
    _i++;
  }
  //if (0 == _i) {printf("WARNING: t=%lf below first knot!\n",t);}
  return _i - 1;

}


__device__
double getKnot(int i)
{
  if (i<=0) {
    i = 0;
  }
  else if (i>=nknots) {
    i = nknots;
  }
  return knots[i];
}


__device__
double getCoeff(double *mat, int r, int c)
{
  return mat[4*r+c];
}




////////////////////////////////////////////////////////////////////////////////





__device__
pycuda::complex<double> getExponentialConvolution(double t, double gamma,
                                                  double omega, double sigma)
{
  pycuda::complex<double> I(0,1);
  pycuda::complex<double> z, fad;
  double sigma2 = sigma*sigma;

  if( t >sigma_threshold*sigma )
  {
    return  //2.*(sqrt(0.5*M_PI))* this was an old factor
  exp(-gamma*t+0.5*gamma*gamma*sigma2-0.5*omega*omega*sigma2)*
  (cos(omega*(t-gamma*sigma2)) + I*sin(omega*(t-gamma*sigma2)));
  }
  else
  {
    z   = (-I*(t-sigma2*gamma) - omega*sigma2)/(sigma*sqrt(2.));
    fad = faddeeva(z);
    fad = (pycuda::real(fad) - I*pycuda::imag(fad));
    return sqrt(0.5*M_PI)*exp(-0.5*t*t/sigma2)*fad;
  }
}
