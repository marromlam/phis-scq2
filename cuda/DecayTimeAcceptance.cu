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

// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double* a, double b) { return b; }
// #endif


////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>
#include "/home3/marcos.romero/phis-scq/cuda/Functions.c"

__device__ double const sigma_threshold = 5.0;
__device__ int const time_acc_bins = 40;
__device__ int const spl_bins = 7;

extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
{





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


}
