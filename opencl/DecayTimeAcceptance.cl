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

//#include <stdio.h>
//#include <math.h>
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>
#include "/home3/marcos.romero/phis-scq/opencl/Functions.cl"

#define sigma_threshold  5.0
#define time_acc_bins  40
#define spl_bins  7

//extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//{





__global
cdouble_t getExponentialConvolution(double t, double gamma,
                                                  double omega, double sigma)
{
  cdouble_t I = cdouble_new(0,1);
  cdouble_t z, fad;
  double sigma2 = sigma*sigma;

  if( t >sigma_threshold*sigma )
  {
    return  //2.*(sqrt(0.5*M_PI))* this was an old factor
            cdouble_mul(
              cdouble_new(exp(-gamma*t+0.5*gamma*gamma*sigma2-0.5*omega*omega*sigma2),0),
              cdouble_add( cdouble_new(cos(omega*(t-gamma*sigma2)),0) ,
                         cdouble_mul( I , cdouble_new(sin(omega*(t-gamma*sigma2)),0) )
                          )
                        );
  }
  else
  {
    printf("not defined yet");
    //z   = cdouble_add(-I*(t-sigma2*gamma) ,- omega*sigma2)/(sigma*sqrt(2.));
    //fad = faddeeva(z);
    //fad = (fad.real - I*fad.imag);
    return cdouble_mul( cdouble_new(0*sqrt(0.5*M_PI)*exp(-0.5*t*t/sigma2),0), fad );
  }
}


//}
