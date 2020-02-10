////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay rate Bs -> mumuKK                        //
//                                                                            //
//   Created: 2019-11-18                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of p-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file contains the following __kernels:                               //
//    * pyDiffRate: Computes Bs2MuMuKK pdf looping over the events. Now it    //
//                  handles a binned X_M fit without splitting beforehand the //
//                  data --it launches a thread per mass bin.                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

#include "DifferentialCrossRate.cl"





////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



//__global
void getAngularWeights(double *dtrue, double *dreco, long *w10,
                       double G, double DG, double DM, double CSP,
                       double ASlon, double APlon, double APpar, double APper,
                       double pSlon, double pPlon, double pPpar, double pPper,
                       double dSlon, double dPlon, double dPpar, double dPper,
                       double lPlon, double lSlon, double lPpar, double lPper,
                       double tLL, double tUL,
                       double *coeffs)
{
  double fk = 0.0;
  double pdf = getDiffRate(dtrue,
                           G, DG, DM, CSP,
                           APlon, ASlon, APpar, APper,
                           pPlon, pSlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lPlon, lSlon, lPpar, lPper,
                           tLL, tUL,
                           coeffs, 1)/
               getDiffRate(dtrue,
                           G, DG, DM, CSP,
                           APlon, ASlon, APpar, APper,
                           pPlon, pSlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lPlon, lSlon, lPpar, lPper,
                           tLL, tUL,
                           coeffs, 0);

  for(int k = 0; k < 10; k++)
  {
    fk     = getF(dtrue[0],dtrue[1],dtrue[2],k+1);
    w10[k] = convert_long(9./(16.*M_PI)*fk/pdf);
  }
}
