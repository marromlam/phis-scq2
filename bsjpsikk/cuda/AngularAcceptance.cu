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
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>

#include "DifferentialCrossRate.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__device__
void getAngularWeights(double *dtrue, double *dreco, double weight, double *w10,
                       double G, double DG, double DM, double CSP,
                       double ASlon, double APlon, double APpar, double APper,
                       double pSlon, double pPlon, double pPpar, double pPper,
                       double dSlon, double dPlon, double dPpar, double dPper,
                       double lSlon, double lPlon, double lPpar, double lPper,
                       double tLL, double tUL,
                       double *coeffs)
{
  double fk = 0.0;
  double pdf = getDiffRate(dreco,
                           G, DG, DM, CSP,
                           ASlon, APlon, APpar, APper,
                           pSlon, pPlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lSlon, lPlon, lPpar, lPper,
                           tLL, tUL, coeffs, 1)/
               getDiffRate(dtrue,
                           G, DG, DM, CSP,
                           ASlon, APlon, APpar, APper,
                           pSlon, pPlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lSlon, lPlon, lPpar, lPper,
                           tLL, tUL, coeffs, 0);

  for(int k = 0; k < 10; k++)
  {
    fk     = 9./(16.*M_PI)*getF(dreco[0],dreco[1],dreco[2],k+1);
    w10[k] = weight*fk/pdf;
  }
}



////////////////////////////////////////////////////////////////////////////////
