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
void getAngularWeights( double *dtrue, double *dreco, double weight, double *w10,
                        // Time-dependent angular distribution
                        double G, double DG, double DM, double CSP,
                        double ASlon, double APlon, double APpar, double APper,
                        double pSlon, double pPlon, double pPpar, double pPper,
                        double dSlon, double dPlon, double dPpar, double dPper,
                        double lSlon, double lPlon, double lPpar, double lPper,
                        // Time limits
                        double tLL, double tUL,
                        // Time resolution
                        double sigma_offset, double sigma_slope, double sigma_curvature,
                        double mu,
                        // Flavor tagging
                        double eta_bar_os, double eta_bar_ss,
                        double p0_os,  double p1_os, double p2_os,
                        double p0_ss,  double p1_ss, double p2_ss,
                        double dp0_os, double dp1_os, double dp2_os,
                        double dp0_ss, double dp1_ss, double dp2_ss,
                        // Time acceptance
                        int nknots, double *knots,
                        double *coeffs,
                        // Angular acceptance
                        double *angular_weights)
{
  double fk = 0.0;
  double pdf = getDiffRate(dtrue,
                           G, DG, DM, CSP,
                           ASlon, APlon, APpar, APper,
                           pSlon, pPlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lSlon, lPlon, lPpar, lPper,
                           tLL, tUL,
                           sigma_offset, sigma_slope, sigma_curvature,
                           mu,
                           eta_bar_os, eta_bar_ss,
                           p0_os,  p1_os, p2_os,
                           p0_ss,  p1_ss, p2_ss,
                           dp0_os, dp1_os, dp2_os,
                           dp0_ss, dp1_ss, dp2_ss,
                           //nknots, knots,
                           coeffs,
                           angular_weights, 1)/
               getDiffRate(dtrue,
                           G, DG, DM, CSP,
                           ASlon, APlon, APpar, APper,
                           pSlon, pPlon, pPpar, pPper,
                           dSlon, dPlon, dPpar, dPper,
                           lSlon, lPlon, lPpar, lPper,
                           tLL, tUL,
                           sigma_offset, sigma_slope, sigma_curvature,
                           mu,
                           eta_bar_os, eta_bar_ss,
                           p0_os,  p1_os, p2_os,
                           p0_ss,  p1_ss, p2_ss,
                           dp0_os, dp1_os, dp2_os,
                           dp0_ss, dp1_ss, dp2_ss,
                           //nknots, knots,
                           coeffs,
                           angular_weights, 0);

  for(int k = 0; k < 10; k++)
  {
    fk     = 9.0/(16.0*M_PI)*getF(dreco[0],dreco[1],dreco[2],k+1);  // READY!
    w10[k] = weight*fk/pdf;
  }
}



////////////////////////////////////////////////////////////////////////////////
