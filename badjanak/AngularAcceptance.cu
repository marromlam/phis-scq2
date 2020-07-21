////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                           ANGULAR ACCEPTANCE                               //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq package, Santiago's framework for the     //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// getAngularWeights ///////////////////////////////////////////////////////////



WITHIN_KERNEL
void getAngularWeights( ${ftype} *dtrue, ${ftype} *dreco, ${ftype} weight, ${ftype} *w10,
                        // Time-dependent angular distribution
                        ${ftype} G, ${ftype} DG, ${ftype} DM, ${ftype} CSP,
                        ${ftype} ASlon, ${ftype} APlon, ${ftype} APpar, ${ftype} APper,
                        ${ftype} pSlon, ${ftype} pPlon, ${ftype} pPpar, ${ftype} pPper,
                        ${ftype} dSlon, ${ftype} dPlon, ${ftype} dPpar, ${ftype} dPper,
                        ${ftype} lSlon, ${ftype} lPlon, ${ftype} lPpar, ${ftype} lPper,
                        // Time limits
                        ${ftype} tLL, ${ftype} tUL,
                        // Time resolution
                        ${ftype} sigma_offset, ${ftype} sigma_slope, ${ftype} sigma_curvature,
                        ${ftype} mu,
                        // Flavor tagging
                        ${ftype} eta_bar_os, ${ftype} eta_bar_ss,
                        ${ftype} p0_os,  ${ftype} p1_os, ${ftype} p2_os,
                        ${ftype} p0_ss,  ${ftype} p1_ss, ${ftype} p2_ss,
                        ${ftype} dp0_os, ${ftype} dp1_os, ${ftype} dp2_os,
                        ${ftype} dp0_ss, ${ftype} dp1_ss, ${ftype} dp2_ss,
                        // Time acceptance
                        GLOBAL_MEM ${ftype} *coeffs,
                        // Angular acceptance
                        GLOBAL_MEM ${ftype} *angular_weights)
{
  ${ftype} fk = 0.0;
  ${ftype} pdf = getDiffRate(dtrue,
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
                           coeffs,
                           angular_weights,
                           1, 0, 0, 0, 0, 0)/
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
                           coeffs,
                           angular_weights,
                           0, 0, 0, 0, 0, 0);

  for(int k = 0; k < 10; k++)
  {
    fk     = 9.0/(16.0*M_PI)*getF(dreco[0],dreco[1],dreco[2],k+1);  // READY!
    w10[k] = weight*fk/pdf;
  }
}



////////////////////////////////////////////////////////////////////////////////
