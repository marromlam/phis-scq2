////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                         DIFFERENTIAL CROSS RATE                            //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include "decay_time_acceptance.h"
#include "cross_rate_bs.h"


KERNEL
void pyrateBs(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *lkhd,
    // Time-dependent angular distribution
    const ftype G, const ftype DG, const ftype DM,
    GLOBAL_MEM const ftype * CSP,
    GLOBAL_MEM const ftype *ASlon, GLOBAL_MEM const ftype *APlon,
    GLOBAL_MEM const ftype *APpar, GLOBAL_MEM const ftype *APper,
    const ftype pSlon, const ftype pPlon, const ftype pPpar,
    const ftype pPper, GLOBAL_MEM const ftype *dSlon,
    const ftype dPlon, const ftype dPpar, const ftype dPper,
    const ftype lSlon, const ftype lPlon, const ftype lPpar,
    const ftype lPper,
    // Time limits
    const ftype tLL, const ftype tUL,
    const ftype cosKLL, const ftype cosKUL,
    const ftype cosLLL, const ftype cosLUL,
    const ftype hphiLL, const ftype hphiUL,
    // Time resolution
    const ftype sigma_offset, const ftype sigma_slope,
    const ftype sigma_curvature, const ftype mu,
    // Flavor tagging
    const ftype eta_bar_os, const ftype eta_bar_ss,
    const ftype p0_os, const ftype p1_os, const ftype p2_os,
    const ftype p0_ss, const ftype p1_ss, const ftype p2_ss,
    const ftype dp0_os, const ftype dp1_os, const ftype dp2_os,
    const ftype dp0_ss, const ftype dp1_ss, const ftype dp2_ss,
    // Time acceptance
    GLOBAL_MEM const ftype *coeffs,
    // Angular acceptance
    GLOBAL_MEM const ftype *angular_weights,
    // Flags
    const int USE_FK, const int BINS, const int USE_ANGACC,
    const int USE_TIMEACC, const int USE_TIMEOFFSET,
    const int SET_TAGGING, const int USE_TIMERES, const int NEVT)
{
  const int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  const ftype mass = data[evt*10+4];
  const ftype arr[9] = {
    data[evt*10+0], // cosK
    data[evt*10+1], // cosL
    data[evt*10+2], // hphi
    data[evt*10+3], // time
    data[evt*10+5], // sigma_t
    data[evt*10+6], // qOS
    data[evt*10+7], // qSS
    data[evt*10+8], // etaOS
    data[evt*10+9]  // etaSS
  };
  const int bin = BINS>1 ? getMassBin(mass) : 0;

  lkhd[evt] = rateBs(arr,
      G, DG, DM, CSP[bin],
      ASlon[bin], APlon[bin], APpar[bin], APper[bin],
      pSlon,      pPlon,      pPpar,      pPper,
      dSlon[bin], dPlon,      dPpar,      dPper,
      lSlon,      lPlon,      lPpar,      lPper,
      tLL, tUL, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL,
      sigma_offset, sigma_slope, sigma_curvature, mu,
      eta_bar_os, eta_bar_ss,
      p0_os,  p1_os, p2_os,
      p0_ss,  p1_ss, p2_ss,
      dp0_os, dp1_os, dp2_os,
      dp0_ss, dp1_ss, dp2_ss,
      coeffs,
      angular_weights,
      USE_FK, USE_ANGACC, USE_TIMEACC,
      USE_TIMEOFFSET, SET_TAGGING, USE_TIMERES);
}


// vim: fdm=marker
