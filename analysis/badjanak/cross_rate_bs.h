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


#ifndef _CROSSRATEBS_H_
#define _CROSSRATEBS_H_


#include <lib99ocl/complex.h>

#include "time_angular_distribution.h"
#include "tagging.h"
#include "decay_time_acceptance.h"


  WITHIN_KERNEL
ftype rateBs(const ftype *data,
    // Time-dependent angular distribution
    const ftype G, const ftype DG, const ftype DM, const ftype CSP,
    const ftype ASlon, const ftype APlon, const ftype APpar,
    const ftype APper, const ftype pSlon, const ftype pPlon,
    const ftype pPpar, const ftype pPper, const ftype dSlon,
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
    const ftype p0_os,  const ftype p1_os, const ftype p2_os,
    const ftype p0_ss,  const ftype p1_ss, const ftype p2_ss,
    const ftype dp0_os, const ftype dp1_os, const ftype dp2_os,
    const ftype dp0_ss, const ftype dp1_ss, const ftype dp2_ss,
    // Time acceptance
    GLOBAL_MEM const ftype *coeffs,
    // Angular acceptance
    GLOBAL_MEM  const ftype *angular_weights,
    const int USE_FK, const int USE_ANGACC, const int USE_TIMEACC,
    const int USE_TIMEOFFSET, const int SET_TAGGING,
    const int USE_TIMERES
    );


#endif //_CROSSRATEBS_H_


// vim:foldmethod=marker
