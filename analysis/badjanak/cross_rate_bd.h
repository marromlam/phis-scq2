////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _CROSSRATEBD_H_
#define _CROSSRATEBD_H_


#include "time_angular_distribution.h"


  WITHIN_KERNEL
ftype getDiffRateBd(const ftype *data,
    const ftype G, const ftype CSP,
    const ftype ASlon, const ftype APlon, const ftype APpar,
    const ftype APper, const ftype dSlon, const ftype dPlon,
    const ftype dPpar, const ftype dPper,
    const ftype tLL, const ftype tUL,
    GLOBAL_MEM const ftype *angular_weights,
    const int USE_FK, const int USE_ANGACC
    );


#endif //_CROSSRATEBD_H_


// vim:foldmethod=marker
