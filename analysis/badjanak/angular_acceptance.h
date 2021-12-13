////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _ANGULARACCEPTANCE_H_
#define _ANGULARACCEPTANCE_H_


#include <lib99ocl/core.h>
#include <lib99ocl/special.h>

#include "fk_helpers.h"


/**
  This function is INCORRECT!
  Taken from Veronika's code for the toys. It has some of the signs of the normweights
  changed. This may be caused because she was not using a proper implementation for
  the Ylm
  */
WITHIN_KERNEL
void weights_to_moments(ftype *tijk, GLOBAL_MEM const ftype *nw);


/**
 * Translates tijk, the analylical coefficients fitted to data/theory and 
 * described by legendre polynomials
 */
WITHIN_KERNEL
ftype angular_efficiency_weights(const ftype cosK, const ftype cosL,
    const ftype phi, GLOBAL_MEM const ftype *nw);


/**
 * Translates tijk, the analylical coefficients fitted to data/theory and 
 * described by legendre polynomials
 */
WITHIN_KERNEL
ftype angular_efficiency(const ftype cosK, const ftype cosL, const ftype hphi,
    const int order_cosK, const int order_cosL, const int order_hphi,
    GLOBAL_MEM const ftype *cijk);


/**
 * Translates tijk, the analylical coefficients fitted to data/theory and 
 * described by legendre polynomials
 */
WITHIN_KERNEL
void tijk2weights(GLOBAL_MEM ftype *w, GLOBAL_MEM const ftype *tijk,
    const int order_cosK, const int order_cosL, const int order_hphi);


#endif // _ANGULARACCEPTANCE_H_
