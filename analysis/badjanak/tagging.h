////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-11-07                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _TAGGING_H_
#define _TAGGING_H_


  WITHIN_KERNEL
ftype parabola(const ftype sigma, const ftype sigma_offset, 
    const ftype sigma_slope, const ftype sigma_curvature);


  WITHIN_KERNEL
ftype get_omega(const ftype eta, const ftype tag,
    const ftype p0, const ftype p1, const ftype p2,
    const ftype dp0, const ftype dp1, const ftype dp2,
    const ftype eta_bar);


/**
 * Generates tagOS 
 */
  WITHIN_KERNEL
ftype tagOSgen(const ftype x);


/**
 * Generates tagOS 
 */
  WITHIN_KERNEL
ftype tagSSgen(const ftype x);


#endif // _TAGGING_H_
