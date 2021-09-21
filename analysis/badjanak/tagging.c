////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-11-07                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include "tagging.h"


  WITHIN_KERNEL
ftype parabola(const ftype sigma, const ftype sigma_offset, 
    const ftype sigma_slope, const ftype sigma_curvature)
{
  return sigma_curvature*sigma*sigma + sigma_slope*sigma + sigma_offset;
}


  WITHIN_KERNEL
ftype get_omega(const ftype eta, const ftype tag,
    const ftype p0, const ftype p1, const ftype p2,
    const ftype dp0, const ftype dp1, const ftype dp2,
    const ftype eta_bar)
{
  ftype result = 0;
  result += (p0 + tag*0.5*dp0);
  result += (p1 + tag*0.5*dp1)*(eta - eta_bar);
  result += (p2 + tag*0.5*dp2)*(eta - eta_bar)*(eta - eta_bar);

  if(result < 0.0)
  {
    return 0;
  }
  return result;
}


/**
 * Generates tagOS 
 */
  WITHIN_KERNEL
ftype tagOSgen(const ftype x)
{
  return 3.8 - 134.6*x + 1341.0*x*x;
}


/**
 * Generates tagOS 
 */
  WITHIN_KERNEL
ftype tagSSgen(const ftype x)
{
  if (x < 0.46)
  {
    return exp(16.0*x - 0.77);
  }
  return 10.0*(16326.0 - 68488.0*x + 72116.0*x*x);
}
