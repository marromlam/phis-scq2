////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                             TIME ACCEPTANCE                                //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq package, Santiago's framework for the     //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _TIMEACCEPTANCE_H_
#define _TIMEACCEPTANCE_H_


#include <lib99ocl/core.h>
#include <lib99ocl/complex.h>
#include <lib99ocl/special.h>
#include <lib99ocl/cspecial.h>


  WITHIN_KERNEL
unsigned int getTimeBin(ftype const t);
  

  WITHIN_KERNEL
unsigned int getMassBin(ftype const t);


  WITHIN_KERNEL
ftype getKnot(int i);


  WITHIN_KERNEL
ftype getCoeff(GLOBAL_MEM const ftype *mat, int const r, int const c);
;

  WITHIN_KERNEL
ftype time_efficiency(const ftype t, GLOBAL_MEM const ftype *coeffs,
    const ftype tLL, const ftype tUL);


  WITHIN_KERNEL
ctype expconv_simon(const ftype t, const ftype G, const ftype omega,
    const ftype sigma);


  WITHIN_KERNEL
ctype expconv(ftype t, ftype G, ftype omega, ftype sigma);


  WITHIN_KERNEL
ctype expconv_wores(ftype t, ftype G, ftype omega);


  WITHIN_KERNEL
ctype getK(const ctype z, const int n);

  
  WITHIN_KERNEL
ctype getM(ftype x, int n, ftype t, ftype sigma, ftype gamma, ftype omega);


  WITHIN_KERNEL
void intgTimeAcceptance(ftype time_terms[4], const ftype delta_t,
    const ftype G, const ftype DG, const ftype DM,
    GLOBAL_MEM const ftype *coeffs, const ftype t0, const ftype tLL, const ftype tUL);


  WITHIN_KERNEL
void integralFullSpline( ftype result[2],
    const ftype vn[10], const ftype va[10],const ftype vb[10], const ftype vc[10],const ftype vd[10],
    const ftype *norm, const ftype G, const ftype DG, const ftype DM,
    const ftype delta_t,
    const ftype tLL, const ftype tUL,
    const ftype t_offset,
    GLOBAL_MEM const ftype *coeffs);


  WITHIN_KERNEL
ctype getI(ctype z, int n, const ftype xmin, const ftype xmax);

    
  WITHIN_KERNEL
void integralFullSpline( ftype result[2],
    const ftype vn[10], const ftype va[10],const ftype vb[10], const ftype vc[10],const ftype vd[10],
    const ftype *norm, const ftype G, const ftype DG, const ftype DM,
    const ftype delta_t,
    const ftype tLL, const ftype tUL,
    const ftype t_offset,
    GLOBAL_MEM const ftype *coeffs);


  WITHIN_KERNEL
ftype getOneSplineTimeAcc(const ftype t, GLOBAL_MEM const ftype *coeffs,
    const ftype mu, const ftype sigma, const ftype gamma,
    const ftype tLL, const ftype tUL);


  WITHIN_KERNEL
ftype getTwoSplineTimeAcc(const ftype t, GLOBAL_MEM const ftype *coeffs2,
    GLOBAL_MEM const ftype *coeffs1,
    const ftype mu, const ftype sigma, const ftype gamma, const ftype tLL,
    const ftype tUL);


#endif //_TIMEACCEPTANCE_H_


// vim:foldmethod=marker
