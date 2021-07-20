////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//   Created: 2019-01-25                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _TIMEANGULARFUNCTIONS_H_
#define _TIMEANGULARFUNCTIONS_H_


  WITHIN_KERNEL
ftype getN(const ftype A10, const ftype A00, const ftype A1a, const ftype A1e,
    const ftype C01, const int k);

  
  WITHIN_KERNEL
ftype getF(const ftype cosK, const ftype cosL, const ftype hphi, const int k);

  
  WITHIN_KERNEL
ftype getAbd(ftype aslon, ftype aplon, ftype appar, ftype apper,
    ftype dslon, ftype dplon, ftype dppar, ftype dpper,
    ftype csp,  int k);


  WITHIN_KERNEL
ftype getA(const ftype p10, const ftype p00, const ftype p1a, const ftype p1e,
    const ftype d10, const ftype d00, const ftype d1a, const ftype d1e,
    const ftype l10, const ftype l00, const ftype l1a, const ftype l1e,
    const int k);

  
  WITHIN_KERNEL
ftype getB(const ftype p10, const ftype p00, const ftype p1a, const ftype p1e,
    const ftype d10, const ftype d00, const ftype d1a, const ftype d1e,
    const ftype l10, const ftype l00, const ftype l1a, const ftype l1e,
    const int k);


  WITHIN_KERNEL
ftype getC(const ftype p10, const ftype p00, const ftype p1a, const ftype p1e,
    const ftype d10, const ftype d00, const ftype d1a, const ftype d1e,
    const ftype l10, const ftype l00, const ftype l1a, const ftype l1e,
    const int k);


  WITHIN_KERNEL
ftype getD(const ftype p10, const ftype p00, const ftype p1a, const ftype p1e,
    const ftype d10, const ftype d00, const ftype d1a, const ftype d1e,
    const ftype l10, const ftype l00, const ftype l1a, const ftype l1e,
    const int k);


  WITHIN_KERNEL
void integralSimple(ftype result[2],
    const ftype vn[10], const ftype va[10], const ftype vb[10],
    const ftype vc[10], const ftype vd[10],
    const ftype *norm, const ftype G, const ftype DG,
    const ftype DM, const ftype ti, const ftype tf);


#endif //_TIMEANGULARFUNCTIONS_H_


// vim:foldmethod=marker
