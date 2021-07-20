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


  KERNEL
void pySingleTimeAcc(GLOBAL_MEM ftype *time, GLOBAL_MEM ftype *lkhd,
    GLOBAL_MEM ftype *coeffs, ftype mu,
    const ftype sigma, const ftype gamma,
    const ftype tLL, const ftype tUL, const int NEVT)
{
  const int row = get_global_id(0);
  if (row >= NEVT) { return; }
  ftype t = time[row] - mu;
  lkhd[row] = getOneSplineTimeAcc(t, coeffs, sigma, gamma, tLL, tUL);
}


  KERNEL
void pyRatioTimeAcc(GLOBAL_MEM const ftype *time1, GLOBAL_MEM const ftype *time2,
    GLOBAL_MEM ftype *lkhd1, GLOBAL_MEM ftype *lkhd2,
    GLOBAL_MEM const ftype *c1, GLOBAL_MEM const ftype *c2,
    const ftype mu1, const ftype sigma1, const ftype gamma1,
    const ftype mu2, const ftype sigma2, const ftype gamma2,
    const ftype tLL, const ftype tUL,
    const int NEVT1, const int NEVT2)
{
  const int row = get_global_id(0);
  if (row < NEVT1)
  {
    ftype t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }
  if (row < NEVT2)
  {
    ftype t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }
}


  KERNEL
void pyFullTimeAcc(GLOBAL_MEM ftype const *time1, GLOBAL_MEM ftype const *time2,
    GLOBAL_MEM ftype const *time3, GLOBAL_MEM ftype *lkhd1,
    GLOBAL_MEM ftype *lkhd2, GLOBAL_MEM ftype *lkhd3,
    GLOBAL_MEM ftype *c1,
    GLOBAL_MEM ftype *c2,
    GLOBAL_MEM ftype *c3,
    ftype mu1, ftype sigma1, ftype gamma1,
    ftype mu2, ftype sigma2, ftype gamma2,
    ftype mu3, ftype sigma3, ftype gamma3,
    ftype tLL, ftype tUL,
    const int NEVT1, const int NEVT2, const int NEVT3)
{
  const int row = get_global_id(0);
  if (row < NEVT1)
  {
    ftype t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }
  if (row < NEVT2)
  {
    ftype t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }
  if (row < NEVT3)
  {
    ftype t3 = time3[row] - mu3;
    lkhd3[row] = getTwoSplineTimeAcc(t3, c2, c3, sigma3, gamma3, tLL, tUL);
  }
}


  KERNEL
void pySpline(GLOBAL_MEM const ftype *time, GLOBAL_MEM ftype *f,
    GLOBAL_MEM const ftype *coeffs, const int NEVT)
{
  const int row = get_global_id(0);
  if (row >= NEVT) { return; }
  ftype t = time[row];

  // Get spline-time-bin
  int bin = getTimeBin(t);

  // Get spline coeffs
  ftype c0 = getCoeff(coeffs,bin,0);
  ftype c1 = getCoeff(coeffs,bin,1);
  ftype c2 = getCoeff(coeffs,bin,2);
  ftype c3 = getCoeff(coeffs,bin,3);

  // Compute spline
  ftype fpdf = (c0 + t*(c1 + t*(c2 + t*c3)));
  f[row] = fpdf;
}
