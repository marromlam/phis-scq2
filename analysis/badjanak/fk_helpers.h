#include <lib99ocl/core.h>
#include <lib99ocl/complex.h>
#include <lib99ocl/special.h>


#ifndef _FKHELPERS_H_
#define _FKHELPERS_H_


//calculates int x^n * sin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_x(ftype x, int n);


//calculates int x^n * cos(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_x(ftype x, int n);


//calculates int x^n * sin(2x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_2x(ftype x, int n);


//calculates int x^n * cos(2x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_2x(ftype x, int n);


//calculates int x^n * asin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_asin_x(ftype x, int n);


//calculates int x^n * sqrt(1-x^2) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sqrt_1_minus_x2(ftype x, int n);


WITHIN_KERNEL
ftype integral_ijk_f1(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f2(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f3(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f4(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f5(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);
  

WITHIN_KERNEL
ftype integral_ijk_f6(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f7(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f8(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f9(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype integral_ijk_f10(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j);


WITHIN_KERNEL
ftype getFintegral(const ftype cosKs, const ftype cosKe, 
                   const ftype cosLs, const ftype cosLe,
                   const ftype phis, const ftype phie, 
                   const int i, const int j, const int k, const int K);


#endif //_FKHELPERS_H_
