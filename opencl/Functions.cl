////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL some useful functions                          //
//                                                                            //
//   Created: 2019-11-18                                                      //
//  Modified: 2019-11-27                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file contains the following __kernels:                               //
//    * [none]                                                                      //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>
#define errf_const 1.12837916709551
#define xLim 5.33
#define yLim 4.29


__global
double factorial(int n)
{
  if(n <= 0) { return 1.; }
  double x = 1; int b = 0;
  do {
    b++;
    x *= b;
  } while(b!=n);

  return x;
}



__global
cdouble_t cdouble_subt(cdouble_t a, cdouble_t b)
{
  return cdouble_add(a,cdouble_mul(cdouble_new(-1,0),b));
}



__global
cdouble_t faddeeva(cdouble_t z)//, double t)
{
   double in_real = z.real;
   double in_imag = z.imag;
   int n, nc, nu;
   double h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
   double Rx [33];
   double Ry [33];

   x = fabs(in_real);
   y = fabs(in_imag);

   if (y < yLim && x < xLim) {
      q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim) * (x / xLim));
      h  = 1.0 / (3.2 * q);
      nc = 7 + convert_int4(23.0 * q);
//       xl = pow(h, double(1 - nc));
      double h_inv = 1./h;
      xl = h_inv;
      for(int i = 1; i < nc-1; i++)
          xl *= h_inv;

      xh = y + 0.5 / h;
      yh = x;
      nu = 10 + convert_int4(21.0 * q);
      Rx[nu] = 0.;
      Ry[nu] = 0.;
      for (n = nu; n > 0; n--){
         Tx = xh + n * Rx[n];
         Ty = yh - n * Ry[n];
         Tn = Tx*Tx + Ty*Ty;
         Rx[n-1] = 0.5 * Tx / Tn;
         Ry[n-1] = 0.5 * Ty / Tn;
         }
      Sx = 0.;
      Sy = 0.;
      for (n = nc; n>0; n--){
         Saux = Sx + xl;
         Sx = Rx[n-1] * Saux - Ry[n-1] * Sy;
         Sy = Rx[n-1] * Sy + Ry[n-1] * Saux;
         xl = h * xl;
      };
      Wx = errf_const * Sx;
      Wy = errf_const * Sy;
   }
   else {
      xh = y;
      yh = x;
      Rx[0] = 0.;
      Ry[0] = 0.;
      for (n = 9; n>0; n--){
         Tx = xh + n * Rx[0];
         Ty = yh - n * Ry[0];
         Tn = Tx * Tx + Ty * Ty;
         Rx[0] = 0.5 * Tx / Tn;
         Ry[0] = 0.5 * Ty / Tn;
      };
      Wx = errf_const * Rx[0];
      Wy = errf_const * Ry[0];
   }

   if (y == 0.) {
      Wx = exp(-x * x);
   }
   if (in_imag < 0.) {

      double exp_x2_y2 = exp(y * y - x * x);
      Wx =   2.0 * exp_x2_y2 * cos(2.0 * x * y) - Wx;
      Wy = - 2.0 * exp_x2_y2 * sin(2.0 * x * y) - Wy;
      if (in_real > 0.) {
         Wy = -Wy;
      }
   }
   else if (in_real < 0.) {
      Wy = -Wy;
   }
   cdouble_t result = cdouble_new(Wx,Wy);
   return result;
}
