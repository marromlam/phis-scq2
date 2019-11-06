// useful functions
#include <stdio.h>
//#include <math.h>
// #include <thrust/complex.h>
#include <pycuda-complex.hpp>
#define errf_const 1.12837916709551
#define xLim 5.33
#define yLim 4.29
#define sigma_t_threshold 5
#define time_acc_bins 40
#define spl_bins 7


__device__ double Factorial(int n)
{
   if(n <= 0)
    return 1.;

   double x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}




__device__ pycuda::complex<double> faddeeva(pycuda::complex<double> z)//, double t)
{
   double in_real = pycuda::real(z);
   double in_imag = pycuda::imag(z);
   int n, nc, nu;
   double h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
   double Rx [33];
   double Ry [33];

   x = fabs(in_real);
   y = fabs(in_imag);

   if (y < yLim && x < xLim) {
      q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim) * (x / xLim));
      h  = 1.0 / (3.2 * q);
      nc = 7 + int(23.0 * q);
//       xl = pow(h, double(1 - nc));
      double h_inv = 1./h;
      xl = h_inv;
      for(int i = 1; i < nc-1; i++)
          xl *= h_inv;

      xh = y + 0.5 / h;
      yh = x;
      nu = 10 + int(21.0 * q);
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

   return pycuda::complex<double>(Wx,Wy);
}
