////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                FUNCTIONS                                   //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Complex numbers handlers ////////////////////////////////////////////////////

WITHIN_KERNEL
${ctype} cnew(${ftype} re, ${ftype} im)
{
  return COMPLEX_CTR(${ctype}) (re,im);
}



WITHIN_KERNEL
${ctype} cmul(${ctype} z1, ${ctype} z2)
{
  // z1*z2 = (a+ib)*(c+id)
  // ${ftype} a = -z1.x * z1.x + z1.y * z1.y;
  // ${ftype} b = -2. * z1.x * z1.y;
  // ${ftype} c = -z2.x * z2.x + z2.y * z2.y;
  // ${ftype} d = -2. * z2.x * z2.y;
  ${ftype} a = z1.x;
  ${ftype} b = z1.y;
  ${ftype} c = z2.x;
  ${ftype} d = z2.y;
  return cnew(a*c-b*d, a*d+b*c);
}


WITHIN_KERNEL
${ctype} cdiv(${ctype} z1, ${ctype} z2)
{
  ${ftype} a = z1.x;
  ${ftype} b = z1.y;
  ${ftype} c = z2.x;
  ${ftype} d = z2.y;
  ${ftype} den = c*c+d*d;
  return cnew( (a*c+b*d)/den , (b*c-a*d)/den );
}


WITHIN_KERNEL
${ctype} cadd(${ctype} z1, ${ctype} z2)
{
  ${ftype} a = z1.x;
  ${ftype} b = z1.y;
  ${ftype} c = z2.x;
  ${ftype} d = z2.y;
  return cnew(a+c,b+d);
}

WITHIN_KERNEL
${ctype} csub(${ctype} z1, ${ctype} z2)
{
  ${ftype} a = z1.x;
  ${ftype} b = z1.y;
  ${ftype} c = z2.x;
  ${ftype} d = z2.y;
  return cnew(a-c,b-d);
}


WITHIN_KERNEL
${ctype} cexp(${ctype} z)
{
  ${ftype} re = exp(z.x);
  ${ftype} im = z.y;
  return cnew(re * cos(im), re * sin(im));
}



WITHIN_KERNEL
${ctype} csquare(${ctype} z)
{
  ${ftype} re = -z.x * z.x + z.y * z.y;
  ${ftype} im = -2. * z.x * z.y;
  return cnew(re, im);
}




////////////////////////////////////////////////////////////////////////////////
// Factorial function //////////////////////////////////////////////////////////

WITHIN_KERNEL
${ftype} factorial(int n)
{
   if(n <= 0)
    return 1.;

   ${ftype} x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}


////////////////////////////////////////////////////////////////////////////////
// Parabola - 2nd order poly ///////////////////////////////////////////////////

WITHIN_KERNEL
${ftype} parabola(${ftype} sigma, ${ftype} sigma_offset, ${ftype} sigma_slope,
                ${ftype} sigma_curvature)
{
  return sigma_curvature*sigma*sigma + sigma_slope*sigma + sigma_offset;
}


////////////////////////////////////////////////////////////////////////////////
// Faddeva function ////////////////////////////////////////////////////////////


WITHIN_KERNEL ${ctype} faddeeva( ${ctype} z)
{
   ${ftype} in_real = z.x;
   ${ftype} in_imag = z.y;
   int n, nc, nu;
   ${ftype} h, q, Saux, Sx, Sy, Tn, Tx, Ty, Wx, Wy, xh, xl, x, yh, y;
   ${ftype} Rx [33];
   ${ftype} Ry [33];

   x = fabs(in_real);
   y = fabs(in_imag);

   if (y < YLIM && x < XLIM) {
      q = (1.0 - y / YLIM) * sqrt(1.0 - (x / XLIM) * (x / XLIM));
      h  = 1.0 / (3.2 * q);
      #ifdef CUDA
        nc = 7 + int(23.0 * q);
      #else
        nc = 7 + convert_int(23.0 * q);
      #endif

//       xl = pow(h, ${ftype}(1 - nc));
      ${ftype} h_inv = 1./h;
      xl = h_inv;
      for(int i = 1; i < nc-1; i++)
          xl *= h_inv;

      xh = y + 0.5 / h;
      yh = x;
      #ifdef CUDA
        nu = 10 + int(21.0 * q);
      #else
        nu = 10 + convert_int(21.0 * q);
      #endif
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
      Wx = ERRF_CONST * Sx;
      Wy = ERRF_CONST * Sy;
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
      Wx = ERRF_CONST * Rx[0];
      Wy = ERRF_CONST * Ry[0];
   }

   if (y == 0.) {
      Wx = exp(-x * x);
   }
   if (in_imag < 0.) {

      ${ftype} exp_x2_y2 = exp(y * y - x * x);
      Wx =   2.0 * exp_x2_y2 * cos(2.0 * x * y) - Wx;
      Wy = - 2.0 * exp_x2_y2 * sin(2.0 * x * y) - Wy;
      if (in_real > 0.) {
         Wy = -Wy;
      }
   }
   else if (in_real < 0.) {
      Wy = -Wy;
   }

   return COMPLEX_CTR(${ctype}) (Wx,Wy);
}



////////////////////////////////////////////////////////////////////////////////
