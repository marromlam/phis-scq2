
#include "fk_helpers.h"


#ifdef CUDA
  WITHIN_KERNEL
  int convert_float(float in) {
      union fi { int i; float f; } conv;
      conv.f = in;
      return conv.i;
  }
#endif


//this methods are helpful for the phi integration
//calculates int x^n * sin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_x(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return -cos(x);
  else
    return -pow(x, n)*cos(x) + n*integral_x_to_n_times_cos_x(x, n-1);
}

//calculates int x^n * cos(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_x(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return sin(x);
  else
    return pow(x,n)*sin(x) - n*integral_x_to_n_times_sin_x(x, n-1);
}

//calculates int x^n * sin(2x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_2x(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return -0.5*cos(2.0*x);
  else
    return -pow(x,n)*0.5*cos(2.0*x)
      +0.5*n*integral_x_to_n_times_cos_2x(x,n-1);
}

//calculates int x^n * cos(2x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_2x(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return 0.5*sin(2.0*x);
  else
    return +0.5*pow(x,n)*sin(2.0*x)
      -0.5*n*integral_x_to_n_times_sin_2x(x,n-1);
}

//calculates int x^n * cos(x)^2 dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_x_2(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  return +1.0/(1.0 + n)*pow(x,n+1)*cos(x)*cos(x)
    +1.0/(1.0+n)*integral_x_to_n_times_sin_2x(x, n+1);
}

//calculates int x^n * sin(x)^2 dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_x_2(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  return +1.0/(1.0 + n)*pow(x,n+1)*sin(x)*sin(x)
    -1.0/(1.0+n)*integral_x_to_n_times_sin_2x(x, n+1);
}

//calculates int x^n * asin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_asin_x(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return x*asin(x)+sqrt(1-x*x);
  else
    return 1.0/(n+1.0)*pow(x,n)*(x*asin(x)+sqrt(1-x*x))
      -n*integral_x_to_n_times_sqrt_1_minus_x2(x, n-1);
}

//calculates int x^n * sqrt(1-x^2) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sqrt_1_minus_x2(const ftype x, const int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return 0.5*asin(x)+0.5*x*sqrt(1-x*x);
  else
    return 2.0/(n+2.0)*pow(x, n)*(0.5*asin(x)+0.5*x*sqrt(1-x*x))
      -n/(n+2.0)*integral_x_to_n_times_asin_x(x, n-1);
}




WITHIN_KERNEL
ftype integral_ijk_f1(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
    2.0*c0
    *(pow(cosKb,k+3)-pow(cosKa,k+3))/(k+3) //cosK
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosL
    *(pow(phib,j+1)-pow(phia,j+1))/(j+1); //phi
}



WITHIN_KERNEL
ftype integral_ijk_f2(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  ftype c0 = 9.0/(32.0*M_PI);
  return
   c0
   *(pow(cosKb,k+1)/(k+1)-pow(cosKb,k+3)/(k+3)-pow(cosKa,k+1)/(k+1)+pow(cosKa,k+3)/(k+3)) //cosK
   *(
     (pow(phib,j+1)-pow(phia,j+1))/(j+1)//phi1
     *(pow(cosLb,i+1)-pow(cosLa,i+1))/(i+1)//cosK1
     -(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosK2
     *(integral_x_to_n_times_cos_x_2(phib, j)-integral_x_to_n_times_cos_x_2(phia, j))//phi2
     );
}


WITHIN_KERNEL
ftype integral_ijk_f3(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0
    *(pow(cosKb,k+1)/(k+1)-pow(cosKb,k+3)/(k+3)-pow(cosKa,k+1)/(k+1)+pow(cosKa,k+3)/(k+3)) //cosL
    *(
      (pow(phib,j+1)-pow(phia,j+1))/(j+1)//phi1
      *(pow(cosLb,i+1)-pow(cosLa,i+1))/(i+1)//cosK1
      -(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosK2
      *(integral_x_to_n_times_sin_x_2(phib, j)-integral_x_to_n_times_sin_x_2(phia, j))//phi2
      );
}


WITHIN_KERNEL
ftype integral_ijk_f4(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0
    *(pow(cosKb,k+1)/(k+1)-pow(cosKb,k+3)/(k+3)-pow(cosKa,k+1)/(k+1)+pow(cosKa,k+3)/(k+3)) //cosL
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosK
    *(integral_x_to_n_times_sin_2x(phib, j) - integral_x_to_n_times_sin_2x(phia, j));//phi
}


WITHIN_KERNEL
ftype integral_ijk_f5(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    c0/sqrt(2.0)
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k+1))//cosL
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1))//cosK
    *(integral_x_to_n_times_cos_x(phib, j) - integral_x_to_n_times_cos_x(phia, j));//phi
}


WITHIN_KERNEL
ftype integral_ijk_f6(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    -c0/sqrt(2.0)
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k+1))//cosL
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1))//cosK
    *(integral_x_to_n_times_sin_x(phib, j) - integral_x_to_n_times_sin_x(phia, j));//phi
}


WITHIN_KERNEL
ftype integral_ijk_f7(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*2.0
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosK
    *(pow(phib,j+1)-pow(phia,j+1))/(j+1) //phi
    *(pow(cosKb,k+1)-pow(cosKa,k+1))/(k+1); //cosL
}


WITHIN_KERNEL
ftype integral_ijk_f8(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*sqrt(6.0)
    *(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k)-integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k))//cosL
    *(integral_x_to_n_times_cos_x(phib, j) - integral_x_to_n_times_cos_x(phia, j))//phi
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1));//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f9(const ftype cosKa, const ftype cosKb,
                      const ftype cosLa, const ftype cosLb,
                      const ftype phia, const ftype phib,
                      const int k, const int i, const int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    -c0*sqrt(6.0)
    *(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k)-integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k))//cosL
    *(integral_x_to_n_times_sin_x(phib, j) - integral_x_to_n_times_sin_x(phia, j))//phi
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1));//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f10(const ftype cosKa, const ftype cosKb,
                       const ftype cosLa, const ftype cosLb,
                       const ftype phia, const ftype phib,
                       const int k, const int i, const int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*4.0*sqrt(3.0)
    *(pow(cosKb,k+2)-pow(cosKa,k+2))/(k+2) //cosL
    *(pow(phib,j+1)-pow(phia,j+1))/(j+1) //phi
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)); //cosK
}


WITHIN_KERNEL
ftype getFintegral(const ftype cosKs, const ftype cosKe, 
                   const ftype cosLs, const ftype cosLe,
                   const ftype phis, const ftype phie, 
                   const int i, const int j, const int k, const int K)
{
  if (i>15){ return 0;}
  if (j>15){ return 0;}
  if (k>15){ return 0;}
  ftype fk;
  switch(K) {
    case 1:  fk = integral_ijk_f1( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 2:  fk = integral_ijk_f2( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 3:  fk = integral_ijk_f3( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 4:  fk = integral_ijk_f4( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 5:  fk = integral_ijk_f5( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 6:  fk = integral_ijk_f6( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 7:  fk = integral_ijk_f7( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 8:  fk = integral_ijk_f8( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 9:  fk = integral_ijk_f9( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    case 10: fk = integral_ijk_f10(cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k); break;
    default: 
#ifdef CUDA
      printf("Wrong k index in fk, please check code %d\\n", K);
      return 0.;
#else
      return 0.;
#endif
  }
  return fk;
}


