//this methods are helpful for the phi integration
//calculates int x^n * sin(x) dx
#include <ipanema/complex.hpp>

#ifdef CUDA
  WITHIN_KERNEL
  int convert_float(float in) {
      union fi { int i; float f; } conv;
      conv.f = in;
      return conv.i;
  }
#endif


WITHIN_KERNEL ftype integral_x_to_n_times_sin_x(ftype x, int n);
WITHIN_KERNEL ftype integral_x_to_n_times_cos_x(ftype x, int n);
WITHIN_KERNEL ftype integral_x_to_n_times_sin_2x(ftype x, int n);
WITHIN_KERNEL ftype integral_x_to_n_times_cos_2x(ftype x, int n);
WITHIN_KERNEL ftype integral_x_to_n_times_sqrt_1_minus_x2(ftype x, int n);

//this methods are helpful for the phi integration
//calculates int x^n * sin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_x(ftype x, int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return -cos(x);
  else
    return -pow(x, n)*cos(x) + n*integral_x_to_n_times_cos_x(x, n-1);
}

//calculates int x^n * cos(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_cos_x(ftype x, int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return sin(x);
  else
    return pow(x,n)*sin(x) - n*integral_x_to_n_times_sin_x(x, n-1);
}

//calculates int x^n * sin(2x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_2x(ftype x, int n)
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
ftype integral_x_to_n_times_cos_2x(ftype x, int n)
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
ftype integral_x_to_n_times_cos_x_2(ftype x, int n)
{
  //ftype N = convert_float(n);
  return +1.0/(1.0 + n)*pow(x,n+1)*cos(x)*cos(x)
    +1.0/(1.0+n)*integral_x_to_n_times_sin_2x(x, n+1);
}

//calculates int x^n * sin(x)^2 dx
WITHIN_KERNEL
ftype integral_x_to_n_times_sin_x_2(ftype x, int n)
{
  //ftype N = convert_float(n);
  return +1.0/(1.0 + n)*pow(x,n+1)*sin(x)*sin(x)
    -1.0/(1.0+n)*integral_x_to_n_times_sin_2x(x, n+1);
}

//calculates int x^n * asin(x) dx
WITHIN_KERNEL
ftype integral_x_to_n_times_asin_x(ftype x, int n)
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
ftype integral_x_to_n_times_sqrt_1_minus_x2(ftype x, int n)
{
  //ftype N = convert_float(n);
  if (n == 0)
    return 0.5*asin(x)+0.5*x*sqrt(1-x*x);
  else
    return 2.0/(n+2.0)*pow(x, n)*(0.5*asin(x)+0.5*x*sqrt(1-x*x))
      -n/(n+2.0)*integral_x_to_n_times_asin_x(x, n-1);
}




WITHIN_KERNEL
ftype integral_ijk_f1(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
  ftype c0 = 9.0/(32.0*M_PI);
  return
    2.0*c0
    *(pow(cosKb,k+3)-pow(cosKa,k+3))/(k+3) //cosK
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)) //cosL
    *(pow(phib,j+1)-pow(phia,j+1))/(j+1); //phi
}



WITHIN_KERNEL
ftype integral_ijk_f2(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
{
  //assert(i>=0);
  //assert(j>=0);
  //assert(k>=0);
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
ftype integral_ijk_f3(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
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
ftype integral_ijk_f4(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
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
ftype integral_ijk_f5(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
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
ftype integral_ijk_f6(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
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
ftype integral_ijk_f7(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
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
ftype integral_ijk_f8(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*sqrt(6.0)
    *(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k)-integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k))//cosL
    *(integral_x_to_n_times_cos_x(phib, j) - integral_x_to_n_times_cos_x(phia, j))//phi
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1));//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f9(ftype cosKa, ftype cosKb,
                                      ftype cosLa, ftype cosLb,
                                      ftype phia, ftype phib,
                                      int k, int i, int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    -c0*sqrt(6.0)
    *(integral_x_to_n_times_sqrt_1_minus_x2(cosKb, k)-integral_x_to_n_times_sqrt_1_minus_x2(cosKa, k))//cosL
    *(integral_x_to_n_times_sin_x(phib, j) - integral_x_to_n_times_sin_x(phia, j))//phi
    *2.0*(integral_x_to_n_times_sqrt_1_minus_x2(cosLb, i+1) - integral_x_to_n_times_sqrt_1_minus_x2(cosLa, i+1));//cosK
}


WITHIN_KERNEL
ftype integral_ijk_f10(ftype cosKa, ftype cosKb,
                                       ftype cosLa, ftype cosLb,
                                       ftype phia, ftype phib,
                                       int k, int i, int j)
{
  const ftype c0 = +3.0/(32.0*M_PI);
  return
    c0*4.0*sqrt(3.0)
    *(pow(cosKb,k+2)-pow(cosKa,k+2))/(k+2) //cosL
    *(pow(phib,j+1)-pow(phia,j+1))/(j+1) //phi
    *(pow(cosLb,i+1)/(i+1)-pow(cosLb,i+3)/(i+3)-pow(cosLa,i+1)/(i+1)+pow(cosLa,i+3)/(i+3)); //cosK
}



KERNEL
void integral_ijk_fx(ftype cosKs, ftype cosKe, ftype cosLs, ftype cosLe, 
                     ftype phis, ftype phie, int i, int j, int k,
                     GLOBAL_MEM ftype * fx)
{
  fx[0] = integral_ijk_f1( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[1] = integral_ijk_f2( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[2] = integral_ijk_f3( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[3] = integral_ijk_f4( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[4] = integral_ijk_f5( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[5] = integral_ijk_f6( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[6] = integral_ijk_f7( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[7] = integral_ijk_f8( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[8] = integral_ijk_f9( cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
  fx[9] = integral_ijk_f10(cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k);
}







WITHIN_KERNEL
ftype ang_eff(const ftype cosK, const ftype cosL, const ftype phi, ftype *moments)
{
    ftype eff = 0.;

    eff += moments[0] * legendre_poly(0, 0, cosK) * sph_harm(0, 0, cosL, phi);
    eff += moments[1] * legendre_poly(0, 0, cosK) * sph_harm(2, 0, cosL, phi);
    eff += moments[2] * legendre_poly(0, 0, cosK) * sph_harm(2, 2, cosL, phi);
    eff += moments[3] * legendre_poly(0, 0, cosK) * sph_harm(2, 1, cosL, phi);
    eff += moments[4] * legendre_poly(0, 0, cosK) * sph_harm(2,-1, cosL, phi);
    eff += moments[5] * legendre_poly(0, 0, cosK) * sph_harm(2,-2, cosL, phi);
    eff += moments[6] * legendre_poly(1, 0, cosK) * sph_harm(0, 0, cosL, phi);
    eff += moments[7] * legendre_poly(1, 0, cosK) * sph_harm(2, 1, cosL, phi);
    eff += moments[8] * legendre_poly(1, 0, cosK) * sph_harm(2,-1, cosL, phi);
    eff += moments[9] * legendre_poly(2, 0, cosK) * sph_harm(0, 0, cosL, phi);

    eff *= 2.*sqrt(M_PI);
//     printf("Ang. eff = %lf \n", eff);
    return eff;
}




WITHIN_KERNEL
void angWeightsToMoments(ftype* moments, GLOBAL_MEM const ftype* normweights)
{
    //c0000
    moments[0]  =   1. / 3. * ( normweights[0] + normweights[1] + normweights[2] );//
    //c0020
    moments[1]  =   1. / 3. * sqrt(5.)             * ( normweights[0] + normweights[1] + normweights[2] - 3. * normweights[6] );//
    //c0022
    moments[2]  =            -sqrt(5. / 3.)        * ( normweights[1] - normweights[2] );//
    //c0021
    moments[3]  = - 8. / 3. * sqrt( 5. / 2. ) / M_PI *   normweights[7];//
    //c002-1
    moments[4]  = - 8. / 3. * sqrt( 5. / 2. ) / M_PI *  (normweights[8]);//-normweights should be +normweights?
    //c002-2
    moments[5]  =             sqrt( 5. / 3. )      *  (normweights[3]);//-normweights should be +normweights?
    //c1000
    moments[6]  =   1. / 2. * sqrt(3.)             *   normweights[9];//
    //c1021
    moments[7]  = - 32. / 3. * sqrt( 5. / 6. ) / M_PI *   normweights[4];//
    //c102-1
    moments[8]  = + 32. / 3. * sqrt( 5. / 6. ) / M_PI *  (normweights[5]);//-normweights should be +normweights?
    //c2000
    moments[9]  =  5. / 2.                        * ( normweights[0] - normweights[6] );//
}






KERNEL
void plot_moments(GLOBAL_MEM const ftype *normweights, GLOBAL_MEM ftype *out,
                  GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL, 
                  GLOBAL_MEM const ftype *hphi)
{
  const int i = get_global_id(0);

  ftype moments[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  // get the moments
  angWeightsToMoments(moments, normweights);
  out[i] = ang_eff(cosK[i], cosL[i], hphi[i], moments);

  //ftype ang_acc = ang_eff(x, y, z, moments); // these are angular weights again

}
