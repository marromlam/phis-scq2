////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay-time acceptance                          //
//                                                                            //
//   Created: 2019-01-28                                                      //
//  Modified: 2019-11-27                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file contains the following __kernels:                               //
//    * [none]                                                                //
//                                                                            //
//  TODO: The way complex numbers are handled is a bit neolithic, but as      //
//        far as openCL does not provide a standard library, this is the      //
//        only solution avaliable                                             //
//  TODO: Finish openCL translation of decay-time acceptance getM and getK    //
//        functions, that are incomplete por dta analysis but right for       //
//        computing standard pdf                                              //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>

#include "Functions.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__device__
int getTimeBin(double t)
{
  int _i = 0;
  int _n = NKNOTS-1;
  //printf("%d\n", _n);
  while(_i <= _n )
  {
    if( t < KNOTS[_i] ) {break;}
    _i++;
  }
  if (0 == _i) {printf("WARNING: t=%lf below first knot!\n",t);}
  return _i - 1;

}



__device__
double getKnot(int i)
{
  if (i<=0) {
    i = 0;
  }
  else if (i>=NKNOTS) {
    i = NKNOTS;
  }
  return KNOTS[i];
}



__device__
double getCoeff(double *mat, int r, int c)
{
  return mat[4*r+c];
}




__device__
double calcTimeAcceptance(double t, double *coeffs, double tLL, double tUL)
{
  int bin   = getTimeBin(t);
  if (t < tLL) { return 0.0; }
  if (t > tUL) { return 0.0; }

  double c0 = getCoeff(coeffs,bin,0);
  double c1 = getCoeff(coeffs,bin,1);
  double c2 = getCoeff(coeffs,bin,2);
  double c3 = getCoeff(coeffs,bin,3);

  double result = (c0 + t*(c1 + t*(c2 + t*c3)));
  if (DEBUG >= 3 && ( threadIdx.x + blockDim.x * blockIdx.x == 0))
  {
    printf("TIME ACC  : t=%lf\tbin=%d\tc=[%+lf\t%+lf\t%+lf\t%+lf]\tdta=%lf\n",
           t,bin,c0,c1,c2,c3,result);
  }

  return result;
}



__device__
pycuda::complex<double> getExponentialConvolution(double t, double G, double omega, double sigma)
{

  double sigma2 = sigma*sigma;
  pycuda::complex<double> I(0,1);

  if( t >SIGMA_THRESHOLD*sigma )
  {//2.*(sqrt(0.5*M_PI))* this was an old factor
    return exp(-G*t+0.5*G*G*sigma2-0.5*omega*omega*sigma2)*(cos(omega*(t-G*sigma2)) + I*sin(omega*(t-G*sigma2)));
  }
  else
  {
    pycuda::complex<double> z, fad, result;
    z   = (-I*(t-sigma2*G) - omega*sigma2)/(sigma*sqrt(2.));
    fad = faddeeva(z);
    fad = (pycuda::real(fad) - I*pycuda::imag(fad));
    return sqrt(0.5*M_PI)*exp(-0.5*t*t/sigma2)*fad;
  }
}






































__device__
pycuda::complex<double> getK(pycuda::complex<double> z, int n)
{
  if (n == 0)
  {
    return 1./(2.*z);
  }
  else if (n == 1)
  {
    return 1./(2.*z*z);
  }
  else if (n == 2)
  {
    return 1./z*(1.+1./(z*z));
  }
  else if (n == 3)
  {
    return 3./(z*z)*(1.+1./(z*z));
  }
  else if (n == 4)
  {
    return 6./z*(1.+2./(z*z)+2./(z*z*z*z));
  }
  else if (n == 5)
  {
    return 30./(z*z)*(1.+2./(z*z)+2./(z*z*z*z));
  }
  else if (n == 6)
  {
    return 60./z*(1.+3./(z*z)+6./(z*z*z*z)+6./(z*z*z*z*z*z));
  }

  return pycuda::complex<double>(0.,0.);

}















__device__
pycuda::complex<double> getM(double x, int n, double t, double sigma,
                              double gamma, double omega)
{
  pycuda::complex<double> conv_term;
  conv_term = getExponentialConvolution(t,gamma,omega,sigma)/(sqrt(0.5*M_PI));

  if (n == 0)
  {
    return pycuda::complex<double>(erf(x),0.)-conv_term;
  }
  else if (n == 1)
  {
    return 2.*(-pycuda::complex<double>(sqrt(1./M_PI)*exp(-x*x),0.)-x*conv_term);
  }
  else if (n == 2)
  {
    return 2.*(-2.*x*exp(-x*x)*pycuda::complex<double>(sqrt(1./M_PI),0.)-(2.*x*x-1.)*conv_term);
  }
  else if (n == 3)
  {
    return 4.*(-(2.*x*x-1.)*exp(-x*x)*pycuda::complex<double>(sqrt(1./M_PI),0.)-x*(2.*x*x-3.)*conv_term);
  }
  else if (n == 4)
  {
    return 4.*(exp(-x*x)*(6.*x+4.*x*x*x)*pycuda::complex<double>(sqrt(1./M_PI),0.)-(3.-12.*x*x+4.*x*x*x*x)*conv_term);
  }
  else if (n == 5)
  {
    return 8.*(-(3.-12.*x*x+4.*x*x*x*x)*exp(-x*x)*pycuda::complex<double>(sqrt(1./M_PI),0.)-x*(15.-20.*x*x+4.*x*x*x*x)*conv_term);
  }
  else if (n == 6)
  {
    return 8.*(-exp(-x*x)*(30.*x-40.*x*x*x+8.*x*x*x*x*x)*pycuda::complex<double>(sqrt(1./M_PI),0.)-(-15.+90.*x*x-60.*x*x*x*x+8.*x*x*x*x*x*x)*conv_term);
  }
  return pycuda::complex<double>(0.,0.);
}




__device__
void intgTimeAcceptance(double time_terms[4], double sigma,
                        double G, double DG, double DM,
                        double *knots, double *coeffs, int n, double t0)
{
  // Add tUL to knots list
  knots[NTIMEBINS] = 15; n += 1;
  double x[NTIMEBINS];

  double aux1 = 1./(sqrt(2.0)*sigma);
  pycuda::complex<double> aux2 = (sigma/(sqrt(2.0)),0);

  for(int i = 0; i < SPL_BINS+1; i++)
  {
    x[i] = (knots[i] - t0)*aux1;
  }

  // Fill S matrix                (TODO speed to be gained here - S is constant)
  double S[SPL_BINS][4][4];
  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int i=0; i<4; ++i)
    {
      for (int j=0; j<4; ++j)
      {
        if(i+j < 4)
        {
          S[bin][i][j] = getCoeff(coeffs,bin,i+j)
                         *factorial(i+j)/factorial(j)/factorial(i)/pow(2.0,i+j);
        }
        else
        {
          S[bin][i][j] = 0.;
        }
      }
    }
  }

  pycuda::complex<double> z_sinh, K_sinh[4], M_sinh[SPL_BINS+1][4];
  pycuda::complex<double> z_cosh, K_cosh[4], M_cosh[SPL_BINS+1][4];
  pycuda::complex<double> z_trig, K_trig[4], M_trig[SPL_BINS+1][4];

  z_cosh = aux2 * pycuda::complex<double>(G-0.5*DG,  0);
  z_sinh = aux2 * pycuda::complex<double>(G+0.5*DG,  0);
  z_trig = aux2 * pycuda::complex<double>(       G,-DM);

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_cosh[j] = getK(z_cosh,j);
    K_sinh[j] = getK(z_sinh,j);
    K_trig[j] = getK(z_trig,j);
  }

  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < SPL_BINS+1; ++bin)
    {
      M_sinh[bin][j] = getM(x[bin],j,knots[bin]-t0,sigma,G-0.5*DG,0.);
      M_cosh[bin][j] = getM(x[bin],j,knots[bin]-t0,sigma,G+0.5*DG,0.);
      M_trig[bin][j] = getM(x[bin],j,knots[bin]-t0,sigma,G,DM);
    }
  }

  // Fill the delta factors to multiply by the integrals
  double sigma_fact[4];
  for (int i=0; i<4; ++i)
  {
    sigma_fact[i] = pow(sigma*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, sinh, cos, sin terms
  double int_sinh = 0;
  double int_cosh = 0;
  pycuda::complex<double> int_trig = pycuda::complex<double>(0.,0.);

  for (int ibin=0; ibin < SPL_BINS; ++ibin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        int_sinh += pycuda::real(S[ibin][j][k]*
                                 (M_sinh[ibin+1][j] - M_sinh[ibin][j])*
                                 K_cosh[k])*
                                 sigma_fact[j+k];

        int_cosh += pycuda::real(S[ibin][j][k]*
                                 (M_cosh[ibin+1][j] - M_cosh[ibin][j])*
                                 K_sinh[k])*
                                 sigma_fact[j+k];

        int_trig +=  S[ibin][j][k]*
                     (M_trig[ibin+1][j] - M_trig[ibin][j])*
                     K_trig[k]*
                     sigma_fact[j+k];
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = 0.5*(int_sinh + int_cosh);
  time_terms[1] = 0.5*(int_sinh - int_cosh);
  time_terms[2] = pycuda::real(int_trig);
  time_terms[3] = pycuda::imag(int_trig);

  if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {
    printf("INTEGRAL           : ta=%.8lf\ttb=%.8lf\ttc=%.8lf\ttd=%.8lf\n",
           time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
  }
}



__device__
void integralFullSpline( double result[2],
                         double vn[10], double va[10],double vb[10], double vc[10],double vd[10],
                         double *norm, double G, double DG, double DM,
                         double delta_t,
                         double t_ll,
                         double t_offset,
                         double nknots, double *knots,
                         double *spline_coeffs)
{
  double integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM,
                     knots, spline_coeffs, nknots, t_offset) ;
  double ta = integrals[0];
  double tb = integrals[1];
  double tc = integrals[2];
  double td = integrals[3];

  for(int k=0; k<10; k++)
  {
    result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
    result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
  }
}



/*
__device__
pycuda::complex<double> getExponentialConvolution(double t, double gamma,
                                                  double omega, double sigma)
{
  pycuda::complex<double> I(0,1);
  pycuda::complex<double> z, fad;
  double sigma2 = sigma*sigma;

  if( t >sigma_threshold*sigma )
  {
    return  //2.*(sqrt(0.5*M_PI))* this was an old factor
  exp(-gamma*t+0.5*gamma*gamma*sigma2-0.5*omega*omega*sigma2)*
  (cos(omega*(t-gamma*sigma2)) + I*sin(omega*(t-gamma*sigma2)));
  }
  else
  {
    z   = (-I*(t-sigma2*gamma) - omega*sigma2)/(sigma*sqrt(2.));
    fad = faddeeva(z);
    fad = (pycuda::real(fad) - I*pycuda::imag(fad));
    return sqrt(0.5*M_PI)*exp(-0.5*t*t/sigma2)*fad;
  }
}
*/



////////////////////////////////////////////////////////////////////////////////
