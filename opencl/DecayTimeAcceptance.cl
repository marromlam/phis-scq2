////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay-time acceptance                          //
//                                                                            //
//   Created: 2019-11-18                                                      //
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

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

// Spline parameters
__constant int sigma_threshold = 5.0;
__constant int time_acc_bins = 40;
#define spl_bins 7

#include "Functions.cl"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__global
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



__global
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



__global
double getCoeff(double *mat, int r, int c)
{
  return mat[4*r+c];
}




__global
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
  if (DEBUG >= 3 && ( get_global_id(0) == 0))
  {
    printf("TIME ACC  : t=%lf\tbin=%d\tc=[%+lf\t%+lf\t%+lf\t%+lf]\tdta=%lf\n",
           t,bin,c0,c1,c2,c3,result);
  }

  return result;
}



__global
cdouble_t getExponentialConvolution(double t, double G, double omega, double sigma)
{
  cdouble_t I  = cdouble_new(0,+1);
  cdouble_t I2 = cdouble_new(-1,0);
  cdouble_t I3 = cdouble_new(0,-1);
  cdouble_t z, fad, result;
  double sigma2 = sigma*sigma;

  if( t >sigma_threshold*sigma )
  {
    return  //2.*(sqrt(0.5*M_PI))* this was an old factor
            cdouble_mul(
              cdouble_new(exp(-G*t+0.5*G*G*sigma2-0.5*omega*omega*sigma2),0),
              cdouble_add( cdouble_new(cos(omega*(t-G*sigma2)),0) ,
                         cdouble_mul( I , cdouble_new(sin(omega*(t-G*sigma2)),0) )
                          )
                        );
  }
  else
  {
    // z   = (-I*(t-sigma2*G) - omega*sigma2)/(sigma*sqrt(2.));
    // fad = faddeeva(z);
    // fad = (pycuda::real(fad) - I*pycuda::imag(fad));
    // return sqrt(0.5*M_PI)*exp(-0.5*t*t/sigma2)*fad;
    z   = cdouble_new( 0 , -(t-sigma2*G) );
    z   = cdouble_add( z , cdouble_new(-omega*sigma2,0) );
    z   = cdouble_mul( z , cdouble_new(1/(sigma*sqrt(2.)),0) );
    fad = faddeeva(z);
    fad = cdouble_add( cdouble_new(fad.real,0), cdouble_mul(I3,cdouble_new(fad.imag,0)) );
    result = cdouble_new( sqrt(0.5*M_PI), 0);
    result = cdouble_mul( result, cdouble_new(exp(-0.5*t*t/sigma2),0) );
    result = cdouble_mul( result, fad );
    return result;
  }
}



__global
cdouble_t getK(cdouble_t z, int n)
{
  cdouble_t z2 = cdouble_mul(z,z);
  cdouble_t z3 = cdouble_mul(z,z2);
  cdouble_t z4 = cdouble_mul(z,z3);
  cdouble_t z5 = cdouble_mul(z,z4);
  cdouble_t z6 = cdouble_mul(z,z5);
  cdouble_t w;

  if (n == 0)      {
    w = cdouble_mul( cdouble_new(2,0), z);
    return cdouble_divide(cdouble_new( 1.,0), w );
  }
  else if (n == 1) {
    w = cdouble_mul( cdouble_new(2,0), z2);
    return cdouble_divide(cdouble_new( 1.,0), w );
  }
  else if (n == 2) {
    w = cdouble_divide( cdouble_new(1,0), z2 );
    w = cdouble_add( cdouble_new(1,0), w );
    w = cdouble_mul( z, w );
    return cdouble_divide(cdouble_new( 1.,0), w );
  }
  else if (n == 3) {
    w = cdouble_divide( cdouble_new(1,0), z2 );
    w = cdouble_add( cdouble_new(1,0), w );
    w = cdouble_mul( z2, w );
    return cdouble_divide(cdouble_new( 3.,0), w );
  }
  // else if (n == 4) {
  //   return cdouble_divide(cdouble_new( 6.,0), z*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 5) {
  //   return cdouble_divide(cdouble_new(30.,0), (z*z)*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 6) {
  //   return cdouble_divide(cdouble_new(60.,0), z*(1.+3./(z*z)+6./(z*z*z*z)+6./(z*z*z*z*z*z))  );
  // }

  return cdouble_new(0.,0.);
}



__global
cdouble_t getM(double x, int n, double t, double sigma, double G, double omega)
{
  cdouble_t conv_term;
  cdouble_t I  = cdouble_new(0,+1);
  cdouble_t I2 = cdouble_new(-1,0);
  cdouble_t I3 = cdouble_new(0,-1);

  conv_term = getExponentialConvolution(t, G, omega, sigma);
  conv_term = cdouble_mul( conv_term, cdouble_new(1/(sqrt(0.5*M_PI)),0) );

  if (n == 0)
  {
    cdouble_t a = cdouble_new(erf(x),0.);
    cdouble_t b = conv_term;
    return cdouble_subt(a,b);
  }
  else if (n == 1)
  {
    cdouble_t a = cdouble_new(sqrt(1./M_PI)*exp(-x*x),0.);
    cdouble_t b = cdouble_new(x,0);
    b = cdouble_mul(b,conv_term);
    return cdouble_mul(cdouble_new(2,0),cdouble_subt(a,b));
  }
  else if (n == 2)
  {
    // return 2.*(-2.*x*exp(-x*x)*cdouble_t(sqrt(1./M_PI),0.)-(2.*x*x-1.)*conv_term);
    cdouble_t a = cdouble_new(-2.*x*exp(-x*x)*sqrt(1./M_PI),0.);
    cdouble_t b = cdouble_new(2*x*x-1,0);
    b = cdouble_mul(b,conv_term);
    return cdouble_mul(cdouble_new(2,0),cdouble_subt(a,b));
  }
  else if (n == 3)
  {
    // return 4.*(-(2.*x*x-1.)*exp(-x*x)*cdouble_t(sqrt(1./M_PI),0.)-x*(2.*x*x-3.)*conv_term);
    cdouble_t a = cdouble_new(-(2.*x*x-1.)*exp(-x*x)*sqrt(1./M_PI),0.);
    cdouble_t b = cdouble_new(x*(2*x*x-3),0);
    b = cdouble_mul(b,conv_term);
    return cdouble_mul(cdouble_new(4,0),cdouble_subt(a,b));
  }
  // else if (n == 4)
  // {
  //   return 4.*(exp(-x*x)*(6.*x+4.*x*x*x)*cdouble_t(sqrt(1./M_PI),0.)-(3.-12.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 5)
  // {
  //   return 8.*(-(3.-12.*x*x+4.*x*x*x*x)*exp(-x*x)*cdouble_t(sqrt(1./M_PI),0.)-x*(15.-20.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 6)
  // {
  //   return 8.*(-exp(-x*x)*(30.*x-40.*x*x*x+8.*x*x*x*x*x)*cdouble_t(sqrt(1./M_PI),0.)-(-15.+90.*x*x-60.*x*x*x*x+8.*x*x*x*x*x*x)*conv_term);
  // }
  return cdouble_new(0.,0.);
}



__global
void intgTimeAcceptance(double time_terms[4], double sigma,
                        double G, double DG, double DM,
                        double *knots, double *coeffs, int n, double t0)
{
  // Add tUL to knots list
  knots[7] = 15; n += 1;
  double x[NTIMEBINS];

  double aux1 = 1./(sqrt(2.0)*sigma);
  cdouble_t aux2 = cdouble_new(sigma/(sqrt(2.0)),0);

  for(int i = 0; i < spl_bins+1; i++)
  {
    x[i] = (knots[i] - t0)*aux1;
  }

  // Fill S matrix                (TODO speed to be gained here - S is constant)
  double S[spl_bins][4][4];
  for (int bin=0; bin < spl_bins; ++bin)
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


  cdouble_t z_sinh, K_sinh[4], M_sinh[spl_bins+1][4];
  cdouble_t z_cosh, K_cosh[4], M_cosh[spl_bins+1][4];
  cdouble_t z_trig, K_trig[4], M_trig[spl_bins+1][4];

  z_cosh = cdouble_mul( aux2, cdouble_new(G-0.5*DG,  0) );
  z_sinh = cdouble_mul( aux2, cdouble_new(G+0.5*DG,  0) );
  z_trig = cdouble_mul( aux2, cdouble_new(       G,-DM) );

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
    for(int bin=0; bin < spl_bins+1; ++bin)
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
  cdouble_t int_sinh = cdouble_new(0.,0.); cdouble_t int_cosh = cdouble_new(0.,0.);
  cdouble_t int_trig = cdouble_new(0.,0.);

  for (int bin=0; bin < spl_bins; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        cdouble_t aux3      = cdouble_new(S[bin][j][k]*sigma_fact[j+k],0);

        cdouble_t int_sinh_aux = cdouble_subt(M_sinh[bin+1][j],M_sinh[bin][j]);
        int_sinh_aux = cdouble_mul(int_sinh_aux,K_cosh[k]);
        int_sinh_aux = cdouble_mul(int_sinh_aux,aux3);
        int_sinh     = cdouble_add( int_sinh, int_sinh_aux );

        cdouble_t int_cosh_aux = cdouble_subt(M_cosh[bin+1][j],M_cosh[bin][j]);
        int_cosh_aux = cdouble_mul(int_cosh_aux,K_sinh[k]);
        int_cosh_aux = cdouble_mul(int_cosh_aux,aux3);
        int_cosh     = cdouble_add( int_cosh, int_cosh_aux );

        cdouble_t int_trig_aux = cdouble_subt(M_trig[bin+1][j],M_trig[bin][j]);
        int_trig_aux = cdouble_mul(int_trig_aux,K_trig[k]);
        int_trig_aux = cdouble_mul(int_trig_aux,aux3);
        int_trig     = cdouble_add( int_trig, int_trig_aux );

        // printf("INTEGRAL  :aux3=%.8lf\tsubt=%.8lf\tK=%.8lf\n",
        //         aux3,cdouble_subt(M_sinh[bin+1][j], M_sinh[bin][j]).real, K_cosh[k].real);
        // printf("INTEGRAL  :int_sinh=%.8lf\tint_cosh=%.8lf\n",
        //         int_sinh.real,int_cosh.real);
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = 0.5*cdouble_add(int_sinh,int_cosh).real;
  time_terms[1] = 0.5*cdouble_add(int_sinh,cdouble_mul(cdouble_new(-1,0),int_cosh)).real;
  time_terms[2] = int_trig.real;
  time_terms[3] = int_trig.imag;

  if (DEBUG > 3 && ( get_global_id(0) == 0) )
  {
    printf("INTEGRAL           : ta=%.8lf\ttb=%.8lf\ttc=%.8lf\ttd=%.8lf\n",
           time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
  }
}



__global
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



////////////////////////////////////////////////////////////////////////////////
