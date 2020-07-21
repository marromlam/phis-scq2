////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//   Created: 2019-01-28                                                      //
//  Modified: 2019-11-27                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
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
  double c0 = getCoeff(coeffs,bin,0);
  double c1 = getCoeff(coeffs,bin,1);
  double c2 = getCoeff(coeffs,bin,2);
  double c3 = getCoeff(coeffs,bin,3);

  #ifdef DEBUG
  if (DEBUG >= 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT))
  {
    printf("\nTIME ACC           : t=%lf\tbin=%d\tc=[%+lf\t%+lf\t%+lf\t%+lf]\tdta=%+.16lf\n",
           t,bin,c0,c1,c2,c3, (c0 + t*(c1 + t*(c2 + t*c3))) );
  }
  #endif

  return (c0 + t*(c1 + t*(c2 + t*c3)));
}


__device__
pycuda::complex<double> ipanema_erfc(pycuda::complex<double> z)
{
  double re = -z.real() * z.real() + z.imag() * z.imag();
  double im = -2. * z.real() * z.imag();
  re = exp(re) * cos(im);
  im = exp(re) * sin(im);

  if (z.imag() < 0.0) {
    return (2.-     pycuda::complex<double> (re, im) * faddeeva(pycuda::complex<double> (z.imag(), -z.real())));
  }
  if (z.real() >= 0.0) {
    return (     pycuda::complex<double> (re, im) * faddeeva(pycuda::complex<double> (-z.imag(), z.real())));
  }
  else{
    return (2. - pycuda::complex<double> (re, im) * faddeeva(pycuda::complex<double> (z.imag(), -z.real())));
  }
}



__device__
pycuda::complex<double> cErrF_2(pycuda::complex<double> x)
{
  pycuda::complex<double> I(0.0,1.0);
  pycuda::complex<double> z(I*x);
  pycuda::complex<double> result = exp(-x*x)*faddeeva(z);

  if (x.real() > 20.0)// && fabs(x.imag()) < 20.0)
    result = 0.0;
  if (x.real() < -20.0)// && fabs(x.imag()) < 20.0)
    result = 2.0;

  return result;
}


__device__
pycuda::complex<double> getExponentialConvolution_simon(double t, double G, double omega, double sigma)
{
  double sigma2 = sigma*sigma;
  pycuda::complex<double> I(0,1);

  if (omega == 0)
  {
    if( t > -6.0*sigma ){
      double exp_part = exp(-t*G + 0.5*G*G*sigma2 -0.5*omega*omega*sigma2);
      pycuda::complex<double> my_erfc = ipanema_erfc(sigma*G/sqrt(2.0) - t/sigma/sqrt(2.0));
      return 0.5 * exp_part * my_erfc;
    }
    else{
      return 0.0;
    }
  }
  else //(omega != 0)
  {
    double c1 = 0.5;

    double exp1arg = 0.5*sigma2*(G*G - omega*omega) - t*G;
    double exp1 = exp(exp1arg);

    double exp2arg = -omega*(t - sigma2*G);
    pycuda::complex<double> exp2(cos(exp2arg), sin(exp2arg));

    pycuda::complex<double> cerfarg(sigma*G/sqrt(2.0) - t/(sigma*sqrt(2.0)) , +omega*sigma/sqrt(2.0));
    pycuda::complex<double> cerf;

    if  (cerfarg.real() < -20.0)
    {
      cerf = pycuda::complex<double>(2.0,0.0);
    }
    else
    {
      cerf = cErrF_2(cerfarg);//best complex error function
    }
    pycuda::complex<double> c2(exp2*cerf);
    double im = -c2.imag();//exp*sin
    double re = +c2.real();//exp*cos

    return c1*exp1* (pycuda::real(c2) - I*pycuda::imag(c2));
  }

}






__device__
pycuda::complex<double> getExponentialConvolution(double t, double G, double omega, double sigma)
{

  double sigma2 = sigma*sigma;
  pycuda::complex<double> I(0,1);

  if( t >SIGMA_THRESHOLD*sigma )
  {//2.*(sqrt(0.5*M_PI))*
    return exp(-G*t+0.5*G*G*sigma2-0.5*omega*omega*sigma2)*(cos(omega*(t-G*sigma2)) + I*sin(omega*(t-G*sigma2)));
  }
  else
  {//sqrt(0.5*M_PI)
    pycuda::complex<double> z, fad, result;
    z   = (-I*(t-sigma2*G) - omega*sigma2)/(sigma*sqrt(2.));
    fad = faddeeva(z);
    fad = (pycuda::real(fad) - I*pycuda::imag(fad));
    return 0.5*exp(-0.5*t*t/sigma2)*fad;
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
  pycuda::complex<double> conv_term, z;
  pycuda::complex<double> I(0,1);
  z = sigma/(sqrt(2.0)) * pycuda::complex<double>(gamma,-omega);
  //conv_term = 5.0*getExponentialConvolution(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));
  // warning there are improvement to do here!!!
  if (omega == 0){
    conv_term = exp(z*z-2*x*z)*(  ipanema_erfc(-I*(I*(z-x)))  );
  }
  else{
    conv_term = exp(z*z-2*x*z)*(  cErrF_2(-I*(I*(z-x)))  );
    //conv_term = 2.0*getExponentialConvolution_simon(t,gamma,omega,sigma);
    //conv_term = 2.0*exp(-gamma*t+0.5*gamma*gamma*sigma*sigma-0.5*omega*omega*sigma*sigma)*(cos(omega*(t-gamma*sigma*sigma)) + I*sin(omega*(t-gamma*sigma*sigma)));
  }
  //conv_term = 2.0*getExponentialConvolution_simon(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));

  // #ifdef DEBUG
  // if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) ){
  //   printf("\nerfc*exp = %+.16lf %+.16lfi\n",  pycuda::real(conv_term), pycuda::imag(conv_term));
  //   printf("erfc = %+.16lf %+.16lfi\n",  pycuda::real(ipanema_erfc(-I*(I*(z-x)))), pycuda::imag(ipanema_erfc(-I*(I*(z-x)))));
  //   printf("cErrF_2 = %+.16lf %+.16lfi\n",  pycuda::real(cErrF_2(-I*(I*(z-x)))), pycuda::imag(cErrF_2(-I*(I*(z-x)))));
  //   printf("exp  = %+.16lf %+.16lfi\n",  pycuda::real(exp(z*z-2*x*z)), pycuda::imag(exp(z*z-2*x*z)));
  //   printf("z    = %+.16lf %+.16lfi     %+.16lf %+.16lf %+.16lf        x = %+.16lf\n",  z.real(), z.imag(), gamma, omega, sigma, x);
  // }
  // #endif

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
void intgTimeAcceptance(double time_terms[4], double delta_t,
                        double G, double DG, double DM,
                        double *coeffs, double t0)
{
  // Some constants
  double cte1 = 1./(sqrt(2.0)*delta_t);
  double cte2 = delta_t/(sqrt(2.0));
  if (delta_t <= 0.0)
  {
    printf("WARNING            : delta_t = %.4f is not a valid value.\n", delta_t);
  }

  // Add tUL to knots list
  double x[NTIMEBINS] = {0.};
  double knots[NTIMEBINS] = {0.};
  for(int i = 0; i < NKNOTS; i++)
  {
    knots[i] = KNOTS[i];
    x[i] = (knots[i] - t0)*cte1;
  }
  knots[NKNOTS] = 15; x[NKNOTS] = (knots[NKNOTS] - t0)*cte1; // WARNING! HARDCODED NUMBER

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

  pycuda::complex<double> z_expm, K_expm[4], M_expm[SPL_BINS+1][4];
  pycuda::complex<double> z_expp, K_expp[4], M_expp[SPL_BINS+1][4];
  pycuda::complex<double> z_trig, K_trig[4], M_trig[SPL_BINS+1][4];

  z_expm = cte2 * pycuda::complex<double>(G-0.5*DG,  0);
  z_expp = cte2 * pycuda::complex<double>(G+0.5*DG,  0);
  z_trig = cte2 * pycuda::complex<double>(       G,-DM);

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_expp[j] = getK(z_expp,j);
    K_expm[j] = getK(z_expm,j);
    K_trig[j] = getK(z_trig,j);
    #ifdef DEBUG
    if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
    {
      printf("K_expp[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n",  j,z_expp.real(),z_expp.imag(),K_expp[j].real(),K_expp[j].imag());
      printf("K_expm[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n",  j,z_expm.real(),z_expm.imag(),K_expm[j].real(),K_expm[j].imag());
      printf("K_trig[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n\n",j,z_trig.real(),z_trig.imag(),K_trig[j].real(),K_trig[j].imag());
    }
    #endif
  }


  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < SPL_BINS+1; ++bin)
    {
      M_expm[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G-0.5*DG,0.);
      M_expp[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G+0.5*DG,0.);
      M_trig[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G,DM);
      if (bin>0){
        #ifdef DEBUG
        if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
        {
          pycuda::complex<double> aja = M_expp[bin][j]-M_expp[bin-1][j];
          pycuda::complex<double> eje = M_expm[bin][j]-M_expm[bin-1][j];
          pycuda::complex<double> iji = M_trig[bin][j]-M_trig[bin-1][j];
          printf("bin=%d M_expp[%d] = %+.14lf%+.14lf\n",  bin,j,aja.real(),aja.imag());
          printf("bin=%d M_expm[%d] = %+.14lf%+.14lf\n",  bin,j,eje.real(),eje.imag());
          printf("bin=%d M_trig[%d] = %+.14lf%+.14lf\n\n",bin,j,iji.real(),iji.imag());
        }
        #endif
      }
    }
  }

  // Fill the delta factors to multiply by the integrals
  double delta_t_fact[4];
  for (int i=0; i<4; ++i)
  {
    delta_t_fact[i] = pow(delta_t*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, sinh, cos, sin terms
  double int_expm = 0;
  double int_expp = 0;
  pycuda::complex<double> int_trig = pycuda::complex<double>(0.,0.);

  for (int ibin=0; ibin < SPL_BINS; ++ibin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        int_expm += pycuda::real(S[ibin][j][k]*
                                 (M_expm[ibin+1][j] - M_expm[ibin][j])*
                                 K_expm[k])*
                                 delta_t_fact[j+k];

        int_expp += pycuda::real(S[ibin][j][k]*
                                 (M_expp[ibin+1][j] - M_expp[ibin][j])*
                                 K_expp[k])*
                                 delta_t_fact[j+k];

        int_trig +=  S[ibin][j][k]*
                     (M_trig[ibin+1][j] - M_trig[ibin][j])*
                     K_trig[k]*
                     delta_t_fact[j+k];

       #ifdef DEBUG
       if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
       {
         printf("bin=%d int_expm[%d,%d] = %+.14lf%+.14lf\n",  ibin,j,k,int_expm);
       }
       #endif
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = sqrt(0.5)*0.5*(int_expm + int_expp);
  time_terms[1] = sqrt(0.5)*0.5*(int_expm - int_expp);
  time_terms[2] = sqrt(0.5)*pycuda::real(int_trig);
  time_terms[3] = sqrt(0.5)*pycuda::imag(int_trig);

  #ifdef DEBUG
  if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
  {
    printf("\nNORMALIZATION      : ta=%.16lf\ttb=%.16lf\ttc=%.16lf\ttd=%.16lf\n",
           time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
    printf("                   : int_expm=%.16lf\tint_expp=%.16lf\ttc=%.16lf\ttd=%.16lf\n",
           int_expm,int_expp,time_terms[2],time_terms[3]);
    printf("                   : sigma=%.16lf\tgamma+=%.16lf\tgamma-=%.16lf\n",
           delta_t, G+0.5*DG, G-0.5*DG);

  }
  #endif
}



__device__
void integralFullSpline( double result[2],
                         double vn[10], double va[10],double vb[10], double vc[10],double vd[10],
                         double *norm, double G, double DG, double DM,
                         double delta_t,
                         double t_ll,
                         double t_offset,
                         double *spline_coeffs)
{
  double integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM, spline_coeffs, t_offset);

  double ta = integrals[0];
  double tb = integrals[1];
  double tc = integrals[2];
  double td = integrals[3];
  //ta = 1.190604219926328; tb = 0.09503489583578451; tc = 0.0236229120942433; td = 0.01432583715181457;

  for(int k=0; k<10; k++)
  {
    result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
    result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
  }
  #ifdef DEBUG
  if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
  {
    printf("                   : t_offset=%+.16lf  delta_t=%+.16lf\n", t_offset, delta_t);
  }
  #endif
}








////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 ////////////////////////////////////////////////////////////

__device__
double getOneSplineTimeAcc(double t, double *coeffs,
                           double sigma, double gamma,
                           double tLL, double tUL)
{

  // Compute pdf value
  double erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  double fpdf = 1.0; double ipdf = 0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2.0*t) ) * (erf_value);
  fpdf *= calcTimeAcceptance(t, coeffs , tLL, tUL);

  // if ( threadIdx.x + blockDim.x * blockIdx.x == 0 ) {
  // printf("COEFFS             : %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  // }
  // if ( threadIdx.x + blockDim.x * blockIdx.x < 3 )
  // {
  //   printf("TIME ACC           : t=%lf, sigma=%lf, gamma=%lf, tLL=%lf, tUL=%lf,     fpdf=%lf\n",
  //          t, sigma, gamma, tLL, tUL, fpdf);
  // }

  // Compute per event normatization
  double ti  = 0.0;  double tf  =  0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    double c0 = getCoeff(coeffs,k,0);
    double c1 = getCoeff(coeffs,k,1);
    double c2 = getCoeff(coeffs,k,2);
    double c3 = getCoeff(coeffs,k,3);

    ipdf += (exp((pow(gamma,2)*pow(sigma,2))/2.)*((c1*(-exp(-(gamma*tf))
    + exp(-(gamma*ti)) -
    (gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) - (gamma*tf)/exp(gamma*tf) +
    (gamma*ti)/exp(gamma*ti) +
    erf(tf/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) + ((1 +
    gamma*tf)*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) -
    erf(ti/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erf((gamma*sigma)/sqrt(2.0) - ti/(sqrt(2.0)*sigma))/exp(gamma*ti) -
    (gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,2) -
    (c2*(2/exp(gamma*tf) - 2/exp(gamma*ti) +
    (2*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) -
    (2*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) + (2*gamma*tf)/exp(gamma*tf) +
    (pow(gamma,2)*sqrt(2/M_PI)*sigma*tf)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (pow(gamma,2)*pow(tf,2))/exp(gamma*tf) - (2*gamma*ti)/exp(gamma*ti) -
    (pow(gamma,2)*sqrt(2/M_PI)*sigma*ti)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) -
    (pow(gamma,2)*pow(ti,2))/exp(gamma*ti) - ((2 +
    pow(gamma,2)*pow(sigma,2))*erf(tf/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) - ((2 + 2*gamma*tf +
    pow(gamma,2)*pow(tf,2))*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) +
    (2*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,2)*pow(sigma,2))/2.) +
    (pow(gamma,2)*pow(sigma,2)*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) + (2*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (2*gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (pow(gamma,2)*pow(ti,2)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,3) -
    (c3*(6/exp(gamma*tf) - 6/exp(gamma*ti) +
    (6*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) -
    (6*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) +
    (2*pow(gamma,3)*sqrt(2/M_PI)*pow(sigma,3))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(tf,2))/(2.*pow(sigma,2))) -
    (2*pow(gamma,3)*sqrt(2/M_PI)*pow(sigma,3))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(ti,2))/(2.*pow(sigma,2))) + (6*gamma*tf)/exp(gamma*tf) +
    (3*pow(gamma,2)*sqrt(2/M_PI)*sigma*tf)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (3*pow(gamma,2)*pow(tf,2))/exp(gamma*tf) +
    (pow(gamma,3)*sqrt(2/M_PI)*sigma*pow(tf,2))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(tf,2))/(2.*pow(sigma,2))) +
    (pow(gamma,3)*pow(tf,3))/exp(gamma*tf) - (6*gamma*ti)/exp(gamma*ti) -
    (3*pow(gamma,2)*sqrt(2/M_PI)*sigma*ti)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) -
    (3*pow(gamma,2)*pow(ti,2))/exp(gamma*ti) -
    (pow(gamma,3)*sqrt(2/M_PI)*sigma*pow(ti,2))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(ti,2))/(2.*pow(sigma,2))) -
    (pow(gamma,3)*pow(ti,3))/exp(gamma*ti) - (3*(2 +
    pow(gamma,2)*pow(sigma,2))*erf(tf/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) - ((6 + 6*gamma*tf + 3*pow(gamma,2)*pow(tf,2) +
    pow(gamma,3)*pow(tf,3))*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) +
    (6*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,2)*pow(sigma,2))/2.) +
    (3*pow(gamma,2)*pow(sigma,2)*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(
    gamma,2)*pow(sigma,2))/2.) + (6*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (6*gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (3*pow(gamma,2)*pow(ti,2)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (pow(gamma,3)*pow(ti,3)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,4) +
    (c0*(erf(tf/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erf(ti/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erfc((gamma*pow(sigma,2) - tf)/(sqrt(2.0)*sigma))/exp(gamma*tf) +
    erfc((gamma*pow(sigma,2) -
    ti)/(sqrt(2.0)*sigma))/exp(gamma*ti)))/gamma))/2.;
  }

  // if ( threadIdx.x + blockDim.x * blockIdx.x == 0)
  // {
  //   printf("TIME ACC           : integral=%.16lf\n",
  //          ipdf);
  // }
  return fpdf/ipdf;
}



////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 x sp2 //////////////////////////////////////////////////////

__device__
double getTwoSplineTimeAcc(double t, double *coeffs2, double *coeffs1,
                           double sigma, double gamma, double tLL, double tUL)
{
  // Compute pdf
  double erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  double fpdf = 1.0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2*t) ) * (erf_value);
  fpdf *= calcTimeAcceptance(t, coeffs1, tLL, tUL);
  fpdf *= calcTimeAcceptance(t, coeffs2, tLL, tUL);

  // Compute per event normatization
  double ipdf = 0.0; double ti  = 0.0;  double tf  =  0.0;
  double term1i = 0.0; double term2i = 0.0;
  double term1f = 0.0; double term2f = 0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    double r0 = getCoeff(coeffs1,k,0);
    double r1 = getCoeff(coeffs1,k,1);
    double r2 = getCoeff(coeffs1,k,2);
    double r3 = getCoeff(coeffs1,k,3);
    double b0 = getCoeff(coeffs2,k,0);
    double b1 = getCoeff(coeffs2,k,1);
    double b2 = getCoeff(coeffs2,k,2);
    double b3 = getCoeff(coeffs2,k,3);

    term1i = -((exp(gamma*ti - (ti*(2*gamma*pow(sigma,2) +
    ti))/(2.*pow(sigma,2)))*sigma*(b3*(720*r3 + 120*gamma*(r2 + 3*r3*ti)
    + 12*pow(gamma,2)*(2*r1 + 5*r2*ti + 10*r3*(2*pow(sigma,2) +
    pow(ti,2))) + 2*pow(gamma,3)*(3*r0 + 6*r1*ti + 10*r2*(2*pow(sigma,2)
    + pow(ti,2)) + 15*r3*ti*(3*pow(sigma,2) + pow(ti,2))) +
    pow(gamma,5)*(3*pow(sigma,2)*(r1 + 5*r3*pow(sigma,2))*ti + (r1 +
    5*r3*pow(sigma,2))*pow(ti,3) + r3*pow(ti,5) + r0*(2*pow(sigma,2) +
    pow(ti,2)) + r2*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(ti,2) +
    pow(ti,4))) + pow(gamma,4)*(3*(r0 + 5*r2*pow(sigma,2))*ti +
    5*r2*pow(ti,3) + 4*r1*(2*pow(sigma,2) + pow(ti,2)) +
    6*r3*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(ti,2) + pow(ti,4)))) +
    gamma*(120*b2*r3 + b1*gamma*(24*r3 + 6*gamma*(r2 + 2*r3*ti) +
    pow(gamma,2)*(2*r1 + 8*r3*pow(sigma,2) + 3*r2*ti + 4*r3*pow(ti,2)) +
    pow(gamma,3)*(r0 + 2*r2*pow(sigma,2) + r1*ti + 3*r3*pow(sigma,2)*ti +
    r2*pow(ti,2) + r3*pow(ti,3))) + b2*gamma*(24*r2 + 60*r3*ti +
    pow(gamma,2)*(2*r0 + 8*r2*pow(sigma,2) + 3*(r1 +
    5*r3*pow(sigma,2))*ti + 4*r2*pow(ti,2) + 5*r3*pow(ti,3)) +
    pow(gamma,3)*(2*pow(sigma,2)*(r1 + 4*r3*pow(sigma,2)) + (r0 +
    3*r2*pow(sigma,2))*ti + (r1 + 4*r3*pow(sigma,2))*pow(ti,2) +
    r2*pow(ti,3) + r3*pow(ti,4)) + 2*gamma*(3*r1 + 6*r2*ti +
    10*r3*(2*pow(sigma,2) + pow(ti,2)))) + b0*pow(gamma,2)*(6*r3 +
    gamma*(2*r2 + 3*r3*ti + gamma*(r1 + 2*r3*pow(sigma,2) + r2*ti +
    r3*pow(ti,2)))))))/(pow(gamma,6)*sqrt(2*M_PI)));
    term1f = -((exp(gamma*tf - (tf*(2*gamma*pow(sigma,2) +
    tf))/(2.*pow(sigma,2)))*sigma*(b3*(720*r3 + 120*gamma*(r2 + 3*r3*tf)
    + 12*pow(gamma,2)*(2*r1 + 5*r2*tf + 10*r3*(2*pow(sigma,2) +
    pow(tf,2))) + 2*pow(gamma,3)*(3*r0 + 6*r1*tf + 10*r2*(2*pow(sigma,2)
    + pow(tf,2)) + 15*r3*tf*(3*pow(sigma,2) + pow(tf,2))) +
    pow(gamma,5)*(3*pow(sigma,2)*(r1 + 5*r3*pow(sigma,2))*tf + (r1 +
    5*r3*pow(sigma,2))*pow(tf,3) + r3*pow(tf,5) + r0*(2*pow(sigma,2) +
    pow(tf,2)) + r2*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(tf,2) +
    pow(tf,4))) + pow(gamma,4)*(3*(r0 + 5*r2*pow(sigma,2))*tf +
    5*r2*pow(tf,3) + 4*r1*(2*pow(sigma,2) + pow(tf,2)) +
    6*r3*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(tf,2) + pow(tf,4)))) +
    gamma*(120*b2*r3 + b1*gamma*(24*r3 + 6*gamma*(r2 + 2*r3*tf) +
    pow(gamma,2)*(2*r1 + 8*r3*pow(sigma,2) + 3*r2*tf + 4*r3*pow(tf,2)) +
    pow(gamma,3)*(r0 + 2*r2*pow(sigma,2) + r1*tf + 3*r3*pow(sigma,2)*tf +
    r2*pow(tf,2) + r3*pow(tf,3))) + b2*gamma*(24*r2 + 60*r3*tf +
    pow(gamma,2)*(2*r0 + 8*r2*pow(sigma,2) + 3*(r1 +
    5*r3*pow(sigma,2))*tf + 4*r2*pow(tf,2) + 5*r3*pow(tf,3)) +
    pow(gamma,3)*(2*pow(sigma,2)*(r1 + 4*r3*pow(sigma,2)) + (r0 +
    3*r2*pow(sigma,2))*tf + (r1 + 4*r3*pow(sigma,2))*pow(tf,2) +
    r2*pow(tf,3) + r3*pow(tf,4)) + 2*gamma*(3*r1 + 6*r2*tf +
    10*r3*(2*pow(sigma,2) + pow(tf,2)))) + b0*pow(gamma,2)*(6*r3 +
    gamma*(2*r2 + 3*r3*tf + gamma*(r1 + 2*r3*pow(sigma,2) + r2*tf +
    r3*pow(tf,2)))))))/(pow(gamma,6)*sqrt(2*M_PI)));
    term2i = (exp(gamma*ti)*(3*b3*(240*r3 + gamma*(2*pow(gamma,2)*r0 +
    8*gamma*r1 + 40*r2 + gamma*(pow(gamma,3)*r0 + 4*pow(gamma,2)*r1 +
    20*gamma*r2 + 120*r3)*pow(sigma,2) + pow(gamma,3)*(pow(gamma,2)*r1 +
    5*gamma*r2 + 30*r3)*pow(sigma,4) + 5*pow(gamma,5)*r3*pow(sigma,6))) +
    gamma*(120*b2*r3 + b2*gamma*(2*pow(gamma,2)*r0 + 6*gamma*r1 + 24*r2 +
    gamma*(pow(gamma,3)*r0 + 3*pow(gamma,2)*r1 + 12*gamma*r2 +
    60*r3)*pow(sigma,2) + 3*pow(gamma,3)*(gamma*r2 + 5*r3)*pow(sigma,4))
    + b0*pow(gamma,2)*(6*r3 + gamma*(2*r2 + gamma*(gamma*r0 + r1 +
    (gamma*r2 + 3*r3)*pow(sigma,2)))) + b1*gamma*(24*r3 + gamma*(6*r2 +
    gamma*(gamma*r0 + 2*r1 + (pow(gamma,2)*r1 + 3*gamma*r2 +
    12*r3)*pow(sigma,2) +
    3*pow(gamma,2)*r3*pow(sigma,4))))))*erf(ti/(sqrt(2.0)*sigma)) -
    exp((pow(gamma,2)*pow(sigma,2))/2.)*(b3*(720*r3 + 120*gamma*(r2 +
    6*r3*ti) + pow(gamma,5)*pow(ti,2)*(3*r0 + 4*r1*ti + 5*r2*pow(ti,2) +
    6*r3*pow(ti,3)) + 24*pow(gamma,2)*(r1 + 5*ti*(r2 + 3*r3*ti)) +
    pow(gamma,6)*pow(ti,3)*(r0 + ti*(r1 + ti*(r2 + r3*ti))) +
    6*pow(gamma,3)*(r0 + 2*ti*(2*r1 + 5*ti*(r2 + 2*r3*ti))) +
    2*pow(gamma,4)*ti*(3*r0 + ti*(6*r1 + 5*ti*(2*r2 + 3*r3*ti)))) +
    gamma*(b2*(120*r3 + 24*gamma*(r2 + 5*r3*ti) + 6*pow(gamma,2)*(r1 +
    4*r2*ti + 10*r3*pow(ti,2)) + pow(gamma,4)*ti*(2*r0 + 3*r1*ti +
    4*r2*pow(ti,2) + 5*r3*pow(ti,3)) + 2*pow(gamma,3)*(r0 + 3*r1*ti +
    6*r2*pow(ti,2) + 10*r3*pow(ti,3)) + pow(gamma,5)*pow(ti,2)*(r0 +
    ti*(r1 + ti*(r2 + r3*ti)))) + gamma*(b1*(24*r3 + 6*gamma*(r2 +
    4*r3*ti) + pow(gamma,3)*(r0 + 2*r1*ti + 3*r2*pow(ti,2) +
    4*r3*pow(ti,3)) + 2*pow(gamma,2)*(r1 + 3*ti*(r2 + 2*r3*ti)) +
    pow(gamma,4)*ti*(r0 + ti*(r1 + ti*(r2 + r3*ti)))) + b0*gamma*(6*r3 +
    gamma*(2*r2 + 6*r3*ti + gamma*(r1 + ti*(2*r2 + 3*r3*ti)) +
    pow(gamma,2)*(r0 + ti*(r1 + ti*(r2 + r3*ti))))))))*erfc((gamma*sigma
    - ti/sigma)/sqrt(2.0)))/(2.*exp(gamma*ti)*pow(gamma,7));
    term2f = (exp(gamma*tf)*(3*b3*(240*r3 + gamma*(2*pow(gamma,2)*r0 +
    8*gamma*r1 + 40*r2 + gamma*(pow(gamma,3)*r0 + 4*pow(gamma,2)*r1 +
    20*gamma*r2 + 120*r3)*pow(sigma,2) + pow(gamma,3)*(pow(gamma,2)*r1 +
    5*gamma*r2 + 30*r3)*pow(sigma,4) + 5*pow(gamma,5)*r3*pow(sigma,6))) +
    gamma*(120*b2*r3 + b2*gamma*(2*pow(gamma,2)*r0 + 6*gamma*r1 + 24*r2 +
    gamma*(pow(gamma,3)*r0 + 3*pow(gamma,2)*r1 + 12*gamma*r2 +
    60*r3)*pow(sigma,2) + 3*pow(gamma,3)*(gamma*r2 + 5*r3)*pow(sigma,4))
    + b0*pow(gamma,2)*(6*r3 + gamma*(2*r2 + gamma*(gamma*r0 + r1 +
    (gamma*r2 + 3*r3)*pow(sigma,2)))) + b1*gamma*(24*r3 + gamma*(6*r2 +
    gamma*(gamma*r0 + 2*r1 + (pow(gamma,2)*r1 + 3*gamma*r2 +
    12*r3)*pow(sigma,2) +
    3*pow(gamma,2)*r3*pow(sigma,4))))))*erf(tf/(sqrt(2.0)*sigma)) -
    exp((pow(gamma,2)*pow(sigma,2))/2.)*(b3*(720*r3 + 120*gamma*(r2 +
    6*r3*tf) + pow(gamma,5)*pow(tf,2)*(3*r0 + 4*r1*tf + 5*r2*pow(tf,2) +
    6*r3*pow(tf,3)) + 24*pow(gamma,2)*(r1 + 5*tf*(r2 + 3*r3*tf)) +
    pow(gamma,6)*pow(tf,3)*(r0 + tf*(r1 + tf*(r2 + r3*tf))) +
    6*pow(gamma,3)*(r0 + 2*tf*(2*r1 + 5*tf*(r2 + 2*r3*tf))) +
    2*pow(gamma,4)*tf*(3*r0 + tf*(6*r1 + 5*tf*(2*r2 + 3*r3*tf)))) +
    gamma*(b2*(120*r3 + 24*gamma*(r2 + 5*r3*tf) + 6*pow(gamma,2)*(r1 +
    4*r2*tf + 10*r3*pow(tf,2)) + pow(gamma,4)*tf*(2*r0 + 3*r1*tf +
    4*r2*pow(tf,2) + 5*r3*pow(tf,3)) + 2*pow(gamma,3)*(r0 + 3*r1*tf +
    6*r2*pow(tf,2) + 10*r3*pow(tf,3)) + pow(gamma,5)*pow(tf,2)*(r0 +
    tf*(r1 + tf*(r2 + r3*tf)))) + gamma*(b1*(24*r3 + 6*gamma*(r2 +
    4*r3*tf) + pow(gamma,3)*(r0 + 2*r1*tf + 3*r2*pow(tf,2) +
    4*r3*pow(tf,3)) + 2*pow(gamma,2)*(r1 + 3*tf*(r2 + 2*r3*tf)) +
    pow(gamma,4)*tf*(r0 + tf*(r1 + tf*(r2 + r3*tf)))) + b0*gamma*(6*r3 +
    gamma*(2*r2 + 6*r3*tf + gamma*(r1 + tf*(2*r2 + 3*r3*tf)) +
    pow(gamma,2)*(r0 + tf*(r1 + tf*(r2 + r3*tf))))))))*erfc((gamma*sigma
    - tf/sigma)/sqrt(2.0)))/(2.*exp(gamma*tf)*pow(gamma,7));

    ipdf += (term1f + term2f) - (term1i + term2i);

  }
  //printf("%lf\n",ipdf);
  return fpdf/ipdf;

}


__global__
void pySpline(double *time, double *f, double *coeffs, int Nevt)
{
  // Elementwise iterator
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= Nevt) { return; }
  double t = time[row];

  // Get spline-time-bin
  int bin   = getTimeBin(t);

  // Get spline coeffs
  double c0 = getCoeff(coeffs,bin,0);
  double c1 = getCoeff(coeffs,bin,1);
  double c2 = getCoeff(coeffs,bin,2);
  double c3 = getCoeff(coeffs,bin,3);

  // Compute spline
  double fpdf = (c0 + t*(c1 + t*(c2 + t*c3)));
  f[row] = (fpdf);

}

////////////////////////////////////////////////////////////////////////////////
