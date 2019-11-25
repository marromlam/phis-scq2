////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//  Created: 2019-01-25                                                       //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// #else
// __device__ double atomicAdd(double* a, double b) { return b; }
// #endif


////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
// #include <thrust/complex.h>
#include <pycuda-complex.hpp>
//#include <curand.h>
//#include <curand_kernel.h>
//#include "/scratch15/diego/gitcrap4/cuda/tag_gen.c"
//#include "/home3/marcos.romero/JpsiKKAna/cuda/somefunctions.c"
#include "/home3/marcos.romero/phis-scq/cuda/DecayTimeAcceptance.cu"
#include "/home3/marcos.romero/phis-scq/cuda/TimeAngularDistribution.cu"


extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
{








/*
__device__
pycuda::complex<double> calcM(double x, int n, double t, double sigma,
                              double gamma, double omega)
{
  pycuda::complex<double> conv_term;
  conv_term = getExponentialConvolution(t, gamma, omega, sigma)/(sqrt(0.5*M_PI));

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
void intgTimeAcceptance(double time_terms[4], double sigma, double gamma,
                            double dgamma, double dm, double *knots,
                            double *coeffs, int n, double t0)
{
  // Add tUL to knots list
  knots[7] = 15; n += 1;
  int const N = 7+1;
  double x[N];

  double aux1 = 1./(sqrt(2.)*sigma);

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
                         *Factorial(i+j)/Factorial(j)/Factorial(i)/pow(2.0,i+j);
        }
        else
        {
          S[bin][i][j] = 0.;
        }
      }
    }
  }


  pycuda::complex<double> z_sinh, K_sinh[4], M_sinh[spl_bins+1][4];
  pycuda::complex<double> z_cosh, K_cosh[4], M_cosh[spl_bins+1][4];
  pycuda::complex<double> z_trig, K_trig[4], M_trig[spl_bins+1][4];

  z_cosh = sigma*pycuda::complex<double>(gamma-0.5*dgamma,0.)/sqrt(2.);
  z_sinh = sigma*pycuda::complex<double>(gamma+0.5*dgamma,0.)/sqrt(2.);
  z_trig = sigma*pycuda::complex<double>(gamma,-dm)/sqrt(2.);

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_cosh[j] = Kn(z_cosh,j);
    K_sinh[j] = Kn(z_sinh,j);
    K_trig[j] = Kn(z_trig,j);
  }

  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < spl_bins+1; ++bin)
    {
      M_sinh[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma-0.5*dgamma,0.);
      M_cosh[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma+0.5*dgamma,0.);
      M_trig[bin][j] = calcM(x[bin],j,knots[bin]-t0,sigma,gamma,dm);
    }
  }

  // Fill the delta factors to multiply by the integrals
  double sigma_fact[4];
  for (int i=0; i<4; ++i)
  {
    sigma_fact[i] = pow(sigma*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, sinh, cos, sin terms
  double int_sinh = 0; double int_cosh = 0;
  pycuda::complex<double> int_trig = pycuda::complex<double>(0.,0.);

  for (int bin=0; bin < spl_bins; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        int_sinh += pycuda::real(S[bin][j][k]*(M_sinh[bin+1][j]-M_sinh[bin][j])
                    *K_cosh[k])*sigma_fact[j+k];

        int_cosh += pycuda::real(S[bin][j][k]*(M_cosh[bin+1][j]-M_cosh[bin][j])
                    *K_sinh[k])*sigma_fact[j+k];

        int_trig += S[bin][j][k]*(M_trig[bin+1][j] - M_trig[bin][j])
                    *K_trig[k]*sigma_fact[j+k];
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = 0.5*(int_sinh + int_cosh);
  time_terms[1] = 0.5*(int_sinh - int_cosh);
  time_terms[2] = pycuda::real(int_trig);
  time_terms[3] = pycuda::imag(int_trig);

}



*/
/*
//This integral works for all decay times.
__device__
void integral4pitime_full_spline( double integral[2], double vNk[10], double vak[10],double vbk[10],
                                  double vck[10],double vdk[10], double *normweights, double G, double DG, double DM,
                                  double delta_t, double t_ll, double t_offset, int spline_Nknots, double *spline_knots, double *spline_coeffs)
{
    double time_terms[4] = {0., 0., 0., 0.};
    intgTimeAcceptance(time_terms, delta_t, G, DG, DM, spline_knots, spline_coeffs, spline_Nknots, t_offset) ;

    double int_ta = time_terms[0];
    double int_tb = time_terms[1];
    double int_tc = time_terms[2];
    double int_td = time_terms[3];

    for(int k=0; k<10; k++)
    {
        integral[0] += vNk[k]*normweights[k]*(vak[k]*int_ta + vbk[k]*int_tb + vck[k]*int_tc + vdk[k]*int_td);
        integral[1] += vNk[k]*normweights[k]*(vak[k]*int_ta + vbk[k]*int_tb - vck[k]*int_tc - vdk[k]*int_td);
    }
}
*/


__device__ double IntegralTimeA(double t_0, double t_1, double G,double DG)
{
    return (2*(DG*sinh(.5*DG*t_0) + 2*G*cosh(.5*DG*t_0))*exp(G*t_1) - 2*(DG*sinh(.5*DG*t_1) + 2*G*cosh(.5*DG*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(-pow(DG, 2) + 4 *pow(G, 2));
}
__device__ double IntegralTimeB(double t_0, double t_1,double G,double DG)
{
    return (2*(DG*cosh(.5*DG*t_0) + 2*G*sinh(.5*DG*t_0))*exp(G*t_1) - 2*(DG*cosh(.5*DG*t_1) + 2*G*sinh(.5*DG*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(-pow(DG, 2) + 4*pow(G, 2));
}
__device__ double IntegralTimeC(double t_0, double t_1,double G,double DM)
{
    return ((-DM*sin(DM*t_0) + G*cos(DM*t_0))*exp(G*t_1) + (DM*sin(DM*t_1) - G*cos(DM*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(pow(DM, 2) + pow(G, 2));
}
__device__ double IntegralTimeD(double t_0, double t_1,double G,double DM)
{
    return ((DM*cos(DM*t_0) + G*sin(DM*t_0))*exp(G*t_1) - (DM*cos(DM*t_1) + G*sin(DM*t_1))*exp(G*t_0))*exp(-G*(t_0 + t_1))/(pow(DM, 2) + pow(G, 2));
}

__device__ void Integral4PiTime(double result[2],
                                double vnk[10],
                                double vak[10],
                                double vbk[10],
                                double vck[10],
                                double vdk[10],
                                double *normweights,
                                double Gamma, double DeltaGamma, double DeltaM,
                                double tLL, double tUL, double TimeOffset)
{
  double IntTimeA = IntegralTimeA(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeB = IntegralTimeB(tLL, tUL, Gamma, DeltaGamma);
  double IntTimeC = IntegralTimeC(tLL, tUL, Gamma, DeltaM);
  double IntTimeD = IntegralTimeD(tLL, tUL, Gamma, DeltaM);

  for(int k=0; k<10 ; k++)
  {
    result[0] += vnk[k]*normweights[k]*(vak[k]*IntTimeA +
                                        vbk[k]*IntTimeB +
                                        vck[k]*IntTimeC +
                                        vdk[k]*IntTimeD);
    result[1] += vnk[k]*normweights[k]*(vak[k]*IntTimeA +
                                        vbk[k]*IntTimeB -
                                        vck[k]*IntTimeC -
                                        vdk[k]*IntTimeD);
  }
  // printf("   INTEGRALS: %.4lf\t%.4lf\n",result[0],result[1] );
  // printf("       RANGE: %.4lf\t%.4lf\n",tLL,tUL );
  // printf("   INTEGRALS: %.4lf\t%.4lf\t%.4lf\t%.4lf\n",result[0],IntTimeA,IntTimeB,IntTimeC,IntTimeD );
}




__device__
double getDiffRate(double *data, double G, double DG, double DM, double CSP,
                    double APlon, double ASlon, double APpar, double APper,
                    double phisPlon, double phisSlon, double phisPpar, double phisPper,
                    double deltaSlon, double deltaPlon, double deltaPpar, double deltaPper,
                    double lPlon, double lSlon, double lPpar, double lPper, bool USE_FK)
{
  // variables
  double cosK = data[0];
  double cosL = data[1];
  double hphi = data[2];
  double time = data[3];
  //printf("%lf,%lf,%lf,%lf\n",cosK,cosL,hphi,time);

  double normweights[10] = {1,1,1,0,0,0,1,0,0,0};

  // double sigma_t 		= data[4];
  // double q_OS 			= data[5];
  // double qSlonSK 		= data[6];
  // double eta_OS 		= data[7];
  // double etaSlonSK 	= data[8];
  // int year 					= data[9];




/*
  double delta_t =  delta(sigma_t, sigma_t_a, sigma_t_b, sigma_t_c);

  double delta_t_1 = delta_1(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);
  double delta_t_2 = delta_2(sigma_t, fSlonigma_t, r_offset_pr, r_offsetSlonc, rSlonlope_pr, rSlonlopeSlonc, sigma_t_bar);

  double omega_OS = omega(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
  double omega_bar_OS = omega_bar(eta_OS, p0_OS, dp0_OS, p1_OS, dp1_OS, p2_OS, dp2_OS, eta_bar_OS);
  double omegaSlonSK = omega(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);
  double omega_barSlonSK = omega_bar(etaSlonSK, p0SlonSK, dp0SlonSK, p1SlonSK, dp1SlonSK, 0., 0., eta_barSlonSK);

  double taggingPparrs_OS[3] = {omega_OS, omega_bar_OS, q_OS};
  double taggingPparrsSlonSK[3] = {omegaSlonSK, omega_barSlonSK, qSlonSK};

  fix_taggingPparrs(taggingPparrs_OS);
  fix_taggingPparrs(taggingPparrsSlonSK);

  omega_OS = taggingPparrs_OS[0];
  omega_bar_OS = taggingPparrs_OS[1];
  omegaSlonSK = taggingPparrsSlonSK[0];
  omega_barSlonSK = taggingPparrsSlonSK[1];

  if((taggingPparrs_OS[0] == 0.5 || taggingPparrs_OS[1] == 0.5) && (taggingPparrs_OS[0] != taggingPparrs_OS[1]))
  printf("OS tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrs_OS[0], taggingPparrs_OS[1], taggingPparrs_OS[2]);
  else
  q_OS = taggingPparrs_OS[2];

  if((taggingPparrsSlonSK[0] == 0.5 || taggingPparrsSlonSK[1] == 0.5) && (taggingPparrsSlonSK[0] != taggingPparrsSlonSK[1]))
  printf("SSK tag mismatch!!! Check code %lf vs %lf and %lf \n", taggingPparrsSlonSK[0], taggingPparrsSlonSK[1], taggingPparrsSlonSK[2]);
  else
  qSlonSK = taggingPparrsSlonSK[2];

*/




  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set delta_t = 0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  pycuda::complex<double> exp_p, exp_m, exp_i;
  double t_offset = 0.0;//delta(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  double delta_t  = 0.0;

  exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., delta_t);
  exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., delta_t);
  exp_i = getExponentialConvolution(time-t_offset,          G, DM, delta_t);

  double ta = pycuda::real(0.5*(exp_m + exp_p));     // cosh = (exp_m + exp_p)/2
  double tb = pycuda::real(0.5*(exp_m - exp_p));     // sinh = (exp_m - exp_p)/2
  double tc = pycuda::real(exp_i);                        // exp_i = cos + I*sin
  double td = pycuda::imag(exp_i);                        // exp_i = cos + I*sin
  //printf("%.8lf\t %.8lf\t %.8lf\t %.8lf\n", ta,tb,tc,td);



  // Flavor tagging ------------------------------------------------------------
  double omegaOSB = 0; double omegaOSBbar = 0;
  double omegaSSB = 0; double omegaSSBbar = 0;
  double tagOS = 0; double tagSS = 0;

  bool useTrueTag = 1;
  if (useTrueTag)
  {
    tagOS = 0.5;
  }

  //   tagOS = meas->tag_decision;
  //   tagSS = meas->tag_decision_ss;
  //
  //   double meas_omega    = meas->tag_omega;
  //   double meas_omega_ss = meas->tag_omega_ss;
  //
  //   double ma(0.99),mi(0);
  //
  //   omega_os_B =    std::max(std::min(params->tag_p0()    + params->tag_deltap0()/2.0    +(params->tag_p1()    + params->tag_deltap1()/2.0)    * (meas_omega - params->tag_eta()),ma),mi);
  //   omega_os_Bbar = std::max(std::min(params->tag_p0()    - params->tag_deltap0()/2.0    +(params->tag_p1()    - params->tag_deltap1()/2.0)    * (meas_omega - params->tag_eta()),ma),mi);
  //
  //   omega_ss_B =    std::max(std::min(params->tag_ss_p0() + params->tag_ss_deltap0()/2.0 +(params->tag_ss_p1() + params->tag_ss_deltap1()/2.0) * (meas_omega_ss - params->tag_ss_eta()),ma),mi);
  //   omega_ss_Bbar = std::max(std::min(params->tag_ss_p0() - params->tag_ss_deltap0()/2.0 +(params->tag_ss_p1() - params->tag_ss_deltap1()/2.0) * (meas_omega_ss - params->tag_ss_eta()),ma),mi);
  //
  //   if((1.0 + tagOS * (1-2*omega_os_B))*(1.0 + tagSS * (1-2*omega_ss_B)) == 0 && (1.0 - tagOS * (1-2*omega_os_Bbar))*(1.0 - tagSS * (1-2*omega_ss_Bbar)) == 0){
  //     omega_ss_Bbar=0.5;
  //     omega_ss_B=0.5;
  //     omega_os_Bbar=0.5;
  //     omega_os_B=0.5;
  //   }
  // }





















  // Decay-time acceptance -----------------------------------------------------
  //     To get rid of decay-time acceptance set dta to 1.0.
  double dta = 1.0;
  /*
  to be implemented
  */


  double vNk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vak[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vbk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vck[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vdk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  double Nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
  double pdfB = 0.0; double pdfBbar = 0.0;

  for(int k = 1; k <= 10; k++)
  {
    Nk = getNcoeffs(APlon,ASlon,APpar,APper,CSP,k);
    if (USE_FK)
    {
      fk = 9./(16.*M_PI)*getFcoeffs(cosK,cosL,hphi,k);
    }
    else
    {
      fk = normweights[k-1];
    }



    ak = getAcoeffs(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getBcoeffs(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getCcoeffs(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getDcoeffs(phisPlon,phisSlon,phisPpar,phisPper,deltaPlon,deltaSlon,deltaPpar,deltaPper,lPlon,lSlon,lPpar,lPper,k);

    hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
    hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);

    pdfB    += Nk*hk_B*fk;
    pdfBbar += Nk*hk_Bbar*fk;

    vNk[k-1] = 1.*Nk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
  }

  double Int4PiTime[2] = {0.,0.};
  Integral4PiTime(Int4PiTime, vNk, vak, vbk, vck, vdk,
                  normweights,
                  G, DG, DM, 0.3, 15., 0.);
  double intB    = Int4PiTime[0];
  double intBbar = Int4PiTime[1];

  // Cooking the output --------------------------------------------------------
  double num = 1.0; double den = 1.0;
  num = dta*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
        );
  den = 1.0*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * intB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * intBbar
        );

  // DEBUG ! -------------------------------------------------------------------
  //printf("t=%+0.3lf\tcosK=%+0.3lf\tcosL=%+0.3lf\thphi=%+0.3lf\tpdf=%+0.3lf\tipdf=%+0.3lf\t --> pdf/ipdf=%+lf\n", time,cosK,cosL,hphi, num,den,num/den);
  //printf("USE_FK=%d\tt=%+lf\tpdf=%+lf\tipdf=%+lf\t --> pdf/ipdf=%+lf\n", USE_FK, time, num,den,num/den);
  // for(int k = 0; k < 10; k++)
  // {
  //   printf("--> %.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\t%.4lf\n", vNk[k], vak[k], vbk[k], vck[k], vdk[k],
  //                    normweights[k]);
  // }
  return num/den;
}







////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyDiffRate //////////////////////////////////////////////////////////

__global__
void pyDiffRate(double *data, double *lkhd,
                double G, double DG, double DM, double CSP, double APlon,
                double ASlon, double APpar, double APper, double phisPlon,
                double phisSlon, double phisPpar, double phisPper,
                double deltaSlon, double deltaPlon, double deltaPpar,
                double deltaPper, double lPlon, double lSlon, double lPpar,
                double lPper,
                int Nevt)
{
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= Nevt) { return; }
  //printf("%lf\n", data[row]);

  double data4[4] = {data[row*4+0],data[row*4+1],data[row*4+2],data[row*4+3]};


  lkhd[row] = getDiffRate(data4,
                          G, DG, DM, CSP, APlon, ASlon, APpar, APper, phisPlon,
                          phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                          deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper, 1);
}

////////////////////////////////////////////////////////////////////////////////





}
