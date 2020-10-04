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



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////

#include <math.h>
#include <stdio.h>
// #include <thrust/complex.h>
#include <pycuda-complex.hpp>
#include<curand.h>
#include<curand_kernel.h>
#include "/home3/marcos.romero/JpsiKKAna/cuda/AngularDistribution.c"

extern "C"

////////////////////////////////////////////////////////////////////////////////


{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


__global__
void Generate(${ftype} *out,  ${ftype} ASlon,
              ${ftype} APper, ${ftype} APlon, ${ftype} APpar,
              ${ftype} ADper, ${ftype} ADlon, ${ftype} ADpar,
              ${ftype} CSP, ${ftype} CSD, ${ftype} CPD,
              ${ftype} phisSlon,
              ${ftype} phisPper, ${ftype} phisPlon, ${ftype} phisPpar,
              ${ftype} phisDper, ${ftype} phisDlon, ${ftype} phisDpar,
              ${ftype} dSlon,
              ${ftype} dPper, ${ftype} dPpar,
              ${ftype} dDper, ${ftype} dDlon, ${ftype} dDpar,
              ${ftype} lamSlon,
              ${ftype} lamPper, ${ftype} lamPlon, ${ftype} lamPpar,
              ${ftype} lamDper, ${ftype} lamDlon, ${ftype} lamDpar,
              ${ftype} Gamma, ${ftype} DeltaGamma, ${ftype} DeltaM,
              ${ftype} tLL, ${ftype} tUL,
              ${ftype} *normweights,
              ${ftype} q, ${ftype} Probmax, int Nevt)


{

  // Elementwise
  int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry

  if (row >= Nevt) { return;}

  int i0        = row*13;
  int idx       =  0 + i0;
  int idy       =  1 + i0;
  int idz       =  2 + i0;
  int idt       =  3 + i0;
  int idsigma_t =  4 + i0;
  int idq_OS    =  5 + i0;
  int idq_SSK   =  6 + i0;
  int ideta_OS  =  7 + i0;
  int ideta_SSK =  8 + i0;
  int i_shit1   =  9 + i0;
  int i_shit2   = 10 + i0;
  int i_shit3   = 11 + i0;
  int iyear     = 12 + i0;

  // Prepare curand
  curandState state;
  curand_init((unsigned long long)clock(), row, 0, &state);
  ${ftype} Niter = 0.0;

  // Decay time resolution HARDCODED!
  ${ftype} sigma_t = 0.04554;
  //${ftype} sigma_t = curand_log_normal(&state,-3.22,0.309);



  // Tagging parameters ////////////////////////////////////////////////////////

  ${ftype} q_OS    = 0.0;
  ${ftype} q_SSK   = 0.0;//data[idq_SSK];
  ${ftype} eta_OS  = 0.5;
  ${ftype} eta_SSK = 0.5;//ta[ideta_SSK];


  // HARDCODDED STUFF --- take a look
  ${ftype} sigma_t_mu_a = 0;
  ${ftype} sigma_t_mu_b = 0;
  ${ftype} sigma_t_mu_c = 0;
  ${ftype} sigma_t_a = 0;
  ${ftype} sigma_t_b = 0.8721;
  ${ftype} sigma_t_c = 0.01225;

    // Hardcoded tagging parameters
  ${ftype} p0_OS = 0.39; ${ftype} dp0_OS = 0.009;
  ${ftype} p1_OS = 0.85; ${ftype} dp1_OS = 0.014;
  ${ftype} p2_OS = 0; ${ftype} dp2_OS = 0;
  ${ftype} eta_bar_OS = 0.379;
  ${ftype} p0_SSK = 0.43; ${ftype} dp0_SSK = 0.0;
  ${ftype} p1_SSK = 0.92; ${ftype} dp1_SSK = 0;
  ${ftype} eta_bar_SSK = 0.4269;


  if ( q == 0 ){
    ${ftype} tag = curand_uniform(&state);
    if (tag < 0.16)
    {
      q_OS = 1.0;
    }
    else if (tag<0.32)
    {
      q_OS = -1.;
    }
    else
    {
      q_OS = 0.;
    }

    tag = curand_uniform(&state);
    if (tag < 0.31)
    {
      q_SSK = 1.;
    }
    else if (tag<0.62)
    {
      q_SSK = -1.;
    }
    else
    {
      q_SSK = 0.;
    }

    ${ftype} OSmax = P_omega_os(0.5);
    ${ftype} SSmax = P_omega_ss(0.5);
    ${ftype} thr;//, mt, pt;

    if (q_OS > 0.5 || q_OS < -0.5)
    {
      while(1)
      {
        tag = .499*curand_uniform(&state);
        thr = OSmax*curand_uniform(&state);
        if (P_omega_os(tag) > thr) break;
      }
      eta_OS = tag;
    }

    if (q_SSK > 0.5 || q_SSK < -0.5)
    {
      while(1)
      {
        tag = .499*curand_uniform(&state);
        thr = SSmax*curand_uniform(&state);
        if (P_omega_ss(tag) > thr) break;
      }
      eta_SSK = tag;
    }

  }


  while(1)
  {
    // Random numbers
    ${ftype} x = - 1.0  +    2.0*curand_uniform(&state);
    ${ftype} y = - 1.0  +    2.0*curand_uniform(&state);
    ${ftype} z = - M_PI + 2*M_PI*curand_uniform(&state);
    ${ftype} t = tLL - log(curand_uniform(&state))/(Gamma-0.5*DeltaGamma);

    // PDF threshold
    ${ftype} thr = Probmax*curand_uniform(&state);

    // Prepare data and pdf variables to DiffRate CUDA function
    ${ftype} data[10] = {x,y,z,t, sigma_t, q_OS, q_SSK, eta_OS, eta_SSK, 2019};
    ${ftype} pdf      = 0.0;

    // Get pdf value from angular distribution
    pdf = DiffRate( data, ASlon, APper, APlon, APpar, ADper, ADlon, ADpar,
                          CSP,  CSD,  CPD,
                          phisSlon, phisPper, phisPlon, phisPpar,
                          phisDper,  phisDlon, phisDpar,
                          dSlon, dPper, dPpar, dDper, dDlon, dDpar,
                          lamSlon, lamPper, lamPlon, lamPpar,
                          lamDper, lamDlon, lamDpar,
                          Gamma, DeltaGamma, DeltaM,
                          p0_OS, dp0_OS, p1_OS,
                          dp1_OS, p2_OS, dp2_OS,
                          eta_bar_OS,
                          p0_SSK, dp0_SSK,
                          p1_SSK, dp1_SSK,
                          eta_bar_SSK,
                          sigma_t_a, sigma_t_b, sigma_t_c,
                          sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c,
                          tLL, tUL, normweights);

    pdf *= exp((Gamma-0.5*DeltaGamma)*(t-tLL));
    pdf *= (Gamma-0.5*DeltaGamma)*(1- exp((Gamma-0.5*DeltaGamma)*(-tUL+tLL)));


    if (pdf > Probmax) {
      printf("WARNING: PDF [ = %lf] > Probmax [ = %lf]\n", pdf, Probmax);
    }

    if (t > tUL) {
      pdf = 0.0;
    }

    Niter++;
    if(Niter > 1000000)
    {
      printf("this p.d.f. is too hard...");
      return;
    }

    if (pdf>= thr)
    {
      // Store
      out[idx] = x; out[idy] = y; out[idz] = z; out[idt] = t;
      out[idsigma_t] = sigma_t;
      out[idq_OS] = q_OS; out[idq_SSK] = q_SSK;
      out[ideta_OS] = eta_OS ; out[ideta_SSK] = eta_SSK;
      out[i_shit1] = 0.0; out[i_shit2] = 0.0; out[i_shit3] = 0.0;
      out[iyear] = 2019;

      return;
    }

  }

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

}
