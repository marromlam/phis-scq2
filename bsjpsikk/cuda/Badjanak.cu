////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay rate Bs -> mumuKK                        //
//                                                                            //
//   Created: 2019-11-18                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file contains the following __global__s:                             //
//    * pyDiffRate: Computes Bs2MuMuKK pdf looping over the events. Now it    //
//                  handles a binned X_M fit without splitting beforehand the //
//                  data --it launches a thread per mass bin.                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>

// Debugging 0 [0,1,2,3,>3]
#define DEBUG {DEBUG}
#define DEBUG_EVT {DEBUG_EVT}

// Flags
#define USE_TIME_ACC {USE_TIME_ACC}
#define USE_TIME_OFFSET {USE_TIME_OFFSET}
#define USE_TIME_RES {USE_TIME_RES}
#define USE_PERFTAG {USE_PERFTAG}
#define USE_TRUETAG {USE_TRUETAG}

// Time resolution parameters
#define SIGMA_T {SIGMA_T}

// Time acceptance parameters
#define NKNOTS {NKNOTS}
#define NTERMS {NTERMS}
#define NTIMEBINS {NTIMEBINS}
__device__ double KNOTS[NKNOTS] = {KNOTS};
__device__ double ANG_ACC[NTERMS] = {ANG_ACC};


__device__ double const SIGMA_THRESHOLD = 5.0;
//__device__ int const TIME_ACC_BINS = 40;
__device__ int const SPL_BINS = 7;


// PDF parameters
#define NMASSBINS {NMASSBINS}
__device__ double const X_M[7] = {X_M};
//__device__ double const TRISTAN[10] = {TRISTAN};

// Include disciplines
//     They follow the next tree, which means that its only necessay to include
//     AngularAcceptance.cu in order to load all of them.
//         AngularAcceptance
//           |– DifferentialCrossRate
//               |- DecayTimeAcceptance
//                   |– Functions
//               |– TimeAngularDistribution
#include "AngularAcceptance.cu"
//#include "DifferentialCrossRate.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyDiffRate //////////////////////////////////////////////////////////

__global__
void pyDiffRate(double *data, double *lkhd,
                // Time-dependent angular distribution
                double G, double DG, double DM,
                double * CSP,
                double *ASlon, double *APlon, double *APpar, double *APper,
                double  pSlon, double  pPlon, double  pPpar, double  pPper,
                double *dSlon, double  dPlon, double  dPpar, double  dPper,
                double  lSlon, double  lPlon, double  lPpar, double  lPper,
                // Time limits
                double tLL, double tUL,
                // Time resolution
                double sigma_offset, double sigma_slope, double sigma_curvature,
                double mu,
                // Flavor tagging
                double eta_bar_os, double eta_bar_ss,
                double p0_os,  double p1_os, double p2_os,
                double p0_ss,  double p1_ss, double p2_ss,
                double dp0_os, double dp1_os, double dp2_os,
                double dp0_ss, double dp1_ss, double dp2_ss,
                // Time acceptance
                double *coeffs,
                // Angular acceptance
                double *angular_weights, int use_fk, int bins,
                int Nevt)
{{
  int evt = threadIdx.x + blockDim.x * blockIdx.x;
  //int bin = threadIdx.y + blockDim.y * blockIdx.y;
  if (evt >= Nevt) {{ return; }}

  double mass = data[evt*10+4];
  double data4[9] = {{data[evt*10+0], // cosK
                      data[evt*10+1], // cosL
                      data[evt*10+2], // hphi
                      data[evt*10+3], // time
                      data[evt*10+5], // sigma_t
                      data[evt*10+6], // qOS
                      data[evt*10+7], // qSS
                      data[evt*10+8], // etaOS
                      data[evt*10+9]  // etaSS
                    }};




  if (bins>1)
  {{
  for (int bin =0; bin<bins; bin++)
  {{
    if ((mass >= X_M[bin]) && (mass < X_M[bin+1]))
    {{
      lkhd[evt] = getDiffRate(data4,
                              G, DG, DM, CSP[bin],
                              ASlon[bin], APlon[bin], APpar[bin], APper[bin],
                              pSlon,      pPlon,      pPpar,      pPper,
                              dSlon[bin], dPlon,      dPpar,      dPper,
                              lSlon,      lPlon,      lPpar,      lPper,
                              tLL, tUL,
                              sigma_offset, sigma_slope, sigma_curvature,
                              mu,
                              eta_bar_os, eta_bar_ss,
                              p0_os,  p1_os, p2_os,
                              p0_ss,  p1_ss, p2_ss,
                              dp0_os, dp1_os, dp2_os,
                              dp0_ss, dp1_ss, dp2_ss,
                              coeffs,
                              angular_weights, use_fk);
    }}
  }}
  }}
  else
  {{
    lkhd[evt] = getDiffRate(data4,
                            G, DG, DM, CSP[0],
                            ASlon[0], APlon[0], APpar[0], APper[0],
                            pSlon,    pPlon,    pPpar,    pPper,
                            dSlon[0], dPlon,    dPpar,    dPper,
                            lSlon,    lPlon,    lPpar,    lPper,
                            tLL, tUL,
                            sigma_offset, sigma_slope, sigma_curvature,
                            mu,
                            eta_bar_os, eta_bar_ss,
                            p0_os,  p1_os, p2_os,
                            p0_ss,  p1_ss, p2_ss,
                            dp0_os, dp1_os, dp2_os,
                            dp0_ss, dp1_ss, dp2_ss,
                            coeffs,
                            angular_weights, use_fk);
  }}


}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyFcoeffs ///////////////////////////////////////////////////////////

__global__
void pyFcoeffs(double *data, double *fk,  int Nevt)
{{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int k = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= Nevt) {{ return; }}
  fk[i*10+k]= 9./(16.*M_PI)*getF(data[i*10+0],data[i*10+1],data[i*10+2],k+1);
}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::getAngularWeights ///////////////////////////////////////////////////

__global__
void pyAngularWeights(double *dtrue, double *dreco, double *weight, double *w,
                      // Time-dependent angular distribution
                      double G, double DG, double DM,
                      double * CSP,
                      double *ASlon, double *APlon, double *APpar, double *APper,
                      double  pSlon, double  pPlon, double  pPpar, double  pPper,
                      double *dSlon, double  dPlon, double  dPpar, double  dPper,
                      double  lSlon, double  lPlon, double  lPpar, double  lPper,
                      double tLL, double tUL,
                      double sigma_offset, double sigma_slope, double sigma_curvature,
                      double mu,
                      // Flavor tagging
                      double eta_bar_os, double eta_bar_ss,
                      double p0_os,  double p1_os, double p2_os,
                      double p0_ss,  double p1_ss, double p2_ss,
                      double dp0_os, double dp1_os, double dp2_os,
                      double dp0_ss, double dp1_ss, double dp2_ss,
                      // Time acceptance
                      int nknots, double *knots, double *coeffs,
                      // Angular acceptance
                      double *angular_weights,
                      int Nevt)
{{
  int evt = threadIdx.x + blockDim.x * blockIdx.x;
  if (evt >= Nevt) {{ return; }}

  double w10[10]     = {{0,0,0,0,0,0,0,0,0,0}};
  double vec_true[9] = {{dtrue[evt*10+0], // cosK
                         dtrue[evt*10+1], // cosL
                         dtrue[evt*10+2], // hphi
                         dtrue[evt*10+3], // time
                         dtrue[evt*10+5], // sigma_t
                         dtrue[evt*10+6], // qOS
                         dtrue[evt*10+6], // qSS
                         0,              // etaOS
                         0               // etaSS
                       }};
  double vec_reco[9] = {{dreco[evt*10+0], // cosK
                         dreco[evt*10+1], // cosL
                         dreco[evt*10+2], // hphi
                         dreco[evt*10+3], // time
                         dreco[evt*10+5], // sigma_t
                         dreco[evt*10+6], // qOS
                         dreco[evt*10+6], // qSS
                         0,              // etaOS
                         0               // etaSS
                       }};

  getAngularWeights(vec_true, vec_reco, weight[evt], w10,
                    G, DG, DM, CSP[0],
                    ASlon[0], APlon[0], APpar[0], APper[0],
                    pSlon,    pPlon,    pPpar,    pPper,
                    dSlon[0], dPlon,    dPpar,    dPper,
                    lSlon,    lPlon,    lPpar,    lPper,
                    tLL, tUL,
                    sigma_offset, sigma_slope, sigma_curvature,
                    mu,
                    eta_bar_os, eta_bar_ss,
                    p0_os,  p1_os, p2_os,
                    p0_ss,  p1_ss, p2_ss,
                    dp0_os, dp1_os, dp2_os,
                    dp0_ss, dp1_ss, dp2_ss,
                    nknots, knots, coeffs,
                    angular_weights);

  __syncthreads();
  for(int k = 0; k < 10; k++)
  {{
    atomicAdd( &w[0]+k , w10[k]);
    //w[k] += w10[k];
  }}

}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::getAngularWeights ///////////////////////////////////////////////////

__global__
void pyAngularCov(double *dtrue, double *dreco, double *weight, double w[10], double cov[10][10], double scale,
  // Time-dependent angular distribution
  double G, double DG, double DM,
  double * CSP,
  double *ASlon, double *APlon, double *APpar, double *APper,
  double  pSlon, double  pPlon, double  pPpar, double  pPper,
  double *dSlon, double  dPlon, double  dPpar, double  dPper,
  double  lSlon, double  lPlon, double  lPpar, double  lPper,
  double tLL, double tUL,
  double sigma_offset, double sigma_slope, double sigma_curvature,
  double mu,
  // Flavor tagging
  double eta_bar_os, double eta_bar_ss,
  double p0_os,  double p1_os, double p2_os,
  double p0_ss,  double p1_ss, double p2_ss,
  double dp0_os, double dp1_os, double dp2_os,
  double dp0_ss, double dp1_ss, double dp2_ss,
  // Time acceptance
  int nknots, double *knots, double *coeffs,
  // Angular acceptance
  double *angular_weights,
                  int Nevt)
{{
  int evt = threadIdx.x + blockDim.x * blockIdx.x;
  if (evt >= Nevt) {{ return; }}

  if ( (DEBUG > 0) && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {{
    printf("\n====================================================================================================");
    printf("\nDEBUGGING IS ENABLED AND RUNNING\n");
    printf("====================================================================================================\n");
  }}

  double w10[10]       = {{0.0}};
  double cov10[10][10] = {{{{0.0}}}};
  double vec_true[9] = {{dtrue[evt*10+0], // cosK
                      dtrue[evt*10+1], // cosL
                      dtrue[evt*10+2], // hphi
                      dtrue[evt*10+3], // time
                      dtrue[evt*10+5], // sigma_t
                      dtrue[evt*10+6],  // qOS
                      dtrue[evt*10+6],  // qSS
                      0,  // etaOS
                      0  // etaSS
                    }};
  double vec_reco[9] = {{dreco[evt*10+0], // cosK
                      dreco[evt*10+1], // cosL
                      dreco[evt*10+2], // hphi
                      dreco[evt*10+3], // time
                      dreco[evt*10+5], // sigma_t
                      dreco[evt*10+6],  // qOS
                      dreco[evt*10+6],  // qSS
                      0,  // etaOS
                      0  // etaSS
                    }};
  //double scale = 3554770.373949724;
  getAngularWeights(vec_true, vec_reco, 1, w10,
    G, DG, DM, CSP[0],
    ASlon[0], APlon[0], APpar[0], APper[0],
    pSlon,    pPlon,    pPpar,    pPper,
    dSlon[0], dPlon,    dPpar,    dPper,
    lSlon,    lPlon,    lPpar,    lPper,
    tLL, tUL,
    sigma_offset, sigma_slope, sigma_curvature,
    mu,
    eta_bar_os, eta_bar_ss,
    p0_os,  p1_os, p2_os,
    p0_ss,  p1_ss, p2_ss,
    dp0_os, dp1_os, dp2_os,
    dp0_ss, dp1_ss, dp2_ss,
    nknots, knots, coeffs,
    angular_weights);

  // __syncthreads();
  // for(int k = 0; k < 10; k++)
  // {{
  //   atomicAdd( &w[0]+k , w10[k]);
  // }}
  // __syncthreads();

  if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x < 3) )
  {{
    printf("\n");
    printf("w10 = %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
           w10[0],w10[1],w10[2],w10[3],w10[4],
           w10[5],w10[6],w10[7],w10[8],w10[9]);
    printf("w  = %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
           w[0]/scale,w[1]/scale,w[2]/scale,w[3]/scale,w[4]/scale,
           w[5]/scale,w[6]/scale,w[7]/scale,w[8]/scale,w[9]/scale);
    printf("\n");
  }}

  for(int i=0; i<10; i++)
  {{
    for(int j=i; j<10; j++)
    {{
      cov10[i][j] = (w10[i]-w[i]/scale)*(w10[j]-w[j]/scale)*weight[evt]*weight[evt];
    }}
  }}

  if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {{
    printf("COV 0\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {{
    printf("COV 1\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 1) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {{
    printf("COV 2\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 2) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == 0) )
  {{
    printf("\n");
  }}

  // if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
  // {{
  //   printf("\n");
  //   printf("%+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g\n",
  //          w[0],w[1],w[2],w[3],w[4],
  //          w[5],w[6],w[7],w[8],w[9]);
  //   printf("%+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g, %+1.5g\n",
  //          w10[0],w10[1],w10[2],w10[3],w10[4],
  //          w10[5],w10[6],w10[7],w10[8],w10[9]);
  //   printf("\n");
  // }}

  __syncthreads();
  for(int i=0; i<10; i++)
  {{
    for(int j=0; j<10; j++)
    {{
      atomicAdd( &cov[i][j], cov10[i][j] );
      //cov10[i][j] = (w10[i]-w[i])*(w10[j]-w[j]);
    }}
    if (DEBUG > 0 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov[i][0],cov[i][1],cov[i][2],cov[i][3],cov[i][4],
             cov[i][5],cov[i][6],cov[i][7],cov[i][8],cov[i][9]);
    }}
  }}

}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Acceptance //////////////////////////////////////////////////////////////////

__global__
void pySingleTimeAcc(double *time, double *lkhd, double *coeffs,
                     double mu, double sigma, double gamma,
                     double tLL, double tUL, int Nevt)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row >= Nevt) {{ return; }}
  double t = time[row] - mu;
  // if (row==0){{
  //   printf("mu = %lf, sigma = %lf, gamma = %lf, tLL = %lf, tUL= %lf\n", mu, sigma, gamma, tLL, tUL);
  // }}
  lkhd[row] = getOneSplineTimeAcc(t, coeffs, sigma, gamma, tLL, tUL);

}}



__global__
void pyRatioTimeAcc(double *time1, double *time2,
                    double *lkhd1, double *lkhd2,
                    double *c1, double *c2,
                    double mu1, double sigma1, double gamma1,
                    double mu2, double sigma2, double gamma2,
                    double tLL, double tUL,
                    int Nevt1, int Nevt2)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row = threadIdx.x + blockDim.x * blockIdx.x;
  if (row < Nevt1)
  {{
    double t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }}
  if (row < Nevt2)
  {{
    double t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }}
}}



__global__
void pyFullTimeAcc(double *time1, double *time2, double *time3,
                    double *lkhd1, double *lkhd2, double *lkhd3,
                    double *c1, double *c2, double *c3,
                    double mu1, double sigma1, double gamma1,
                    double mu2, double sigma2, double gamma2,
                    double mu3, double sigma3, double gamma3,
                    double tLL, double tUL,
                    int Nevt1, int Nevt2, int Nevt3)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row1 = threadIdx.x + blockDim.x * blockIdx.x;
  //int row2 = threadIdx.y + blockDim.y * blockIdx.y;
  //int row3 = threadIdx.z + blockDim.z * blockIdx.z;
  if (row1 < Nevt1)
  {{
    double t1 = time1[row1] - mu1;
    lkhd1[row1] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }}
  if (row1 < Nevt2)
  {{
    double t2 = time2[row1] - mu2;
    lkhd2[row1] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }}
  if (row1 < Nevt3)
  {{
    double t3 = time3[row1] - mu3;
    lkhd3[row1] = getTwoSplineTimeAcc(t3, c2, c3, sigma3, gamma3, tLL, tUL);
  }}

}}
