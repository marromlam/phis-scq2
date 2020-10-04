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
//  This file contains the following KERNELs:                                 //
//    * pyDiffRate: Computes Bs2MuMuKK pdf looping over the events. Now it    //
//                  handles a binned X_M fit without splitting beforehand the //
//                  data --it launches a thread per mass bin.                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#ifndef CUDA
  #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
  WITHIN_KERNEL
  void atomicAdd(volatile __global double *addr, double val)
  {{
    union {{
      long u;
      double f;
    }} next, expected, current;
    current.f = *addr;
    do {{
      expected.f = current.f;
      next.f = expected.f + val;
      current.u = atomic_cmpxchg( (volatile __global long *) addr, expected.u, next.u);
    }} while( current.u != expected.u );
  }}
#endif



// Debugging 0 [0,1,2,3,>3]
#define DEBUG {DEBUG}
#define DEBUG_EVT {DEBUG_EVT} // the events that is being debugged



// Flags
#define FAST_INTEGRAL {FAST_INTEGRAL}
// #define USE_TIME_OFFSET {{USE_TIME_OFFSET}}
// #define USE_TIME_RES {{USE_TIME_RES}}
// #define USE_PERFTAG {{USE_PERFTAG}}
// #define USE_TRUETAG {{USE_TRUETAG}}

// Time resolution parameters
#define SIGMA_T {SIGMA_T}
#define SIGMA_THRESHOLD 5.0

// Time acceptance parameters
#define NKNOTS {NKNOTS}
#define SPL_BINS {SPL_BINS}
#define NTIMEBINS {NTIMEBINS}
const CONSTANT_MEM ${{ftype}} KNOTS[NKNOTS] = {KNOTS};




// PDF parameters
#define NTERMS {NTERMS}
#define MKNOTS {NMASSKNOTS}
const CONSTANT_MEM ${{ftype}} X_M[7] = {X_M};
const CONSTANT_MEM ${{ftype}} TRISTAN[{NTERMS}] = {TRISTAN};

// Other definitions
#define ERRF_CONST 1.12837916709551
#define XLIM 5.33
#define YLIM 4.29

//#include "Functions.cu"
{FUNCTIONS_CU}

//#include "TimeAngularDistribution.cu"
{TIMEANGULARDISTRIBUTION_CU}

//#include "DecayTimeAcceptance.cu"
{DECAYTIMEACCEPTANCE_CU}

//#include "DifferentialCrossRate.cu"
{DIFFERENTIALCROSSRATE_CU}

//#include "AngularAcceptance.cu"
{ANGULARACCEPTANCE_CU}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// KERNEL::pyDiffRate //////////////////////////////////////////////////////////

KERNEL
void pyDiffRate(GLOBAL_MEM ${{ftype}} *data, GLOBAL_MEM ${{ftype}} *lkhd,
                // Time-dependent angular distribution
                ${{ftype}} G, ${{ftype}} DG, ${{ftype}} DM,
                GLOBAL_MEM ${{ftype}} * CSP,
                GLOBAL_MEM ${{ftype}} *ASlon, GLOBAL_MEM ${{ftype}} *APlon, GLOBAL_MEM ${{ftype}} *APpar, GLOBAL_MEM ${{ftype}} *APper,
                ${{ftype}}  pSlon, ${{ftype}}  pPlon, ${{ftype}}  pPpar, ${{ftype}}  pPper,
                GLOBAL_MEM ${{ftype}} *dSlon, ${{ftype}}  dPlon, ${{ftype}}  dPpar, ${{ftype}}  dPper,
                ${{ftype}}  lSlon, ${{ftype}}  lPlon, ${{ftype}}  lPpar, ${{ftype}}  lPper,
                // Time limits
                ${{ftype}} tLL, ${{ftype}} tUL,
                // Time resolution
                ${{ftype}} sigma_offset, ${{ftype}} sigma_slope, ${{ftype}} sigma_curvature,
                ${{ftype}} mu,
                // Flavor tagging
                ${{ftype}} eta_bar_os, ${{ftype}} eta_bar_ss,
                ${{ftype}} p0_os,  ${{ftype}} p1_os, ${{ftype}} p2_os,
                ${{ftype}} p0_ss,  ${{ftype}} p1_ss, ${{ftype}} p2_ss,
                ${{ftype}} dp0_os, ${{ftype}} dp1_os, ${{ftype}} dp2_os,
                ${{ftype}} dp0_ss, ${{ftype}} dp1_ss, ${{ftype}} dp2_ss,
                // Time acceptance
                GLOBAL_MEM ${{ftype}} *coeffs,
                // Angular acceptance
                GLOBAL_MEM ${{ftype}} *angular_weights,
                // Flags
                int USE_FK, int BINS, int USE_ANGACC, int USE_TIMEACC,
                int USE_TIMEOFFSET, int SET_TAGGING, int USE_TIMERES,
                int NEVT)
{{
  int evt = get_global_id(0);
  if (evt >= NEVT) {{ return; }}


  ${{ftype}} mass = data[evt*10+4];
  ${{ftype}} data4[9] = {{data[evt*10+0], // cosK
                          data[evt*10+1], // cosL
                          data[evt*10+2], // hphi
                          data[evt*10+3], // time
                          data[evt*10+5], // sigma_t
                          data[evt*10+6], // qOS
                          data[evt*10+7], // qSS
                          data[evt*10+8], // etaOS
                          data[evt*10+9]  // etaSS
                        }};

  unsigned int bin = BINS>1 ? getMassBin(mass) : 0;
  lkhd[evt] = getDiffRate(data4,
                          G, DG, DM, CSP[bin],
                          ASlon[bin], APlon[bin], APpar[bin], APper[bin],
                          pSlon,      pPlon,      pPpar,      pPper,
                          dSlon[bin], dPlon,      dPpar,      dPper,
                          lSlon,      lPlon,      lPpar,      lPper,
                          tLL, tUL,
                          sigma_offset, sigma_slope, sigma_curvature, mu,
                          eta_bar_os, eta_bar_ss,
                          p0_os,  p1_os, p2_os,
                          p0_ss,  p1_ss, p2_ss,
                          dp0_os, dp1_os, dp2_os,
                          dp0_ss, dp1_ss, dp2_ss,
                          coeffs,
                          angular_weights,
                          USE_FK, USE_ANGACC, USE_TIMEACC,
                          USE_TIMEOFFSET, SET_TAGGING, USE_TIMERES);

}}

//////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// KERNEL::pyFcoeffs ///////////////////////////////////////////////////////////

KERNEL
void pyFcoeffs(GLOBAL_MEM ${{ftype}} *data, GLOBAL_MEM ${{ftype}} *fk, int NEVT)
{{
  const SIZE_T i = get_global_id(0);
  //const SIZE_T k = get_global_id(1);
  if (i >= NEVT) {{ return; }}
  for(int k=0; k<10; k++)
  {{
    fk[i*10+k]= 9./(16.*M_PI)*getF(data[i*3+0],data[i*3+1],data[i*3+2],k+1);
  }}
}}

////////////////////////////////////////////////////////////////////////////////


KERNEL
void pyfaddeeva(GLOBAL_MEM ${{ctype}} *z, GLOBAL_MEM ${{ctype}} *out)
{{
   const SIZE_T idx = get_global_id(0);
   out[idx] = faddeeva(z[idx]);
}}


KERNEL
void pycerfc(GLOBAL_MEM ${{ctype}} *z, GLOBAL_MEM ${{ctype}} *out)
{{
   const SIZE_T idx = get_global_id(0);
   out[idx] = cerfc(z[idx]);
   printf("erfc(%+.4f%+.4fi) = %+.4f%+.4fi\n",z[idx].x,z[idx].y,out[idx].x,out[idx].y);
}}

KERNEL
void pyipacerfc(GLOBAL_MEM ${{ctype}} *z, GLOBAL_MEM ${{ctype}} *out)
{{
   const SIZE_T idx = get_global_id(0);
   out[idx] = ipanema_erfc(z[idx]);
}}

KERNEL
void pycexp(GLOBAL_MEM ${{ctype}} *z, GLOBAL_MEM ${{ctype}} *out)
{{
   const SIZE_T idx = get_global_id(0);
   out[idx] = cexp(z[idx]);
}}



////////////////////////////////////////////////////////////////////////////////
// KERNEL::getAngularWeights ///////////////////////////////////////////////////

KERNEL
void pyAngularWeights(GLOBAL_MEM ${{ftype}} *dtrue, GLOBAL_MEM ${{ftype}} *dreco, GLOBAL_MEM ${{ftype}} *weight, GLOBAL_MEM ${{ftype}} *w,
                      // Time-dependent angular distribution
                      ${{ftype}} G, ${{ftype}} DG, ${{ftype}} DM,
                      GLOBAL_MEM ${{ftype}} * CSP,
                      GLOBAL_MEM ${{ftype}} *ASlon, GLOBAL_MEM ${{ftype}} *APlon, GLOBAL_MEM ${{ftype}} *APpar, GLOBAL_MEM ${{ftype}} *APper,
                      ${{ftype}}  pSlon, ${{ftype}}  pPlon, ${{ftype}}  pPpar, ${{ftype}}  pPper,
                      GLOBAL_MEM ${{ftype}} *dSlon, ${{ftype}}  dPlon, ${{ftype}}  dPpar, ${{ftype}}  dPper,
                      ${{ftype}}  lSlon, ${{ftype}}  lPlon, ${{ftype}}  lPpar, ${{ftype}}  lPper,
                      ${{ftype}} tLL, ${{ftype}} tUL,
                      ${{ftype}} sigma_offset, ${{ftype}} sigma_slope, ${{ftype}} sigma_curvature,
                      ${{ftype}} mu,
                      // Flavor tagging
                      ${{ftype}} eta_bar_os, ${{ftype}} eta_bar_ss,
                      ${{ftype}} p0_os,  ${{ftype}} p1_os, ${{ftype}} p2_os,
                      ${{ftype}} p0_ss,  ${{ftype}} p1_ss, ${{ftype}} p2_ss,
                      ${{ftype}} dp0_os, ${{ftype}} dp1_os, ${{ftype}} dp2_os,
                      ${{ftype}} dp0_ss, ${{ftype}} dp1_ss, ${{ftype}} dp2_ss,
                      // Time acceptance
                      GLOBAL_MEM ${{ftype}} *coeffs,
                      // Angular acceptance
                      GLOBAL_MEM ${{ftype}} *angular_weights,
                      int NEVT)
{{
  int evt = get_global_id(0);
  if (evt >= NEVT) {{ return; }}

  ${{ftype}} w10[10]     = {{0,0,0,0,0,0,0,0,0,0}};
  ${{ftype}} vec_true[9] = {{dtrue[evt*10+0], // cosK
                         dtrue[evt*10+1], // cosL
                         dtrue[evt*10+2], // hphi
                         dtrue[evt*10+3], // time
                         dtrue[evt*10+5], // sigma_t
                         dtrue[evt*10+6], // qOS
                         dtrue[evt*10+6], // qSS
                         0,              // etaOS
                         0               // etaSS
                       }};
  ${{ftype}} vec_reco[9] = {{dreco[evt*10+0], // cosK
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
                    coeffs,
                    angular_weights);

  LOCAL_BARRIER;
  for(int k = 0; k < 10; k++)
  {{
    atomicAdd( &w[0]+k , w10[k]);
  }}

}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// KERNEL::getAngularWeights ///////////////////////////////////////////////////

KERNEL
void pyAngularCov(GLOBAL_MEM ${{ftype}} *dtrue, GLOBAL_MEM ${{ftype}} *dreco, GLOBAL_MEM ${{ftype}} *weight, GLOBAL_MEM ${{ftype}} w[10], GLOBAL_MEM ${{ftype}} cov[10][10], ${{ftype}} scale,
  // Time-dependent angular distribution
  ${{ftype}} G, ${{ftype}} DG, ${{ftype}} DM,
  GLOBAL_MEM ${{ftype}} * CSP,
  GLOBAL_MEM ${{ftype}} *ASlon, GLOBAL_MEM ${{ftype}} *APlon, GLOBAL_MEM ${{ftype}} *APpar, GLOBAL_MEM ${{ftype}} *APper,
  ${{ftype}}  pSlon, ${{ftype}}  pPlon, ${{ftype}}  pPpar, ${{ftype}}  pPper,
  GLOBAL_MEM ${{ftype}} *dSlon, ${{ftype}}  dPlon, ${{ftype}}  dPpar, ${{ftype}}  dPper,
  ${{ftype}}  lSlon, ${{ftype}}  lPlon, ${{ftype}}  lPpar, ${{ftype}}  lPper,
  ${{ftype}} tLL, ${{ftype}} tUL,
  ${{ftype}} sigma_offset, ${{ftype}} sigma_slope, ${{ftype}} sigma_curvature,
  ${{ftype}} mu,
  // Flavor tagging
  ${{ftype}} eta_bar_os, ${{ftype}} eta_bar_ss,
  ${{ftype}} p0_os,  ${{ftype}} p1_os, ${{ftype}} p2_os,
  ${{ftype}} p0_ss,  ${{ftype}} p1_ss, ${{ftype}} p2_ss,
  ${{ftype}} dp0_os, ${{ftype}} dp1_os, ${{ftype}} dp2_os,
  ${{ftype}} dp0_ss, ${{ftype}} dp1_ss, ${{ftype}} dp2_ss,
  // Time acceptance
  GLOBAL_MEM ${{ftype}} *coeffs,
  // Angular acceptance
  GLOBAL_MEM ${{ftype}} *angular_weights,
                  int NEVT)
{{
  int evt = get_global_id(0);
  if (evt >= NEVT) {{ return; }}

  ${{ftype}} w10[10] = {{0.0}};
  ${{ftype}} cov10[10][10] = {{{{0.0}}}};
  ${{ftype}} vec_true[9] = {{dtrue[evt*10+0], // cosK
                      dtrue[evt*10+1], // cosL
                      dtrue[evt*10+2], // hphi
                      dtrue[evt*10+3], // time
                      dtrue[evt*10+5], // sigma_t
                      dtrue[evt*10+6],  // qOS
                      dtrue[evt*10+6],  // qSS
                      0,  // etaOS
                      0  // etaSS
                    }};
  ${{ftype}} vec_reco[9] = {{dreco[evt*10+0], // cosK
                      dreco[evt*10+1], // cosL
                      dreco[evt*10+2], // hphi
                      dreco[evt*10+3], // time
                      dreco[evt*10+5], // sigma_t
                      dreco[evt*10+6],  // qOS
                      dreco[evt*10+6],  // qSS
                      0,  // etaOS
                      0  // etaSS
                    }};
  //${{ftype}} scale = 3554770.373949724;
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
    coeffs,
    angular_weights);

  // LOCAL_BARRIER;
  // for(int k = 0; k < 10; k++)
  // {{
  //   atomicAdd( &w[0]+k , w10[k]);
  // }}
  // LOCAL_BARRIER;

  if (DEBUG > 0 && ( get_global_id(0) < 3) )
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

  if (DEBUG > 0 && ( get_global_id(0) == 0) )
  {{
    printf("COV 0\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( get_global_id(0) == 0) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( get_global_id(0) == 0) )
  {{
    printf("COV 1\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( get_global_id(0) == 1) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( get_global_id(0) == 0) )
  {{
    printf("COV 2\n");
  }}
  for(int i=0; i<10; i++)
  {{
    if (DEBUG > 0 && ( get_global_id(0) == 2) )
    {{
      printf("%+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E  %+.5E\n",
             cov10[i][0],cov10[i][1],cov10[i][2],cov10[i][3],cov10[i][4],
             cov10[i][5],cov10[i][6],cov10[i][7],cov10[i][8],cov10[i][9]);
    }}
  }}
  if (DEBUG > 0 && ( get_global_id(0) == 0) )
  {{
    printf("\n");
  }}

  // if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
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

  LOCAL_BARRIER;
  for(int i=0; i<10; i++)
  {{
    for(int j=0; j<10; j++)
    {{
      atomicAdd( &cov[i][j], cov10[i][j] );
      //cov10[i][j] = (w10[i]-w[i])*(w10[j]-w[j]);
    }}
    if (DEBUG > 0 && ( get_global_id(0) == DEBUG_EVT) )
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

KERNEL
void pySingleTimeAcc(GLOBAL_MEM ${{ftype}} *time, GLOBAL_MEM ${{ftype}} *lkhd,
                     GLOBAL_MEM ${{ftype}} *coeffs, ${{ftype}} mu,
                     ${{ftype}} sigma, ${{ftype}} gamma,
                     ${{ftype}} tLL, ${{ftype}} tUL, int NEVT)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row = get_global_id(0);
  if (row >= NEVT) {{ return; }}
  ${{ftype}} t = time[row] - mu;
  lkhd[row] = getOneSplineTimeAcc(t, coeffs, sigma, gamma, tLL, tUL);

}}



KERNEL
void pyRatioTimeAcc(GLOBAL_MEM ${{ftype}} *time1, GLOBAL_MEM ${{ftype}} *time2,
                    GLOBAL_MEM ${{ftype}} *lkhd1, GLOBAL_MEM ${{ftype}} *lkhd2,
                    GLOBAL_MEM ${{ftype}} *c1, GLOBAL_MEM ${{ftype}} *c2,
                    ${{ftype}} mu1, ${{ftype}} sigma1, ${{ftype}} gamma1,
                    ${{ftype}} mu2, ${{ftype}} sigma2, ${{ftype}} gamma2,
                    ${{ftype}} tLL, ${{ftype}} tUL,
                    int NEVT1, int NEVT2)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row = get_global_id(0);
  if (row < NEVT1)
  {{
    ${{ftype}} t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }}
  if (row < NEVT2)
  {{
    ${{ftype}} t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }}
}}



KERNEL
void pyFullTimeAcc(GLOBAL_MEM ${{ftype}} *time1, GLOBAL_MEM ${{ftype}} *time2,
                   GLOBAL_MEM ${{ftype}} *time3, GLOBAL_MEM ${{ftype}} *lkhd1,
                   GLOBAL_MEM ${{ftype}} *lkhd2, GLOBAL_MEM ${{ftype}} *lkhd3,
                   GLOBAL_MEM ${{ftype}} *c1,
                   GLOBAL_MEM ${{ftype}} *c2,
                   GLOBAL_MEM ${{ftype}} *c3,
                   ${{ftype}} mu1, ${{ftype}} sigma1, ${{ftype}} gamma1,
                   ${{ftype}} mu2, ${{ftype}} sigma2, ${{ftype}} gamma2,
                   ${{ftype}} mu3, ${{ftype}} sigma3, ${{ftype}} gamma3,
                   ${{ftype}} tLL, ${{ftype}} tUL,
                   int NEVT1, int NEVT2, int NEVT3)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{{

  int row1 = get_global_id(0);
  //int row2 = get_global_id(1);
  //int row3 = threadIdx.z + blockDim.z * blockIdx.z;
  if (row1 < NEVT1)
  {{
    ${{ftype}} t1 = time1[row1] - mu1;
    lkhd1[row1] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }}
  if (row1 < NEVT2)
  {{
    ${{ftype}} t2 = time2[row1] - mu2;
    lkhd2[row1] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }}
  if (row1 < NEVT3)
  {{
    ${{ftype}} t3 = time3[row1] - mu3;
    lkhd3[row1] = getTwoSplineTimeAcc(t3, c2, c3, sigma3, gamma3, tLL, tUL);
  }}

}}

KERNEL
void pySpline(GLOBAL_MEM ${{ftype}} *time, GLOBAL_MEM ${{ftype}} *f,
              GLOBAL_MEM ${{ftype}} *coeffs, int NEVT)
{{
  // Elementwise iterator
  int row = get_global_id(0);
  if (row >= NEVT) {{ return; }}
  ${{ftype}} t = time[row];

  // Get spline-time-bin
  int bin   = getTimeBin(t);

  // Get spline coeffs
  ${{ftype}} c0 = getCoeff(coeffs,bin,0);
  ${{ftype}} c1 = getCoeff(coeffs,bin,1);
  ${{ftype}} c2 = getCoeff(coeffs,bin,2);
  ${{ftype}} c3 = getCoeff(coeffs,bin,3);

  // Compute spline
  ${{ftype}} fpdf = (c0 + t*(c1 + t*(c2 + t*c3)));
  f[row] = fpdf;

}}
