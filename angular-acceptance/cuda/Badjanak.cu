////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      OPENCL decay rate Bs -> mumuKK                        //
//                                                                            //
//   Created: 2019-11-18                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of p-scq packages, Santiago's framework for the       //
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
#define NTIMEBINS {NTIMEBINS}
__device__ double const KNOTS[NKNOTS] = {KNOTS};

// PDF parameters
#define NMASSBINS {NMASSBINS}
__device__ double const X_M[8] = {X_M};
__device__ double const TRISTAN[10] = {TRISTAN};

// Include disciplines
//     They follow the next tree, which means that its only necessay to include
//     AngularAcceptance.cu in order to load all of them.
//         AngularAcceptance
//           |– DifferentialCrossRate
//               |- DecayTimeAcceptance
//                   |– Functions
//               |– TimeAngularDistribution
#include "AngularAcceptance.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyDiffRate //////////////////////////////////////////////////////////

__global__
void pyDiffRate(double *data, double *lkhd,
                double G, double DG, double DM,
                const double * CSP,
                const double * ASlon,
                const double * APlon,
                const double * APpar,
                const double * APper,
                double pSlon,
                double pPlon, double pPpar, double pPper,
                const double * deltaSlon,
                double deltaPlon, double deltaPpar, double deltaPper,
                double lPlon,
                double lSlon, double lPpar, double lPper,
                double tLL, double tUL,
                double *coeffs,
                int Nevt)
{{
  int evt = get_global_id(0);
  int bin = get_global_id(1);
  if (evt >= Nevt) {{ return; }}

  double shit[28];                                // check why this is mandatory
  for (int index =0; index < 28; index++)
  {{
    shit[index] = coeffs[index];
  }}

  double mass = data[evt*7+4];
  //printf("mass=%+lf",mass);
  double data4[6] = {{data[evt*7+0], // cosK
                      data[evt*7+1], // cosL
                      data[evt*7+2], // hphi
                      data[evt*7+3], // time
                      data[evt*7+5], // sigma_t
                      data[evt*7+6]  // flavour
                    }};

  if (get_local_size(1) > 1)                   // if fitting binned X_M spectrum
  {{
    if ((mass >= X_M[bin]) && (mass < X_M[bin+1]))
    {{
      lkhd[evt] = getDiffRate(data4,
                              G, DG, DM, CSP[bin],
                              APlon[bin], ASlon[bin], APpar[bin], APper[bin],
                              pPlon, pSlon, pPpar, pPper,
                              deltaSlon[bin], deltaPlon, deltaPpar, deltaPper,
                              lPlon, lSlon, lPpar, lPper,
                              tLL, tUL,
                              shit, 1);
    }}
  }}
  else
  {{
    lkhd[evt] = getDiffRate(data4,
                            G, DG, DM, CSP[0],
                            APlon[0], ASlon[0], APpar[0], APper[0],
                            pPlon, pSlon, pPpar, pPper,
                            deltaSlon[0], deltaPlon, deltaPpar, deltaPper,
                            lPlon, lSlon, lPpar, lPper,
                            tLL, tUL,
                            shit, 1);
  }}


}}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyFcoeffs ///////////////////////////////////////////////////////////

__global__
void pyFcoeffs(double *data, double *fk,  int Nevt)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int k = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= Nevt) { return; }
  fk[i*10+k]= 9./(16.*M_PI)*getFcoeffs(data[i*4+0],data[i*4+1],data[i*4+2],k+1);
}

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::getAngularWeights ///////////////////////////////////////////////////

__global__
void pyAngularWeights(double *dtrue, double *dreco,
                      double *w,
                      double G, double DG, double DM, double CSP,
                      double APlon, double ASlon, double APpar, double APper,
                      double pPlon, double pSlon, double pPpar, double pPper,
                      double dSlon, double dPlon, double dPpar, double dPper,
                      double lPlon, double lSlon, double lPpar, double lPper,
                      double tLL, double tUL,
                      double *coeffs,
                      int Nevt)
{{
  int i = get_global_id(0);
  if (i >= Nevt) {{ return; }}

  double shit[28];                                // check why this is mandatory
  for (int index =0; index < 28; index++)
  {{
    shit[index] = coeffs[index];
  }}

  double w10[10]     = {{0,0,0,0,0,0,0,0,0,0}};
  double vec_true[4] = {{dtrue[i*4+0],dtrue[i*4+1],dtrue[i*4+2],dtrue[i*4+3]}};
  double vec_reco[4] = {{dreco[i*4+0],dreco[i*4+1],dreco[i*4+2],dreco[i*4+3]}};
  getAngularWeights(vec_true, vec_reco, w10,
                    G, DG, DM, CSP,
                    ASlon, APlon, APpar, APper,
                    pSlon, pPlon, pPpar, pPper,
                    dSlon, dPlon, dPpar, dPper,
                    lPlon, lSlon, lPpar, lPper,
                    tLL, tUL,
                    shit);

  for(int k = 0; k < 10; k++)
  {{
    atomicAdd( &w[0]+k , w10[k] );
  }}
  __syncthreads();

}}

////////////////////////////////////////////////////////////////////////////////
