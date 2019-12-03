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
//  This file contains the following __kernels:                               //
//    * pyDiffRate: Computes Bs2MuMuKK pdf looping over the events. Now it    //
//                  handles a binned X_M fit without splitting beforehand the //
//                  data --it launches a thread per mass bin.                 //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

// Debugging 0 [0,1,2,3,>3]
#define DEBUG {DEBUG}

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>




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
__constant double KNOTS[NKNOTS] = {KNOTS};

// PDF parameters
#define NMASSBINS {NMASSBINS}
__constant double X_M[8] = {X_M};
__constant double TRISTAN[10] = {TRISTAN};


// Include disciplines
//     They follow the next tree, which means that its only necessay to include
//     AngularAcceptance.cl in order to load all of them.
//         AngularAcceptance
//           |– DifferentialCrossRate
//               |- DecayTimeAcceptance
//                   |– Functions
//               |– TimeAngularDistribution
//#include "AngularAcceptance.cl" but this file is not yet translated to openCL.
#include "DifferentialCrossRate.cl"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// GLOBAL::pyDiffRate //////////////////////////////////////////////////////////

__kernel
void pyDiffRate(__global double *data, __global double *lkhd,
                double G, double DG, double DM,
                __global const double * CSP,
                __global const double * ASlon,
                __global const double * APlon,
                __global const double * APpar,
                __global const double * APper,
                double phisSlon,
                double phisPlon, double phisPpar, double phisPper,
                __global const double * deltaSlon,
                double deltaPlon, double deltaPpar, double deltaPper,
                double lPlon,
                double lSlon, double lPpar, double lPper,
                double tLL, double tUL,
                __global double *coeffs,
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
                              phisPlon, phisSlon, phisPpar, phisPper,
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
                            phisPlon, phisSlon, phisPpar, phisPper,
                            deltaSlon[0], deltaPlon, deltaPpar, deltaPper,
                            lPlon, lSlon, lPpar, lPper,
                            tLL, tUL,
                            shit, 1);
  }}


}}




////////////////////////////////////////////////////////////////////////////////
