///////////////////////////////////////////////////////////////////////////////
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

#include "/home3/marcos.romero/phis-scq/opencl/DifferentialCrossRate.cu"


extern "C"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
{


__device__
void getAngularWeights(double *dtrue, double *dreco, double *w10,
                       double G, double DG, double DM, double CSP,
                       double APlon, double ASlon, double APpar, double APper,
                       double phisPlon, double phisSlon, double phisPpar,
                       double phisPper, double deltaSlon, double deltaPlon,
                       double deltaPpar, double deltaPper, double lPlon,
                       double lSlon, double lPpar, double lPper)
{
  double fk = 0.0;
  double pdf = getDiffRate(dtrue,
                           G, DG, DM, CSP, APlon, ASlon, APpar, APper, phisPlon,
                           phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                           deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper, 1)/
               getDiffRate(dtrue,
                           G, DG, DM, CSP, APlon, ASlon, APpar, APper, phisPlon,
                           phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                           deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper, 0);

  for(int k = 0; k < 10; k++)
  {
    fk     = getFcoeffs(dtrue[0],dtrue[1],dtrue[2],k+1);
    w10[k] = 9./(16.*M_PI)*fk/pdf;
  }
}


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
void pyAngularWeights(double *dtrue, double *dreco, double *w,
                      double G, double DG, double DM, double CSP,
                      double APlon, double ASlon, double APpar, double APper,
                      double phisPlon, double phisSlon, double phisPpar,
                      double phisPper, double deltaSlon, double deltaPlon,
                      double deltaPpar, double deltaPper, double lPlon,
                      double lSlon, double lPpar, double lPper,
                      int Nevt)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x; // loop over events
  if (i >= Nevt) { return; }

  double w10[10]     = {0,0,0,0,0,0,0,0,0,0};
  double vec_true[4] = {dtrue[i*4+0],dtrue[i*4+1],dtrue[i*4+2],dtrue[i*4+3]};
  double vec_reco[4] = {dreco[i*4+0],dreco[i*4+1],dreco[i*4+2],dreco[i*4+3]};
  getAngularWeights(vec_true, vec_reco, w10,
                    G, DG, DM, CSP, APlon, ASlon, APpar, APper, phisPlon,
                    phisSlon, phisPpar, phisPper, deltaSlon, deltaPlon,
                    deltaPpar, deltaPper, lPlon, lSlon, lPpar, lPper);

  for(int k = 0; k < 10; k++)
  {
    atomicAdd( &w[0]+k , w10[k] );
  }
  __syncthreads();

  //printf("                   %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  %+.4lf  \n", w10[0], w10[1], w10[2], w10[3], w10[4], w10[5], w10[6], w10[7], w10[8], w10[9]);

}

////////////////////////////////////////////////////////////////////////////////





}
