#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pycuda-complex.hpp>
#include<curand.h>
#include<curand_kernel.h>

// Diego Martinez Santos

extern "C" 
{

  __global__ void cx_mutate( pycuda::complex<double> *xg, pycuda::complex<double> *ug1, double F, double CR, int D, int NP)
  {
    int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
    int i0 = row*D;
    int r1 = row;
    int r2 = row; 
    int i, ri; 
    double rand;
    curandState state;
    if (row >= NP) return;

    curand_init((unsigned long long)clock(), row, 0, &state);
  
    while (r1 == row) r1 = round(curand_uniform(&state)*NP);
    while (r2 == row or r2 == r1) r2 = round(curand_uniform(&state)*NP);
    ri = round(curand_uniform(&state)*D);
    
    
    for(i=0;i<D;i++) 
    {
      rand = curand_uniform(&state);
      if (rand <= CR || i == ri) ug1[i0+i] = xg[i0+i] + F*(xg[r1*D +i] - xg[r2*D + i]);
      else ug1[i0 + i] = xg[i0 + i];
      
    }
  }
 
  __global__ void re_mutate( double *xg, double *ug1, double *vmin, double *vmax, double F, double CR, int D, int NP)
  {
    int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
    int i0 = row*D;
    int r1 = row;
    int r2 = row; 
    int i, ri; 
    double rand;
    curandState state;
    if (row >= NP) return;

    curand_init((unsigned long long)clock(), row, 0, &state);
  
    while (r1 == row) r1 = round(curand_uniform(&state)*NP);
    while (r2 == row or r2 == r1) r2 = round(curand_uniform(&state)*NP);
    ri = round(curand_uniform(&state)*D);
    
    
    for(i=0;i<D;i++) 
    {
      rand = curand_uniform(&state);
      if (rand <= CR || i == ri) ug1[i0+i] = xg[i0+i] + F*(xg[r1*D +i] - xg[r2*D + i]);
      else ug1[i0 + i] = xg[i0 + i];
      if (ug1[i0+i] > vmax[i])  ug1[i0 + i] = xg[i0 + i]; //ug1[i0+i] = vmax[i];
      else if (ug1[i0+i] < vmin[i])  ug1[i0 + i] = xg[i0 + i]; //ug1[i0+i] = vmax[i];
      
    }
  }
  
 
  __global__ void cx_select( pycuda::complex<double> *xg, pycuda::complex<double> *ug1, double *darwin, int D, int NP)
 {
   int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
   int i0 = row*D;
   int i;
   
   if (row >= NP) return;
   for(i=0;i<D;i++) xg[i0+i] += darwin[row]*(ug1[i0+i]-xg[i0+i]);
 }
 
  __global__ void re_select( double *xg, double *ug1, double *darwin, int D, int NP)
 {
   int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
   int i0 = row*D;
   int i;
   
   if (row >= NP) return;
   for(i=0;i<D;i++) xg[i0+i] += darwin[row]*(ug1[i0+i]-xg[i0+i]);
 }
  
  
 
}
