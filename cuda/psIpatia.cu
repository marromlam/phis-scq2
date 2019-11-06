#include <math.h>
//#include <stdio.h>
//#include <math_functions.h>
  __device__ double log_apIpatia(double x, double mu, double sigma, double l, double beta,  double a, double n,  double a2, double n2)
  {
  double d = x-mu;
  double delta, delta2, cons1, phi, B, k1, k2;
  double logA, logk1,logcons1;
  double asigma = a*sigma;
  double a2sigma = a2*sigma;
  double out = 0.;
  cons1 = -2.*l;
  if (l<=-1.0)  delta = sigma *sqrt(-2+cons1);
  else delta = sigma;
  delta2 = delta*delta;
     if (d < -asigma ) {
       logcons1 = -beta*asigma;
       phi = 1. + asigma*asigma/delta2;
       //k1 = cons1*TMath::Power(phi,l-0.5);
       logk1 = logcons1 + (l-0.5)*log(phi);
       cons1 = exp(logcons1);
       k1 = exp(logk1);
       k2 = beta*k1- cons1*(l-0.5)*pow(phi,l-1.5)*2*asigma/delta2;
       B = -asigma + n*k1/k2;
       logA = logk1 + n*log(B+asigma);
       //out = A*TMath::Power(B-d,-n);
       // we want the log
       out = logA -n*log(B-d);
     }
     else if (d > a2sigma) {
       //cons1 = TMath::Exp(beta*a2sigma);
       logcons1 = beta*a2sigma;
       phi = 1. + a2sigma*a2sigma/delta2;
       //k1 = cons1*TMath::Power(phi,l-0.5);
       logk1 = logcons1 + (l-0.5)*log(phi);
       cons1 = exp(logcons1);
       k1 = exp(logk1);
       k2 = beta*k1+ cons1*(l-0.5)*pow(phi,l-1.5)*2.*a2sigma/delta2;
       B = -a2sigma - n2*k1/k2;
       // A = k1*TMath::Power(B+a2sigma,n2);
       logA = log(k1) + n2*log(B+a2sigma);
       //out =  A*TMath::Power(B+d,-n2);
       out = logA -n2*log(B+d);
     }
     else  out = beta*d + (l-0.5)*log(1. + d*d/delta2);
       
     

   return out;

  }
  __global__ void logIpatia(double *in, double *out,double mu, double sigma, double l, double beta, double a, double n, double a2, double n2)
  {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;//+ threadIdx.y*4;
    out[idx] = log_apIpatia(in[idx],mu, sigma, l, beta, a, n, a2, n2);
  }
  __global__ void Ipatia(double *in, double *out,double mu, double sigma, double l, double beta, double a, double n, double a2, double n2)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;//+ threadIdx.y*4;
    out[idx] = exp(log_apIpatia(in[idx],mu, sigma, l, beta, a, n, a2, n2));
  }
