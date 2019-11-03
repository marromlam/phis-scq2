#include <math.h>

extern "C" 
{
  __device__ double P_omega_os(double x)
  {
    return 3.8 - 134.6*x + 1341.*x*x;
    
  }
  __device__ double P_omega_ss(double x)
  {
    if (x < 0.46) return exp(16*x -.77);
    else return 10*(16326 - 68488*x + 72116*x*x);
    
  }
  
}
