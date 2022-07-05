#include "angular_acceptance.h"


KERNEL
void py_tijk2weights(GLOBAL_MEM ftype *w, GLOBAL_MEM const ftype *tijk,
    const int order_cosK, const int order_cosL,
    const int order_hphi)
{

  tijk2weights(w, tijk, order_cosK, order_cosL, order_hphi);

}


KERNEL
void kangular_efficiency_weights(GLOBAL_MEM ftype *eff, GLOBAL_MEM const ftype *cosK,
        GLOBAL_MEM const ftype *cosL, GLOBAL_MEM const ftype *phi,
        GLOBAL_MEM const ftype *nw)
{
  int idx = get_global_id(0);
  eff[idx] = angular_efficiency_weights(cosK[idx], cosL[idx], phi[idx], nw);
}


/**
  this function wants cijk already translated to a polynomial basis. Not legendre

*/
KERNEL
void pyangular_efficiency(GLOBAL_MEM ftype *out, GLOBAL_MEM const ftype *cijk,
    GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL, GLOBAL_MEM const ftype *hphi,
    const int bin_cosK, const int bin_cosL, const int bin_hphi,
    const int order_cosK, const int order_cosL, const int order_hphi)
{
  const unsigned int l = get_global_id(0); //if (l >= bin_cosK) {return;}
const unsigned int k = get_global_id(1); //if (k >= bin_cosL) {return;}
const unsigned int j = get_global_id(2); //if (j >= bin_hphi) {return;}

unsigned int hbin = 0;

#ifdef CUDA
hbin = int(j + bin_cosK*k + bin_cosK*bin_hphi*l);
#else
hbin = convert_int(j + bin_cosK*k + bin_cosK*bin_hphi*l);
#endif

out[hbin] = angular_efficiency(cosK[j], cosL[l], M_PI*hphi[k],
    order_cosK, order_cosL, order_hphi,
    cijk);

}


  KERNEL
void magia_borras(
    GLOBAL_MEM ftype *chi2,
    GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL, GLOBAL_MEM const ftype *hphi,
    GLOBAL_MEM const ftype *data_3d, GLOBAL_MEM const ftype *prediction_3d,
    GLOBAL_MEM const ftype *pars,
    const int bin_cosK, const int bin_cosL, const int bin_hphi,
    const int order_cosK, const int order_cosL, const int order_hphi)
{
  ftype fitted = 0.0; ftype pred = 0.0; ftype value = 0.0;
  int hbin = 0; int lbin = 0;

  const int l = get_global_id(0); //if (l >= bin_cosK) {return;}
const int k = get_global_id(1); //if (k >= bin_cosL) {return;}
const int j = get_global_id(2); //if (j >= bin_hphi) {return;}

#ifdef CUDA
hbin = int(j + bin_cosK*k + bin_cosK*bin_hphi*l);
#else
hbin = convert_int(j + bin_cosK*k + bin_cosK*bin_hphi*l);
#endif

fitted = 0.0;
for( int p=0; p<order_cosL+1; p++ )
{
  for( int o=0; o<order_hphi+1; o++ )
  {
    for( int n=0; n<order_cosK+1; n++ )
    {
#ifdef CUDA
      lbin = int(n + (order_cosK+1)*o + (order_cosK+1)*(order_hphi+1)*p);
#else
      lbin = convert_int(n + (order_cosK+1)*o + (order_cosK+1)*(order_hphi+1)*p);
#endif
      if (pars[lbin] != 0.0)
      {
        fitted += pars[lbin]*lpmv(0.,n,cosK[j])*lpmv(0.,p,cosL[l])*lpmv(0.,o,hphi[k]);
      }
      // if (l+k+j == 0){
      //   printf("%d\n", lbin);
      // }
    }
  }
}
value = data_3d[hbin];
pred = prediction_3d[hbin];
if ( (fitted > 0.0) & (pred > 0.0) )
{
  // cross-entropy minimization
  chi2[hbin] -= ( 2.0 * ( value*log(fitted*pred) - fitted*pred ) );
  // chi2 minimization
  // chi2[hbin] += ( pow(value - pred*fitted,2)/pow(pred,1) );
}

}


// vim:foldmethod=marker
