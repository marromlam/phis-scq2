#include <ipanema/special.h>


/**
 This function is INCORRECT!
 Taken from Veronika's code for the toys. It has some of the signs of the normweights
 changed. This may be caused because she was not using a proper implementation for
 the Ylm
 */
WITHIN_KERNEL
void angWeightsToMoments(ftype *tijk, GLOBAL_MEM const ftype *nw)
{
    //c0000
    tijk[0]  =   1./3.                  * (nw[0] + nw[1] + nw[2]);
    //c0020
    tijk[1]  =   1./3.*sqrt(5.)         * (nw[0] + nw[1] + nw[2] - 3. * nw[6]);
    //c0022
    tijk[2]  =        -sqrt(5./3.)      * (nw[1] - nw[2]);
    //c0021
    tijk[3]  = - 8./3.*sqrt(5./2.)/M_PI *  nw[7];
    //c002-1
    tijk[4]  = - 8./3.*sqrt(5./2.)/M_PI *  nw[8];//-normweights should be +nw?
    //c002-2
    tijk[5]  =         sqrt(5./3.)      *  nw[3];//-normweights should be +nw?
    //c1000
    tijk[6]  =   1./2.*sqrt(3.)         *  nw[9];
    //c1021
    tijk[7]  = -32./3.*sqrt(5./6.)/M_PI *  nw[4];
    //c102-1
    tijk[8]  = +32./3.*sqrt(5./6.)/M_PI *  nw[5];//-normweights should be +nw?
    //c2000
    tijk[9]  =  5./2.                   * (nw[0] - nw[6]);
}



WITHIN_KERNEL
ftype angular_wefficiency(const ftype cosK, const ftype cosL, const ftype phi,
                          GLOBAL_MEM const ftype *nw)
{
  ftype eff = 0.;
  ftype tijk[10];
 
  angWeightsToMoments(tijk, nw);

  // correct for (-1) wrt Veronika's Ylm implementation --------------*
  eff += tijk[0] * lpmv(0, 0, cosK) * sph_harm(0, 0, cosL, phi);
  eff += tijk[1] * lpmv(0, 0, cosK) * sph_harm(0, 2, cosL, phi);
  eff += tijk[2] * lpmv(0, 0, cosK) * sph_harm(2, 2, cosL, phi);
  eff += tijk[3] * lpmv(0, 0, cosK) * sph_harm(1, 2, cosL, phi)   * (-1);
  eff += tijk[4] * lpmv(0, 0, cosK) * sph_harm(-1,2, cosL, phi)   * (-1);
  eff += tijk[5] * lpmv(0, 0, cosK) * sph_harm(-2,2, cosL, phi);
  eff += tijk[6] * lpmv(0, 1, cosK) * sph_harm(0, 0, cosL, phi);
  eff += tijk[7] * lpmv(0, 1, cosK) * sph_harm(1, 2, cosL, phi)   * (-1);
  eff += tijk[8] * lpmv(0, 1, cosK) * sph_harm(-1,2, cosL, phi)   * (-1);
  eff += tijk[9] * lpmv(0, 2, cosK) * sph_harm(0, 0, cosL, phi);

  eff *= 2.*sqrt(M_PI);

  return eff;
}


/*
================================================================================ 

  THIS CODE SHOULD BE DELETED AT SOME STAGE

================================================================================ 
__device__ double P_lm(int l, int m, double cos_psi)
{
//     double factor = 1./(l+0.5);
    double factor = 1.;
    
    if(l == 0 && m == 0)
    {
        return factor*1.;
    }
    else if(l == 1 && m == 0)
    {
        return factor*cos_psi;
    }
    else if(l == 1 && m == -1)
    {
        return 0;
    }
    else if(l == 1 && m == 1)
    {
        return 0;
    }
    else if(l == 2 && m == 0)
    {
        return factor*0.5*(3.*cos_psi*cos_psi - 1.);
    }
    else if(l == 2 && m == 1 )
    {
        return -factor*3*cos_psi*sqrt(1.-cos_psi*cos_psi);
    }
    else if(l == 2 && m == -1 )
    {
        return factor*0.5*cos_psi*sqrt(1.-cos_psi*cos_psi);
    }
    else if(l == 2 && m == 2)
    {
        return factor*3.*(1.-cos_psi*cos_psi);
    }
    else if(l == 2 && m == -2)
    {
        return factor*0.125*(1.-cos_psi*cos_psi);
    }
    else
        printf("ATTENTION: Legendre polynomial index l,m is out of the range of this function. Check code. %d   %d", l, m);
    
    return 0.;
}


//Spherical harmonics up to l = 2
__device__ double Y_lm(int l, int m, double cos_theta, double phi)
{
    double P_l;
//     double factor = 1./(l+0.5);
    double factor = 1.;
    
    if(l == 0)
    {
        P_l = factor*1.;
    }
    else if (l == 1)
    {
        P_l = factor*cos_theta;
    }
    else if (l == 2)
    {
        P_l = factor*0.5*(3*cos_theta*cos_theta-1.);
    }
    else if (l > 2)
    {
        printf("ATTENTION: Ylm polynomial index l is out of the range of this function. Check code.");
        return 0.;
    }
    
    if(m == 0)
    {
//         return sqrt((2*l + 1)/(4.*M_PI))*P_lm(l, m, cos_theta);
        return sqrt((2*l + 1)/(4.*M_PI))*P_l;
    }
    else if(m > 0)
    {
//         return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*cos(m*phi);
        return sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*cos(m*phi);
    }
    else
    {
//         return pow(-1.,m)*sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-(-1.*m)))/sqrt(factorial(l-1.*m)))*P_lm(l, -1.*m, cos_theta)*sin(-1.*m*phi);
      m = abs(m);
        return sqrt(2.)*sqrt((2*l + 1)/(4.*M_PI))*(sqrt(factorial(l-m))/sqrt(factorial(l+m)))*P_lm(l, m, cos_theta)*sin(m*phi);
    }

    return 0.;
}

__device__ double ang_eff(double helcosthetaK, double helcosthetaL, double helphi, const double *nw)
{
  ftype eff = 0.;
  ftype tijk[10];
 
  angWeightsToMoments(tijk, nw);
  
  int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
//  if (row==0) 
//  for (int merda=0; merda<10; ++merda){
//      printf("tijk[%d] = %f",merda, tijk[merda]);
//  }
    if (row==0) {printf("\nYlm = %f -- %f -- %f -- %f -- %f -- %f -- %f -- %f -- %f -- %f \n", 
            Y_lm(0, 0, helcosthetaL, helphi),//1
            Y_lm(2, 0, helcosthetaL, helphi),//2
            Y_lm(2, 2, helcosthetaL, helphi),//3
            Y_lm(2, 1, helcosthetaL, helphi),//4
            Y_lm(2,-1, helcosthetaL, helphi),//5
            Y_lm(2,-2, helcosthetaL, helphi),//6
            Y_lm(0, 0, helcosthetaL, helphi),//7
            Y_lm(2, 1, helcosthetaL, helphi),//8
            Y_lm(2,-1, helcosthetaL, helphi),//9
            Y_lm(0, 0, helcosthetaL, helphi)//1
            );}
    eff += tijk[0]*P_lm(0, 0, helcosthetaK)*Y_lm(0, 0, helcosthetaL, helphi);
    eff += tijk[1]*P_lm(0, 0, helcosthetaK)*Y_lm(2, 0, helcosthetaL, helphi);
    eff += tijk[2]*P_lm(0, 0, helcosthetaK)*Y_lm(2, 2, helcosthetaL, helphi);
   // if (row==0) {printf("eff = %f", eff);}
    eff += tijk[3]*P_lm(0, 0, helcosthetaK)*Y_lm(2, 1, helcosthetaL, helphi);
    //if (row==0) {printf("eff = %f", Y_lm(2, 1, helcosthetaL, helphi));}
    eff += tijk[4]*P_lm(0, 0, helcosthetaK)*Y_lm(2, -1, helcosthetaL, helphi);
    if (row==0) {printf("eff = %f", eff);}
    eff += tijk[5]*P_lm(0, 0, helcosthetaK)*Y_lm(2, -2, helcosthetaL, helphi);
    eff += tijk[6]*P_lm(1, 0, helcosthetaK)*Y_lm(0, 0, helcosthetaL, helphi);
    eff += tijk[7]*P_lm(1, 0, helcosthetaK)*Y_lm(2, 1, helcosthetaL, helphi);
    eff += tijk[8]*P_lm(1, 0, helcosthetaK)*Y_lm(2, -1, helcosthetaL, helphi);
    eff += tijk[9]*P_lm(2, 0, helcosthetaK)*Y_lm(0, 0, helcosthetaL, helphi);
    
    eff *= 2.*sqrt(M_PI);
//     printf("Ang. eff = %lf \n", eff);
    return eff;
}
*/


KERNEL
void magia_borras(
    GLOBAL_MEM double *chi2,
    GLOBAL_MEM const double *cosK, GLOBAL_MEM const double *cosL, GLOBAL_MEM const double *hphi,
    GLOBAL_MEM const double *data_3d, GLOBAL_MEM const double *prediction_3d,
    GLOBAL_MEM const double *pars,
    const int bin_cosK, const int bin_cosL, const int bin_hphi,
    const int order_cosK, const int order_cosL, const int order_hphi)
{
  double fitted = 0.0; double pred = 0.0; double value = 0.0;
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



WITHIN_KERNEL
ftype angular_efficiency(const ftype cosK, const ftype cosL, const ftype hphi,
                         const int order_cosK, const int order_cosL, const int order_hphi,
                         GLOBAL_MEM const ftype *cijk)
{
  ftype eff = 0.;
  int lbin = 0;

  for( int p=0; p<order_cosL+1; p++ )
  {
    for( int o=0; o<order_hphi+1; o++ )
    {
      for( int n=0; n<order_cosK+1; n++ )
      {
        #ifdef CUDA
          lbin = int(n + (order_cosK+1)*o + (order_cosK+1)*(order_hphi+1)*p);
        #else
          lbin = convert_int(n + (order_cosL+1)*o + (order_cosL+1)*(order_hphi+1)*p);
        #endif
        eff += cijk[lbin] * pow(cosL,p) * pow(hphi,o) * pow(cosK,n);
      }
    }
  }

  return eff;
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



/**
 * Translates tijk, the analylical coefficients fitted to data/theory and 
 * described by legendre polynomials
 */
WITHIN_KERNEL
void tijk2weights(GLOBAL_MEM ftype *w, GLOBAL_MEM const ftype *tijk, const int order_cosK, 
                  const int order_cosL, const int order_hphi)
{

  ftype it; int lbin;

  for (unsigned int p=0; p<=order_cosK; p++)
  {
    for (unsigned int o=0; o<=order_hphi; o++)
    {
      for (unsigned int n=0; n<=order_cosL; n++)
      {
        lbin = n + (order_cosL+1) * o + p * (order_cosL+1) * (order_hphi+1);
        //  printf("coeff[%d] = %f\n", lbin, tijk[lbin]);
        //  printf("f%d = %f\n", 1, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 1));
        //  printf("f%d = %f\n", 2, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 2));
        //  printf("f%d = %f\n", 3, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 3));
        //  printf("f%d = %f\n", 4, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 4));
        //  printf("f%d = %f\n", 5, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 5));
        //  printf("f%d = %f\n", 6, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 6));
        //  printf("f%d = %f\n", 7, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 7));
        //  printf("f%d = %f\n", 8, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 8));
        //  printf("f%d = %f\n", 8, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 9));
        //  printf("f%d = %f\n", 10, getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, 10));
        for (unsigned int m=0; m<NTERMS; m++)
        {
          it = getFintegral(-1.0, 1.0, -1.0, +1.0, -M_PI, M_PI, p, n, o, m+1);
          w[m] += tijk[lbin] * it; 
        }
      }
    }
  }

}



KERNEL
void py_tijk2weights(GLOBAL_MEM ftype *w, GLOBAL_MEM const ftype *tijk, 
                     const int order_cosK, const int order_cosL, 
                     const int order_hphi)
{

  tijk2weights(w, tijk, order_cosK, order_cosL, order_hphi);

}
