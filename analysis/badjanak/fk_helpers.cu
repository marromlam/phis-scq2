#include "fk_helpers.h"


KERNEL
void integral_ijk_fx(const ftype cosKs, const ftype cosKe, const ftype cosLs, const ftype cosLe,
                     const ftype phis, const ftype phie, const int i, const int j, const int k,
                     GLOBAL_MEM ftype * fx)
{
  // const int idx = get_global_id(0);
  for (int K=0; K<NTERMS; K++)
  {
    fx[K] = getFintegral(cosKs, cosKe, cosLs, cosLe, phis, phie, i, j, k, K);
  }
}


KERNEL
void pyEff(
    GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL, GLOBAL_MEM const ftype *hphi,
    GLOBAL_MEM const ftype *data_3d, GLOBAL_MEM const ftype *prediction_3d,
    GLOBAL_MEM const ftype *pars,
    const int bin_cosK, const int bin_cosL, const int bin_hphi,
    const int order_cosK, const int order_cosL, const int order_hphi,
    const int NEVT
)
{
  ftype f1  = 0.0;
  ftype f2  = 0.0;
  ftype f3  = 0.0;
  ftype f4  = 0.0;
  ftype f5  = 0.0;
  ftype f6  = 0.0;
  ftype f7  = 0.0;
  ftype f8  = 0.0;
  ftype f9  = 0.0;
  ftype f10 = 0.0;

  int lbin = 0;

  // need to determine fx from 3d eff!

  for (unsigned int p = 0; p<order_cosK+1; p++)
  {
    for (unsigned int o = 0; o<order_hphi+1; o++)
    {
      for (unsigned int n = 0; n<order_cosL+1; n++)
      {
            #ifdef CUDA
            lbin = int(n + (order_cosL+1)*o + (order_cosL+1)*(order_hphi+1)*p);
            #else
            lbin = convert_int(n + (order_cosL+1)*o + (order_cosL+1)*(order_hphi+1)*p);
            #endif
            f1  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 1);
            f2  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 2);
            f3  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 3);
            f4  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 4);
            f5  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 5);
            f6  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 6);
            f7  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 7);
            f8  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 8);
            f9  += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 9);
            f10 += pars[lbin] * getFintegral(-1.0, 1.0, -1.0, 1.0, -M_PI, M_PI, n, o, p, 10);
      }
    }
  }
    // ftype scale = 1.0;//(f1_analytic.value + f2_analytic.value + f3_analytic.value)/3.0;
    // f1_analytic /= entry(scale, 0.0);
    // f2_analytic /= entry(scale, 0.0);
    // f3_analytic /= entry(scale, 0.0);
    // f4_analytic /= entry(scale, 0.0);
    // f5_analytic /= entry(scale, 0.0);
    // f6_analytic /= entry(scale, 0.0);
}


/*


WITHIN_KERNEL
ftype ang_eff(const ftype cosK, const ftype cosL, const ftype phi, ftype *moments)
{
    ftype eff = 0.;

    eff += moments[0] *lpmv(0, 0, cosK) * sph_harm(0, 0, cosL, phi);
    eff += moments[1] *lpmv(0, 0, cosK) * sph_harm(2, 0, cosL, phi);
    eff += moments[2] *lpmv(0, 0, cosK) * sph_harm(2, 2, cosL, phi);
    eff += moments[3] *lpmv(0, 0, cosK) * sph_harm(2, 1, cosL, phi);
    eff += moments[4] *lpmv(0, 0, cosK) * sph_harm(2,-1, cosL, phi);
    eff += moments[5] *lpmv(0, 0, cosK) * sph_harm(2,-2, cosL, phi);
    eff += moments[6] *lpmv(1, 0, cosK) * sph_harm(0, 0, cosL, phi);
    eff += moments[7] *lpmv(1, 0, cosK) * sph_harm(2, 1, cosL, phi);
    eff += moments[8] *lpmv(1, 0, cosK) * sph_harm(2,-1, cosL, phi);
    eff += moments[9] *lpmv(2, 0, cosK) * sph_harm(0, 0, cosL, phi);

    eff *= 2.*sqrt(M_PI);
    return eff;
}




WITHIN_KERNEL
void angWeightsToMoments(ftype* moments, GLOBAL_MEM const ftype* normweights)
{
    //c0000
    moments[0]  =   1. / 3. * ( normweights[0] + normweights[1] + normweights[2] );//
    //c0020
    moments[1]  =   1. / 3. * sqrt(5.)             * ( normweights[0] + normweights[1] + normweights[2] - 3. * normweights[6] );//
    //c0022
    moments[2]  =            -sqrt(5. / 3.)        * ( normweights[1] - normweights[2] );//
    //c0021
    moments[3]  = - 8. / 3. * sqrt( 5. / 2. ) / M_PI *   normweights[7];//
    //c002-1
    moments[4]  = - 8. / 3. * sqrt( 5. / 2. ) / M_PI *  (normweights[8]);//-normweights should be +normweights?
    //c002-2
    moments[5]  =             sqrt( 5. / 3. )      *  (normweights[3]);//-normweights should be +normweights?
    //c1000
    moments[6]  =   1. / 2. * sqrt(3.)             *   normweights[9];//
    //c1021
    moments[7]  = - 32. / 3. * sqrt( 5. / 6. ) / M_PI *   normweights[4];//
    //c102-1
    moments[8]  = + 32. / 3. * sqrt( 5. / 6. ) / M_PI *  (normweights[5]);//-normweights should be +normweights?
    //c2000
    moments[9]  =  5. / 2.                        * ( normweights[0] - normweights[6] );//
}






KERNEL
void plot_moments(GLOBAL_MEM const ftype *normweights, GLOBAL_MEM ftype *out,
                  GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL,
                  GLOBAL_MEM const ftype *hphi)
{
  const int i = get_global_id(0);

  ftype moments[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  // get the moments
  angWeightsToMoments(moments, normweights);
  out[i] = ang_eff(cosK[i], cosL[i], hphi[i], moments);

  //ftype ang_acc = ang_eff(x, y, z, moments); // these are angular weights again

}

*/

