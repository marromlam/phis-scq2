////////////////////////////////////////////////////////////////////////////////
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



////////////////////////////////////////////////////////////////////////////////
// Inlude headers //////////////////////////////////////////////////////////////


#include <curand.h>
#include <curand_kernel.h>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

WITHIN_KERNEL
double tagOSgen(double x)
{
  return 3.8 - 134.6*x + 1341.*x*x;
}

WITHIN_KERNEL
double tagSSgen(double x)
{
  if (x < 0.46) return exp(16*x -.77);
  else return 10*(16326 - 68488*x + 72116*x*x);
}



KERNEL
void dG5toy(GLOBAL_MEM ${ftype} * out,
            ${ftype} G, ${ftype} DG, ${ftype} DM,
            GLOBAL_MEM ${ftype} * CSP,
            //GLOBAL_MEM ${ftype} * CSD,
            //GLOBAL_MEM ${ftype} * CPD,
            GLOBAL_MEM ${ftype} * ASlon,
            GLOBAL_MEM ${ftype} * APlon,
            GLOBAL_MEM ${ftype} * APpar,
            GLOBAL_MEM ${ftype} * APper,
            //${ftype} ADlon, ${ftype} ADpar, ${ftype} ADper,
            ${ftype} pSlon,
            ${ftype} pPlon, ${ftype} pPpar, ${ftype} pPper,
            //${ftype} pDlon, ${ftype} pDpar, ${ftype} pDper,
            GLOBAL_MEM ${ftype} *dSlon,
            ${ftype} dPlon, ${ftype} dPpar, ${ftype} dPper,
            //${ftype} dDlon, ${ftype} dDpar, ${ftype} dDper,
            ${ftype} lSlon,
            ${ftype} lPlon, ${ftype} lPpar, ${ftype} lPper,
            //${ftype} lDlon, ${ftype} lDpar, ${ftype} lDper,
            // Time limits
            ${ftype} tLL, ${ftype} tUL,
            // Time resolution
            ${ftype} sigma_offset, ${ftype} sigma_slope, ${ftype} sigma_curvature,
            ${ftype} mu,
            // Flavor tagging
            ${ftype} eta_bar_os, ${ftype} eta_bar_ss,
            ${ftype} p0_os,  ${ftype} p1_os, ${ftype} p2_os,
            ${ftype} p0_ss,  ${ftype} p1_ss, ${ftype} p2_ss,
            ${ftype} dp0_os, ${ftype} dp1_os, ${ftype} dp2_os,
            ${ftype} dp0_ss, ${ftype} dp1_ss, ${ftype} dp2_ss,
            // Time acceptance
            GLOBAL_MEM ${ftype} *coeffs,
            // Angular acceptance
            GLOBAL_MEM  ${ftype} *angular_weights,
            int USE_FK, int BINS, int USE_ANGACC, int USE_TIMEACC,
            int USE_TIMEOFFSET, int SET_TAGGING, int USE_TIMERES,
            ${ftype} PROB_MAX, int NEVT)


{
  int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  // Prepare curand
  curandState state;
  curand_init((unsigned long long)clock(), evt, 0, &state);

  ${ftype} iter = 0.0;

  // Decay time resolution HARDCODED!
  ${ftype} sigmat = 0.0;
  if (USE_TIMERES)
  {
    sigmat = curand_log_normal(&state,-3.22,0.309);
  }
  else
  {
    sigmat = 0*0.04554;
  }



  // Flavor tagging ------------------------------------------------------------
  ${ftype} qOS = 0;
  ${ftype} qSS = 0;
  ${ftype} etaOS = 0;
  ${ftype} etaSS = 0;

  if (SET_TAGGING == 1) // DATA
  {
    ${ftype} tagOS = curand_uniform(&state);
    ${ftype} tagSS = curand_uniform(&state);
    ${ftype} OSmax = tagOSgen(0.5);
    ${ftype} SSmax = tagSSgen(0.5);
    ${ftype} tag = 0;
    ${ftype} threshold;

    // generate qOS
    if (tagOS < 0.16) {
      qOS = 1.;
    }
    else if (tagOS<0.32){
      qOS = -1.;
    }
    else {
      qOS = 0.;
    }
    // generate qSS
    if (tagSS < 0.31) {
      qSS = 1.;
    }
    else if (tagSS<0.62){
      qSS = -1.;
    }
    else {
      qSS = 0.;
    }

    // generate etaOS
    if (qOS > 0.5 || qOS < -0.5)
    {
      while(1)
      {
        tag = 0.49*curand_uniform(&state);
        threshold = OSmax*curand_uniform(&state);
        if (tagOSgen(tag) > threshold) break;
      }
      etaOS = tag;
    }
    // generate etaSS
    if (qSS > 0.5 || qSS < -0.5)
    {
      while(1)
      {
        tag = 0.49*curand_uniform(&state);
        threshold = SSmax*curand_uniform(&state);
        if (tagSSgen(tag) > threshold) break;
      }
      etaSS = tag;
    }
  }
  else if (SET_TAGGING == 0) // PERFECT, MC
  {
    ${ftype} tag = curand_uniform(&state);
    if (tag < 0.5){
      qOS = +1.0;
      qSS = +1.0;
    }
    else
    {
      qOS = -1.0;
      qSS = -1.0;
    }
    etaOS = 0.5;
    etaSS = 0.5;
  }
  else //TRUE
  {
    qOS = 0.0;
    qSS = 0.0;
    etaOS = 0.5;
    etaSS = 0.5;
  }


  // Loop and generate ---------------------------------------------------------
  while(1)
  {
    // Random numbers
    ${ftype} cosK = - 1.0  +    2.0*curand_uniform(&state);
    ${ftype} cosL = - 1.0  +    2.0*curand_uniform(&state);
    ${ftype} hphi = - M_PI + 2*M_PI*curand_uniform(&state);
    ${ftype} time = tLL - log(curand_uniform(&state))/(G-0.5*DG);

    // PDF threshold
    ${ftype} threshold = PROB_MAX*curand_uniform(&state);

    // Prepare data and pdf variables to DiffRate CUDA function
    ${ftype} mass = out[evt*10+4];
    ${ftype} data[9] = {cosK, cosL, hphi, time, sigmat, qOS, qSS, etaOS, etaSS};
    //printf("cosK=%lf cosL=%lf hphi=%lf time=%lf sigmat=%lf qOS=%lf qSS=%lf etaOS=%lf etaSS=%lf", cosK, cosL, hphi, time, sigmat, qOS, qSS, etaOS, etaSS);
    ${ftype} pdf = 0.0;

    // Get pdf value from angular distribution
    // if time is larger than asked, put pdf to zero
    if ((time < tLL) || (time > tUL))
    {
      pdf = 0.0;
    }
    else
    {
      unsigned int bin = BINS>1 ? getMassBin(mass) : 0;
      pdf = getDiffRate(data,
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

      pdf *= exp((G-0.5*DG)*(time-tLL));
      pdf *= (G-0.5*DG)*(1- exp((G-0.5*DG)*(-tUL+tLL)));
    }
    // final checks ------------------------------------------------------------

    // check if probability is greater than the PROB_MAX
    if (pdf > PROB_MAX) {
      printf("WARNING: PDF [ = %lf] > PROB_MAX [ = %lf]\n", pdf, PROB_MAX);
    }

    // stop if it's taking too much iterations ---------------------------------
    iter++;
    if(iter > 1000000)
    {
      printf("ERROR: This p.d.f. is too hard...");
      return;
    }

    // Store generated values --------------------------------------------------
    if (pdf >= threshold)
    {
      out[evt*10+0] = data[0]; // cosK
      out[evt*10+1] = data[1]; // cosL
      out[evt*10+2] = data[2]; // hphi
      out[evt*10+3] = data[3]; // time
      // mass (index 4) is already in the array :)
      out[evt*10+5] = data[4]; // sigma_t
      out[evt*10+6] = data[5]; // qOS
      out[evt*10+7] = data[6]; // qSS
      out[evt*10+8] = data[7]; // etaOS
      out[evt*10+9] = data[8];  // etaSS
      return;
    }

  }

}
