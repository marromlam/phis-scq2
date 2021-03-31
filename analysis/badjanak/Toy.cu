////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                        GENERATE TOY CROSS RATE                             //
//                                                                            //
//   Created: 2019-01-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include <ipanema/core.h>
#include <ipanema/random.h>



////////////////////////////////////////////////////////////////////////////////
// Generate toy ////////////////////////////////////////////////////////////////

KERNEL
void dG5toy(GLOBAL_MEM ftype * out,
            const ftype G, const ftype DG, const ftype DM,
            GLOBAL_MEM const ftype * CSP,
            //GLOBAL_MEM const ftype * CSD,
            //GLOBAL_MEM const ftype * CPD,
            GLOBAL_MEM const ftype * ASlon,
            GLOBAL_MEM const ftype * APlon,
            GLOBAL_MEM const ftype * APpar,
            GLOBAL_MEM const ftype * APper,
            //const ftype ADlon, const ftype ADpar, const ftype ADper,
            const ftype pSlon,
            const ftype pPlon, const ftype pPpar, const ftype pPper,
            //const ftype pDlon, const ftype pDpar, const ftype pDper,
            GLOBAL_MEM const ftype *dSlon,
            const ftype dPlon, const ftype dPpar, const ftype dPper,
            //const ftype dDlon, const ftype dDpar, const ftype dDper,
            const ftype lSlon,
            const ftype lPlon, const ftype lPpar, const ftype lPper,
            //const ftype lDlon, const ftype lDpar, const ftype lDper,
            // Time limits
            const ftype tLL, const ftype tUL,
            const ftype cosKLL, const ftype cosKUL,
            const ftype cosLLL, const ftype cosLUL,
            const ftype hphiLL, const ftype hphiUL,
            // Time resolution
            const ftype sigma_offset, const ftype sigma_slope, const ftype sigma_curvature,
            const ftype mu,
            // Flavor tagging
            const ftype eta_bar_os, const ftype eta_bar_ss,
            const ftype p0_os,  const ftype p1_os, const ftype p2_os,
            const ftype p0_ss,  const ftype p1_ss, const ftype p2_ss,
            const ftype dp0_os, const ftype dp1_os, const ftype dp2_os,
            const ftype dp0_ss, const ftype dp1_ss, const ftype dp2_ss,
            // Time acceptance
            GLOBAL_MEM const ftype *coeffs,
            // Angular acceptance
            GLOBAL_MEM const ftype *tijk,
            const int order_cosK, const int order_cosL, const int order_hphi,
            const int USE_FK, const int BINS, const int USE_ANGACC, const int USE_TIMEACC,
            const int USE_TIMEOFFSET, const int SET_TAGGING, const int USE_TIMERES,
            const ftype PROB_MAX, const int SEED, const int NEVT)
{
  int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  ftype threshold;

  // Prepare curand
  #ifdef CUDA
    curandState state;
    curand_init((unsigned long long)clock(), evt, 0, &state);
  #else
    int _seed = SEED;
    int *state = &_seed;
  #endif

  ftype iter = 0.0;

  // Decay time resolution HARDCODED!
  ftype sigmat = 0.0;
  if (USE_TIMERES)
  {
    sigmat = rngLogNormal(-3.22,0.309, &state, 100);
  }
  else
  {
    sigmat = 0*0.04554;
  }



  // Flavor tagging ------------------------------------------------------------
  ftype qOS = 0;
  ftype qSS = 0;
  ftype etaOS = 0.5;
  ftype etaSS = 0.5;

  if (SET_TAGGING == 1) // DATA
  {
    ftype tagOS = rng_uniform(&state, 100);
    ftype tagSS = rng_uniform(&state, 100);
    ftype OSmax = tagOSgen(0.5);
    ftype SSmax = tagSSgen(0.5);
    ftype tag = 0;
    

    // generate qOS
    if (tagOS < 0.175)
    {
      qOS = 1.;
    }
    else if (tagOS<0.35)
    {
      qOS = -1.;
    }
    else
    {
      qOS = 0.;
    }
    // generate qSS
    if (tagSS < 0.33)
    {
      qSS = 1.;
    }
    else if (tagSS<0.66)
    {
      qSS = -1.;
    }
    else
    {
      qSS = 0.;
    }

    // generate etaOS
    if (qOS > 0.5 || qOS < -0.5)
    {
      while(1)
      {
        tag = 0.49*rng_uniform(&state, 100);
        threshold = OSmax*rng_uniform(&state, 100);
        if (tagOSgen(tag) > threshold) break;
      }
      etaOS = tag;
    }
    // generate etaSS
    if (qSS > 0.5 || qSS < -0.5)
    {
      while(1)
      {
        tag = 0.49*rng_uniform(&state, 100);
        threshold = OSmax*rng_uniform(&state, 100);
        if (tagSSgen(tag) > threshold) break;
      }
      etaSS = tag;
    }
  }
  else if (SET_TAGGING == 0) // PERFECT, MC
  {
    ftype tag = rng_uniform(&state, 100);
    if (tag < 0.5)
    {
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
  //qOS = -1; qSS = -1;  etaOS=+0.24959514;  etaSS=+0.46155431;
  qOS *= 531; // put same number used in EvtGen 
  qSS *= 531; // put same number used in EvtGen

  // Loop and generate ---------------------------------------------------------
  ftype cosK, cosL, hphi, time, mass, pdf, timeacc, angacc;
  //ftype data[9];
  unsigned int bin = 0; 
  
  while(1)
  {
    pdf = 0;
    // Random numbers
    cosK = - 1.0  +    2.0*rng_uniform(&state, 100);
    cosL = - 1.0  +    2.0*rng_uniform(&state, 100);
    hphi = - M_PI + 2*M_PI*rng_uniform(&state, 100);
    time =   tLL  -    log(rng_uniform(&state, 100)) / (G-0.5*DG);

  //cosK=+0.45774651; cosL=-0.20873141; hphi=-0.00354153; time=+3.2032610199999998;
    // PDF threshold
    threshold = PROB_MAX*rng_uniform(&state, 100);

    // Prepare data and pdf variables to DiffRate CUDA function
    mass = out[evt*10+4];
    ftype data[9] = {cosK, cosL, hphi, time, sigmat, qOS, qSS, etaOS, etaSS};
    //printf("cosK=%lf cosL=%lf hphi=%lf time=%lf sigmat=%lf qOS=%lf qSS=%lf etaOS=%lf etaSS=%lf", cosK, cosL, hphi, time, sigmat, qOS, qSS, etaOS, etaSS);
    pdf = 0.0;

    // Get pdf value from angular distribution
    // if time is larger than asked, put pdf to zero
    if ((time >= tLL) & (time <= tUL))
    {
      timeacc = 1.0;
      angacc = 1.0;
      if (USE_TIMEACC)
      {
        timeacc = time_efficiency(time, coeffs, tLL, tUL);
      }
      if (USE_ANGACC)
      {
        angacc = angular_wefficiency(data[0], data[1], data[2], tijk);
      }
      if (evt  == 0){
      printf("angacc = %f", angacc);}
      bin = BINS>1 ? getMassBin(mass) : 0;
      pdf = rateBs(data,
                    G, DG, DM, CSP[bin],
                    ASlon[bin], APlon[bin], APpar[bin], APper[bin],
                    pSlon,      pPlon,      pPpar,      pPper,
                    dSlon[bin], dPlon,      dPpar,      dPper,
                    lSlon,      lPlon,      lPpar,      lPper,
                    tLL, tUL, cosKLL, cosKUL, cosLLL, cosLUL, hphiLL, hphiUL,
                    sigma_offset, sigma_slope, sigma_curvature, mu,
                    eta_bar_os, eta_bar_ss,
                    p0_os,  p1_os, p2_os,
                    p0_ss,  p1_ss, p2_ss,
                    dp0_os, dp1_os, dp2_os,
                    dp0_ss, dp1_ss, dp2_ss,
                    coeffs,
                    tijk,
                    USE_FK, USE_ANGACC, USE_TIMEACC,
                    USE_TIMEOFFSET, SET_TAGGING, USE_TIMERES);
      pdf *= angacc * timeacc * exp((G-0.5*DG)*(time-tLL));
      pdf *= (G-0.5*DG) * (1 - exp((G-0.5*DG)*(-tUL+tLL)));
    }
    // final checks ------------------------------------------------------------

    // check if probability is greater than the PROB_MAX
    if ( (pdf > PROB_MAX) && (get_global_id(0)<100) ) 
    {
      printf("WARNING: PDF [=%f] > PROB_MAX [=%f]\n", pdf, PROB_MAX);
    }

    // stop if it's taking too much iterations ---------------------------------
    iter++;
    if( (iter > 100000) && (get_global_id(0)<100) )
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
      out[evt*10+9] = data[8]; // etaSS
      return;
    }

  }

}

////////////////////////////////////////////////////////////////////////////////
