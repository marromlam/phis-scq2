////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                         DIFFERENTIAL CROSS RATE                            //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#include <lib99ocl/complex.h>

#include "cross_rate_bs.h"
#include "decay_time_acceptance.h"
#include "tagging.h"
#include "time_angular_distribution.h"

WITHIN_KERNEL
ftype rateBs(const ftype *data,
             // Time-dependent angular distribution
             const ftype G, const ftype DG, const ftype DM, const ftype CSP,
             const ftype ASlon, const ftype APlon, const ftype APpar,
             const ftype APper, const ftype pSlon, const ftype pPlon,
             const ftype pPpar, const ftype pPper, const ftype dSlon,
             const ftype dPlon, const ftype dPpar, const ftype dPper,
             const ftype lSlon, const ftype lPlon, const ftype lPpar,
             const ftype lPper,
             // Time limits
             const ftype tLL, const ftype tUL, const ftype cosKLL,
             const ftype cosKUL, const ftype cosLLL, const ftype cosLUL,
             const ftype hphiLL, const ftype hphiUL,
             // Time resolution
             const ftype sigma_offset, const ftype sigma_slope,
             const ftype sigma_curvature, const ftype mu,
             // Flavor tagging
             const ftype eta_bar_os, const ftype eta_bar_ss, const ftype p0_os,
             const ftype p1_os, const ftype p2_os, const ftype p0_ss,
             const ftype p1_ss, const ftype p2_ss, const ftype dp0_os,
             const ftype dp1_os, const ftype dp2_os, const ftype dp0_ss,
             const ftype dp1_ss, const ftype dp2_ss,
             // Time acceptance
             GLOBAL_MEM const ftype *coeffs,
             // Angular acceptance
             GLOBAL_MEM const ftype *angular_weights, const int USE_FK,
             const int USE_ANGACC, const int USE_TIMEACC,
             const int USE_TIMEOFFSET, const int SET_TAGGING,
             const int USE_TIMERES) {
  // Print inputs {{{

#if DEBUG
  if (DEBUG > 4 && get_global_id(0) == DEBUG_EVT) {
    printf("*USE_FK            : %d\n", USE_FK);
    printf("*USE_ANGACC        : %d\n", USE_ANGACC);
    printf("*USE_TIMEACC       : %d\n", USE_TIMEACC);
    printf("*USE_TIMEOFFSET    : %d\n", USE_TIMEOFFSET);
    printf("*USE_TIMERES       : %d\n", USE_TIMERES);
    printf("*SET_TAGGING       : %d [0:perfect,1:real,2:true]\n", SET_TAGGING);
    printf("G                  : %+.8f\n", G);
    printf("DG                 : %+.8f\n", DG);
    printf("DM                 : %+.8f\n", DM);
    printf("CSP                : %+.8f\n", CSP);
    printf("ASlon              : %+.8f\n", ASlon);
    printf("APlon              : %+.8f\n", APlon);
    printf("APpar              : %+.8f\n", APpar);
    printf("APper              : %+.8f\n", APper);
    printf("pSlon              : %+.8f\n", pSlon);
    printf("pPlon              : %+.8f\n", pPlon);
    printf("pPpar              : %+.8f\n", pPpar);
    printf("pPper              : %+.8f\n", pPper);
    printf("dSlon              : %+.8f\n", dSlon);
    printf("dPlon              : %+.8f\n", dPlon);
    printf("dPper              : %+.8f\n", dPper);
    printf("dPpar              : %+.8f\n", dPpar);
    printf("lSlon              : %+.8f\n", lSlon);
    printf("lPlon              : %+.8f\n", lPlon);
    printf("lPper              : %+.8f\n", lPper);
    printf("lPpar              : %+.8f\n", lPpar);
    printf("tLL                : %+.8f\n", tLL);
    printf("cosKLL             : %+.8f\n", cosKLL);
    printf("cosLLL             : %+.8f\n", cosLLL);
    printf("hphiLL             : %+.8f\n", hphiLL);
    printf("tUL                : %+.8f\n", tUL);
    printf("cosKUL             : %+.8f\n", cosKUL);
    printf("cosLUL             : %+.8f\n", cosLUL);
    printf("hphiUL             : %+.8f\n", hphiUL);
    printf("mu                 : %+.8f\n", mu);
    printf("sigma_offset       : %+.8f\n", sigma_offset);
    printf("sigma_slope        : %+.8f\n", sigma_slope);
    printf("sigma_curvature    : %+.8f\n", sigma_curvature);
    printf("etaOS, etaSS       : %+.8f, %+.8f\n", eta_bar_os, eta_bar_ss);
    printf("p0OS, p0SS         : %+.8f, %+.8f\n", p0_os, p0_ss);
    printf("p1OS, p1SS         : %+.8f, %+.8f\n", p1_os, p1_ss);
    printf("p2OS, p2SS         : %+.8f, %+.8f\n", p2_os, p2_ss);
    printf("dp0OS, dp0SS       : %+.8f, %+.8f\n", dp0_os, dp0_ss);
    printf("dp1OS, dp1SS       : %+.8f, %+.8f\n", dp1_os, dp1_ss);
    printf("dp2OS, dp2SS       : %+.8f, %+.8f\n", dp2_os, dp2_ss);
    printf("COEFFS             : %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[0 * 4 + 0], coeffs[0 * 4 + 1], coeffs[0 * 4 + 2],
           coeffs[0 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[1 * 4 + 0], coeffs[1 * 4 + 1], coeffs[1 * 4 + 2],
           coeffs[1 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[2 * 4 + 0], coeffs[2 * 4 + 1], coeffs[2 * 4 + 2],
           coeffs[2 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[3 * 4 + 0], coeffs[3 * 4 + 1], coeffs[3 * 4 + 2],
           coeffs[3 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[4 * 4 + 0], coeffs[4 * 4 + 1], coeffs[4 * 4 + 2],
           coeffs[4 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[5 * 4 + 0], coeffs[5 * 4 + 1], coeffs[5 * 4 + 2],
           coeffs[5 * 4 + 3]);
    printf("                     %+.8f\t%+.8f\t%+.8f\t%+.8f\n",
           coeffs[6 * 4 + 0], coeffs[6 * 4 + 1], coeffs[6 * 4 + 2],
           coeffs[6 * 4 + 3]);
  }
#endif

  // }}}

  // Variables {{{
  //     Make sure that the input it's in this order.
  //     lalala
  const ftype cosK = data[0]; // Time-angular distribution
  const ftype cosL = data[1];
  const ftype hphi = data[2];
  const ftype time = data[3];
  // ftype sigma_t        = 0.04554;                          // Time resolution
  const ftype sigma_t = data[4]; // Time resolution
  const ftype qOS = data[5];     // Tagging
  const ftype qSS = data[6];
  const ftype etaOS = data[7];
  const ftype etaSS = data[8];

#if DEBUG
  if (DEBUG > 99 && ((time >= tUL) || (time <= tLL))) {
    printf("WARNING            : Event with time not within [%.4f,%.4f].\n",
           tLL, tUL);
  }
#endif

#if DEBUG
  if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT) {
    printf("\nINPUT              : cosK=%+.8f  cosL=%+.8f  hphi=%+.8f  "
           "time=%+.8f\n",
           cosK, cosL, hphi, time);
    printf("                   : sigma_t=%+.8f  qOS=%+.8f  qSS=%+.8f  "
           "etaOS=%+.8f  etaSS=%+.8f\n",
           sigma_t, qOS, qSS, etaOS, etaSS);
  }
#endif

  const bool FULL_RANGE = (cosKLL == -1) && (cosKUL == 1) && (cosLLL == -1) &&
                          (cosLUL == 1) && (hphiLL == -M_PI) &&
                          (hphiUL == M_PI);
  // }}}

  // Time resolution {{{
  //     In order to remove the effects of conv, set sigma_t=0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  ftype delta_t = sigma_t;
  // WARNING: These should be arguments for the cross rate
  ftype t_offset = 0.0;
  ftype sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;

  if (USE_TIMEOFFSET) {
    // t_offset = parabola(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
    t_offset = mu;
  }

  if (USE_TIMERES) // use_per_event_res
  {
    delta_t = parabola(sigma_t, sigma_offset, sigma_slope, sigma_curvature);
  }

#if DEBUG
  if (DEBUG > 3 && get_global_id(0) == DEBUG_EVT) {
    printf("\nTIME RESOLUTION    : delta_t=%.8f\n", delta_t);
    printf("                   : time-t_offset=%.8f\n", time - t_offset);
  }
#endif

  // }}}

  // Time-dependent part {{{

  ctype exp_p = C(0, 0);
  ctype exp_m = C(0, 0);
  ctype exp_i = C(0, 0);

  if ((USE_TIMEACC & !USE_TIMERES) || (delta_t == 0)) {
    // exp_p = expconv_wores(time-t_offset, G + 0.5*DG, 0.);
    // exp_m = expconv_wores(time-t_offset, G - 0.5*DG, 0.);
    // exp_i = expconv_wores(time-t_offset,          G, DM);
    exp_p = expconv(time - t_offset, G + 0.5 * DG, 0., delta_t);
    exp_m = expconv(time - t_offset, G - 0.5 * DG, 0., delta_t);
    exp_i = expconv(time - t_offset, G, DM, delta_t);
  } else {
    exp_p = expconv_simon(time - t_offset, G + 0.5 * DG, 0., delta_t);
    exp_m = expconv_simon(time - t_offset, G - 0.5 * DG, 0., delta_t);
    exp_i = expconv_simon(time - t_offset, G, DM, delta_t);
  }

  // TODO: temporal fix to get HD fitter results
  //       we should try if this can/cannot be removed
  // if ( delta_t == 0 ) // MC samples need to solve some problems
  // {
  //   exp_p = expconv(time-t_offset, G + 0.5*DG, 0., delta_t);
  //   exp_m = expconv(time-t_offset, G - 0.5*DG, 0., delta_t);
  //   exp_i = expconv(time-t_offset,          G, DM, delta_t);
  // }
  // else
  // {
  //   exp_p = expconv(time-t_offset, G + 0.5*DG, 0., delta_t);
  //   exp_m = expconv(time-t_offset, G - 0.5*DG, 0., delta_t);
  //   exp_i = expconv(time-t_offset, G         , DM, delta_t);
  // }

  ftype ta = 0.5 * (exp_m.x + exp_p.x);
  ftype tb = 0.5 * (exp_m.x - exp_p.x);
  ftype tc = exp_i.x;
  ftype td = exp_i.y;

#if FAST_INTEGRAL
  // Veronika has these factors wrt HD-fitter
  ta *= sqrt(2 * M_PI);
  tb *= sqrt(2 * M_PI);
  tc *= sqrt(2 * M_PI);
  td *= sqrt(2 * M_PI);
#endif

#if DEBUG
  if (DEBUG >= 3 && get_global_id(0) == DEBUG_EVT) {
    printf("\nTIME TERMS         : ta=%.8f  tb=%.8f  tc=%.8f  td=%.8f\n", ta,
           tb, tc, td);
    printf(
        "                   : exp_m=%.8f  exp_p=%.8f  exp_i=%.8f  exp_i=%.8f\n",
        sqrt(2 * M_PI) * exp_m.x, sqrt(2 * M_PI) * exp_p.x,
        sqrt(2 * M_PI) * exp_i.x, exp_i.y);
  }
#endif

  // }}}

  // Flavor tagging {{{

  ftype omegaOSB = 0;
  ftype omegaOSBbar = 0;
  ftype tagOS = 0;
  ftype omegaSSB = 0;
  ftype omegaSSBbar = 0;
  ftype tagSS = 0;

  if (SET_TAGGING == 1) // DATA
  {
    if (qOS != 0) {
      tagOS = qOS / fabs(qOS);
    }
    if (qSS != 0) {
      tagSS = qSS / fabs(qSS);
    }
    omegaOSB = get_omega(etaOS, +1, p0_os, p1_os, p2_os, dp0_os, dp1_os, dp2_os,
                         eta_bar_os);
    omegaOSBbar = get_omega(etaOS, -1, p0_os, p1_os, p2_os, dp0_os, dp1_os,
                            dp2_os, eta_bar_os);
    omegaSSB = get_omega(etaSS, +1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss, dp2_ss,
                         eta_bar_ss);
    omegaSSBbar = get_omega(etaSS, -1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss,
                            dp2_ss, eta_bar_ss);
  } else if (SET_TAGGING == 0) // PERFECT, MC
  {
    if (qOS != 0) {
      tagOS = qOS / fabs(qOS);
    }
    if (qSS != 0) {
      tagSS = qSS / fabs(qSS);
    }
  } else // TRUE
  {
    tagOS = 0.0;
    tagSS = 0.0;
  }

  // Print warning if tagOS|tagSS == 0
#if DEBUG
  if (DEBUG > 99 && ((tagOS == 0) | (tagSS == 0))) {
    printf("This event is not tagged!\n");
  }
#endif

#if DEBUG
  if (DEBUG > 3 && get_global_id(0) == DEBUG_EVT) {
    printf("\nFLAVOR TAGGING     : delta_t=%.8f\n", delta_t);
    printf("                   : tagOS=%.8f, tagSS=%.8f\n", tagOS, tagSS);
    printf("                   : omegaOSB=%.8f, omegaOSBbar=%.8f\n", omegaOSB,
           omegaOSBbar);
    printf("                   : omegaSSB=%.8f, omegaSSBbar=%.8f\n", omegaSSB,
           omegaSSBbar);
  }
#endif

  // }}}

  // Decay-time acceptance {{{
  //     To get rid of decay-time acceptance set USE_TIMEACC to False. If True
  //     then time_efficiency locates the time bin of the event and returns
  //     the value of the cubic spline.

  ftype dta = 1.0;
  if (USE_TIMEACC) {
    dta = time_efficiency(time, coeffs, tLL, tUL);
  }

  // }}}

  // Compute per event pdf {{{

  // Allocate some variables {{{

  ftype vnk[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
#if DEBUG
  ftype vfk[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
#endif
  ftype vak[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  ftype vbk[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  ftype vck[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  ftype vdk[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
  ftype angnorm[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

  ftype nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
  ftype pdfB = 0.0;
  ftype pdfBbar = 0.0;

  // }}}

  // Angular-dependent part {{{

  for (int k = 1; k <= NTERMS; k++) {
    nk = getN(APlon, ASlon, APpar, APper, CSP, k);
    if (USE_FK) {
#if FAST_INTEGRAL
      fk = getF(cosK, cosL, hphi, k);
#else
      fk = (9.0 / (16.0 * M_PI)) * getF(cosK, cosL, hphi, k);
#endif
    } else {
      if (FULL_RANGE)
        fk = TRISTAN[k - 1]; // these are 0s or 1s
      else
        // fk = FULL_RANGE ? TRISTAN[k - 1]
        //                 : getFintegral(cosKLL, cosKUL, cosLLL, cosLUL,
        //                 hphiLL,
        //                                hphiUL, 0, 0, 0, k);
        fk = FULL_RANGE ? TRISTAN[k - 1] : TRISTAN[k - 1];
    }

    ak = getA(pPlon, pSlon, pPpar, pPper, dPlon, dSlon, dPpar, dPper, lPlon,
              lSlon, lPpar, lPper, k);
    bk = getB(pPlon, pSlon, pPpar, pPper, dPlon, dSlon, dPpar, dPper, lPlon,
              lSlon, lPpar, lPper, k);
    ck = getC(pPlon, pSlon, pPpar, pPper, dPlon, dSlon, dPpar, dPper, lPlon,
              lSlon, lPpar, lPper, k);
    dk = getD(pPlon, pSlon, pPpar, pPper, dPlon, dSlon, dPpar, dPper, lPlon,
              lSlon, lPpar, lPper, k);

    if ((int)fabs(qOS) != 511) { // Bs2JpsiPhi p.d.f
      hk_B = (ak * ta + bk * tb + ck * tc + dk * td);
      hk_Bbar = (ak * ta + bk * tb - ck * tc - dk * td);
    } else { // Bd2JpsiKstar p.d.f (time dependent)
      hk_B = ak * ta + ck * tc;
      if ((k == 4) || (k == 6) || (k == 9)) {
        hk_Bbar = tagOS * ak * ta + tagOS * ck * tc;
      } else {
        hk_Bbar = ak * ta + ck * tc;
      }
    }
#if FAST_INTEGRAL
    hk_B = 3. / (4. * M_PI) * hk_B;
    hk_Bbar = 3. / (4. * M_PI) * hk_Bbar;
#endif
    pdfB += nk * fk * hk_B;
    pdfBbar += nk * fk * hk_Bbar;
    vnk[k - 1] = 1. * nk;
    vak[k - 1] = 1. * ak;
    vbk[k - 1] = 1. * bk;
    vck[k - 1] = 1. * ck;
    vdk[k - 1] = 1. * dk;

    if (FULL_RANGE || USE_ANGACC)
      angnorm[k - 1] = angular_weights[k - 1];
    else
      angnorm[k - 1] = angular_weights[k - 1];
      // angnorm[k - 1] = getFintegral(cosKLL, cosKUL, cosLLL, cosLUL, hphiLL,
      //                               hphiUL, 0, 0, 0, k);

#if DEBUG
    vfk[k - 1] = 1. * fk;
#endif
  }

#if DEBUG
  if (DEBUG > 3 && get_global_id(0) == DEBUG_EVT) {
    printf("\nANGULAR PART       :  n            a            b            c   "
           "         d            f            angnorm\n");
    for (int k = 0; k < 10; k++) {
      printf("               (%d) : %+.8f  %+.8f  %+.8f  %+.8f  %+.8f  %+.8f  "
             "%+.8f\n",
             k, vnk[k], vak[k], vbk[k], vck[k], vdk[k], vfk[k], angnorm[k]);
    }
  }
#endif

  // }}}

  // Compute pdf integral {{{

  ftype intBBar[2] = {0., 0.};
  if (USE_TIMEACC == 0) {
    // Here we can use the simplest 4xPi integral of the pdf since there are no
    // resolution effects
    integralSimple(intBBar, vnk, vak, vbk, vck, vdk, angnorm, G, DG, DM, tLL,
                   tUL);
  } else if (USE_TIMERES) {
    // This integral works for all decay times, remember delta_t != 0.
#if FAST_INTEGRAL
    integralSpline(intBBar, vnk, vak, vbk, vck, vdk, angnorm, G, DG, DM,
                   delta_t, tLL, tUL, t_offset, coeffs);
#else
    const int simon_j = sigma_t / (SIGMA_T / 80);
    integralFullSpline(
        intBBar, vnk, vak, vbk, vck, vdk, angnorm, G, DG, DM,
        /* what should beused */ delta_t,
        /* HD-fitter used */ // parabola((0.5+simon_j)*(SIGMA_T/80),
                             //  sigma_offset, sigma_slope, sigma_curvature),
        tLL, tUL, t_offset, coeffs);
#endif
  } else {
    integralFullSpline_wores(intBBar, vnk, vak, vbk, vck, vdk, angnorm, G, DG,
                             DM, tLL, tUL, coeffs);
  }
  const ftype intB = intBBar[0];
  const ftype intBbar = intBBar[1];

  // }}}

  // Cooking the output {{{

  ftype num = 1.0;
  ftype den = 1.0;
  num = (1 + tagOS * (1 - 2 * omegaOSB)) * (1 + tagSS * (1 - 2 * omegaSSB)) *
            pdfB +
        (1 - tagOS * (1 - 2 * omegaOSBbar)) *
            (1 - tagSS * (1 - 2 * omegaSSBbar)) * pdfBbar;
  den = (1 + tagOS * (1 - 2 * omegaOSB)) * (1 + tagSS * (1 - 2 * omegaSSB)) *
            intB +
        (1 - tagOS * (1 - 2 * omegaOSBbar)) *
            (1 - tagSS * (1 - 2 * omegaSSBbar)) * intBbar;

#if FAST_INTEGRAL
  num = num;
#else
  num = num / 4;
  den = den / 4; // this is only to agree with Peilian
#endif

#if DEBUG
  if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT) {
    printf("\nRESULT             : <  pdf/ipdf = %+.8f  >\n", num / den);
    if (DEBUG >= 2) {
      printf("                   : pdf=%+.8f  ipdf=%+.8f\n", num, den);
      printf("                   : pdfB=%+.8f  pdBbar=%+.8f  ipdfB=%+.8f  "
             "ipdfBbar=%+.8f\n",
             pdfB, pdfBbar, intB, intBbar);
    }
  }
#endif

  // }}}

  // }}}

  return num / den;
}

// vim:foldmethod=marker
