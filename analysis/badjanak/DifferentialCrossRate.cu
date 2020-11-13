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



////////////////////////////////////////////////////////////////////////////////











////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



WITHIN_KERNEL
${ftype} getDiffRate( ${ftype} *data,
                    // Time-dependent angular distribution
                    ${ftype} G, ${ftype} DG, ${ftype} DM, ${ftype} CSP,
                    ${ftype} ASlon, ${ftype} APlon, ${ftype} APpar, ${ftype} APper,
                    ${ftype} pSlon, ${ftype} pPlon, ${ftype} pPpar, ${ftype} pPper,
                    ${ftype} dSlon, ${ftype} dPlon, ${ftype} dPpar, ${ftype} dPper,
                    ${ftype} lSlon, ${ftype} lPlon, ${ftype} lPpar, ${ftype} lPper,
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
                    int USE_FK, int USE_ANGACC, int USE_TIMEACC,
                    int USE_TIMEOFFSET, int SET_TAGGING, int USE_TIMERES
                  )
{
  #if DEBUG
  //printf("EVT = %d (%d)\n", get_global_id(0), DEBUG_EVT);
  if ( DEBUG > 4 && get_global_id(0) == DEBUG_EVT )
  {
    printf("*USE_FK            : %d\n", USE_FK);
    printf("*USE_ANGACC        : %d\n", USE_ANGACC);
    printf("*USE_TIMEACC       : %d\n", USE_TIMEACC);
    printf("*USE_TIMEOFFSET    : %d\n", USE_TIMEOFFSET);
    printf("*USE_TIMERES       : %d\n", USE_TIMERES);
    printf("*SET_TAGGING       : %d [0:perfect,1:real,2:true]\n", SET_TAGGING);
    printf("G                  : %+.16f\n", G);
    printf("DG                 : %+.16f\n", DG);
    printf("DM                 : %+.16f\n", DM);
    printf("CSP                : %+.16f\n", CSP);
    printf("ASlon              : %+.16f\n", ASlon);
    printf("APlon              : %+.16f\n", APlon);
    printf("APpar              : %+.16f\n", APpar);
    printf("APper              : %+.16f\n", APper);
    printf("pSlon              : %+.16f\n", pSlon);
    printf("pPlon              : %+.16f\n", pPlon);
    printf("pPpar              : %+.16f\n", pPpar);
    printf("pPper              : %+.16f\n", pPper);
    printf("dSlon              : %+.16f\n", dSlon);
    printf("dPlon              : %+.16f\n", dPlon);
    printf("dPper              : %+.16f\n", dPper);
    printf("dPpar              : %+.16f\n", dPpar);
    printf("lSlon              : %+.16f\n", lSlon);
    printf("lPlon              : %+.16f\n", lPlon);
    printf("lPper              : %+.16f\n", lPper);
    printf("lPpar              : %+.16f\n", lPpar);
    printf("tLL                : %+.16f\n", tLL);
    printf("tUL                : %+.16f\n", tUL);
    printf("mu                 : %+.16f\n", mu);
    printf("sigma_offset       : %+.16f\n", sigma_offset);
    printf("sigma_slope        : %+.16f\n", sigma_slope);
    printf("sigma_curvature    : %+.16f\n", sigma_curvature);
    printf("COEFFS             : %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
    printf("                     %+.16f\t%+.16f\t%+.16f\t%+.16f\n",
            coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  }
  #endif



  // Variables -----------------------------------------------------------------
  //     Make sure that the input it's in this order.
  //     lalala
  ${ftype} cosK       = data[0];                      // Time-angular distribution
  ${ftype} cosL       = data[1];
  ${ftype} hphi       = data[2];
  ${ftype} time       = data[3];

  //${ftype} sigma_t    = 0.04554;                            // Time resolution
  ${ftype} sigma_t    = data[4];                              // Time resolution

  ${ftype} qOS        = data[5];                                      // Tagging
  ${ftype} qSS        = data[6];
  ${ftype} etaOS 	  	= data[7];
  ${ftype} etaSS 	    = data[8];

  #if DEBUG
  if ( DEBUG > 99 && ( (time>=tUL) || (time<=tLL) ) )
  {
    printf("WARNING            : Event with time not within [%.4f,%.4f].\n",
           tLL, tUL);
  }
  #endif

  #if DEBUG
  if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT )
  {
    printf("\nINPUT              : cosK=%+.8f  cosL=%+.8f  hphi=%+.8f  time=%+.8f\n",
           cosK,cosL,hphi,time);
    printf("                   : sigma_t=%+.8f  qOS=%+.8f  qSS=%+.8f  etaOS=%+.8f  etaSS=%+.8f\n",
           sigma_t,qOS,qSS,etaOS,etaSS);
  }
  #endif


  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set sigma_t=0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  ${ctype} exp_p, exp_m, exp_i;
  ${ftype} t_offset = 0.0; ${ftype} delta_t = sigma_t;
  ${ftype} sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;

  if (USE_TIMEOFFSET)
  {
    t_offset = parabola(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  }

  if (USE_TIMERES) // use_per_event_res
  {
    delta_t  = parabola(sigma_t, sigma_offset, sigma_slope, sigma_curvature);
  }

  #if DEBUG
  if (DEBUG > 3 && get_global_id(0) == DEBUG_EVT )
  {
    printf("\nTIME RESOLUTION    : delta_t=%.8f\n", delta_t);
  }
  #endif


  if ( delta_t == 0 ) // MC samples need to solve some problems
  {
    exp_p = expconv(time-t_offset, G + 0.5*DG, 0., delta_t);
    exp_m = expconv(time-t_offset, G - 0.5*DG, 0., delta_t);
    exp_i = expconv(time-t_offset,          G, DM, delta_t);
  }
  else
  {
    exp_p = expconv(time-t_offset, G + 0.5*DG, 0., delta_t);
    exp_m = expconv(time-t_offset, G - 0.5*DG, 0., delta_t);
    exp_i = expconv(time-t_offset,          G, DM, delta_t);
  }
  //printf("                   : exp_p=%+.8f%+.8fi   exp_m=%+.8f%+.8fi   exp_i=%+.8f%+.8fi\n", exp_p.x, exp_p.y, exp_m.x, exp_m.y, exp_i.x, exp_i.y);

  // ${ftype} ta = pycuda::real(0.5*(exp_m + exp_p));     // cosh = (exp_m + exp_p)/2
  // ${ftype} tb = pycuda::real(0.5*(exp_m - exp_p));     // sinh = (exp_m - exp_p)/2
  // ${ftype} tc = pycuda::real(exp_i);                        // exp_i = cos + I*sin
  // ${ftype} td = pycuda::imag(exp_i);                        // exp_i = cos + I*sin
  ${ftype} ta = 0.5*(exp_m.x+exp_p.x);
  ${ftype} tb = 0.5*(exp_m.x-exp_p.x);
  ${ftype} tc = exp_i.x;
  ${ftype} td = exp_i.y;
  #if FAST_INTEGRAL
    ta *= sqrt(2*M_PI); tb *= sqrt(2*M_PI); tc *= sqrt(2*M_PI); td *= sqrt(2*M_PI);
  #endif
  #if DEBUG
  if (DEBUG >= 3 && get_global_id(0) == DEBUG_EVT)
  {
    printf("\nTIME TERMS         : ta=%.16f  tb=%.16f  tc=%.16f  td=%.16f\n",
           ta,tb,tc,td);
    printf("\nTIME TERMS         : exp_m=%.16f  exp_p=%.16f  exp_i=%.16f  exp_i=%.16f\n",
           sqrt(2*M_PI)*exp_m.x,sqrt(2*M_PI)*exp_p.x,sqrt(2*M_PI)*exp_i.x,exp_i.y);
  }
  #endif

  // Flavor tagging ------------------------------------------------------------
  ${ftype} omegaOSB = 0; ${ftype} omegaOSBbar = 0; ${ftype} tagOS = 0;
  ${ftype} omegaSSB = 0; ${ftype} omegaSSBbar = 0; ${ftype} tagSS = 0;


  if (SET_TAGGING == 1) // DATA
  {
    if (qOS != 0) { tagOS = qOS/fabs(qOS);}
    if (qSS != 0) { tagSS = qSS/fabs(qSS);}
    omegaOSB    = get_omega(etaOS, +1, p0_os, p1_os, p2_os, dp0_os, dp1_os, dp2_os, eta_bar_os);
    omegaOSBbar = get_omega(etaOS, -1, p0_os, p1_os, p2_os, dp0_os, dp1_os, dp2_os, eta_bar_os);
    omegaSSB    = get_omega(etaSS, +1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss, dp2_ss, eta_bar_ss);
    omegaSSBbar = get_omega(etaSS, -1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss, dp2_ss, eta_bar_ss);
  }
  else if (SET_TAGGING == 0) // PERFECT, MC
  {
    if (qOS != 0) {tagOS = qOS/fabs(qOS);}
    if (qSS != 0) {tagSS = qSS/fabs(qSS);}
  }
  else //TRUE
  {
    tagOS = 0.0;
    tagSS = 0.0;
  }

  // Print warning if tagOS|tagSS == 0
  #if DEBUG
  if ( DEBUG > 99 && ( (tagOS == 0)|(tagSS == 0) ) )
  {
    printf("This event is not tagged!\n");
  }
  #endif

  #if DEBUG
  if ( DEBUG > 3  && get_global_id(0) == DEBUG_EVT )
  {
    printf("\nFLAVOR TAGGING     : delta_t=%.16f\n", delta_t);
    printf("                   : tagOS=%.8f, tagSS=%.8f\n",
           tagOS, tagSS);
    printf("                   : omegaOSB=%.8f, omegaOSBbar=%.8f\n",
           omegaOSB, omegaOSBbar);
    printf("                   : omegaSSB=%.8f, omegaSSBbar=%.8f\n",
           omegaSSB, omegaSSBbar);
  }
  #endif

  // Decay-time acceptance -----------------------------------------------------
  //     To get rid of decay-time acceptance set USE_TIMEACC to False. If True
  //     then calcTimeAcceptance locates the time bin of the event and returns
  //     the value of the cubic spline.
  ${ftype} dta = 1.0;
  if (USE_TIMEACC)
  {
    dta = calcTimeAcceptance(time, coeffs, tLL, tUL);
  }

  // Compute per event pdf -----------------------------------------------------
  ${ftype} vnk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  #if DEBUG
    ${ftype} vfk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  #endif
  ${ftype} vak[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  ${ftype} vbk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  ${ftype} vck[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  ${ftype} vdk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  ${ftype} nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
  ${ftype} pdfB = 0.0; ${ftype} pdfBbar = 0.0;

  for(int k = 1; k <= 10; k++)
  {
    nk = getN(APlon,ASlon,APpar,APper,CSP,k);
    if (USE_FK)
    {
      #if FAST_INTEGRAL
        fk = getF(cosK,cosL,hphi,k);
      #else
        fk = ( 9.0/(16.0*M_PI) )*getF(cosK,cosL,hphi,k);
      #endif
    }
    else
    {
      fk = TRISTAN[k-1]; // these are 0s or 1s
    }

    ak = getA(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getB(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getC(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getD(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);

    // WARNING: now I know if is Bs or Bd with DM, but I should change it asap (its clearly misleading)
    //if (fabs(qOS) == 511) // Bd pdf
    if (fabs(qOS) != 511) // Bs PDF
    {
      hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);
      hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);
      if (get_global_id(0) <=1){
      printf("voy por aqui jefe");
    }
    }
    else
    {
      // this is the Bd2JpsiKstar p.d.f
      hk_B = ak*ta + ck*tc;
      if ( (k==4) || (k==6)  || (k==9) )
      {
        hk_Bbar = tagOS*ak*ta + tagOS*ck*tc;
      }
      else
      {
        hk_Bbar = ak*ta + ck*tc;
      }
    }
    #if FAST_INTEGRAL
      hk_B = 3./(4.*M_PI)*hk_B;
      hk_Bbar = 3./(4.*M_PI)*hk_Bbar;
    #endif
    pdfB += nk*fk*hk_B; pdfBbar += nk*fk*hk_Bbar;
    vnk[k-1] = 1.*nk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;

    #if DEBUG
      vfk[k-1] = 1.*fk;
    #endif
  }

  #if DEBUG
  if ( DEBUG > 3  && get_global_id(0) == DEBUG_EVT )
  {
    printf("\nANGULAR PART       :  n            a            b            c            d            f            ang_acc\n");
    for(int k = 0; k < 10; k++)
    {
      printf("               (%d) : %+.16f  %+.16f  %+.16f  %+.16f  %+.16f  %+.16f  %+.16f\n",
             k,vnk[k], vak[k], vbk[k], vck[k], vdk[k], vfk[k], angular_weights[k]);
    }
  }
  #endif


  // Compute pdf integral ------------------------------------------------------
  ${ftype} intBBar[2] = {0.,0.};
  if ( (delta_t == 0) & (USE_TIMEACC == 0) )
  {
    // Here we can use the simplest 4xPi integral of the pdf since there are no
    // resolution effects
    integralSimple(intBBar,
                   vnk, vak, vbk, vck, vdk, angular_weights, G, DG, DM, tLL, tUL);
  }
  else
  {
    // This integral works for all decay times, remember delta_t != 0.
    #if FAST_INTEGRAL
    // if ( get_global_id(0) == DEBUG_EVT)
    // {
    //   printf("fast integral");
    // }
      integralSpline( intBBar,
                       vnk, vak, vbk, vck, vdk,
                       angular_weights, G, DG, DM,
                       delta_t,
                       tLL, tUL, t_offset,
                       coeffs);
    #else
    // if ( get_global_id(0) == DEBUG_EVT)
    // {
    //   printf("slow integral");
    // }
    int simon_j = sigma_t/(SIGMA_T/80);
    // if (DEBUG >= 1)
    // {
    //   if ( get_global_id(0) == DEBUG_EVT){
    // {
    //   printf("simon_j = %+d = round(%+.8f)\n", simon_j, sigma_t/(SIGMA_T/80) );
    // }
      integralFullSpline(intBBar,
                       vnk, vak, vbk, vck, vdk,
                       angular_weights, G, DG, DM,
                       delta_t,
                       //sigma_t,
                       //parabola(  (0.5+simon_j)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
                       tLL, tUL, t_offset,
                       coeffs);
     #endif
  }
  ${ftype} intB = intBBar[0]; ${ftype} intBbar = intBBar[1];


  // Cooking the output --------------------------------------------------------
  ${ftype} num = 1.0; ${ftype} den = 1.0;
  num = dta*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
        );
  den = 1.0*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * intB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * intBbar
        );
  num = num/4; den = den/4; // this is only to agree with Peilian

  #if DEBUG
  if ( DEBUG >= 1  && get_global_id(0) == DEBUG_EVT)
  {
    printf("\nRESULT             : <  pdf/ipdf = %+.16f  >\n",
           num/den);
    if ( DEBUG >= 2 )
    {
     printf("                   : pdf=%+.16f  ipdf=%+.16f\n",
            num,den);
     printf("                   : pdfB=%+.16f  pdBbar=%+.16f  ipdfB=%+.16f  ipdfBbar=%+.16f\n",
            pdfB,pdfBbar,intB,intBbar);
    }
  }
  #endif
  // That's all folks!
  return num/den;
}



////////////////////////////////////////////////////////////////////////////////
