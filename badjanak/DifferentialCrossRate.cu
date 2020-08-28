////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//   Created: 2019-01-25                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////



// these lines should be moved to a taggin.cu file

WITHIN_KERNEL
${ftype} get_omega(${ftype} eta, ${ftype} tag,
                 ${ftype} p0,  ${ftype} p1, ${ftype} p2,
                 ${ftype} dp0, ${ftype} dp1, ${ftype} dp2,
                 ${ftype} eta_bar)
{
    ${ftype} result = 0;
    result += (p0 + tag*0.5*dp0);
    result += (p1 + tag*0.5*dp1)*(eta - eta_bar);
    result += (p2 + tag*0.5*dp2)*(eta - eta_bar)*(eta - eta_bar);

    if(result < 0.0)
    {
      return 0;
    }
    return result;
}



WITHIN_KERNEL ${ftype} get_int_ta_spline(${ftype} delta_t,${ftype} G,${ftype} DM,${ftype} DG,${ftype} a,${ftype} b,${ftype} c,${ftype} d,${ftype} t_0,${ftype} t_1)
{
    ${ftype} G_sq = G*G;
    ${ftype} G_cub = G_sq*G;
    ${ftype} DG_sq = DG*DG;
    ${ftype} DG_cub = DG_sq*DG;
    ${ftype} delta_t_sq = delta_t*delta_t;
    ${ftype} t_0_sq = t_0*t_0;
    ${ftype} t_0_cub = t_0_sq*t_0;
    ${ftype} t_1_sq = t_1*t_1;
    ${ftype} t_1_cub = t_1_sq*t_1;

    return -0.5*sqrt(2.)*sqrt(delta_t)
    *(-2*a*pow(DG - 2*G, 4.)*pow(DG + 2*G, 3.)*(-exp(-0.5*t_1*(DG + 2*G)) + exp(-0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq)
     + 2*a*pow(DG - 2*G, 3.)*pow(DG + 2*G, 4.)*(exp(0.5*t_0*(DG - 2*G)) - exp(0.5*t_1*(DG - 2*G)))
     + b*pow(DG - 2*G, 4.)*pow(DG + 2*G, 2.)*(-2*(DG*t_0 + 2*G*t_0 + 2.)*exp(0.5*t_1*(DG + 2*G)) + 2*(DG*t_1 + 2*G*t_1 + 2.)*exp(0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq - 0.5*(DG + 2*G)*(t_0 + t_1))
     - 2*b*pow(DG - 2*G, 2.)*pow(DG + 2*G, 4.)*((-DG*t_0 + 2*G*t_0 + 2.)*exp(0.5*DG*t_0 + G*t_1) + (DG*t_1 - 2*G*t_1 - 2.)*exp(0.5*DG*t_1 + G*t_0))*exp(-G*(t_0 + t_1))
     + 2*c*pow(DG - 2*G, 4.)*(DG + 2*G)*(-(DG_sq*t_0_sq + 4*DG*t_0*(G*t_0 + 1) + 4*G_sq*t_0_sq + 8*G*t_0 + 8)*exp(0.5*t_1*(DG + 2*G)) + (DG_sq*t_1_sq + 4*DG*t_1*(G*t_1 + 1) + 4*G_sq*t_1_sq + 8*G*t_1 + 8)*exp(0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq - 0.5*(DG + 2*G)*(t_0 + t_1))
     - 2*c*(DG - 2*G)*pow(DG + 2*G, 4.)*(-(DG_sq*t_0_sq - 4*DG*t_0*(G*t_0 + 1) + 4*G_sq*t_0_sq + 8*G*t_0 + 8)*exp(0.5*DG*t_0 + G*t_1) + (DG_sq*t_1_sq - 4*DG*t_1*(G*t_1 + 1) + 4*G_sq*t_1_sq + 8*G*t_1 + 8)*exp(0.5*DG*t_1 + G*t_0))*exp(-G*(t_0 + t_1))
     + 2*d*pow(DG - 2*G, 4.)*((-DG_cub*t_0_cub - 6*DG_sq*t_0_sq*(G*t_0 + 1) - 12*DG*t_0*(G_sq*t_0_sq + 2*G*t_0 + 2.) - 8*G_cub*t_0_cub - 24*G_sq*t_0_sq - 48*G*t_0 + 48*exp(0.5*t_0*(DG + 2*G)) - 48)*exp(-0.5*t_0*(DG + 2*G)) + (DG_cub*t_1_cub + 6*DG_sq*t_1_sq*(G*t_1 + 1) + 12*DG*t_1*(G_sq*t_1_sq + 2*G*t_1 + 2.) + 8*G_cub*t_1_cub + 24*G_sq*t_1_sq + 48*G*t_1 - 48*exp(0.5*t_1*(DG + 2*G)) + 48)*exp(-0.5*t_1*(DG + 2*G)))*exp(DG*G*delta_t_sq)
     + 2*d*pow(DG + 2*G, 4.)*(((DG_cub*t_0_cub - 6*DG_sq*t_0_sq*(G*t_0 + 1) + 12*DG*t_0*(G_sq*t_0_sq + 2*G*t_0 + 2.) - 8*G_cub*t_0_cub - 24*G_sq*t_0_sq - 48*G*t_0 - 48)*exp(0.5*DG*t_0) + 48*exp(G*t_0))*exp(-G*t_0) - ((DG_cub*t_1_cub - 6*DG_sq*t_1_sq*(G*t_1 + 1) + 12*DG*t_1*(G_sq*t_1_sq + 2*G*t_1 + 2.) - 8*G_cub*t_1_cub - 24*G_sq*t_1_sq - 48*G*t_1 - 48)*exp(0.5*DG*t_1) + 48*exp(G*t_1))*exp(-G*t_1)))
    *exp(0.125*delta_t_sq*pow(DG - 2*G, 2.))/pow(DG_sq - 4*G_sq, 4.);
}

WITHIN_KERNEL ${ftype} get_int_tb_spline(${ftype} delta_t,${ftype} G,${ftype} DM,${ftype} DG,${ftype} a,${ftype} b,${ftype} c,${ftype} d,${ftype} t_0,${ftype} t_1)
{
    ${ftype} G_sq = G*G;
    ${ftype} G_cub = G_sq*G;
    ${ftype} DG_sq = DG*DG;
    ${ftype} DG_cub = DG_sq*DG;
    ${ftype} delta_t_sq = delta_t*delta_t;
    ${ftype} t_0_sq = t_0*t_0;
    ${ftype} t_0_cub = t_0_sq*t_0;
    ${ftype} t_1_sq = t_1*t_1;
    ${ftype} t_1_cub = t_1_sq*t_1;

    return 0.5*sqrt(2.)*sqrt(delta_t)
    *(-2*a*pow(DG - 2*G, 4.)*pow(DG + 2*G, 3.)*(-exp(-0.5*t_1*(DG + 2*G)) + exp(-0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq)
    - 2*a*pow(DG - 2*G, 3.)*pow(DG + 2*G, 4.)*(exp(0.5*t_0*(DG - 2*G)) - exp(0.5*t_1*(DG - 2*G))) + b*pow(DG - 2*G, 4.)*pow(DG + 2*G, 2.)*(-2*(DG*t_0 + 2*G*t_0 + 2.)*exp(0.5*t_1*(DG + 2*G)) + 2*(DG*t_1 + 2*G*t_1 + 2.)*exp(0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq - 0.5*(DG + 2*G)*(t_0 + t_1))
    + 2*b*pow(DG - 2*G, 2.)*pow(DG + 2*G, 4.)*((-DG*t_0 + 2*G*t_0 + 2.)*exp(0.5*DG*t_0 + G*t_1) + (DG*t_1 - 2*G*t_1 - 2.)*exp(0.5*DG*t_1 + G*t_0))*exp(-G*(t_0 + t_1))
    + 2*c*pow(DG - 2*G, 4.)*(DG + 2*G)*(-(DG_sq*t_0_sq + 4*DG*t_0*(G*t_0 + 1) + 4*G_sq*t_0_sq + 8*G*t_0 + 8)*exp(0.5*t_1*(DG + 2*G)) + (DG_sq*t_1_sq + 4*DG*t_1*(G*t_1 + 1) + 4*G_sq*t_1_sq + 8*G*t_1 + 8)*exp(0.5*t_0*(DG + 2*G)))*exp(DG*G*delta_t_sq - 0.5*(DG + 2*G)*(t_0 + t_1))
    + 2*c*(DG - 2*G)*pow(DG + 2*G, 4.)*(-(DG_sq*t_0_sq - 4*DG*t_0*(G*t_0 + 1) + 4*G_sq*t_0_sq + 8*G*t_0 + 8)*exp(0.5*DG*t_0 + G*t_1) + (DG_sq*t_1_sq - 4*DG*t_1*(G*t_1 + 1) + 4*G_sq*t_1_sq + 8*G*t_1 + 8)*exp(0.5*DG*t_1 + G*t_0))*exp(-G*(t_0 + t_1))
    + 2*d*pow(DG - 2*G, 4.)*((-DG_cub*t_0_cub - 6*DG_sq*t_0_sq*(G*t_0 + 1) - 12*DG*t_0*(G_sq*t_0_sq + 2*G*t_0 + 2.) - 8*G_cub*t_0_cub - 24*G_sq*t_0_sq - 48*G*t_0 + 48*exp(0.5*t_0*(DG + 2*G)) - 48)*exp(-0.5*t_0*(DG + 2*G)) + (DG_cub*t_1_cub + 6*DG_sq*t_1_sq*(G*t_1 + 1) + 12*DG*t_1*(G_sq*t_1_sq + 2*G*t_1 + 2.) + 8*G_cub*t_1_cub + 24*G_sq*t_1_sq + 48*G*t_1 - 48*exp(0.5*t_1*(DG + 2*G)) + 48)*exp(-0.5*t_1*(DG + 2*G)))*exp(DG*G*delta_t_sq)
    - 2*d*pow(DG + 2*G, 4.)*(((DG_cub*t_0_cub - 6*DG_sq*t_0_sq*(G*t_0 + 1) + 12*DG*t_0*(G_sq*t_0_sq + 2*G*t_0 + 2.) - 8*G_cub*t_0_cub - 24*G_sq*t_0_sq - 48*G*t_0 - 48)*exp(0.5*DG*t_0) + 48*exp(G*t_0))*exp(-G*t_0) - ((DG_cub*t_1_cub - 6*DG_sq*t_1_sq*(G*t_1 + 1) + 12*DG*t_1*(G_sq*t_1_sq + 2*G*t_1 + 2.) - 8*G_cub*t_1_cub - 24*G_sq*t_1_sq - 48*G*t_1 - 48)*exp(0.5*DG*t_1) + 48*exp(G*t_1))*exp(-G*t_1)))
    *exp(0.125*delta_t_sq*pow(DG - 2*G, 2.))/pow(DG_sq - 4*G_sq, 4.);
}

WITHIN_KERNEL ${ftype} get_int_tc_spline(${ftype} delta_t,${ftype} G,${ftype} DM,${ftype} DG,${ftype} a,${ftype} b,${ftype} c,${ftype} d,${ftype} t_0,${ftype} t_1)
{
    ${ftype} G_sq = G*G;
    ${ftype} G_cub = G_sq*G;
    ${ftype} DM_sq = DM*DM;
    ${ftype} DM_fr = DM_sq*DM_sq;
    ${ftype} DM_sx = DM_fr*DM_sq;
    ${ftype} G_fr = G_sq*G_sq;
    ${ftype} delta_t_sq = delta_t*delta_t;
    ${ftype} t_0_sq = t_0*t_0;
    ${ftype} t_0_cub = t_0_sq*t_0;
    ${ftype} t_1_sq = t_1*t_1;
    ${ftype} t_1_cub = t_1_sq*t_1;
    ${ftype} exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    ${ftype} exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    ${ftype} exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    ${ftype} exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);

    return (a*pow(DM_sq + G_sq, 3.)*(-DM*exp_0_sin_1_term + DM*exp_1_sin_0_term - G*exp_0_cos_1_term + G*exp_1_cos_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM*((DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_sin_0_term - (DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_sin_1_term) + (DM_sq*(G*t_0 - 1) + G_sq*(G*t_0 + 1))*exp_1_cos_0_term - (DM_sq*(G*t_1 - 1) + G_sq*(G*t_1 + 1))*exp_0_cos_1_term)
    + c*(DM_sq + G_sq)*(DM*((DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_sin_0_term - (DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_sin_1_term) + (DM_fr*t_0*(G*t_0 - 2.) + 2*DM_sq*G*(G_sq*t_0_sq - 3.) + G_cub*(G_sq*t_0_sq + 2*G*t_0 + 2.))*exp_1_cos_0_term - (DM_fr*t_1*(G*t_1 - 2.) + 2*DM_sq*G*(G_sq*t_1_sq - 3.) + G_cub*(G_sq*t_1_sq + 2*G*t_1 + 2.))*exp_0_cos_1_term)
    + d*(DM*((DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_sin_0_term - (DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_sin_1_term) + (DM_sx*t_0_sq*(G*t_0 - 3.) + 3*DM_fr*(G_cub*t_0_cub - G_sq*t_0_sq - 6*G*t_0 + 2.) + 3*DM_sq*G_sq*(G_cub*t_0_cub + G_sq*t_0_sq - 4*G*t_0 - 12.) + G_fr*(G_cub*t_0_cub + 3*G_sq*t_0_sq + 6*G*t_0 + 6))*exp_1_cos_0_term - (DM_sx*t_1_sq*(G*t_1 - 3.) + 3*DM_fr*(G_cub*t_1_cub - G_sq*t_1_sq - 6*G*t_1 + 2.) + 3*DM_sq*G_sq*(G_cub*t_1_cub + G_sq*t_1_sq - 4*G*t_1 - 12.) + G_fr*(G_cub*t_1_cub + 3*G_sq*t_1_sq + 6*G*t_1 + 6))*exp_0_cos_1_term))*sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}

WITHIN_KERNEL ${ftype} get_int_td_spline(${ftype} delta_t,${ftype} G,${ftype} DM,${ftype} DG,${ftype} a,${ftype} b,${ftype} c,${ftype} d,${ftype} t_0,${ftype} t_1)
{
    ${ftype} G_sq = G*G;
    ${ftype} G_cub = G_sq*G;
    ${ftype} G_fr = G_sq*G_sq;
    ${ftype} G_fv = G_cub*G_sq;
    ${ftype} DM_sq = DM*DM;
    ${ftype} DM_fr = DM_sq*DM_sq;
    ${ftype} DM_sx = DM_fr*DM_sq;
    ${ftype} delta_t_sq = delta_t*delta_t;
    ${ftype} t_0_sq = t_0*t_0;
    ${ftype} t_0_cub = t_0_sq*t_0;
    ${ftype} t_1_sq = t_1*t_1;
    ${ftype} t_1_cub = t_1_sq*t_1;
    ${ftype} exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    ${ftype} exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    ${ftype} exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    ${ftype} exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);


    return -(a*pow(DM_sq + G_sq, 3.)*(DM*exp_0_cos_1_term - DM*exp_1_cos_0_term - G*exp_0_sin_1_term + G*exp_1_sin_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM_sq*G*t_0*exp_1_sin_0_term - DM_sq*G*t_1*exp_0_sin_1_term + DM_sq*exp_0_sin_1_term - DM_sq*exp_1_sin_0_term - DM*(DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_cos_0_term + DM*(DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_cos_1_term + G_cub*t_0*exp_1_sin_0_term - G_cub*t_1*exp_0_sin_1_term - G_sq*exp_0_sin_1_term + G_sq*exp_1_sin_0_term)
    + c*(DM_sq + G_sq)*(DM_fr*G*t_0_sq*exp_1_sin_0_term - DM_fr*G*t_1_sq*exp_0_sin_1_term - 2*DM_fr*t_0*exp_1_sin_0_term + 2*DM_fr*t_1*exp_0_sin_1_term + 2*DM_sq*G_cub*t_0_sq*exp_1_sin_0_term - 2*DM_sq*G_cub*t_1_sq*exp_0_sin_1_term + 6*DM_sq*G*exp_0_sin_1_term - 6*DM_sq*G*exp_1_sin_0_term - DM*(DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_cos_0_term + DM*(DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_cos_1_term + G_fv*t_0_sq*exp_1_sin_0_term - G_fv*t_1_sq*exp_0_sin_1_term + 2*G_fr*t_0*exp_1_sin_0_term - 2*G_fr*t_1*exp_0_sin_1_term - 2*G_cub*exp_0_sin_1_term + 2*G_cub*exp_1_sin_0_term)
    + d*(DM_sx*G*t_0_cub*exp_1_sin_0_term - DM_sx*G*t_1_cub*exp_0_sin_1_term - 3*DM_sx*t_0_sq*exp_1_sin_0_term + 3*DM_sx*t_1_sq*exp_0_sin_1_term + 3*DM_fr*G_cub*t_0_cub*exp_1_sin_0_term - 3*DM_fr*G_cub*t_1_cub*exp_0_sin_1_term - 3*DM_fr*G_sq*t_0_sq*exp_1_sin_0_term + 3*DM_fr*G_sq*t_1_sq*exp_0_sin_1_term - 18*DM_fr*G*t_0*exp_1_sin_0_term + 18*DM_fr*G*t_1*exp_0_sin_1_term - 6*DM_fr*exp_0_sin_1_term + 6*DM_fr*exp_1_sin_0_term + 3*DM_sq*G_fv*t_0_cub*exp_1_sin_0_term - 3*DM_sq*G_fv*t_1_cub*exp_0_sin_1_term + 3*DM_sq*G_fr*t_0_sq*exp_1_sin_0_term - 3*DM_sq*G_fr*t_1_sq*exp_0_sin_1_term - 12*DM_sq*G_cub*t_0*exp_1_sin_0_term + 12*DM_sq*G_cub*t_1*exp_0_sin_1_term + 36*DM_sq*G_sq*exp_0_sin_1_term - 36*DM_sq*G_sq*exp_1_sin_0_term - DM*(DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_cos_0_term + DM*(DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_cos_1_term + pow(G, 7)*t_0_cub*exp_1_sin_0_term - pow(G, 7)*t_1_cub*exp_0_sin_1_term + 3*pow(G, 6)*t_0_sq*exp_1_sin_0_term - 3*pow(G, 6)*t_1_sq*exp_0_sin_1_term + 6*G_fv*t_0*exp_1_sin_0_term - 6*G_fv*t_1*exp_0_sin_1_term - 6*G_fr*exp_0_sin_1_term + 6*G_fr*exp_1_sin_0_term))
    *sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}


WITHIN_KERNEL
void integralSpline( ${ftype} result[2],
                     ${ftype} vn[10], ${ftype} va[10],${ftype} vb[10], ${ftype} vc[10],${ftype} vd[10],
                     GLOBAL_MEM ${ftype} *norm, ${ftype} G, ${ftype} DG, ${ftype} DM,
                     ${ftype} delta_t,
                     ${ftype} tLL, ${ftype} tUL,
                     ${ftype} t_offset,
                     GLOBAL_MEM ${ftype} *coeffs)
{
  //int bin0 = 0;
  // ${ftype} tS = tLL-t_offset;
  ${ftype} tS = 0;
  // ${ftype} tE = KNOTS[bin0+1]-t_offset;
  ${ftype} tE = 0;
  for(int bin = 0; bin < NKNOTS; bin++)
  {
    if (bin == NKNOTS-1){
      tS = KNOTS[bin+0];
      tE = tUL-t_offset;
    }
    else{
      tS = KNOTS[bin+0] - t_offset;
      tE = KNOTS[bin+1] - t_offset;
    }
    // if ( get_global_id(0) == 0){
    // printf("integral: bin %d (%f,%f)\n", bin ,tS, tE);
    // }

    ${ftype} c0 = getCoeff(coeffs,bin,0);
    ${ftype} c1 = getCoeff(coeffs,bin,1);
    ${ftype} c2 = getCoeff(coeffs,bin,2);
    ${ftype} c3 = getCoeff(coeffs,bin,3);

    ${ftype} ta = get_int_ta_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ${ftype} tb = get_int_tb_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ${ftype} tc = get_int_tc_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ${ftype} td = get_int_td_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);

    for(int k=0; k<10; k++)
    {
      result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
      result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
    }
  }
}







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

  ${ftype} sigma_t    = data[4];                                // Time resolution

  ${ftype} qOS        = data[5];                                        // Tagging
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
  ${ftype} vfk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
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
    if (DM != 0) // Bd pdf
    {
      hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);
      hk_Bbar = (ak*ta + bk*tb - ck*tc - dk*td);
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
    vnk[k-1] = 1.*nk; vfk[k-1] = 1.*fk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
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
                       //delta_t,
                       //sigma_t,
                       //parabola(  (0.5+simon_j-1)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
                       parabola(  (0.5+simon_j)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
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
