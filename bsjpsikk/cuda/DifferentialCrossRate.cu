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
// Include headers /////////////////////////////////////////////////////////////

#include <stdio.h>
#include <math.h>
#include <pycuda-complex.hpp>

#include "DecayTimeAcceptance.cu"
#include "TimeAngularDistribution.cu"

////////////////////////////////////////////////////////////////////////////////



// these lines should be moved to a taggin.cu file

__device__
double get_omega(double eta, double tag,
                 double p0,  double p1, double p2,
                 double dp0, double dp1, double dp2,
                 double eta_bar)
{
    double result = 0;
    result += (p0 + tag*0.5*dp0);
    result += (p1 + tag*0.5*dp1)*(eta - eta_bar);
    result += (p2 + tag*0.5*dp2)*(eta - eta_bar)*(eta - eta_bar);

    if(result < 0.0)
    {
      return 0;
    }
    return result;
}



__device__ double get_int_ta_spline(double delta_t,double G,double DM,double DG,double a,double b,double c,double d,double t_0,double t_1)
{
    double G_sq = G*G;
    double G_cub = G_sq*G;
    double DG_sq = DG*DG;
    double DG_cub = DG_sq*DG;
    double delta_t_sq = delta_t*delta_t;
    double t_0_sq = t_0*t_0;
    double t_0_cub = t_0_sq*t_0;
    double t_1_sq = t_1*t_1;
    double t_1_cub = t_1_sq*t_1;

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

__device__ double get_int_tb_spline(double delta_t,double G,double DM,double DG,double a,double b,double c,double d,double t_0,double t_1)
{
    double G_sq = G*G;
    double G_cub = G_sq*G;
    double DG_sq = DG*DG;
    double DG_cub = DG_sq*DG;
    double delta_t_sq = delta_t*delta_t;
    double t_0_sq = t_0*t_0;
    double t_0_cub = t_0_sq*t_0;
    double t_1_sq = t_1*t_1;
    double t_1_cub = t_1_sq*t_1;

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

__device__ double get_int_tc_spline(double delta_t,double G,double DM,double DG,double a,double b,double c,double d,double t_0,double t_1)
{
    double G_sq = G*G;
    double G_cub = G_sq*G;
    double DM_sq = DM*DM;
    double DM_fr = DM_sq*DM_sq;
    double DM_sx = DM_fr*DM_sq;
    double G_fr = G_sq*G_sq;
    double delta_t_sq = delta_t*delta_t;
    double t_0_sq = t_0*t_0;
    double t_0_cub = t_0_sq*t_0;
    double t_1_sq = t_1*t_1;
    double t_1_cub = t_1_sq*t_1;
    double exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    double exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    double exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    double exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);

    return (a*pow(DM_sq + G_sq, 3.)*(-DM*exp_0_sin_1_term + DM*exp_1_sin_0_term - G*exp_0_cos_1_term + G*exp_1_cos_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM*((DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_sin_0_term - (DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_sin_1_term) + (DM_sq*(G*t_0 - 1) + G_sq*(G*t_0 + 1))*exp_1_cos_0_term - (DM_sq*(G*t_1 - 1) + G_sq*(G*t_1 + 1))*exp_0_cos_1_term)
    + c*(DM_sq + G_sq)*(DM*((DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_sin_0_term - (DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_sin_1_term) + (DM_fr*t_0*(G*t_0 - 2.) + 2*DM_sq*G*(G_sq*t_0_sq - 3.) + G_cub*(G_sq*t_0_sq + 2*G*t_0 + 2.))*exp_1_cos_0_term - (DM_fr*t_1*(G*t_1 - 2.) + 2*DM_sq*G*(G_sq*t_1_sq - 3.) + G_cub*(G_sq*t_1_sq + 2*G*t_1 + 2.))*exp_0_cos_1_term)
    + d*(DM*((DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_sin_0_term - (DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_sin_1_term) + (DM_sx*t_0_sq*(G*t_0 - 3.) + 3*DM_fr*(G_cub*t_0_cub - G_sq*t_0_sq - 6*G*t_0 + 2.) + 3*DM_sq*G_sq*(G_cub*t_0_cub + G_sq*t_0_sq - 4*G*t_0 - 12.) + G_fr*(G_cub*t_0_cub + 3*G_sq*t_0_sq + 6*G*t_0 + 6))*exp_1_cos_0_term - (DM_sx*t_1_sq*(G*t_1 - 3.) + 3*DM_fr*(G_cub*t_1_cub - G_sq*t_1_sq - 6*G*t_1 + 2.) + 3*DM_sq*G_sq*(G_cub*t_1_cub + G_sq*t_1_sq - 4*G*t_1 - 12.) + G_fr*(G_cub*t_1_cub + 3*G_sq*t_1_sq + 6*G*t_1 + 6))*exp_0_cos_1_term))*sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}

__device__ double get_int_td_spline(double delta_t,double G,double DM,double DG,double a,double b,double c,double d,double t_0,double t_1)
{
    double G_sq = G*G;
    double G_cub = G_sq*G;
    double G_fr = G_sq*G_sq;
    double G_fv = G_cub*G_sq;
    double DM_sq = DM*DM;
    double DM_fr = DM_sq*DM_sq;
    double DM_sx = DM_fr*DM_sq;
    double delta_t_sq = delta_t*delta_t;
    double t_0_sq = t_0*t_0;
    double t_0_cub = t_0_sq*t_0;
    double t_1_sq = t_1*t_1;
    double t_1_cub = t_1_sq*t_1;
    double exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    double exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    double exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    double exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);


    return -(a*pow(DM_sq + G_sq, 3.)*(DM*exp_0_cos_1_term - DM*exp_1_cos_0_term - G*exp_0_sin_1_term + G*exp_1_sin_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM_sq*G*t_0*exp_1_sin_0_term - DM_sq*G*t_1*exp_0_sin_1_term + DM_sq*exp_0_sin_1_term - DM_sq*exp_1_sin_0_term - DM*(DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_cos_0_term + DM*(DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_cos_1_term + G_cub*t_0*exp_1_sin_0_term - G_cub*t_1*exp_0_sin_1_term - G_sq*exp_0_sin_1_term + G_sq*exp_1_sin_0_term)
    + c*(DM_sq + G_sq)*(DM_fr*G*t_0_sq*exp_1_sin_0_term - DM_fr*G*t_1_sq*exp_0_sin_1_term - 2*DM_fr*t_0*exp_1_sin_0_term + 2*DM_fr*t_1*exp_0_sin_1_term + 2*DM_sq*G_cub*t_0_sq*exp_1_sin_0_term - 2*DM_sq*G_cub*t_1_sq*exp_0_sin_1_term + 6*DM_sq*G*exp_0_sin_1_term - 6*DM_sq*G*exp_1_sin_0_term - DM*(DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_cos_0_term + DM*(DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_cos_1_term + G_fv*t_0_sq*exp_1_sin_0_term - G_fv*t_1_sq*exp_0_sin_1_term + 2*G_fr*t_0*exp_1_sin_0_term - 2*G_fr*t_1*exp_0_sin_1_term - 2*G_cub*exp_0_sin_1_term + 2*G_cub*exp_1_sin_0_term)
    + d*(DM_sx*G*t_0_cub*exp_1_sin_0_term - DM_sx*G*t_1_cub*exp_0_sin_1_term - 3*DM_sx*t_0_sq*exp_1_sin_0_term + 3*DM_sx*t_1_sq*exp_0_sin_1_term + 3*DM_fr*G_cub*t_0_cub*exp_1_sin_0_term - 3*DM_fr*G_cub*t_1_cub*exp_0_sin_1_term - 3*DM_fr*G_sq*t_0_sq*exp_1_sin_0_term + 3*DM_fr*G_sq*t_1_sq*exp_0_sin_1_term - 18*DM_fr*G*t_0*exp_1_sin_0_term + 18*DM_fr*G*t_1*exp_0_sin_1_term - 6*DM_fr*exp_0_sin_1_term + 6*DM_fr*exp_1_sin_0_term + 3*DM_sq*G_fv*t_0_cub*exp_1_sin_0_term - 3*DM_sq*G_fv*t_1_cub*exp_0_sin_1_term + 3*DM_sq*G_fr*t_0_sq*exp_1_sin_0_term - 3*DM_sq*G_fr*t_1_sq*exp_0_sin_1_term - 12*DM_sq*G_cub*t_0*exp_1_sin_0_term + 12*DM_sq*G_cub*t_1*exp_0_sin_1_term + 36*DM_sq*G_sq*exp_0_sin_1_term - 36*DM_sq*G_sq*exp_1_sin_0_term - DM*(DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_cos_0_term + DM*(DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_cos_1_term + pow(G, 7)*t_0_cub*exp_1_sin_0_term - pow(G, 7)*t_1_cub*exp_0_sin_1_term + 3*pow(G, 6)*t_0_sq*exp_1_sin_0_term - 3*pow(G, 6)*t_1_sq*exp_0_sin_1_term + 6*G_fv*t_0*exp_1_sin_0_term - 6*G_fv*t_1*exp_0_sin_1_term - 6*G_fr*exp_0_sin_1_term + 6*G_fr*exp_1_sin_0_term))
    *sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}


__device__
void integralSpline( double result[2],
                     double vn[10], double va[10],double vb[10], double vc[10],double vd[10],
                     double *norm, double G, double DG, double DM,
                     double delta_t,
                     double tLL, double tUL,
                     double t_offset,
                     double *coeffs)
{
  //int bin0 = 0;
  // double tS = tLL-t_offset;
  double tS = 0;
  // double tE = KNOTS[bin0+1]-t_offset;
  double tE = 0;
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
    // if ( threadIdx.x + blockDim.x * blockIdx.x == 0){
    // printf("integral: bin %d (%lf,%lf)\n", bin ,tS, tE);
    // }

    double c0 = getCoeff(coeffs,bin,0);
    double c1 = getCoeff(coeffs,bin,1);
    double c2 = getCoeff(coeffs,bin,2);
    double c3 = getCoeff(coeffs,bin,3);

    double ta = get_int_ta_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    double tb = get_int_tb_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    double tc = get_int_tc_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    double td = get_int_td_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);

    for(int k=0; k<10; k++)
    {
      result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
      result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
    }
  }
}







////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



__device__
double getDiffRate( double *data,
                    // Time-dependent angular distribution
                    double G, double DG, double DM, double CSP,
                    double ASlon, double APlon, double APpar, double APper,
                    double pSlon, double pPlon, double pPpar, double pPper,
                    double dSlon, double dPlon, double dPpar, double dPper,
                    double lSlon, double lPlon, double lPpar, double lPper,
                    // Time limits
                    double tLL, double tUL,
                    // Time resolution
                    double sigma_offset, double sigma_slope, double sigma_curvature,
                    double mu,
                    // Flavor tagging
                    double eta_bar_os, double eta_bar_ss,
                    double p0_os,  double p1_os, double p2_os,
                    double p0_ss,  double p1_ss, double p2_ss,
                    double dp0_os, double dp1_os, double dp2_os,
                    double dp0_ss, double dp1_ss, double dp2_ss,
                    // Time acceptance
                    double *coeffs,
                    // Angular acceptance
                    double *angular_weights,bool USE_FK
                  )
{
  // OK double *data,
  // OK double *out,
  // OK double G, double DG, double DM, double CSP,
  // OK double A_0_abs, double A_S_abs, double A_pa_abs, double A_pe_abs,
  // OK double phis_0, double phis_S, double phis_pa, double phis_pe,
  // OK double delta_S, double delta_pa, double delta_pe,
  // OK double l_0_abs, double l_S_abs, double l_pa_abs, double l_pe_abs,


  // double p0_OS, double dp0_OS, double p1_OS, double dp1_OS, double p2_OS, double dp2_OS, double eta_bar_OS,
  // double p0_SSK, double dp0_SSK, double p1_SSK, double dp1_SSK, double eta_bar_SSK,

  // double sigma_t_a, double sigma_t_b, double sigma_t_c, double t_ll,
  // double sigma_t_mu_a, double sigma_t_mu_b, double sigma_t_mu_c,

  // double f_sigma_t, double r_offset_pr, double r_offset_sc, double r_slope_pr, double r_slope_sc,
  // double sigma_t_bar, double t_res_scale, double t_mu,


  // OK int spline_Nknots,
  // OK double *normweights,
  // OK double *spline_knots,
  // OK double *spline_coeffs,

  // double *low_time_acc_bins_ul, double *low_time_acc,

  // OK int Nevt


  #ifdef DEBUG
  if ( DEBUG > 4 && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT )
  {
    printf("*USE_FK            : %d\n", USE_FK);
    printf("*USE_TIME_ACC      : %d\n", USE_TIME_ACC);
    printf("*USE_TIME_OFFSET   : %d\n", USE_TIME_OFFSET);
    printf("*USE_TIME_RES      : %d\n", USE_TIME_RES);
    printf("*USE_PERFTAG       : %d\n", USE_PERFTAG);
    printf("*USE_TRUETAG       : %d\n", USE_TRUETAG);
    printf("G                  : %+.16lf\n", G);
    printf("DG                 : %+.16lf\n", DG);
    printf("DM                 : %+.16lf\n", DM);
    printf("CSP                : %+.16lf\n", CSP);
    printf("ASlon              : %+.16lf\n", ASlon);
    printf("APlon              : %+.16lf\n", APlon);
    printf("APpar              : %+.16lf\n", APpar);
    printf("APper              : %+.16lf\n", APper);
    printf("pSlon              : %+.16lf\n", pSlon);
    printf("pPlon              : %+.16lf\n", pPlon);
    printf("pPpar              : %+.16lf\n", pPpar);
    printf("pPper              : %+.16lf\n", pPper);
    printf("dSlon              : %+.16lf\n", dSlon);
    printf("dPlon              : %+.16lf\n", dPlon);
    printf("dPper              : %+.16lf\n", dPper);
    printf("dPpar              : %+.16lf\n", dPpar);
    printf("lSlon              : %+.16lf\n", lSlon);
    printf("lPlon              : %+.16lf\n", lPlon);
    printf("lPper              : %+.16lf\n", lPper);
    printf("lPpar              : %+.16lf\n", lPpar);
    printf("tLL                : %+.16lf\n", tLL);
    printf("tUL                : %+.16lf\n", tUL);
    printf("mu                 : %+.16lf\n", mu);
    printf("sigma_offset       : %+.16lf\n", sigma_offset);
    printf("sigma_slope        : %+.16lf\n", sigma_slope);
    printf("sigma_curvature    : %+.16lf\n", sigma_curvature);
    printf("COEFFS             : %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
    printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
            coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  }
  #endif

  double normweights[10] = {1,1,1,0,0,0,1,0,0,0};

  // Variables -----------------------------------------------------------------
  //     Make sure that the input it's in this order.
  //     lalala
  double cosK       = data[0];                      // Time-angular distribution
  double cosL       = data[1];
  double hphi       = data[2];
  double time       = data[3];

  double sigma_t    = data[4];                                // Time resolution

  double qOS        = data[5];                                        // Tagging
  double qSS        = data[6];
  double etaOS 	  	= data[7];
  double etaSS 	    = data[8];

  #ifdef DEBUG
  if ( DEBUG > 99 && ( (time>=tUL) || (time<=tLL) ) )
  {
    printf("WARNING            : Event with time not within [%.4lf,%.4lf].\n",
           tLL, tUL);
  }
  #endif

  #ifdef DEBUG
  if (DEBUG >= 1 && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT )
  {
    printf("\nINPUT              : cosK=%+.16lf  cosL=%+.16lf  hphi=%+.16lf  time=%+.16lf\n",
           cosK,cosL,hphi,time);
    printf("                   : sigma_t=%+.16lf  qOS=%+.16lf  qSS=%+.16lf  etaOS=%+.16lf  etaSS=%+.16lf\n",
           sigma_t,qOS,qSS,etaOS,etaSS);
  }
  #endif


  // Time resolution -----------------------------------------------------------
  //     In order to remove the effects of conv, set sigma_t=0, so in this way
  //     you are running the first branch of getExponentialConvolution.
  pycuda::complex<double> exp_p, exp_m, exp_i;
  double t_offset = 0.0; double delta_t = sigma_t;
  double sigma_t_mu_a = 0, sigma_t_mu_b = 0, sigma_t_mu_c = 0;

  if (USE_TIME_OFFSET)
  {
    t_offset = getTimeCal(sigma_t, sigma_t_mu_a, sigma_t_mu_b, sigma_t_mu_c);
  }

  if (USE_TIME_RES) // use_per_event_res
  {
    delta_t  = getTimeCal(sigma_t, sigma_offset, sigma_slope, sigma_curvature);
  }

  #ifdef DEBUG
  if (DEBUG > 3 && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT )
  {
    printf("\nTIME RESOLUTION    : delta_t=%.16f\n", delta_t);
  }
  #endif


  if ( delta_t == 0 ) // MC samples need to solve some problems
  {
    exp_p = getExponentialConvolution(time-t_offset, G + 0.5*DG, 0., delta_t);
    exp_m = getExponentialConvolution(time-t_offset, G - 0.5*DG, 0., delta_t);
    exp_i = getExponentialConvolution(time-t_offset,          G, DM, delta_t);
  }
  else
  {
    exp_p = getExponentialConvolution_simon(time-t_offset, G + 0.5*DG, 0., delta_t);
    exp_m = getExponentialConvolution_simon(time-t_offset, G - 0.5*DG, 0., delta_t);
    exp_i = getExponentialConvolution_simon(time-t_offset,          G, DM, delta_t);
  }

  // printf("                   : exp_p=%+.16lf%+.16lfi   exp_m=%+.16lf%+.16lfi   exp_i=%+.16lf%+.16lfi\n",
  //        pycuda::real(exp_p), pycuda::imag(exp_p), pycuda::real(exp_m), pycuda::imag(exp_m), pycuda::real(exp_i), pycuda::imag(exp_i));

  double ta = pycuda::real(0.5*(exp_m + exp_p));     // cosh = (exp_m + exp_p)/2
  double tb = pycuda::real(0.5*(exp_m - exp_p));     // sinh = (exp_m - exp_p)/2
  double tc = pycuda::real(exp_i);                        // exp_i = cos + I*sin
  double td = pycuda::imag(exp_i);                        // exp_i = cos + I*sin

  #ifdef DEBUG
  if (DEBUG >= 3 && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT)
  {
    printf("\nTIME TERMS         : ta=%.16f  tb=%.16f  tc=%.16f  td=%.16f\n",
           ta,tb,tc,td);
  }
  #endif

  // Flavor tagging ------------------------------------------------------------
  double omegaOSB = 0; double omegaOSBbar = 0; double tagOS = 0;
  double omegaSSB = 0; double omegaSSBbar = 0; double tagSS = 0;

  if (USE_TRUETAG)
  {
    tagOS = 0.0;
    tagSS = 0.0;
  }
  else if (USE_PERFTAG)
  {
    if (qOS != 0) {tagOS = qOS/fabs(qOS);}
    if (qSS != 0) {tagSS = qSS/fabs(qSS);}
  }
  else
  {
    if (qOS != 0) { tagOS = qOS/fabs(qOS);}
    if (qSS != 0) { tagSS = qSS/fabs(qSS);}
    omegaOSB    = get_omega(etaOS, +1, p0_os, p1_os, p2_os, dp0_os, dp1_os, dp2_os, eta_bar_os);
    omegaOSBbar = get_omega(etaOS, -1, p0_os, p1_os, p2_os, dp0_os, dp1_os, dp2_os, eta_bar_os);
    omegaSSB    = get_omega(etaSS, +1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss, dp2_ss, eta_bar_ss);
    omegaSSBbar = get_omega(etaSS, -1, p0_ss, p1_ss, p2_ss, dp0_ss, dp1_ss, dp2_ss, eta_bar_ss);
  }

  // Print warning if tagOS|tagSS == 0
  #ifdef DEBUG
  if ( DEBUG > 99 && ( (tagOS == 0)|(tagSS == 0) ) )
  {
    printf("This event is not tagged!\n");
  }
  #endif

  #ifdef DEBUG
  if ( DEBUG > 3  && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT )
  {
    printf("\nFLAVOR TAGGING     : delta_t=%.16f\n", delta_t);
    printf("                   : tagOS=%.16f, tagSS=%.16f\n",
           tagOS, tagSS);
    printf("                   : omegaOSB=%.16f, omegaOSBbar=%.16f\n",
           omegaOSB, omegaOSBbar);
    printf("                   : omegaSSB=%.16f, omegaSSBbar=%.16f\n",
           omegaSSB, omegaSSBbar);
  }
  #endif

  // Decay-time acceptance -----------------------------------------------------
  //     To get rid of decay-time acceptance set USE_TIME_ACC to False. If True
  //     then calcTimeAcceptance locates the time bin of the event and returns
  //     the value of the cubic spline.
  double dta = 1.0;
  if (USE_TIME_ACC)
  {
    dta = calcTimeAcceptance(time, coeffs, tLL, tUL);
  }

  // Compute per event pdf -----------------------------------------------------
  double vnk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vfk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vak[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vbk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vck[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  double vdk[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};

  double nk, fk, ak, bk, ck, dk, hk_B, hk_Bbar;
  double pdfB = 0.0; double pdfBbar = 0.0;

  for(int k = 1; k <= 10; k++)
  {
    nk = getN(APlon,ASlon,APpar,APper,CSP,k);
    if (USE_FK)
    {
      fk = ( 9.0/(16.0*M_PI) )*getF(cosK,cosL,hphi,k);
      //fk = getF(cosK,cosL,hphi,k);
    }
    else
    {
      fk = normweights[k-1]; // these are 0s or 1s
    }

    ak = getA(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    bk = getB(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    ck = getC(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);
    dk = getD(pPlon,pSlon,pPpar,pPper,dPlon,dSlon,dPpar,dPper,lPlon,lSlon,lPpar,lPper,k);

    if (DM != 0)
    {
      hk_B    = (ak*ta + bk*tb + ck*tc + dk*td);//old factor: 3./(4.*M_PI)*
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
    pdfB += nk*fk*hk_B; pdfBbar += nk*fk*hk_Bbar;
    vnk[k-1] = 1.*nk; vfk[k-1] = 1.*fk;
    vak[k-1] = 1.*ak; vbk[k-1] = 1.*bk; vck[k-1] = 1.*ck; vdk[k-1] = 1.*dk;
  }

  #ifdef DEBUG
  if ( DEBUG > 3  && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT )
  {
    printf("\nANGULAR PART       :  n            a            b            c            d            f            ang_acc\n");
    for(int k = 0; k < 10; k++)
    {
      printf("               (%d) : %+.16lf  %+.16lf  %+.16lf  %+.16lf  %+.16lf  %+.16lf  %+.16lf\n",
             k,vnk[k], vak[k], vbk[k], vck[k], vdk[k], vfk[k], angular_weights[k]);
    }
  }
  #endif


  // Compute pdf integral ------------------------------------------------------
  double intBBar[2] = {0.,0.};
  if ( (delta_t == 0) & (USE_TIME_ACC == 0) )
  {
    // Here we can use the simplest 4xPi integral of the pdf since there are no
    // resolution effects
    integralSimple(intBBar,
                   vnk, vak, vbk, vck, vdk, angular_weights, G, DG, DM, tLL, tUL);
  }
  else
  {
    // This integral works for all decay times, remember delta_t != 0.
    int simon_j = sigma_t/(SIGMA_T/80);
    // if (DEBUG >= 1)
    // {
    //   if ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT){
    // {
    //   printf("simon_j = %+d = round(%+.16lf)\n", simon_j, sigma_t/(SIGMA_T/80) );
    // }
    integralFullSpline(intBBar,
                       vnk, vak, vbk, vck, vdk,
                       angular_weights, G, DG, DM,
                       //delta_t,
                       //sigma_t,
                       //getTimeCal(  (0.5+simon_j-1)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
                       getTimeCal(  (0.5+simon_j)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
                       tLL,
                       t_offset,
                       coeffs);
     // integralSpline( intBBar,
     //                  vnk, vak, vbk, vck, vdk,
     //                  angular_weights, G, DG, DM,
     //                  delta_t,
     //                  //getTimeCal(  (0.5+simon_j)*(SIGMA_T/80)  , sigma_offset, sigma_slope, sigma_curvature),
     //                  tLL, tUL,
     //                  t_offset,
     //                  coeffs);
  }
  double intB = intBBar[0]; double intBbar = intBBar[1];



  // Cooking the output --------------------------------------------------------
  double num = 1.0; double den = 1.0;
  num = dta*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * pdfB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * pdfBbar
        );
  den = 1.0*(
        (1+tagOS*(1-2*omegaOSB)   ) * (1+tagSS*(1-2*omegaSSB)   ) * intB +
        (1-tagOS*(1-2*omegaOSBbar)) * (1-tagSS*(1-2*omegaSSBbar)) * intBbar
        );
  num = num/4;
  den = den/4;

  #ifdef DEBUG
  if ( DEBUG >= 1  && threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT)
  {
    printf("\nRESULT             : <  pdf/ipdf = %+.16lf  >\n",
           num/den);
    if ( DEBUG >= 2 )
    {
     printf("                   : pdf=%+.16lf  ipdf=%+.16lf\n",
            num,den);
     printf("                   : pdfB=%+.16lf  pdBbar=%+.16lf  ipdfB=%+.16lf  ipdfBbar=%+.16lf\n",
            pdfB,pdfBbar,intB,intBbar);
    }
  }
  #endif
  // That's all folks!
  return num/den;
}



////////////////////////////////////////////////////////////////////////////////
