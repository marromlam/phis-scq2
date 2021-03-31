////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                             TIME ACCEPTANCE                                //
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq package, Santiago's framework for the     //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////

#include <ipanema/core.h>
#include <ipanema/complex.h>
#include <ipanema/special.h>


WITHIN_KERNEL
unsigned int getTimeBin(ftype const t)
{
  int _i = 0;
  int _n = NKNOTS-1;
  while(_i <= _n )
  {
    if( t < KNOTS[_i] ) {break;}
    _i++;
  }
  if ((0 == _i) & (DEBUG > 3)) {
    printf("WARNING: t=%.16f below first knot!\n",t);
  }
  return _i - 1;

}


WITHIN_KERNEL
unsigned int getMassBin(ftype const t)
{
  int _i = 0;
  int _n = MKNOTS-1;
  while(_i <= _n )
  {
    if( t < MHH[_i] ) {break;}
    _i++;
  }
  if ((0 == _i) & (DEBUG > 3)) {
    printf("WARNING: t=%.16f below first knot!\n",t);
  }
  return _i - 1;
}



WITHIN_KERNEL
ftype getKnot(int i)
{
  if (i<=0) {
    i = 0;
  }
  else if (i>=NKNOTS) {
    i = NKNOTS;
  }
  return KNOTS[i];
}



WITHIN_KERNEL
ftype getCoeff(GLOBAL_MEM const ftype *mat, int const r, int const c)
{
  return mat[4*r+c];
}



WITHIN_KERNEL
ftype time_efficiency(const ftype t, GLOBAL_MEM const ftype *coeffs, const ftype tLL, const ftype tUL)
{
  int bin   = getTimeBin(t);
  ftype c0 = getCoeff(coeffs,bin,0);
  ftype c1 = getCoeff(coeffs,bin,1);
  ftype c2 = getCoeff(coeffs,bin,2);
  ftype c3 = getCoeff(coeffs,bin,3);
  #ifdef DEBUG
  if (DEBUG >= 3 && ( get_global_id(0) == DEBUG_EVT))
  {
    printf("\nTIME ACC           : t=%.16f\tbin=%d\tc=[%+f\t%+f\t%+f\t%+f]\tdta=%+.16f\n",
           t,bin,c0,c1,c2,c3, (c0 + t*(c1 + t*(c2 + t*c3))) );
  }
  #endif

  return (c0 + t*(c1 + t*(c2 + t*c3)));
}










WITHIN_KERNEL
ctype expconv_simon(const ftype t, const ftype G, const ftype omega, const ftype sigma)
{
  ctype I2 = C(-1, 0);
  ctype I3 = C( 0,-1);

  const ftype sigma2 = sigma*sigma;
  const ftype omega2 = omega*omega;

  if (omega == 0)
  {
    if( t > -6.0*sigma )
    {
      const ftype t_exp = 0.5*exp(-t*G + 0.5*G*G*sigma2 -0.5*omega2*sigma2);
      const ctype t_cerfc = cerfc(C(sigma*G/sqrt(2.0) - t/sigma/sqrt(2.0),0));
      return cmul( C(t_exp,0) , t_cerfc );
    }
    else
    {
      return C(0,0);
    }
  }
  else //(omega != 0)
  {
    //ftype c1 = 0.5;

    ftype exp1arg = 0.5*sigma2*(G*G - omega*omega) - t*G;
    ctype exp1 = C( 0.5*exp(exp1arg) ,0);

    ftype exp2arg = -omega*(t - sigma2*G);
    ctype exp2 = C(cos(exp2arg), sin(exp2arg));

    ctype cerfarg = C(sigma*G/sqrt(2.0) - t/(sigma*sqrt(2.0)) , +omega*sigma/sqrt(2.0));
    ctype cerf;

    if  (cerfarg.x < -20.0)
    {
      cerf = C(2.0,0.0);
    }
    else
    {
      cerf = cerfc(cerfarg);//best complex error function
    }
    ctype c2 = cmul(exp2, cerf);
    //ftype im = -c2.x;//exp*sin
    //ftype re = +c2.real();//exp*cos

    return cmul( exp1 , cadd( C(c2.x,0), cmul( I3, C(c2.y,0) ) ) );
  }

}



WITHIN_KERNEL
ctype expconv(ftype t, ftype G, ftype omega, ftype sigma)
{
  const ftype sigma2 = sigma*sigma;
  const ftype omega2 = omega*omega;

  if( t > SIGMA_THRESHOLD*sigma )
  {
    ftype a = exp(-G*t+0.5*G*G*sigma2-0.5*omega2*sigma2);
    ftype b = omega*(t-G*sigma2);
    return C(a*cos(b),a*sin(b));
  }
  else
  {
    ctype z, fad;
    z   = C(-omega*sigma2/(sigma*sqrt(2.)), -(t-sigma2*G)/(sigma*sqrt(2.)));
    fad = cwofz(z);
/*
 if ( (t>0.3006790) && (t<0.3006792) ){
           printf("z   = %+.16f%+.16f\n",z.x,z.y );
           printf("fad = %+.16f%+.16f\n",fad.x,fad.y );
           //printf("shit = %+.16f%+.16f\n",shit.x,shit.y );
           double2 res = C(0.5*exp(-0.5*t*t/sigma2),0.0);
           printf("res   = %+.16f\n",res.x,res.y );
     }
 */
    return cmul( C(fad.x,-fad.y), C(0.5*exp(-0.5*t*t/sigma2),0) );
  }
}



WITHIN_KERNEL
ctype getK(const ctype z, const int n)
{
  ctype z2 = cmul(z,z);
  ctype z3 = cmul(z,z2);
  ctype z4 = cmul(z,z3);
  ctype z5 = cmul(z,z4);
  ctype z6 = cmul(z,z5);
  ctype w;

  if (n == 0)
  {
    w = cmul( C(2.0,0.0), z);
    return cdiv(C( 1.0,0.0), w );
  }
  else if (n == 1)
  {
    w = cmul( C(2.0,0.0), z2);
    return cdiv(C(1.0,0.0), w );
  }
  else if (n == 2)
  {
    w = cdiv( C(1.0,0.0), z2 );
    w = cadd( C(1.0,0.0), w );
    return cmul( cdiv(C(1.0,0.0),z) , w );
  }
  else if (n == 3)
  {
    w = cdiv( C(1.0,0.0), z2 );
    w = cadd( C(1.0,0.0), w );
    return cmul( cdiv(C(3.0,0.0),z2) , w );
  }
  // else if (n == 4) {
  //   return cdiv(C( 6.,0), z*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 5) {
  //   return cdiv(C(30.,0), (z*z)*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 6) {
  //   return cdiv(C(60.,0), z*(1.+3./(z*z)+6./(z*z*z*z)+6./(z*z*z*z*z*z))  );
  // }

  return C(0.,0.);
}



WITHIN_KERNEL
ctype getM(ftype x, int n, ftype t, ftype sigma, ftype gamma, ftype omega)
{
  ctype conv_term, z;
  ctype I2 = C(-1,0);
  ctype I3 = C(0,-1);

  z = C(gamma*sigma/sqrt(2.0),-omega*sigma/sqrt(2.0));
  ctype arg1 = csub( cmul(z,z), cmul(C(2*x,0),z) );
  ctype arg2 = csub(z,C(x,0));
  //conv_term = 5.0*expconv(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));
  // warning there are improvement to do here!!!
  if (omega == 0){
    //conv_term = cmul( cexp(arg1), ipanema_erfc(arg2) );
    conv_term = cmul( cexp(arg1), cerfc(arg2) );
  }
  else{
    conv_term = cmul( cexp(arg1), cerfc(arg2) );
    //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);
    //conv_term = 2.0*exp(-gamma*t+0.5*gamma*gamma*sigma*sigma-0.5*omega*omega*sigma*sigma)*(cos(omega*(t-gamma*sigma*sigma)) + I*sin(omega*(t-gamma*sigma*sigma)));
  }
  //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));

  #ifdef DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) ){
    printf("\nerfc*exp = %+.16f %+.16fi\n",  conv_term.x, conv_term.y);
    // printf("erfc = %+.16f %+.16fi\n",  ipanema_erfc(arg2).x, ipanema_erfc(arg2).y );
    // printf("cErrF_2 = %+.16f %+.16fi\n",  cErrF_2(arg2).x, cErrF_2(arg2).y );
    // printf("exp  = %+.16f %+.16fi\n",  cexp(arg1).x, cexp(arg1).y );
    // printf("z    = %+.16f %+.16fi     %+.16f %+.16f %+.16f        x = %+.16f\n",  z.x, z.y, gamma, omega, sigma, x);
  }
  #endif

  if (n == 0)
  {
    ctype a = C(erf(x),0.);
    ctype b = conv_term;
    return csub(a,b);
  }
  else if (n == 1)
  {
    // return 2.*(-pycuda::complex<double>(sqrt(1./M_PI)*exp(-x*x),0.)-x*conv_term);
    //ctype a = C(sqrt(1./M_PI)*exp(-x*x),0.);
    ctype a = C(sqrt(1.0/M_PI)*exp(-x*x),0.);
    ctype b = C(x,0);
    b = cmul(b,conv_term);
    return cmul(C(-2.0,0.0),cadd(a,b));
  }
  else if (n == 2)
  {
    // return 2.*(-2.*x*exp(-x*x)*ctype(sqrt(1./M_PI),0.)-(2.*x*x-1.)*conv_term);
    ctype a = C(-2.*x*exp(-x*x)*sqrt(1./M_PI),0.);
    ctype b = C(2*x*x-1,0);
    b = cmul(b,conv_term);
    return cmul(C(2,0),csub(a,b));
  }
  else if (n == 3)
  {
    // return 4.*(-(2.*x*x-1.)*exp(-x*x)*ctype(sqrt(1./M_PI),0.)-x*(2.*x*x-3.)*conv_term);
    ctype a = C(-(2.*x*x-1.)*exp(-x*x)*sqrt(1./M_PI),0.);
    ctype b = C(x*(2*x*x-3),0);
    b = cmul(b,conv_term);
    return cmul(C(4,0),csub(a,b));
  }
  // else if (n == 4)
  // {
  //   return 4.*(exp(-x*x)*(6.*x+4.*x*x*x)*ctype(sqrt(1./M_PI),0.)-(3.-12.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 5)
  // {
  //   return 8.*(-(3.-12.*x*x+4.*x*x*x*x)*exp(-x*x)*ctype(sqrt(1./M_PI),0.)-x*(15.-20.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 6)
  // {
  //   return 8.*(-exp(-x*x)*(30.*x-40.*x*x*x+8.*x*x*x*x*x)*ctype(sqrt(1./M_PI),0.)-(-15.+90.*x*x-60.*x*x*x*x+8.*x*x*x*x*x*x)*conv_term);
  // }
  return C(0.,0.);
}



WITHIN_KERNEL
void intgTimeAcceptance(ftype time_terms[4], const ftype delta_t,
                        const ftype G, const ftype DG, const ftype DM,
                        GLOBAL_MEM const ftype *coeffs, const ftype t0, const ftype tLL, const ftype tUL)
{
  // Some constants
  ftype cte1 = 1.0/(sqrt(2.0)*delta_t);
  ctype cte2 = C( delta_t/(sqrt(2.0)) , 0 );
  if (DEBUG > 3 && delta_t <= 0.0)
  {
    printf("WARNING            : delta_t = %.4f is not a valid value.\n", delta_t);
  }

  // Add tUL to knots list
  ftype x[NTIMEBINS] = {0.};
  ftype knots[NTIMEBINS] = {0.};
  knots[0] = tLL; x[0] = (knots[0] - t0)*cte1;
  for(int i = 1; i < NKNOTS; i++)
  {
    knots[i] = KNOTS[i];
    x[i] = (knots[i] - t0)*cte1;
  }
  knots[NKNOTS] = tUL; x[NKNOTS] = (knots[NKNOTS] - t0)*cte1;

  // Fill S matrix                (TODO speed to be gained here - S is constant)
  ftype S[SPL_BINS][4][4];
  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int i=0; i<4; ++i)
    {
      for (int j=0; j<4; ++j)
      {
        if(i+j < 4)
        {
          S[bin][i][j] = getCoeff(coeffs,bin,i+j)
                         *factorial(i+j)/factorial(j)/factorial(i)/pow(2.0,i+j);
        }
        else
        {
          S[bin][i][j] = 0.;
        }
      }
    }
  }

  ctype z_expm, K_expm[4], M_expm[SPL_BINS+1][4];
  ctype z_expp, K_expp[4], M_expp[SPL_BINS+1][4];
  ctype z_trig, K_trig[4], M_trig[SPL_BINS+1][4];

  z_expm = cmul( cte2 , C(G-0.5*DG,  0) );
  z_expp = cmul( cte2 , C(G+0.5*DG,  0) );
  z_trig = cmul( cte2 , C(       G,-DM) );

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_expp[j] = getK(z_expp,j);
    K_expm[j] = getK(z_expm,j);
    K_trig[j] = getK(z_trig,j);
    #ifdef DEBUG
    if (DEBUG > 3 && (get_global_id(0) == DEBUG_EVT) )
    {
      printf("K_expp[%d](%+.14f%+.14f) = %+.14f%+.14f\n",  j,z_expp.x,z_expp.y,K_expp[j].x,K_expp[j].y);
      printf("K_expm[%d](%+.14f%+.14f) = %+.14f%+.14f\n",  j,z_expm.x,z_expm.y,K_expm[j].x,K_expm[j].y);
      printf("K_trig[%d](%+.14f%+.14f) = %+.14f%+.14f\n\n",j,z_trig.x,z_trig.y,K_trig[j].x,K_trig[j].y);
    }
    #endif
  }

  // Fill Mn
  for (int j=0; j<4; ++j)
  {
    for(int bin=0; bin < SPL_BINS+1; ++bin)
    {
      M_expm[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G-0.5*DG,0.);
      M_expp[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G+0.5*DG,0.);
      M_trig[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G       ,DM);
      if (bin>0){
        #ifdef DEBUG
        if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
        {
          ctype aja = M_expp[bin][j];//-M_expp[bin-1][j];
          ctype eje = M_expm[bin][j];//-M_expm[bin-1][j];
          ctype iji = M_trig[bin][j];//-M_trig[bin-1][j];
          printf("bin=%d M_expp[%d] = %+.14f%+.14f\n",  bin,j,aja.x,aja.y);
          printf("bin=%d M_expm[%d] = %+.14f%+.14f\n",  bin,j,eje.x,eje.y);
          printf("bin=%d M_trig[%d] = %+.14f%+.14f\n\n",bin,j,iji.x,iji.y);
        }
        #endif
      }
    }
  }

  // Fill the delta factors to multiply by the integrals
  ftype delta_t_fact[4];
  for (int i=0; i<4; ++i)
  {
    delta_t_fact[i] = pow(delta_t*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, expm, cos, sin terms
  ctype int_expm = C(0.,0.);
  ctype int_expp = C(0.,0.);
  ctype int_trig = C(0.,0.);
  ctype aux, int_expm_aux, int_expp_aux, int_trig_aux;

  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        aux = C( S[bin][j][k]*delta_t_fact[j+k] , 0 );

        int_expm_aux = csub(M_expm[bin+1][j],M_expm[bin][j]);
        int_expm_aux = cmul(int_expm_aux,K_expm[k]);
        int_expm_aux = cmul(int_expm_aux,aux);
        int_expm     = cadd( int_expm, int_expm_aux );

        int_expp_aux = csub(M_expp[bin+1][j],M_expp[bin][j]);
        int_expp_aux = cmul(int_expp_aux,K_expp[k]);
        int_expp_aux = cmul(int_expp_aux,aux);
        int_expp     = cadd( int_expp, int_expp_aux );

        int_trig_aux = csub(M_trig[bin+1][j],M_trig[bin][j]);
        int_trig_aux = cmul(int_trig_aux,K_trig[k]);
        int_trig_aux = cmul(int_trig_aux,aux);
        int_trig     = cadd( int_trig, int_trig_aux );

        #ifdef DEBUG
        if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
        {
          //printf("bin=%d int_expm_aux[%d,%d] = %+.14f%+.14f\n",  bin,j,k,int_expm_aux.x,int_expm_aux.y);
          printf("bin=%d int_expm[%d,%d] = %+.14f%+.14f\n",  bin,j,k,int_expm.x,int_expm.y);
        }
        #endif
      }
    }
  }

  // Fill itengral terms - 0:cosh, 1:sinh, 2:cos, 3:sin
  time_terms[0] = sqrt(0.5)*0.5*(int_expm.x+int_expp.x);
  time_terms[1] = sqrt(0.5)*0.5*(int_expm.x-int_expp.x);
  time_terms[2] = sqrt(0.5)*int_trig.x;
  time_terms[3] = sqrt(0.5)*int_trig.y;

  #ifdef DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
  {
    printf("\nNORMALIZATION      : ta=%.16f\ttb=%.16f\ttc=%.16f\ttd=%.16f\n",
           time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
    printf("                   : sigma=%.16f\tgamma+=%.16f\tgamma-=%.16f\n",
           delta_t, G+0.5*DG, G-0.5*DG);
  }
  #endif
}



WITHIN_KERNEL ftype get_int_ta_spline(ftype delta_t,ftype G,ftype DM,ftype DG,ftype a,ftype b,ftype c,ftype d,ftype t_0,ftype t_1)
{
    ftype G_sq = G*G;
    ftype G_cub = G_sq*G;
    ftype DG_sq = DG*DG;
    ftype DG_cub = DG_sq*DG;
    ftype delta_t_sq = delta_t*delta_t;
    ftype t_0_sq = t_0*t_0;
    ftype t_0_cub = t_0_sq*t_0;
    ftype t_1_sq = t_1*t_1;
    ftype t_1_cub = t_1_sq*t_1;

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

WITHIN_KERNEL ftype get_int_tb_spline(ftype delta_t,ftype G,ftype DM,ftype DG,ftype a,ftype b,ftype c,ftype d,ftype t_0,ftype t_1)
{
    ftype G_sq = G*G;
    ftype G_cub = G_sq*G;
    ftype DG_sq = DG*DG;
    ftype DG_cub = DG_sq*DG;
    ftype delta_t_sq = delta_t*delta_t;
    ftype t_0_sq = t_0*t_0;
    ftype t_0_cub = t_0_sq*t_0;
    ftype t_1_sq = t_1*t_1;
    ftype t_1_cub = t_1_sq*t_1;

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

WITHIN_KERNEL ftype get_int_tc_spline(ftype delta_t,ftype G,ftype DM,ftype DG,ftype a,ftype b,ftype c,ftype d,ftype t_0,ftype t_1)
{
    ftype G_sq = G*G;
    ftype G_cub = G_sq*G;
    ftype DM_sq = DM*DM;
    ftype DM_fr = DM_sq*DM_sq;
    ftype DM_sx = DM_fr*DM_sq;
    ftype G_fr = G_sq*G_sq;
    ftype delta_t_sq = delta_t*delta_t;
    ftype t_0_sq = t_0*t_0;
    ftype t_0_cub = t_0_sq*t_0;
    ftype t_1_sq = t_1*t_1;
    ftype t_1_cub = t_1_sq*t_1;
    ftype exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    ftype exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    ftype exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    ftype exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);

    return (a*pow(DM_sq + G_sq, 3.)*(-DM*exp_0_sin_1_term + DM*exp_1_sin_0_term - G*exp_0_cos_1_term + G*exp_1_cos_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM*((DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_sin_0_term - (DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_sin_1_term) + (DM_sq*(G*t_0 - 1) + G_sq*(G*t_0 + 1))*exp_1_cos_0_term - (DM_sq*(G*t_1 - 1) + G_sq*(G*t_1 + 1))*exp_0_cos_1_term)
    + c*(DM_sq + G_sq)*(DM*((DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_sin_0_term - (DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_sin_1_term) + (DM_fr*t_0*(G*t_0 - 2.) + 2*DM_sq*G*(G_sq*t_0_sq - 3.) + G_cub*(G_sq*t_0_sq + 2*G*t_0 + 2.))*exp_1_cos_0_term - (DM_fr*t_1*(G*t_1 - 2.) + 2*DM_sq*G*(G_sq*t_1_sq - 3.) + G_cub*(G_sq*t_1_sq + 2*G*t_1 + 2.))*exp_0_cos_1_term)
    + d*(DM*((DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_sin_0_term - (DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_sin_1_term) + (DM_sx*t_0_sq*(G*t_0 - 3.) + 3*DM_fr*(G_cub*t_0_cub - G_sq*t_0_sq - 6*G*t_0 + 2.) + 3*DM_sq*G_sq*(G_cub*t_0_cub + G_sq*t_0_sq - 4*G*t_0 - 12.) + G_fr*(G_cub*t_0_cub + 3*G_sq*t_0_sq + 6*G*t_0 + 6))*exp_1_cos_0_term - (DM_sx*t_1_sq*(G*t_1 - 3.) + 3*DM_fr*(G_cub*t_1_cub - G_sq*t_1_sq - 6*G*t_1 + 2.) + 3*DM_sq*G_sq*(G_cub*t_1_cub + G_sq*t_1_sq - 4*G*t_1 - 12.) + G_fr*(G_cub*t_1_cub + 3*G_sq*t_1_sq + 6*G*t_1 + 6))*exp_0_cos_1_term))*sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}

WITHIN_KERNEL ftype get_int_td_spline(ftype delta_t,ftype G,ftype DM,ftype DG,ftype a,ftype b,ftype c,ftype d,ftype t_0,ftype t_1)
{
    ftype G_sq = G*G;
    ftype G_cub = G_sq*G;
    ftype G_fr = G_sq*G_sq;
    ftype G_fv = G_cub*G_sq;
    ftype DM_sq = DM*DM;
    ftype DM_fr = DM_sq*DM_sq;
    ftype DM_sx = DM_fr*DM_sq;
    ftype delta_t_sq = delta_t*delta_t;
    ftype t_0_sq = t_0*t_0;
    ftype t_0_cub = t_0_sq*t_0;
    ftype t_1_sq = t_1*t_1;
    ftype t_1_cub = t_1_sq*t_1;
    ftype exp_0_sin_1_term = exp(G*t_0)*sin(DM*G*delta_t_sq - DM*t_1);
    ftype exp_1_sin_0_term = exp(G*t_1)*sin(DM*G*delta_t_sq - DM*t_0);
    ftype exp_0_cos_1_term = exp(G*t_0)*cos(DM*G*delta_t_sq - DM*t_1);
    ftype exp_1_cos_0_term = exp(G*t_1)*cos(DM*G*delta_t_sq - DM*t_0);


    return -(a*pow(DM_sq + G_sq, 3.)*(DM*exp_0_cos_1_term - DM*exp_1_cos_0_term - G*exp_0_sin_1_term + G*exp_1_sin_0_term)
    + b*pow(DM_sq + G_sq, 2.)*(DM_sq*G*t_0*exp_1_sin_0_term - DM_sq*G*t_1*exp_0_sin_1_term + DM_sq*exp_0_sin_1_term - DM_sq*exp_1_sin_0_term - DM*(DM_sq*t_0 + G_sq*t_0 + 2*G)*exp_1_cos_0_term + DM*(DM_sq*t_1 + G_sq*t_1 + 2*G)*exp_0_cos_1_term + G_cub*t_0*exp_1_sin_0_term - G_cub*t_1*exp_0_sin_1_term - G_sq*exp_0_sin_1_term + G_sq*exp_1_sin_0_term)
    + c*(DM_sq + G_sq)*(DM_fr*G*t_0_sq*exp_1_sin_0_term - DM_fr*G*t_1_sq*exp_0_sin_1_term - 2*DM_fr*t_0*exp_1_sin_0_term + 2*DM_fr*t_1*exp_0_sin_1_term + 2*DM_sq*G_cub*t_0_sq*exp_1_sin_0_term - 2*DM_sq*G_cub*t_1_sq*exp_0_sin_1_term + 6*DM_sq*G*exp_0_sin_1_term - 6*DM_sq*G*exp_1_sin_0_term - DM*(DM_fr*t_0_sq + 2*DM_sq*(G_sq*t_0_sq + 2*G*t_0 - 1) + G_sq*(G_sq*t_0_sq + 4*G*t_0 + 6))*exp_1_cos_0_term + DM*(DM_fr*t_1_sq + 2*DM_sq*(G_sq*t_1_sq + 2*G*t_1 - 1) + G_sq*(G_sq*t_1_sq + 4*G*t_1 + 6))*exp_0_cos_1_term + G_fv*t_0_sq*exp_1_sin_0_term - G_fv*t_1_sq*exp_0_sin_1_term + 2*G_fr*t_0*exp_1_sin_0_term - 2*G_fr*t_1*exp_0_sin_1_term - 2*G_cub*exp_0_sin_1_term + 2*G_cub*exp_1_sin_0_term)
    + d*(DM_sx*G*t_0_cub*exp_1_sin_0_term - DM_sx*G*t_1_cub*exp_0_sin_1_term - 3*DM_sx*t_0_sq*exp_1_sin_0_term + 3*DM_sx*t_1_sq*exp_0_sin_1_term + 3*DM_fr*G_cub*t_0_cub*exp_1_sin_0_term - 3*DM_fr*G_cub*t_1_cub*exp_0_sin_1_term - 3*DM_fr*G_sq*t_0_sq*exp_1_sin_0_term + 3*DM_fr*G_sq*t_1_sq*exp_0_sin_1_term - 18*DM_fr*G*t_0*exp_1_sin_0_term + 18*DM_fr*G*t_1*exp_0_sin_1_term - 6*DM_fr*exp_0_sin_1_term + 6*DM_fr*exp_1_sin_0_term + 3*DM_sq*G_fv*t_0_cub*exp_1_sin_0_term - 3*DM_sq*G_fv*t_1_cub*exp_0_sin_1_term + 3*DM_sq*G_fr*t_0_sq*exp_1_sin_0_term - 3*DM_sq*G_fr*t_1_sq*exp_0_sin_1_term - 12*DM_sq*G_cub*t_0*exp_1_sin_0_term + 12*DM_sq*G_cub*t_1*exp_0_sin_1_term + 36*DM_sq*G_sq*exp_0_sin_1_term - 36*DM_sq*G_sq*exp_1_sin_0_term - DM*(DM_sx*t_0_cub + 3*DM_fr*t_0*(G_sq*t_0_sq + 2*G*t_0 - 2.) + 3*DM_sq*G*(G_cub*t_0_cub + 4*G_sq*t_0_sq + 4*G*t_0 - 8) + G_cub*(G_cub*t_0_cub + 6*G_sq*t_0_sq + 18*G*t_0 + 24.))*exp_1_cos_0_term + DM*(DM_sx*t_1_cub + 3*DM_fr*t_1*(G_sq*t_1_sq + 2*G*t_1 - 2.) + 3*DM_sq*G*(G_cub*t_1_cub + 4*G_sq*t_1_sq + 4*G*t_1 - 8) + G_cub*(G_cub*t_1_cub + 6*G_sq*t_1_sq + 18*G*t_1 + 24.))*exp_0_cos_1_term + pow(G, 7)*t_0_cub*exp_1_sin_0_term - pow(G, 7)*t_1_cub*exp_0_sin_1_term + 3*pow(G, 6)*t_0_sq*exp_1_sin_0_term - 3*pow(G, 6)*t_1_sq*exp_0_sin_1_term + 6*G_fv*t_0*exp_1_sin_0_term - 6*G_fv*t_1*exp_0_sin_1_term - 6*G_fr*exp_0_sin_1_term + 6*G_fr*exp_1_sin_0_term))
    *sqrt(2.)*sqrt(delta_t)*exp(-G*(t_0 + t_1) + 0.5*delta_t_sq*(-DM_sq + G_sq))/pow(DM_sq + G_sq, 4.);
}


WITHIN_KERNEL
void integralSpline( ftype result[2],
                     const ftype vn[10], const ftype va[10],const ftype vb[10], const ftype vc[10],const ftype vd[10],
                    const ftype *norm, const ftype G, const ftype DG, const ftype DM,
                     const ftype delta_t,
                     const ftype tLL, const ftype tUL,
                     const ftype t_offset,
                     GLOBAL_MEM const ftype *coeffs)
{
  //int bin0 = 0;
  // ftype tS = tLL-t_offset;
  ftype tS = 0;
  // ftype tE = KNOTS[bin0+1]-t_offset;
  ftype tE = 0;
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

    ftype c0 = getCoeff(coeffs,bin,0);
    ftype c1 = getCoeff(coeffs,bin,1);
    ftype c2 = getCoeff(coeffs,bin,2);
    ftype c3 = getCoeff(coeffs,bin,3);

    ftype ta = get_int_ta_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ftype tb = get_int_tb_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ftype tc = get_int_tc_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);
    ftype td = get_int_td_spline( delta_t, G, DM, DG, c0, c1, c2, c3, tS, tE);

    for(int k=0; k<10; k++)
    {
      result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
      result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
    }
  }
}



















WITHIN_KERNEL
void integralFullSpline( ftype result[2],
                         const ftype vn[10], const ftype va[10],const ftype vb[10], const ftype vc[10],const ftype vd[10],
                         const ftype *norm, const ftype G, const ftype DG, const ftype DM,
                         const ftype delta_t,
                         const ftype tLL, const ftype tUL,
                         const ftype t_offset,
                         GLOBAL_MEM const ftype *coeffs)
{
  ftype integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);

  ftype ta = integrals[0];
  ftype tb = integrals[1];
  ftype tc = integrals[2];
  ftype td = integrals[3];

  for(int k=0; k<10; k++)
  {
    result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
    result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
  }
  #ifdef DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
  {
    printf("                   : t_offset=%+.16f  delta_t=%+.16f\n", t_offset, delta_t);
  }
  #endif
}








////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 ////////////////////////////////////////////////////////////

WITHIN_KERNEL
ftype getOneSplineTimeAcc(const ftype t,
                             GLOBAL_MEM const ftype *coeffs,
                             const ftype sigma, const ftype gamma,
                             const ftype tLL, const ftype tUL)
{
  // Compute pdf value
  ftype erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  ftype fpdf = 1.0; ftype ipdf = 0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2.0*t) ) * (erf_value);
  fpdf *= time_efficiency(t, coeffs , tLL, tUL);

  // Compute per event normatization
  ftype ti  = 0.0; ftype tf  =  0.0;
  ftype c0 = 0.0; ftype c1 = 0.0; ftype c2 = 0.0; ftype c3 = 0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    c0 = getCoeff(coeffs,k,0);
    c1 = getCoeff(coeffs,k,1);
    c2 = getCoeff(coeffs,k,2);
    c3 = getCoeff(coeffs,k,3);

    ipdf += (exp((pow(gamma,2)*pow(sigma,2))/2.)*((c1*(-exp(-(gamma*tf))
    + exp(-(gamma*ti)) -
    (gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) - (gamma*tf)/exp(gamma*tf) +
    (gamma*ti)/exp(gamma*ti) +
    erf(tf/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) + ((1 +
    gamma*tf)*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) -
    erf(ti/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erf((gamma*sigma)/sqrt(2.0) - ti/(sqrt(2.0)*sigma))/exp(gamma*ti) -
    (gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,2) -
    (c2*(2/exp(gamma*tf) - 2/exp(gamma*ti) +
    (2*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) -
    (2*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) + (2*gamma*tf)/exp(gamma*tf) +
    (pow(gamma,2)*sqrt(2/M_PI)*sigma*tf)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (pow(gamma,2)*pow(tf,2))/exp(gamma*tf) - (2*gamma*ti)/exp(gamma*ti) -
    (pow(gamma,2)*sqrt(2/M_PI)*sigma*ti)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) -
    (pow(gamma,2)*pow(ti,2))/exp(gamma*ti) - ((2 +
    pow(gamma,2)*pow(sigma,2))*erf(tf/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) - ((2 + 2*gamma*tf +
    pow(gamma,2)*pow(tf,2))*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) +
    (2*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,2)*pow(sigma,2))/2.) +
    (pow(gamma,2)*pow(sigma,2)*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) + (2*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (2*gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (pow(gamma,2)*pow(ti,2)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,3) -
    (c3*(6/exp(gamma*tf) - 6/exp(gamma*ti) +
    (6*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) -
    (6*gamma*sqrt(2/M_PI)*sigma)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) +
    (2*pow(gamma,3)*sqrt(2/M_PI)*pow(sigma,3))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(tf,2))/(2.*pow(sigma,2))) -
    (2*pow(gamma,3)*sqrt(2/M_PI)*pow(sigma,3))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(ti,2))/(2.*pow(sigma,2))) + (6*gamma*tf)/exp(gamma*tf) +
    (3*pow(gamma,2)*sqrt(2/M_PI)*sigma*tf)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(tf,2))/(2.*pow(sigma,2))) +
    (3*pow(gamma,2)*pow(tf,2))/exp(gamma*tf) +
    (pow(gamma,3)*sqrt(2/M_PI)*sigma*pow(tf,2))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(tf,2))/(2.*pow(sigma,2))) +
    (pow(gamma,3)*pow(tf,3))/exp(gamma*tf) - (6*gamma*ti)/exp(gamma*ti) -
    (3*pow(gamma,2)*sqrt(2/M_PI)*sigma*ti)/exp((pow(gamma,2)*pow(sigma,4) +
    pow(ti,2))/(2.*pow(sigma,2))) -
    (3*pow(gamma,2)*pow(ti,2))/exp(gamma*ti) -
    (pow(gamma,3)*sqrt(2/M_PI)*sigma*pow(ti,2))/exp((pow(gamma,2)*pow(sigma,
    4) + pow(ti,2))/(2.*pow(sigma,2))) -
    (pow(gamma,3)*pow(ti,3))/exp(gamma*ti) - (3*(2 +
    pow(gamma,2)*pow(sigma,2))*erf(tf/(sqrt(2.0)*sigma)))/exp((pow(gamma,
    2)*pow(sigma,2))/2.) - ((6 + 6*gamma*tf + 3*pow(gamma,2)*pow(tf,2) +
    pow(gamma,3)*pow(tf,3))*erf((gamma*sigma)/sqrt(2.0) -
    tf/(sqrt(2.0)*sigma)))/exp(gamma*tf) +
    (6*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(gamma,2)*pow(sigma,2))/2.) +
    (3*pow(gamma,2)*pow(sigma,2)*erf(ti/(sqrt(2.0)*sigma)))/exp((pow(
    gamma,2)*pow(sigma,2))/2.) + (6*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (6*gamma*ti*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (3*pow(gamma,2)*pow(ti,2)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti) +
    (pow(gamma,3)*pow(ti,3)*erf((gamma*sigma)/sqrt(2.0) -
    ti/(sqrt(2.0)*sigma)))/exp(gamma*ti)))/pow(gamma,4) +
    (c0*(erf(tf/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erf(ti/(sqrt(2.0)*sigma))/exp((pow(gamma,2)*pow(sigma,2))/2.) -
    erfc((gamma*pow(sigma,2) - tf)/(sqrt(2.0)*sigma))/exp(gamma*tf) +
    erfc((gamma*pow(sigma,2) -
    ti)/(sqrt(2.0)*sigma))/exp(gamma*ti)))/gamma))/2.;
  }

  return fpdf/ipdf;

}



////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 x sp2 //////////////////////////////////////////////////////

WITHIN_KERNEL
ftype getTwoSplineTimeAcc(const ftype t, GLOBAL_MEM const ftype *coeffs2,
                          GLOBAL_MEM const ftype *coeffs1,
                          const ftype sigma, const ftype gamma, const ftype tLL,
                          const ftype tUL)
{
  // Compute pdf
  ftype erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  ftype fpdf = 1.0; ftype ipdf = 0.0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2*t) ) * (erf_value);
  fpdf *= time_efficiency(t, coeffs1, tLL, tUL);
  fpdf *= time_efficiency(t, coeffs2, tLL, tUL);

  // Compute per event normatization
  ftype ti  = 0.0;  ftype tf  =  0.0;
  ftype b0 = 0.0; ftype b1 = 0.0; ftype b2 = 0.0; ftype b3 = 0.0;
  ftype r0 = 0.0; ftype r1 = 0.0; ftype r2 = 0.0; ftype r3 = 0.0;
  ftype term1i = 0.0; ftype term2i = 0.0;
  ftype term1f = 0.0; ftype term2f = 0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    r0 = getCoeff(coeffs1,k,0);
    r1 = getCoeff(coeffs1,k,1);
    r2 = getCoeff(coeffs1,k,2);
    r3 = getCoeff(coeffs1,k,3);
    b0 = getCoeff(coeffs2,k,0);
    b1 = getCoeff(coeffs2,k,1);
    b2 = getCoeff(coeffs2,k,2);
    b3 = getCoeff(coeffs2,k,3);

    term1i = -((exp(gamma*ti - (ti*(2*gamma*pow(sigma,2) +
    ti))/(2.*pow(sigma,2)))*sigma*(b3*(720*r3 + 120*gamma*(r2 + 3*r3*ti)
    + 12*pow(gamma,2)*(2*r1 + 5*r2*ti + 10*r3*(2*pow(sigma,2) +
    pow(ti,2))) + 2*pow(gamma,3)*(3*r0 + 6*r1*ti + 10*r2*(2*pow(sigma,2)
    + pow(ti,2)) + 15*r3*ti*(3*pow(sigma,2) + pow(ti,2))) +
    pow(gamma,5)*(3*pow(sigma,2)*(r1 + 5*r3*pow(sigma,2))*ti + (r1 +
    5*r3*pow(sigma,2))*pow(ti,3) + r3*pow(ti,5) + r0*(2*pow(sigma,2) +
    pow(ti,2)) + r2*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(ti,2) +
    pow(ti,4))) + pow(gamma,4)*(3*(r0 + 5*r2*pow(sigma,2))*ti +
    5*r2*pow(ti,3) + 4*r1*(2*pow(sigma,2) + pow(ti,2)) +
    6*r3*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(ti,2) + pow(ti,4)))) +
    gamma*(120*b2*r3 + b1*gamma*(24*r3 + 6*gamma*(r2 + 2*r3*ti) +
    pow(gamma,2)*(2*r1 + 8*r3*pow(sigma,2) + 3*r2*ti + 4*r3*pow(ti,2)) +
    pow(gamma,3)*(r0 + 2*r2*pow(sigma,2) + r1*ti + 3*r3*pow(sigma,2)*ti +
    r2*pow(ti,2) + r3*pow(ti,3))) + b2*gamma*(24*r2 + 60*r3*ti +
    pow(gamma,2)*(2*r0 + 8*r2*pow(sigma,2) + 3*(r1 +
    5*r3*pow(sigma,2))*ti + 4*r2*pow(ti,2) + 5*r3*pow(ti,3)) +
    pow(gamma,3)*(2*pow(sigma,2)*(r1 + 4*r3*pow(sigma,2)) + (r0 +
    3*r2*pow(sigma,2))*ti + (r1 + 4*r3*pow(sigma,2))*pow(ti,2) +
    r2*pow(ti,3) + r3*pow(ti,4)) + 2*gamma*(3*r1 + 6*r2*ti +
    10*r3*(2*pow(sigma,2) + pow(ti,2)))) + b0*pow(gamma,2)*(6*r3 +
    gamma*(2*r2 + 3*r3*ti + gamma*(r1 + 2*r3*pow(sigma,2) + r2*ti +
    r3*pow(ti,2)))))))/(pow(gamma,6)*sqrt(2*M_PI)));
    term1f = -((exp(gamma*tf - (tf*(2*gamma*pow(sigma,2) +
    tf))/(2.*pow(sigma,2)))*sigma*(b3*(720*r3 + 120*gamma*(r2 + 3*r3*tf)
    + 12*pow(gamma,2)*(2*r1 + 5*r2*tf + 10*r3*(2*pow(sigma,2) +
    pow(tf,2))) + 2*pow(gamma,3)*(3*r0 + 6*r1*tf + 10*r2*(2*pow(sigma,2)
    + pow(tf,2)) + 15*r3*tf*(3*pow(sigma,2) + pow(tf,2))) +
    pow(gamma,5)*(3*pow(sigma,2)*(r1 + 5*r3*pow(sigma,2))*tf + (r1 +
    5*r3*pow(sigma,2))*pow(tf,3) + r3*pow(tf,5) + r0*(2*pow(sigma,2) +
    pow(tf,2)) + r2*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(tf,2) +
    pow(tf,4))) + pow(gamma,4)*(3*(r0 + 5*r2*pow(sigma,2))*tf +
    5*r2*pow(tf,3) + 4*r1*(2*pow(sigma,2) + pow(tf,2)) +
    6*r3*(8*pow(sigma,4) + 4*pow(sigma,2)*pow(tf,2) + pow(tf,4)))) +
    gamma*(120*b2*r3 + b1*gamma*(24*r3 + 6*gamma*(r2 + 2*r3*tf) +
    pow(gamma,2)*(2*r1 + 8*r3*pow(sigma,2) + 3*r2*tf + 4*r3*pow(tf,2)) +
    pow(gamma,3)*(r0 + 2*r2*pow(sigma,2) + r1*tf + 3*r3*pow(sigma,2)*tf +
    r2*pow(tf,2) + r3*pow(tf,3))) + b2*gamma*(24*r2 + 60*r3*tf +
    pow(gamma,2)*(2*r0 + 8*r2*pow(sigma,2) + 3*(r1 +
    5*r3*pow(sigma,2))*tf + 4*r2*pow(tf,2) + 5*r3*pow(tf,3)) +
    pow(gamma,3)*(2*pow(sigma,2)*(r1 + 4*r3*pow(sigma,2)) + (r0 +
    3*r2*pow(sigma,2))*tf + (r1 + 4*r3*pow(sigma,2))*pow(tf,2) +
    r2*pow(tf,3) + r3*pow(tf,4)) + 2*gamma*(3*r1 + 6*r2*tf +
    10*r3*(2*pow(sigma,2) + pow(tf,2)))) + b0*pow(gamma,2)*(6*r3 +
    gamma*(2*r2 + 3*r3*tf + gamma*(r1 + 2*r3*pow(sigma,2) + r2*tf +
    r3*pow(tf,2)))))))/(pow(gamma,6)*sqrt(2*M_PI)));
    term2i = (exp(gamma*ti)*(3*b3*(240*r3 + gamma*(2*pow(gamma,2)*r0 +
    8*gamma*r1 + 40*r2 + gamma*(pow(gamma,3)*r0 + 4*pow(gamma,2)*r1 +
    20*gamma*r2 + 120*r3)*pow(sigma,2) + pow(gamma,3)*(pow(gamma,2)*r1 +
    5*gamma*r2 + 30*r3)*pow(sigma,4) + 5*pow(gamma,5)*r3*pow(sigma,6))) +
    gamma*(120*b2*r3 + b2*gamma*(2*pow(gamma,2)*r0 + 6*gamma*r1 + 24*r2 +
    gamma*(pow(gamma,3)*r0 + 3*pow(gamma,2)*r1 + 12*gamma*r2 +
    60*r3)*pow(sigma,2) + 3*pow(gamma,3)*(gamma*r2 + 5*r3)*pow(sigma,4))
    + b0*pow(gamma,2)*(6*r3 + gamma*(2*r2 + gamma*(gamma*r0 + r1 +
    (gamma*r2 + 3*r3)*pow(sigma,2)))) + b1*gamma*(24*r3 + gamma*(6*r2 +
    gamma*(gamma*r0 + 2*r1 + (pow(gamma,2)*r1 + 3*gamma*r2 +
    12*r3)*pow(sigma,2) +
    3*pow(gamma,2)*r3*pow(sigma,4))))))*erf(ti/(sqrt(2.0)*sigma)) -
    exp((pow(gamma,2)*pow(sigma,2))/2.)*(b3*(720*r3 + 120*gamma*(r2 +
    6*r3*ti) + pow(gamma,5)*pow(ti,2)*(3*r0 + 4*r1*ti + 5*r2*pow(ti,2) +
    6*r3*pow(ti,3)) + 24*pow(gamma,2)*(r1 + 5*ti*(r2 + 3*r3*ti)) +
    pow(gamma,6)*pow(ti,3)*(r0 + ti*(r1 + ti*(r2 + r3*ti))) +
    6*pow(gamma,3)*(r0 + 2*ti*(2*r1 + 5*ti*(r2 + 2*r3*ti))) +
    2*pow(gamma,4)*ti*(3*r0 + ti*(6*r1 + 5*ti*(2*r2 + 3*r3*ti)))) +
    gamma*(b2*(120*r3 + 24*gamma*(r2 + 5*r3*ti) + 6*pow(gamma,2)*(r1 +
    4*r2*ti + 10*r3*pow(ti,2)) + pow(gamma,4)*ti*(2*r0 + 3*r1*ti +
    4*r2*pow(ti,2) + 5*r3*pow(ti,3)) + 2*pow(gamma,3)*(r0 + 3*r1*ti +
    6*r2*pow(ti,2) + 10*r3*pow(ti,3)) + pow(gamma,5)*pow(ti,2)*(r0 +
    ti*(r1 + ti*(r2 + r3*ti)))) + gamma*(b1*(24*r3 + 6*gamma*(r2 +
    4*r3*ti) + pow(gamma,3)*(r0 + 2*r1*ti + 3*r2*pow(ti,2) +
    4*r3*pow(ti,3)) + 2*pow(gamma,2)*(r1 + 3*ti*(r2 + 2*r3*ti)) +
    pow(gamma,4)*ti*(r0 + ti*(r1 + ti*(r2 + r3*ti)))) + b0*gamma*(6*r3 +
    gamma*(2*r2 + 6*r3*ti + gamma*(r1 + ti*(2*r2 + 3*r3*ti)) +
    pow(gamma,2)*(r0 + ti*(r1 + ti*(r2 + r3*ti))))))))*erfc((gamma*sigma
    - ti/sigma)/sqrt(2.0)))/(2.*exp(gamma*ti)*pow(gamma,7));
    term2f = (exp(gamma*tf)*(3*b3*(240*r3 + gamma*(2*pow(gamma,2)*r0 +
    8*gamma*r1 + 40*r2 + gamma*(pow(gamma,3)*r0 + 4*pow(gamma,2)*r1 +
    20*gamma*r2 + 120*r3)*pow(sigma,2) + pow(gamma,3)*(pow(gamma,2)*r1 +
    5*gamma*r2 + 30*r3)*pow(sigma,4) + 5*pow(gamma,5)*r3*pow(sigma,6))) +
    gamma*(120*b2*r3 + b2*gamma*(2*pow(gamma,2)*r0 + 6*gamma*r1 + 24*r2 +
    gamma*(pow(gamma,3)*r0 + 3*pow(gamma,2)*r1 + 12*gamma*r2 +
    60*r3)*pow(sigma,2) + 3*pow(gamma,3)*(gamma*r2 + 5*r3)*pow(sigma,4))
    + b0*pow(gamma,2)*(6*r3 + gamma*(2*r2 + gamma*(gamma*r0 + r1 +
    (gamma*r2 + 3*r3)*pow(sigma,2)))) + b1*gamma*(24*r3 + gamma*(6*r2 +
    gamma*(gamma*r0 + 2*r1 + (pow(gamma,2)*r1 + 3*gamma*r2 +
    12*r3)*pow(sigma,2) +
    3*pow(gamma,2)*r3*pow(sigma,4))))))*erf(tf/(sqrt(2.0)*sigma)) -
    exp((pow(gamma,2)*pow(sigma,2))/2.)*(b3*(720*r3 + 120*gamma*(r2 +
    6*r3*tf) + pow(gamma,5)*pow(tf,2)*(3*r0 + 4*r1*tf + 5*r2*pow(tf,2) +
    6*r3*pow(tf,3)) + 24*pow(gamma,2)*(r1 + 5*tf*(r2 + 3*r3*tf)) +
    pow(gamma,6)*pow(tf,3)*(r0 + tf*(r1 + tf*(r2 + r3*tf))) +
    6*pow(gamma,3)*(r0 + 2*tf*(2*r1 + 5*tf*(r2 + 2*r3*tf))) +
    2*pow(gamma,4)*tf*(3*r0 + tf*(6*r1 + 5*tf*(2*r2 + 3*r3*tf)))) +
    gamma*(b2*(120*r3 + 24*gamma*(r2 + 5*r3*tf) + 6*pow(gamma,2)*(r1 +
    4*r2*tf + 10*r3*pow(tf,2)) + pow(gamma,4)*tf*(2*r0 + 3*r1*tf +
    4*r2*pow(tf,2) + 5*r3*pow(tf,3)) + 2*pow(gamma,3)*(r0 + 3*r1*tf +
    6*r2*pow(tf,2) + 10*r3*pow(tf,3)) + pow(gamma,5)*pow(tf,2)*(r0 +
    tf*(r1 + tf*(r2 + r3*tf)))) + gamma*(b1*(24*r3 + 6*gamma*(r2 +
    4*r3*tf) + pow(gamma,3)*(r0 + 2*r1*tf + 3*r2*pow(tf,2) +
    4*r3*pow(tf,3)) + 2*pow(gamma,2)*(r1 + 3*tf*(r2 + 2*r3*tf)) +
    pow(gamma,4)*tf*(r0 + tf*(r1 + tf*(r2 + r3*tf)))) + b0*gamma*(6*r3 +
    gamma*(2*r2 + 6*r3*tf + gamma*(r1 + tf*(2*r2 + 3*r3*tf)) +
    pow(gamma,2)*(r0 + tf*(r1 + tf*(r2 + r3*tf))))))))*erfc((gamma*sigma
    - tf/sigma)/sqrt(2.0)))/(2.*exp(gamma*tf)*pow(gamma,7));

    ipdf += (term1f + term2f) - (term1i + term2i);

  }

  return fpdf/ipdf;

}



////////////////////////////////////////////////////////////////////////////////
