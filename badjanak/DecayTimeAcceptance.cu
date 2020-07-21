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



WITHIN_KERNEL
unsigned int getTimeBin(${ftype} const t)
{
  int _i = 0;
  int _n = NKNOTS-1;
  while(_i <= _n )
  {
    if( t < KNOTS[_i] ) {break;}
    _i++;
  }
  if (0 == _i) {printf("WARNING: t=%.16lf below first knot!\n",t);}
  return _i - 1;

}


WITHIN_KERNEL
unsigned int getMassBin(${ftype} const t)
{
  int _i = 0;
  int _n = MKNOTS-1;
  while(_i <= _n )
  {
    if( t < X_M[_i] ) {break;}
    _i++;
  }
  if (0 == _i) {printf("WARNING: t=%.16lf below first knot!\n",t);}
  return _i - 1;
}



WITHIN_KERNEL
${ftype} getKnot(int i)
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
${ftype} getCoeff(GLOBAL_MEM ${ftype} *mat, int r, int c)
{
  return mat[4*r+c];
}



WITHIN_KERNEL
${ftype} calcTimeAcceptance(${ftype} t, GLOBAL_MEM ${ftype} *coeffs, ${ftype} tLL, ${ftype} tUL)
{
  int bin   = getTimeBin(t);
  ${ftype} c0 = getCoeff(coeffs,bin,0);
  ${ftype} c1 = getCoeff(coeffs,bin,1);
  ${ftype} c2 = getCoeff(coeffs,bin,2);
  ${ftype} c3 = getCoeff(coeffs,bin,3);
  #ifdef DEBUG
  if (DEBUG >= 3 && ( get_global_id(0) == DEBUG_EVT))
  {
    printf("\nTIME ACC           : t=%.16lf\tbin=%d\tc=[%+lf\t%+lf\t%+lf\t%+lf]\tdta=%+.16lf\n",
           t,bin,c0,c1,c2,c3, (c0 + t*(c1 + t*(c2 + t*c3))) );
  }
  #endif

  return (c0 + t*(c1 + t*(c2 + t*c3)));
}


WITHIN_KERNEL
${ctype} ipanema_erfc2(${ctype} z)
{
  ${ftype} re = -z.x * z.x + z.y * z.y;
  ${ftype} im = -2. * z.x * z.y;
  ${ctype} expmz = cexp( cnew(re,im) );

  if (z.x >= 0.0) {
    return                 cmul( expmz, faddeeva(cnew(-z.y,+z.x)) );
  }
  else{
    ${ctype} ans = cmul( expmz, faddeeva(cnew(+z.y,-z.x)) );
    return cnew(2.0-ans.x, ans.y);
  }
}



WITHIN_KERNEL
${ctype} ipanema_erfc(${ctype} z)
{
  if (z.y<0)
  {
    ${ctype} ans = ipanema_erfc2( cnew(-z.x, -z.y) );
    return cnew( 2.0-ans.x, -ans.y);
  }
  else{
    return ipanema_erfc2(z);
  }
}



WITHIN_KERNEL
${ctype} cErrF_2(${ctype} x)
{
  ${ctype} I = cnew(0.0,1.0);
  ${ctype} z = cmul(I,x);
  ${ctype} result = cmul( cexp(  cmul(cnew(-1,0),cmul(x,x))   ) , faddeeva(z) );

  //printf("z = %+.16lf %+.16lfi\n", z.x, z.y);
  //printf("fad = %+.16lf %+.16lfi\n", faddeeva(z).x, faddeeva(z).y);

  if (x.x > 20.0){// && fabs(x.y < 20.0)
    result = cnew(0.0,0);
  }
  if (x.x < -20.0){// && fabs(x.y < 20.0)
    result = cnew(2.0,0);
  }

  return result;
}


WITHIN_KERNEL
${ctype} cerfc(${ctype} z)
{
  if (z.y<0)
  {
    ${ctype} ans = cErrF_2( cnew(-z.x, -z.y) );
    return cnew( 2.0-ans.x, -ans.y);
  }
  else{
    return cErrF_2(z);
  }
}








WITHIN_KERNEL
${ctype} expconv_simon(${ftype} t, ${ftype} G, ${ftype} omega, ${ftype} sigma)
{
  ${ctype} I  = cnew( 0, 1);
  ${ctype} I2 = cnew(-1, 0);
  ${ctype} I3 = cnew( 0,-1);
  ${ftype} sigma2 = sigma*sigma;

  if (omega == 0)
  {
    if( t > -6.0*sigma ){
      ${ftype} exp_part = 0.5*exp(-t*G + 0.5*G*G*sigma2 -0.5*omega*omega*sigma2);
      ${ctype} my_erfc = ipanema_erfc(cnew(sigma*G/sqrt(2.0) - t/sigma/sqrt(2.0),0));
      return cmul( cnew(exp_part,0) , my_erfc );
    }
    else{
      return cnew(0,0);
    }
  }
  else //(omega != 0)
  {
    //${ftype} c1 = 0.5;

    ${ftype} exp1arg = 0.5*sigma2*(G*G - omega*omega) - t*G;
    ${ctype} exp1 = cnew( 0.5*exp(exp1arg) ,0);

    ${ftype} exp2arg = -omega*(t - sigma2*G);
    ${ctype} exp2 = cnew(cos(exp2arg), sin(exp2arg));

    ${ctype} cerfarg = cnew(sigma*G/sqrt(2.0) - t/(sigma*sqrt(2.0)) , +omega*sigma/sqrt(2.0));
    ${ctype} cerf;

    if  (cerfarg.x < -20.0)
    {
      cerf = cnew(2.0,0.0);
    }
    else
    {
      cerf = cErrF_2(cerfarg);//best complex error function
    }
    ${ctype} c2 = cmul(exp2, cerf);
    //${ftype} im = -c2.x;//exp*sin
    //${ftype} re = +c2.real();//exp*cos

    return cmul( exp1 , cadd( cnew(c2.x,0), cmul( I3, cnew(c2.y,0) ) ) );
  }

}






WITHIN_KERNEL
${ctype} expconv(${ftype} t, ${ftype} G, ${ftype} omega, ${ftype} sigma)
{
  // OpenCL need beautiful code, doesn't it?
  ${ftype} sigma2 = sigma*sigma;

  if( t > SIGMA_THRESHOLD*sigma )
  {
    ${ftype} a = exp(-G*t+0.5*G*G*sigma2-0.5*omega*omega*sigma2);
    ${ftype} b = omega*(t-G*sigma2);
    return cnew(a*cos(b),a*sin(b));
  }
  else
  {
    //printf("dont like it\n");
    ${ctype} z, fad;
    z   = cnew(-omega*sigma2/(sigma*sqrt(2.)), -(t-sigma2*G)/(sigma*sqrt(2.)));
    fad = faddeeva(z);

 if ( (t>0.3006790) && (t<0.3006792) ){
           printf("z   = %+.16lf%+.16lf\n",z.x,z.y );
           printf("fad = %+.16lf%+.16lf\n",fad.x,fad.y );
           //printf("shit = %+.16lf%+.16lf\n",shit.x,shit.y );
           double2 res = cnew(0.5*exp(-0.5*t*t/sigma2),0.0);
           printf("res   = %+.16lf\n",res.x,res.y );
     }
    
    return cmul( cnew(fad.x,-fad.y), cnew(0.5*exp(-0.5*t*t/sigma2),0) );
  }
}



WITHIN_KERNEL
${ctype} getK(${ctype} z, int n)
{
  ${ctype} z2 = cmul(z,z);
  ${ctype} z3 = cmul(z,z2);
  ${ctype} z4 = cmul(z,z3);
  ${ctype} z5 = cmul(z,z4);
  ${ctype} z6 = cmul(z,z5);
  ${ctype} w;

  if (n == 0)      {
    w = cmul( cnew(2.0,0.0), z);
    return cdiv(cnew( 1.0,0.0), w );
  }
  else if (n == 1) {
    w = cmul( cnew(2.0,0.0), z2);
    return cdiv(cnew(1.0,0.0), w );
  }
  else if (n == 2) {
    w = cdiv( cnew(1.0,0.0), z2 );
    w = cadd( cnew(1.0,0.0), w );
    return cmul( cdiv(cnew(1.0,0.0),z) , w );
  }
  else if (n == 3) {
    w = cdiv( cnew(1.0,0.0), z2 );
    w = cadd( cnew(1.0,0.0), w );
    return cmul( cdiv(cnew(3.0,0.0),z2) , w );
  }
  // else if (n == 4) {
  //   return cdiv(cnew( 6.,0), z*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 5) {
  //   return cdiv(cnew(30.,0), (z*z)*(1.+2./(z*z)+2./(z*z*z*z))  );
  // }
  // else if (n == 6) {
  //   return cdiv(cnew(60.,0), z*(1.+3./(z*z)+6./(z*z*z*z)+6./(z*z*z*z*z*z))  );
  // }

  return cnew(0.,0.);
}



WITHIN_KERNEL
${ctype} getM(${ftype} x, int n, ${ftype} t, ${ftype} sigma, ${ftype} gamma, ${ftype} omega)
{
  ${ctype} conv_term, z;
  ${ctype} I  = cnew(0,+1);
  ${ctype} I2 = cnew(-1,0);
  ${ctype} I3 = cnew(0,-1);

  z = cnew(gamma*sigma/sqrt(2.0),-omega*sigma/sqrt(2.0));
  ${ctype} arg1 = csub( cmul(z,z), cmul(cnew(2*x,0),z) );
  ${ctype} arg2 = csub(z,cnew(x,0));
  //conv_term = 5.0*expconv(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));
  // warning there are improvement to do here!!!
  if (omega == 0){
    conv_term = cmul( cexp(arg1), ipanema_erfc2(arg2) );
  }
  else{
    conv_term = cmul( cexp(arg1), cErrF_2(arg2) );
    //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);
    //conv_term = 2.0*exp(-gamma*t+0.5*gamma*gamma*sigma*sigma-0.5*omega*omega*sigma*sigma)*(cos(omega*(t-gamma*sigma*sigma)) + I*sin(omega*(t-gamma*sigma*sigma)));
  }
  //conv_term = 2.0*expconv_simon(t,gamma,omega,sigma);///(sqrt(0.5*M_PI));

  // #ifdef DEBUG
  // if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) ){
  //   printf("\nerfc*exp = %+.16lf %+.16lfi\n",  conv_term.x, conv_term.y);
  //   printf("erfc = %+.16lf %+.16lfi\n",  ipanema_erfc(arg2).x, ipanema_erfc(arg2).y );
  //   printf("cErrF_2 = %+.16lf %+.16lfi\n",  cErrF_2(arg2).x, cErrF_2(arg2).y );
  //   printf("exp  = %+.16lf %+.16lfi\n",  cexp(arg1).x, cexp(arg1).y );
  //   printf("z    = %+.16lf %+.16lfi     %+.16lf %+.16lf %+.16lf        x = %+.16lf\n",  z.x, z.y, gamma, omega, sigma, x);
  // }
  // #endif

  if (n == 0)
  {
    ${ctype} a = cnew(erf(x),0.);
    ${ctype} b = conv_term;
    return csub(a,b);
  }
  else if (n == 1)
  {
    ${ctype} a = cnew(sqrt(1./M_PI)*exp(-x*x),0.);
    ${ctype} b = cnew(x,0);
    b = cmul(b,conv_term);
    return cmul(cnew(2,0),csub(a,b));
  }
  else if (n == 2)
  {
    // return 2.*(-2.*x*exp(-x*x)*${ctype}(sqrt(1./M_PI),0.)-(2.*x*x-1.)*conv_term);
    ${ctype} a = cnew(-2.*x*exp(-x*x)*sqrt(1./M_PI),0.);
    ${ctype} b = cnew(2*x*x-1,0);
    b = cmul(b,conv_term);
    return cmul(cnew(2,0),csub(a,b));
  }
  else if (n == 3)
  {
    // return 4.*(-(2.*x*x-1.)*exp(-x*x)*${ctype}(sqrt(1./M_PI),0.)-x*(2.*x*x-3.)*conv_term);
    ${ctype} a = cnew(-(2.*x*x-1.)*exp(-x*x)*sqrt(1./M_PI),0.);
    ${ctype} b = cnew(x*(2*x*x-3),0);
    b = cmul(b,conv_term);
    return cmul(cnew(4,0),csub(a,b));
  }
  // else if (n == 4)
  // {
  //   return 4.*(exp(-x*x)*(6.*x+4.*x*x*x)*${ctype}(sqrt(1./M_PI),0.)-(3.-12.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 5)
  // {
  //   return 8.*(-(3.-12.*x*x+4.*x*x*x*x)*exp(-x*x)*${ctype}(sqrt(1./M_PI),0.)-x*(15.-20.*x*x+4.*x*x*x*x)*conv_term);
  // }
  // else if (n == 6)
  // {
  //   return 8.*(-exp(-x*x)*(30.*x-40.*x*x*x+8.*x*x*x*x*x)*${ctype}(sqrt(1./M_PI),0.)-(-15.+90.*x*x-60.*x*x*x*x+8.*x*x*x*x*x*x)*conv_term);
  // }
  return cnew(0.,0.);
}



WITHIN_KERNEL
void intgTimeAcceptance(${ftype} time_terms[4], ${ftype} delta_t,
                        ${ftype} G, ${ftype} DG, ${ftype} DM,
                        GLOBAL_MEM ${ftype} *coeffs, ${ftype} t0, ${ftype} tLL, ${ftype} tUL)
{
  // Some constants
  ${ftype} cte1 = 1.0/(sqrt(2.0)*delta_t);
  ${ctype} cte2 = cnew( delta_t/(sqrt(2.0)) , 0 );
  if (delta_t <= 0.0)
  {
    printf("WARNING            : delta_t = %.4f is not a valid value.\n", delta_t);
  }

  // Add tUL to knots list
  ${ftype} x[NTIMEBINS] = {0.};
  ${ftype} knots[NTIMEBINS] = {0.};
  knots[0] = tLL; x[0] = (knots[0] - t0)*cte1;
  for(int i = 1; i < NKNOTS; i++)
  {
    knots[i] = KNOTS[i];
    x[i] = (knots[i] - t0)*cte1;
  }
  knots[NKNOTS] = tUL; x[NKNOTS] = (knots[NKNOTS] - t0)*cte1;

  // Fill S matrix                (TODO speed to be gained here - S is constant)
  ${ftype} S[SPL_BINS][4][4];
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

  ${ctype} z_expm, K_expm[4], M_expm[SPL_BINS+1][4];
  ${ctype} z_expp, K_expp[4], M_expp[SPL_BINS+1][4];
  ${ctype} z_trig, K_trig[4], M_trig[SPL_BINS+1][4];

  z_expm = cmul( cte2 , cnew(G-0.5*DG,  0) );
  z_expp = cmul( cte2 , cnew(G+0.5*DG,  0) );
  z_trig = cmul( cte2 , cnew(       G,-DM) );

  // Fill Kn                 (only need to calculate this once per minimization)
  for (int j=0; j<4; ++j)
  {
    K_expp[j] = getK(z_expp,j);
    K_expm[j] = getK(z_expm,j);
    K_trig[j] = getK(z_trig,j);
    #ifdef DEBUG
    if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
    {
      printf("K_expp[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n",  j,z_expp.x,z_expp.y,K_expp[j].x,K_expp[j].y);
      printf("K_expm[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n",  j,z_expm.x,z_expm.y,K_expm[j].x,K_expm[j].y);
      printf("K_trig[%d](%+.14lf%+.14lf) = %+.14lf%+.14lf\n\n",j,z_trig.x,z_trig.y,K_trig[j].x,K_trig[j].y);
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
      M_trig[bin][j] = getM(x[bin],j,knots[bin]-t0,delta_t,G,DM);
      if (bin>0){
        #ifdef DEBUG
        if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
        {
          ${ctype} aja = M_expp[bin][j]-M_expp[bin-1][j];
          ${ctype} eje = M_expm[bin][j]-M_expm[bin-1][j];
          ${ctype} iji = M_trig[bin][j]-M_trig[bin-1][j];
          printf("bin=%d M_expp[%d] = %+.14lf%+.14lf\n",  bin,j,aja.x,aja.y);
          printf("bin=%d M_expm[%d] = %+.14lf%+.14lf\n",  bin,j,eje.x,eje.y);
          printf("bin=%d M_trig[%d] = %+.14lf%+.14lf\n\n",bin,j,iji.x,iji.y);
        }
        #endif
      }
    }
  }

  // Fill the delta factors to multiply by the integrals
  ${ftype} delta_t_fact[4];
  for (int i=0; i<4; ++i)
  {
    delta_t_fact[i] = pow(delta_t*sqrt(2.), i+1)/sqrt(2.);
  }

  // Integral calculation for cosh, expm, cos, sin terms
  ${ctype} int_expm = cnew(0.,0.);
  ${ctype} int_expp = cnew(0.,0.);
  ${ctype} int_trig = cnew(0.,0.);
  ${ctype} aux, int_expm_aux, int_expp_aux, int_trig_aux;

  for (int bin=0; bin < SPL_BINS; ++bin)
  {
    for (int j=0; j<=3; ++j)
    {
      for (int k=0; k<=3-j; ++k)
      {
        aux = cnew( S[bin][j][k]*delta_t_fact[j+k] , 0 );

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
        if (DEBUG > 3 && ( threadIdx.x + blockDim.x * blockIdx.x == DEBUG_EVT) )
        {
          //printf("bin=%d int_expm_aux[%d,%d] = %+.14lf%+.14lf\n",  bin,j,k,int_expm_aux.x,int_expm_aux.y);
          printf("bin=%d int_expm[%d,%d] = %+.14lf%+.14lf\n",  bin,j,k,int_expm.x,int_expm.y);
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
    printf("\nNORMALIZATION      : ta=%.16lf\ttb=%.16lf\ttc=%.16lf\ttd=%.16lf\n",
           time_terms[0],time_terms[1],time_terms[2],time_terms[3]);
    printf("                   : sigma=%.16lf\tgamma+=%.16lf\tgamma-=%.16lf\n",
           delta_t, G+0.5*DG, G-0.5*DG);
  }
  #endif
}



WITHIN_KERNEL
void integralFullSpline( ${ftype} result[2],
                         ${ftype} vn[10], ${ftype} va[10],${ftype} vb[10], ${ftype} vc[10],${ftype} vd[10],
                         GLOBAL_MEM ${ftype} *norm, ${ftype} G, ${ftype} DG, ${ftype} DM,
                         ${ftype} delta_t,
                         ${ftype} tLL, ${ftype} tUL,
                         ${ftype} t_offset,
                         GLOBAL_MEM ${ftype} *coeffs)
{
  ${ftype} integrals[4] = {0., 0., 0., 0.};
  intgTimeAcceptance(integrals, delta_t, G, DG, DM, coeffs, t_offset, tLL, tUL);

  ${ftype} ta = integrals[0];
  ${ftype} tb = integrals[1];
  ${ftype} tc = integrals[2];
  ${ftype} td = integrals[3];

  for(int k=0; k<10; k++)
  {
    result[0] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb + vc[k]*tc + vd[k]*td);
    result[1] += vn[k]*norm[k]*(va[k]*ta + vb[k]*tb - vc[k]*tc - vd[k]*td);
  }
  #ifdef DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
  {
    printf("                   : t_offset=%+.16lf  delta_t=%+.16lf\n", t_offset, delta_t);
  }
  #endif
}








////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 ////////////////////////////////////////////////////////////

WITHIN_KERNEL
${ftype} getOneSplineTimeAcc(${ftype} t,
                             GLOBAL_MEM ${ftype} *coeffs,
                             ${ftype} sigma, ${ftype} gamma,
                             ${ftype} tLL, ${ftype} tUL)
{

  // Compute pdf value
  ${ftype} erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  ${ftype} fpdf = 1.0; ${ftype} ipdf = 0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2.0*t) ) * (erf_value);
  fpdf *= calcTimeAcceptance(t, coeffs , tLL, tUL);

  // if ( get_global_id(0) == 0 ) {
  // printf("COEFFS             : %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[0*4+0],coeffs[0*4+1],coeffs[0*4+2],coeffs[0*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[1*4+0],coeffs[1*4+1],coeffs[1*4+2],coeffs[1*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[2*4+0],coeffs[2*4+1],coeffs[2*4+2],coeffs[2*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[3*4+0],coeffs[3*4+1],coeffs[3*4+2],coeffs[3*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[4*4+0],coeffs[4*4+1],coeffs[4*4+2],coeffs[4*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[5*4+0],coeffs[5*4+1],coeffs[5*4+2],coeffs[5*4+3]);
  // printf("                     %+.16lf\t%+.16lf\t%+.16lf\t%+.16lf\n",
  //         coeffs[6*4+0],coeffs[6*4+1],coeffs[6*4+2],coeffs[6*4+3]);
  // }
  // if ( get_global_id(0) < 3 )
  // {
  //   printf("TIME ACC           : t=%.16lf, sigma=%.16lf, gamma=%.16lf, tLL=%.16lf, tUL=%.16lf,     fpdf=%.16lf\n",
  //          t, sigma, gamma, tLL, tUL, fpdf);
  // }

  // Compute per event normatization
  ${ftype} ti  = 0.0;  ${ftype} tf  =  0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    ${ftype} c0 = getCoeff(coeffs,k,0);
    ${ftype} c1 = getCoeff(coeffs,k,1);
    ${ftype} c2 = getCoeff(coeffs,k,2);
    ${ftype} c3 = getCoeff(coeffs,k,3);

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

  // if ( get_global_id(0) == 0)
  // {
  //   printf("TIME ACC           : integral=%.14lf\n",
  //          ipdf);
  // }
  return fpdf/ipdf;
}



////////////////////////////////////////////////////////////////////////////////
// PDF = conv x sp1 x sp2 //////////////////////////////////////////////////////

WITHIN_KERNEL
${ftype} getTwoSplineTimeAcc(${ftype} t,
                             GLOBAL_MEM ${ftype} *coeffs2, GLOBAL_MEM ${ftype} *coeffs1,
                             ${ftype} sigma, ${ftype} gamma, ${ftype} tLL, ${ftype} tUL)
{
  // Compute pdf
  ${ftype} erf_value = 1 - erf((gamma*sigma - t/sigma)/sqrt(2.0));
  ${ftype} fpdf = 1.0;
  fpdf *= 0.5*exp( 0.5*gamma*(sigma*sigma*gamma - 2*t) ) * (erf_value);
  fpdf *= calcTimeAcceptance(t, coeffs1, tLL, tUL);
  fpdf *= calcTimeAcceptance(t, coeffs2, tLL, tUL);

  // Compute per event normatization
  ${ftype} ipdf = 0.0; ${ftype} ti  = 0.0;  ${ftype} tf  =  0.0;
  ${ftype} term1i = 0.0; ${ftype} term2i = 0.0;
  ${ftype} term1f = 0.0; ${ftype} term2f = 0.0;
  for (int k = 0; k < NKNOTS; k++) {

    if (k == NKNOTS-1) {
      ti = KNOTS[NKNOTS-1];
      tf = tUL;
    }
    else {
      ti = KNOTS[k];
      tf = KNOTS[k+1];
    }

    ${ftype} r0 = getCoeff(coeffs1,k,0);
    ${ftype} r1 = getCoeff(coeffs1,k,1);
    ${ftype} r2 = getCoeff(coeffs1,k,2);
    ${ftype} r3 = getCoeff(coeffs1,k,3);
    ${ftype} b0 = getCoeff(coeffs2,k,0);
    ${ftype} b1 = getCoeff(coeffs2,k,1);
    ${ftype} b2 = getCoeff(coeffs2,k,2);
    ${ftype} b3 = getCoeff(coeffs2,k,3);

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
  //printf("%.16lf\n",ipdf);
  return fpdf/ipdf;

}



////////////////////////////////////////////////////////////////////////////////
