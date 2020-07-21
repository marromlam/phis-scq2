////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                       CUDA decay rate Bs -> mumuKK                         //
//                                                                            //
//   Created: 2019-01-25                                                      //
//  Modified: 2019-11-21                                                      //
//    Author: Marcos Romero                                                   //
//                                                                            //
//    This file is part of p-scq packages, Santiago's framework for the       //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
// Functions ///////////////////////////////////////////////////////////////////



WITHIN_KERNEL
${ftype} getN(${ftype} A10, ${ftype} A00, ${ftype} A1a, ${ftype} A1e, ${ftype} C01, int k)
{
  ${ftype} nk;
  switch(k) {
    case 1:  nk = A10*A10; break;
    case 2:  nk = A1a*A1a; break;
    case 3:  nk = A1e*A1e; break;
    case 4:  nk = A1e*A1a; break;
    case 5:  nk = A10*A1a; break;
    case 6:  nk = A10*A1e; break;
    case 7:  nk = A00*A00; break;
    case 8:  nk = C01*A00*A1a; break;
    case 9:  nk = C01*A00*A1e; break;
    case 10: nk = C01*A00*A10; break;
    default: printf("Wrong k index in nk, please check code %d\\n", k);
             return 0.;
  }
  return nk;
}



WITHIN_KERNEL
${ftype} getF(${ftype} cosK, ${ftype} cosL, ${ftype} hphi, int k)
{
  ${ftype} sinK   = sqrt(1. - cosK*cosK);
  ${ftype} sinL   = sqrt(1. - cosL*cosL);
  ${ftype} sinphi = sin(hphi);
  ${ftype} cosphi = cos(hphi);

  ${ftype} fk;
  switch(k) {
    case 1:  fk = cosK*cosK*sinL*sinL; break;
    case 2:  fk = 0.5*sinK*sinK*(1.-cosphi*cosphi*sinL*sinL); break;
    case 3:  fk = 0.5*sinK*sinK*(1.-sinphi*sinphi*sinL*sinL); break;
    case 4:  fk = sinK*sinK*sinL*sinL*sinphi*cosphi; break;
    case 5:  fk = sqrt(2.)*sinK*cosK*sinL*cosL*cosphi; break;
    case 6:  fk = -sqrt(2.)*sinK*cosK*sinL*cosL*sinphi; break;
    case 7:  fk = sinL*sinL/3.; break;
    case 8:  fk = 2.*sinK*sinL*cosL*cosphi/sqrt(6.); break;
    case 9:  fk = -2.*sinK*sinL*cosL*sinphi/sqrt(6.); break;
    case 10: fk = 2.*cosK*sinL*sinL/sqrt(3.); break;
    default: printf("Wrong k index in fk, please check code %d\\n", k);
             return 0.;
  }
  return fk;
}



WITHIN_KERNEL
${ftype} getA(${ftype} p10, ${ftype} p00, ${ftype} p1a, ${ftype} p1e, ${ftype} d10,
						${ftype} d00, ${ftype} d1a, ${ftype} d1e, ${ftype} l10, ${ftype} l00,
						${ftype} l1a, ${ftype} l1e, int k)
{
  ${ftype} ak;
  switch(k) {
    case 1:  ak = 0.5*(1.+l10*l10); break;
    case 2:  ak = 0.5*(1.+l1a*l1a); break;
    case 3:  ak = 0.5*(1.+l1e*l1e); break;
    case 4:  ak = 0.5*(sin(d1e-d1a) - l1e*l1a*sin(d1e-d1a-p1e+p1a)); break;
    case 5:  ak = 0.5*(cos(d10-d1a) + l10*l1a*cos(d10-d1a-p10+p1a)); break;
    case 6:  ak = -0.5*(sin(d10-d1e) - l10*l1e*sin(d10-d1e-p10+p1e)); break;
    case 7:  ak = 0.5*(1.+l00*l00); break;
    case 8:  ak = 0.5*(cos(d00-d1a) - l00*l1a*cos(d00-d1a-p00+p1a)); break;
    case 9:  ak = -0.5*(sin(d00-d1e) + l00*l1e*sin(d00-d1e-p00+p1e)); break;
    case 10: ak = 0.5*(cos(d00-d10) - l00*l10*cos(d00-d10-p00+p10)); break;
    default: printf("Wrong k index in ak, please check code %d\\n", k);
             return 0.;
  }
  return ak;
}



WITHIN_KERNEL
${ftype} getB(${ftype} p10, ${ftype} p00, ${ftype} p1a, ${ftype} p1e, ${ftype} d10,
						${ftype} d00, ${ftype} d1a, ${ftype} d1e, ${ftype} l10, ${ftype} l00,
						${ftype} l1a, ${ftype} l1e, int k)
{
  ${ftype} bk;
  switch(k) {
    case 1:  bk = -l10*cos(p10); break;
    case 2:  bk = -l1a*cos(p1a); break;
    case 3:  bk = l1e*cos(p1e); break;
    case 4:  bk = 0.5*(l1e*sin(d1e-d1a-p1e) + l1a*sin(d1a-d1e-p1a)); break;
    case 5:  bk = -0.5*(l10*cos(d10-d1a-p10) + l1a*cos(d1a-d10-p1a)); break;
    case 6:  bk = 0.5*(l10*sin(d10-d1e-p10) + l1e*sin(d1e-d10-p1e)); break;
    case 7:  bk = l00*cos(p00); break;
    case 8:  bk = 0.5*(l00*cos(d00-d1a-p00) - l1a*cos(d1a-d00-p1a)); break;
    case 9:  bk = -0.5*(l00*sin(d00-d1e-p00) - l1e*sin(d1e-d00-p1e)); break;
    case 10: bk = 0.5*(l00*cos(d00-d10-p00) - l10*cos(d10-d00-p10)); break;
    default: printf("Wrong k index in bk, please check code %d\\n", k);
             return 0.;
  }
  return bk;
}



WITHIN_KERNEL
${ftype} getC(${ftype} p10, ${ftype} p00, ${ftype} p1a, ${ftype} p1e, ${ftype} d10,
						${ftype} d00, ${ftype} d1a, ${ftype} d1e, ${ftype} l10, ${ftype} l00,
						${ftype} l1a, ${ftype} l1e, int k)
{

  ${ftype} ck;
  switch(k) {
    case 1:  ck = 0.5*(1.-l10*l10); break;
    case 2:  ck = 0.5*(1.-l1a*l1a); break;
    case 3:  ck = 0.5*(1.-l1e*l1e); break;
    case 4:  ck = 0.5*(sin(d1e-d1a) + l1e*l1a*sin(d1e-d1a-p1e+p1a)); break;
    case 5:  ck = 0.5*(cos(d10-d1a) - l10*l1a*cos(d10-d1a-p10+p1a)); break;
    case 6:  ck = -0.5*(sin(d10-d1e) + l10*l1e*sin(d10-d1e-p10+p1e)); break;
    case 7:  ck = 0.5*(1.-l00*l00); break;
    case 8:  ck = 0.5*(cos(d00-d1a) + l00*l1a*cos(d00-d1a-p00+p1a)); break;
    case 9:  ck = -0.5*(sin(d00-d1e) - l00*l1e*sin(d00-d1e-p00+p1e)); break;
    case 10: ck = 0.5*(cos(d00-d10) + l00*l10*cos(d00-d10-p00+p10)); break;
    default: printf("Wrong k index in ck, please check code %d\\n", k);
             return 0.;
  }
  return ck;
}



WITHIN_KERNEL
${ftype} getD(${ftype} p10, ${ftype} p00, ${ftype} p1a, ${ftype} p1e, ${ftype} d10,
						${ftype} d00, ${ftype} d1a, ${ftype} d1e, ${ftype} l10, ${ftype} l00,
						${ftype} l1a, ${ftype} l1e, int k)
{

  ${ftype} dk;
  switch(k) {
    case 1:  dk = l10*sin(p10); break;
    case 2:  dk = l1a*sin(p1a); break;
    case 3:  dk = -l1e*sin(p1e); break;
    case 4:  dk = -0.5*(l1e*cos(d1e-d1a-p1e) + l1a*cos(d1a-d1e-p1a)); break;
    case 5:  dk = -0.5*(l10*sin(d10-d1a-p10) + l1a*sin(d1a-d10-p1a)); break;
    case 6:  dk = -0.5*(l10*cos(d10-d1e-p10) + l1e*cos(d1e-d10-p1e)); break;
    case 7:  dk = -l00*sin(p00); break;
    case 8:  dk = 0.5*(l00*sin(d00-d1a-p00) - l1a*sin(d1a-d00-p1a)); break;
    case 9:  dk = -0.5*(-l00*cos(d00-d1e-p00) + l1e*cos(d1e-d00-p1e)); break;
    case 10: dk = 0.5*(l00*sin(d00-d10-p00) - l10*sin(d10-d00-p10)); break;
    default: printf("Wrong k index in dk, please check code %d\\n", k);
             return 0.;
  }
  return dk;
}



WITHIN_KERNEL
void integralSimple(${ftype} result[2],
                    ${ftype} vn[10],
                    ${ftype} va[10], ${ftype} vb[10], ${ftype} vc[10], ${ftype} vd[10],
                    GLOBAL_MEM ${ftype} *norm,
                    ${ftype} G, ${ftype} DG, ${ftype} DM, ${ftype} ti, ${ftype} tf)
{
  ${ftype} ita = (2*(DG*sinh(.5*DG*ti) + 2*G*cosh(.5*DG*ti))*exp(G*tf) - 2*(DG*sinh(.5*DG*tf) + 2*G*cosh(.5*DG*tf))*exp(G*ti))*exp(-G*(ti + tf))/(-pow(DG, 2) + 4 *pow(G, 2));
  ${ftype} itb = (2*(DG*cosh(.5*DG*ti) + 2*G*sinh(.5*DG*ti))*exp(G*tf) - 2*(DG*cosh(.5*DG*tf) + 2*G*sinh(.5*DG*tf))*exp(G*ti))*exp(-G*(ti + tf))/(-pow(DG, 2) + 4*pow(G, 2));
  ${ftype} itc = ((-DM*sin(DM*ti) + G*cos(DM*ti))*exp(G*tf) + (DM*sin(DM*tf) - G*cos(DM*tf))*exp(G*ti))*exp(-G*(ti + tf))/(pow(DM, 2) + pow(G, 2));
  ${ftype} itd = ((DM*cos(DM*ti) + G*sin(DM*ti))*exp(G*tf) - (DM*cos(DM*tf) + G*sin(DM*tf))*exp(G*ti))*exp(-G*(ti + tf))/(pow(DM, 2) + pow(G, 2));;

  for(int k=0; k<10 ; k++)
  {
    result[0] += vn[k]*norm[k]*(va[k]*ita + vb[k]*itb + vc[k]*itc + vd[k]*itd);
    result[1] += vn[k]*norm[k]*(va[k]*ita + vb[k]*itb - vc[k]*itc - vd[k]*itd);
  }

  #ifdef DEBUG
  if (DEBUG > 3 && ( get_global_id(0) == DEBUG_EVT) )
  {
    printf("INTEGRAL           : ta=%.8lf\ttb=%.8lf\ttc=%.8lf\ttd=%.8lf\n",
           ita,itb,itc,itd );
  }
  #endif
}



////////////////////////////////////////////////////////////////////////////////
