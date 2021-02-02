////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                      KERNEL FOR THE PHIS-SCQ ANALYSIS                      //
//                                                                            //
//   Created: 2019-11-18                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
//  This file exposes the following KERNELs:                                  //
//    * pyrateBs: Computes Bs2MuMuKK pdf looping over the events. Now it      //
//                handles a binned X_M fit without splitting beforehand the   //
//                data --it launches a thread per mass bin.                   //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#define ftype double
////////////////////////////////////////////////////////////////////////////////
// Include headers /////////////////////////////////////////////////////////////

#define USE_DOUBLE ${USE_DOUBLE}
#include <ipanema/core.hpp>

// Debugging 0 [0,1,2,3,>3]
#define DEBUG ${DEBUG}
#define DEBUG_EVT ${DEBUG_EVT}               // the event that is being debugged

#define FAST_INTEGRAL ${FAST_INTEGRAL}        // fast integral, means no faddeva

// Time resolution parameters
#define SIGMA_T ${SIGMA_T}
#define SIGMA_THRESHOLD 5.0

// Time acceptance parameters
#define NKNOTS ${NKNOTS}
#define SPL_BINS ${SPL_BINS}
#define NTIMEBINS ${NTIMEBINS}
const CONSTANT_MEM ftype KNOTS[NKNOTS] = ${KNOTS};

// PDF parameters
#define NTERMS ${NTERMS}
#define MKNOTS ${NMASSKNOTS}
const CONSTANT_MEM ftype X_M[MKNOTS] = ${X_M};
const CONSTANT_MEM ftype TRISTAN[NTERMS] = ${TRISTAN};

// Include ipanema
#include <ipanema/complex.cpp>
#include <ipanema/special.cpp>
#include <ipanema/random.cpp>

// Include analysis parts
//#include "FkHelpers.cu"
#include "Tagging.cu"
#include "TimeAngularDistribution.cu"
#include "DecayTimeAcceptance.cu"
#include "CrossRateBs.cu"
#include "AngularAcceptance.cu"
#include "CrossRateBd.cu"
#include "Toy.cu"

////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// KERNEL::pyDiffRate //////////////////////////////////////////////////////////

KERNEL
void pyrateBs(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *lkhd,
              // Time-dependent angular distribution
              const ftype G, const ftype DG, const ftype DM,
              GLOBAL_MEM const ftype * CSP,
              GLOBAL_MEM const ftype *ASlon, GLOBAL_MEM const ftype *APlon,
              GLOBAL_MEM const ftype *APpar, GLOBAL_MEM const ftype *APper,
              const ftype pSlon, const ftype pPlon, const ftype pPpar,
              const ftype pPper, GLOBAL_MEM const ftype *dSlon,
              const ftype dPlon, const ftype dPpar, const ftype dPper,
              const ftype lSlon, const ftype lPlon, const ftype lPpar,
              const ftype lPper,
              // Time limits
              const ftype tLL, const ftype tUL,
              // Time resolution
              const ftype sigma_offset, const ftype sigma_slope,
              const ftype sigma_curvature, const ftype mu,
              // Flavor tagging
              const ftype eta_bar_os, const ftype eta_bar_ss,
              const ftype p0_os, const ftype p1_os, const ftype p2_os,
              const ftype p0_ss, const ftype p1_ss, const ftype p2_ss,
              const ftype dp0_os, const ftype dp1_os, const ftype dp2_os,
              const ftype dp0_ss, const ftype dp1_ss, const ftype dp2_ss,
              // Time acceptance
              GLOBAL_MEM const ftype *coeffs,
              // Angular acceptance
              GLOBAL_MEM const ftype *angular_weights,
              // Flags
              const int USE_FK, const int BINS, const int USE_ANGACC,
              const int USE_TIMEACC, const int USE_TIMEOFFSET,
              const int SET_TAGGING, const int USE_TIMERES, const int NEVT)
{
  const int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  ftype mass = data[evt*10+4];
  ftype arr[9] = {data[evt*10+0], // cosK
                  data[evt*10+1], // cosL
                  data[evt*10+2], // hphi
                  data[evt*10+3], // time
                  data[evt*10+5], // sigma_t
                  data[evt*10+6], // qOS
                  data[evt*10+7], // qSS
                  data[evt*10+8], // etaOS
                  data[evt*10+9]  // etaSS
                 };

  const int bin = BINS>1 ? getMassBin(mass) : 0;
  lkhd[evt] = rateBs(arr,
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
}




KERNEL
void pyrateBd(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *lkhd,
              //inputs
              const ftype G, GLOBAL_MEM const ftype * CSP,
              GLOBAL_MEM const ftype *ASlon, GLOBAL_MEM const ftype *APlon,
              GLOBAL_MEM const ftype *APpar, GLOBAL_MEM const ftype *APper,
              GLOBAL_MEM const ftype *dSlon, const ftype dPlon,
              const ftype dPpar, const ftype dPper,
              // Time limits
              const ftype tLL, const ftype tUL,
              // Angular acceptance
              GLOBAL_MEM const ftype *angular_weights,
              // Flags
              const int USE_FK, const int BINS, const int USE_ANGACC, const int NEVT)
{
  int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  ftype mass = data[evt*10+4];
  ftype arr[9] = {data[evt*10+0], // cosK
                  data[evt*10+1], // cosL
                  data[evt*10+2], // hphi
                  data[evt*10+3], // time
                  data[evt*10+5], // sigma_t
                  data[evt*10+6], // qOS
                  data[evt*10+7], // qSS
                  data[evt*10+8], // etaOS
                  data[evt*10+9]  // etaSS
                 };

  unsigned int bin = BINS>1 ? getMassBin(mass) : 0;
  lkhd[evt] = getDiffRateBd(arr,
                          G, CSP[bin],
                          ASlon[bin], APlon[bin], APpar[bin], APper[bin],
                          dSlon[bin], dPlon,      dPpar,      dPper,
                          tLL, tUL,
                          angular_weights,USE_FK, USE_ANGACC);

}
//////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// KERNEL::pyFcoeffs ///////////////////////////////////////////////////////////

KERNEL
void pyFcoeffs(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *fk,
               const int NEVT)
{
  const int i = get_global_id(0);
  //const int k = get_global_id(1);
  if (i >= NEVT) { return; }
  for(int k=0; k< NTERMS; k++)
  {
    fk[i*NTERMS+k]= 9./(16.*M_PI)*getF(data[i*3+0],data[i*3+1],data[i*3+2],k+1);
  }
}

////////////////////////////////////////////////////////////////////////////////
//CJLM//////////////////////////////////////////////////////////////////////////
KERNEL
void pyCjlms(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *out,
              const int NEVT, const int m)
{
  const int evt = get_global_id(0);
  ftype cosK = data[evt*3+0];
  ftype cosL = data[evt*3+1];
  ftype hphi = data[evt*3+2];
  int index = 0;
  if (evt >= NEVT) { return; }
  for (int i=0; i<=m; i++)
  {
    for (int j=0; j<=m; j++)
    {
      for (int k=-j; k<=j; k++)
      {
        out[int(pow(m+1,3))*evt+index] = lpmv(i,0,cosK)*sph_harm(j,k,cosL,hphi);
        index += 1;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//CJLM_w_ws//////////////////////////////////////////////////////////////////////////
//for the moment only valid for m=4, WARNING.
KERNEL
void pyCs2Ws(GLOBAL_MEM ftype *Cs, GLOBAL_MEM ftype *out,
              const int NEVT, const int m)
{
  const int evt = get_global_id(0);
  const int IDX = evt*pow(m+1,3);
  if ((evt==0) or (evt==1))
  {
    printf("\nfirst 4 Cs: Cs0=%+.8f Cs1=%+.8f Cs2=%+.8f Cs3=%+.8f Cs4=%+.8f\n",
           Cs[0], Cs[1], Cs[2], Cs[3]);
    printf("Cs50=%+.8f Cs6=%+.8f Cs56=%+.8f Cs8=%+.8f Cs58=%+.8f\n",
                  Cs[50], Cs[6], Cs[56], Cs[8], Cs[58]);
  }
  if (evt >= NEVT) { return; }
  out[evt*NTERMS+0] = Cs[IDX+0] + 2./5.*Cs[IDX+50] + 1./sqrt(20.)*(Cs[IDX+6] + 2./5.*Cs[IDX+56]) - sqrt(3./20.)*(Cs[IDX+8]+2./5.*Cs[IDX+58]);
  out[evt*NTERMS+1] = Cs[IDX+0] - 1./5.*Cs[IDX+50] + 1./sqrt(20.)*(Cs[IDX+6] - 1./5.*Cs[IDX+56]) + sqrt(3./20.)*(Cs[IDX+8]-1./5.*Cs[IDX+58]);
  out[evt*NTERMS+2] = Cs[IDX+0] - 1./5.*Cs[IDX+50] - sqrt(1./5.)*(Cs[IDX+6] - 1./5.*Cs[IDX+56]);
  out[evt*NTERMS+3] = sqrt(3./5.)*(Cs[IDX+5] - 1./5.*Cs[IDX+55]);
  out[evt*NTERMS+4] = sqrt(6./5.)*3.*M_PI/32.*(Cs[IDX+29] - 1./4.*Cs[IDX+79]);
  out[evt*NTERMS+5] = sqrt(6./5.)*3.*M_PI/32.*(Cs[IDX+32] - 1./4.*Cs[IDX+82]);
  out[evt*NTERMS+6] = 1./2.*(2.*Cs[IDX+0] + sqrt(1./5.)*Cs[IDX+6] - sqrt(3./5.)*Cs[IDX+8]);
  out[evt*NTERMS+7] = 3.*sqrt(2./5.)*M_PI/8.*(Cs[IDX+4] - 1./8.*Cs[IDX+54] - 1./64.*Cs[IDX+104]);
  out[evt*NTERMS+8] = 3.*sqrt(2./5.)*M_PI/8.*(Cs[IDX+7] - 1./8.*Cs[IDX+57] - 1./64.*Cs[IDX+107]);
  out[evt*NTERMS+9] = 1./6.*(4.*sqrt(3.)*Cs[IDX+25] + 2*sqrt(3./5.)*Cs[IDX+31] - 6.*sqrt(1./5.)*Cs[IDX+33]);
  if ((evt==0) or (evt==1))
  {
    printf("\nfirst 10 ws: w0=%+.8f w1=%+.8f w2=%+.8f w3=%+.8f w4=%+.8f w5=%+.8f, w6=%+.8f, w7=%+.8f w8=%+.8f w9=%+.8f\n",
           out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9]);
  }
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Acceptance //////////////////////////////////////////////////////////////////

KERNEL
void pySingleTimeAcc(GLOBAL_MEM ftype *time, GLOBAL_MEM ftype *lkhd,
                     GLOBAL_MEM ftype *coeffs, ftype mu,
                     const ftype sigma, const ftype gamma,
                     const ftype tLL, const ftype tUL, const int NEVT)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{
  const int row = get_global_id(0);
  if (row >= NEVT) { return; }
  ftype t = time[row] - mu;
  lkhd[row] = getOneSplineTimeAcc(t, coeffs, sigma, gamma, tLL, tUL);
}



KERNEL
void pyRatioTimeAcc(GLOBAL_MEM const ftype *time1, GLOBAL_MEM const ftype *time2,
                    GLOBAL_MEM ftype *lkhd1, GLOBAL_MEM ftype *lkhd2,
                    GLOBAL_MEM const ftype *c1, GLOBAL_MEM const ftype *c2,
                    const ftype mu1, const ftype sigma1, const ftype gamma1,
                    const ftype mu2, const ftype sigma2, const ftype gamma2,
                    const ftype tLL, const ftype tUL,
                    const int NEVT1, const int NEVT2)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{
  const int row = get_global_id(0);
  if (row < NEVT1)
  {
    ftype t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }
  if (row < NEVT2)
  {
    ftype t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }
}



KERNEL
void pyFullTimeAcc(GLOBAL_MEM ftype const *time1, GLOBAL_MEM ftype const *time2,
                   GLOBAL_MEM ftype const *time3, GLOBAL_MEM ftype *lkhd1,
                   GLOBAL_MEM ftype *lkhd2, GLOBAL_MEM ftype *lkhd3,
                   GLOBAL_MEM ftype *c1,
                   GLOBAL_MEM ftype *c2,
                   GLOBAL_MEM ftype *c3,
                   ftype mu1, ftype sigma1, ftype gamma1,
                   ftype mu2, ftype sigma2, ftype gamma2,
                   ftype mu3, ftype sigma3, ftype gamma3,
                   ftype tLL, ftype tUL,
                   const int NEVT1, const int NEVT2, const int NEVT3)
/*
This is a pycuda iterating function. It calls getAcceptanceSingle for each event
in time and returns
*/
{
  const int row = get_global_id(0);
  if (row < NEVT1)
  {
    ftype t1 = time1[row] - mu1;
    lkhd1[row] = getOneSplineTimeAcc(t1, c1,     sigma1, gamma1, tLL, tUL);
  }
  if (row < NEVT2)
  {
    ftype t2 = time2[row] - mu2;
    lkhd2[row] = getTwoSplineTimeAcc(t2, c1, c2, sigma2, gamma2, tLL, tUL);
  }
  if (row < NEVT3)
  {
    ftype t3 = time3[row] - mu3;
    lkhd3[row] = getTwoSplineTimeAcc(t3, c2, c3, sigma3, gamma3, tLL, tUL);
  }
}



KERNEL
void pySpline(GLOBAL_MEM const ftype *time, GLOBAL_MEM ftype *f,
              GLOBAL_MEM const ftype *coeffs, const int NEVT)
{
  const int row = get_global_id(0);
  if (row >= NEVT) { return; }
  ftype t = time[row];

  // Get spline-time-bin
  int bin = getTimeBin(t);

  // Get spline coeffs
  ftype c0 = getCoeff(coeffs,bin,0);
  ftype c1 = getCoeff(coeffs,bin,1);
  ftype c2 = getCoeff(coeffs,bin,2);
  ftype c3 = getCoeff(coeffs,bin,3);

  // Compute spline
  ftype fpdf = (c0 + t*(c1 + t*(c2 + t*c3)));
  f[row] = fpdf;
}

////////////////////////////////////////////////////////////////////////////////
// that's all folks
