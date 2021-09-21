////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#ifndef _CROSSRATEBD_H_
#define _CROSSRATEBD_H_


#include "time_angular_distribution.h"


  WITHIN_KERNEL
ftype getDiffRateBd(const ftype *data,
    const ftype G, const ftype CSP,
    const ftype ASlon, const ftype APlon, const ftype APpar,
    const ftype APper, const ftype dSlon, const ftype dPlon,
    const ftype dPpar, const ftype dPper,
    const ftype tLL, const ftype tUL,
    GLOBAL_MEM const ftype *angular_weights,
    const int USE_FK, const int USE_ANGACC
    )
{

  // Variables {{{
  //     Make sure that the input it's in this order.
  //     lalala
  ftype cosK       = data[0];                      // Time-angular distribution
  ftype cosL       = data[1];
  ftype hphi       = data[2];
  ftype time       = data[3];                            // Time resolution
  ftype qOS        = data[5];                                      // Tagging


#if DEBUG
  if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT )
  {
    printf("\nINPUT              : cosK=%+.16f  cosL=%+.16f  hphi=%+.16f  time=%+.16f\n",
        cosK,cosL,hphi,time);
    printf("                   : qOS=%+.16f\n",qOS);
  }
#endif
#if DEBUG
  if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT )
  {
    printf("\n Amplitudes : AsLon=%+.5f APLon=%+.5f APpar=%+.5f APper=%+.5f CSP=%+.2f\n",
        ASlon, APlon, APpar, APper, CSP);
  }
#endif

  // }}}


  // Flavor tagging {{{

  ftype id;
  id = qOS/fabs(qOS);

  // }}}


  // Compute per event pdf {{{

  ftype fk, ak;
  ftype num_t, tnorm;
  ftype pdfB = 0.0;
  ftype norm = 0.0;
  num_t = exp(-time*G);
  tnorm = (exp(-tLL*G)-exp(-tUL*G))/G;

  //for(int k = 1; k <= 10; k++)
  for(int k = 1; k <= 10; k++)
  {
#if DEBUG
    if (DEBUG >= 1 && get_global_id(0) == DEBUG_EVT )
    {
      printf("\nANGACC              : ang_weight=%+.4f \n",
          angular_weights[k-1]);
    }
#endif
    if (USE_FK)
    {
      fk = ( 9.0/(16.0*M_PI) )*getF(cosK,cosL,hphi,k);
    }
    else
    {
      fk = TRISTAN[k-1];
    }
    ak = getAbd(ASlon, APlon, APpar, APper, dSlon, dPpar, dPpar, dPper, CSP, k);
    if ( (k==4) || (k==6)  || (k==9) )
    {
      pdfB += id*fk*ak;
      norm += id*angular_weights[k-1]*ak;
    }
    else
    {
      pdfB += fk*ak;
      norm += angular_weights[k-1]*ak;
    }
  }
  pdfB = num_t*pdfB;
  norm = tnorm*norm;

#if DEBUG
  if ( DEBUG >= 1  && get_global_id(0) == DEBUG_EVT)
  {
    printf("\nRESULT             : <  pdf/ipdf = %+.16f  >\n",
        pdfB/norm);
    if ( DEBUG >= 2 )
    {
      printf("                   : pdf=%+.16f  ipdf=%+.16f\n",
          pdfB,norm);
    }
  }
#endif
  // }}}


  return pdfB/norm;
}


#endif //_CROSSRATEBD_H_


// vim:foldmethod=marker
