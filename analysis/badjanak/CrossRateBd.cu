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

//Ramon es mejor pasar el data igual para no confundirse en otras partes del codigo

WITHIN_KERNEL
ftype getDiffRateBd(const ftype *data,
                    const ftype G, const ftype CSP,
                    const ftype ASlon, const ftype APlon, const ftype APpar,
                    const ftype APper, const ftype dSlon, const ftype dPlon,
                    const ftype dPpar, const ftype dPper,
                    // Time Limints
                    const ftype tLL, const ftype tUL,
                    // Angular acceptance
                    GLOBAL_MEM  const ftype *angular_weights,
                    const int USE_FK, const int USE_ANGACC
                  )
{


  // Variables -----------------------------------------------------------------
  //     Make sure that the input it's in this order.
  //     lalala
  ftype cosK       = data[0];                      // Time-angular distribution
  ftype cosL       = data[1];
  ftype hphi       = data[2];
  ftype time       = data[3];                            // Time resolution
  ftype qOS        = data[5];                                      // Tagging


  // Flavor tagging ------------------------------------------------------------
  ftype id;
  id = qOS/fabs(qOS);

  // Compute per event pdf -----------------------------------------------------
  ftype fk, ak;
  ftype num_t, tnorm;
  ftype pdfB = 0.0;
  ftype norm = 0.0;

  if (USE_ANGACC)
  {
    num_t = exp(-time*G);
    tnorm = (exp(-tLL*G)-exp(-tUL*G))/G;
  }
  else
  {
    num_t = 1.0;
    tnorm = 1.0;
  }

  for(int k = 1; k <= 10; k++)
  {
    if (USE_FK)
    {
      fk = num_t*( 9.0/(16.0*M_PI) )*getF(cosK,cosL,hphi,k);
    }
    else
    {
      fk = tnorm*TRISTAN[k-1];
    }
    ak = getAbd(ASlon, APlon, APpar, APper, dSlon, dPpar, dPpar, dPper, CSP, k);
    if ( (k==4) || (k==6)  || (k==9) )
    {
      pdfB += id*fk*ak;
      norm += tnorm*id*angular_weights[k-1]*ak;
    }
    else
    {
      pdfB += fk*ak;
      norm += tnorm*angular_weights[k-1]*ak;
    }
  }



  return pdfB/norm;
}
