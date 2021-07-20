////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include "decay_time_acceptance.h"
#include "cross_rate_bd.h"


  KERNEL
void pyrateBd(GLOBAL_MEM const ftype *data, GLOBAL_MEM ftype *lkhd,
    const ftype G, GLOBAL_MEM const ftype * CSP,
    GLOBAL_MEM const ftype *ASlon, GLOBAL_MEM const ftype *APlon,
    GLOBAL_MEM const ftype *APpar, GLOBAL_MEM const ftype *APper,
    GLOBAL_MEM const ftype *dSlon, const ftype dPlon,
    const ftype dPpar, const ftype dPper,
    const ftype tLL, const ftype tUL,
    GLOBAL_MEM const ftype *angular_weights,
    const int USE_FK, const int BINS, const int USE_ANGACC,
    const int NEVT)
{
  int evt = get_global_id(0);
  if (evt >= NEVT) { return; }

  ftype mass = data[evt*10+4];
  ftype arr[9] = {
    data[evt*10+0], // cosK
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
  lkhd[evt] = getDiffRateBd(arr, G, CSP[bin], ASlon[bin], APlon[bin],
      APpar[bin], APper[bin], dSlon[bin], dPlon, dPpar,
      dPper, tLL, tUL, angular_weights,
      USE_FK, USE_ANGACC);

}


// vim:foldmethod=marker
