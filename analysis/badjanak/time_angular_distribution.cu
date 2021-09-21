////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//   Created: 2020-06-25                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////


#include "time_angular_distribution.h"


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


// vim:foldmethod=marker
