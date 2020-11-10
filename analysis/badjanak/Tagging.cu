////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//                                 TAGGING                                    //
//                                                                            //
//   Created: 2020-11-07                                                      //
//    Author: Marcos Romero Lamas (mromerol@cern.ch)                          //
//                                                                            //
//    This file is part of phis-scq packages, Santiago's framework for the    //
//                     phi_s analysis in Bs -> Jpsi K+ K-                     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////



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
