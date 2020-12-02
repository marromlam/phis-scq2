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
ftype parabola(ftype sigma, ftype sigma_offset, ftype sigma_slope,
                ftype sigma_curvature)
{
  return sigma_curvature*sigma*sigma + sigma_slope*sigma + sigma_offset;
}



WITHIN_KERNEL
ftype get_omega(ftype eta, ftype tag,
                 ftype p0,  ftype p1, ftype p2,
                 ftype dp0, ftype dp1, ftype dp2,
                 ftype eta_bar)
{
    ftype result = 0;
    result += (p0 + tag*0.5*dp0);
    result += (p1 + tag*0.5*dp1)*(eta - eta_bar);
    result += (p2 + tag*0.5*dp2)*(eta - eta_bar)*(eta - eta_bar);

    if(result < 0.0)
    {
      return 0;
    }
    return result;
}



WITHIN_KERNEL
ftype tagOSgen(const ftype x)
{
  return 3.8 - 134.6*x + 1341.*x*x;
}



WITHIN_KERNEL
ftype tagSSgen(const ftype x)
{
  if (x < 0.46) {
    return exp(16*x -.77);
  }
  else 
  {
    return 10*(16326 - 68488*x + 72116*x*x);
  }
}
