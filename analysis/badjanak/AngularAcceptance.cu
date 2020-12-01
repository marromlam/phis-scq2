#include <ipanema/special.hpp>



WITHIN_KERNEL
ftype angular_efficiency(const ftype cosK, const ftype cosL, const ftype phi, 
                         ftype *moments)
{
  ftype eff = 0.;

  eff += moments[0] * legendre_poly(0, 0, cosK) * sph_harm(0, 0, cosL, phi);
  eff += moments[1] * legendre_poly(0, 0, cosK) * sph_harm(2, 0, cosL, phi);
  eff += moments[2] * legendre_poly(0, 0, cosK) * sph_harm(2, 2, cosL, phi);
  eff += moments[3] * legendre_poly(0, 0, cosK) * sph_harm(2, 1, cosL, phi);
  eff += moments[4] * legendre_poly(0, 0, cosK) * sph_harm(2,-1, cosL, phi);
  eff += moments[5] * legendre_poly(0, 0, cosK) * sph_harm(2,-2, cosL, phi);
  eff += moments[6] * legendre_poly(1, 0, cosK) * sph_harm(0, 0, cosL, phi);
  eff += moments[7] * legendre_poly(1, 0, cosK) * sph_harm(2, 1, cosL, phi);
  eff += moments[8] * legendre_poly(1, 0, cosK) * sph_harm(2,-1, cosL, phi);
  eff += moments[9] * legendre_poly(2, 0, cosK) * sph_harm(0, 0, cosL, phi);

  eff *= 2.*sqrt(M_PI);
  return eff;
}



WITHIN_KERNEL
void angular_weights2moments(GLOBAL_MEM const ftype* nw, ftype* moments)
{
  //c0000
  moments[0] =  1./3.                    * ( nw[0] + nw[1] + nw[2] );
  //c0020
  moments[1] =  1./3. * sqrt(5.)         * ( nw[0] + nw[1] + nw[2] - 3.*nw[6] );
  //c0022
  moments[2] =         -sqrt(5./3.)      * ( nw[1] - nw[2] );
  //c0021
  moments[3] = -8./3. * sqrt(5./2.)/M_PI * ( nw[7] );
  //c002-1  -> -nw should be +nw?
  moments[4] = -8./3. * sqrt(5./2.)/M_PI * ( nw[8] );
  //c002-2  -> -nw should be +nw?
  moments[5] =          sqrt(5./3.)      * ( nw[3] );
  //c1000
  moments[6] =  1./2. * sqrt(3.)         * ( nw[9] );
  //c1021
  moments[7] =-32./3. * sqrt(5./6.)/M_PI * ( nw[4] );
  //c102-1  -> -nw should be +nw?
  moments[8] =+32./3. * sqrt(5./6.)/M_PI * ( nw[5] );
  //c2000
  moments[9] =  5./2.                    * ( nw[0] - nw[6] );
}



KERNEL
void plot_moments(GLOBAL_MEM const ftype *nw, GLOBAL_MEM ftype *out,
                  GLOBAL_MEM const ftype *cosK, GLOBAL_MEM const ftype *cosL, 
                  GLOBAL_MEM const ftype *hphi)
{
  const int evt = get_global_id(0);
  ftype moments[NTERMS] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  angular_weights2moments(nw, moments);
  out[evt] = angular_efficiency(cosK[evt], cosL[evt], hphi[evt], moments);
}