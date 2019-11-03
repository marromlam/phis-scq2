#include <math.h>
// #include <thrust/complex.h>
#include <pycuda-complex.hpp>

__device__ pycuda::complex<double> TimeEvolution(pycuda::complex<double> gplus, pycuda::complex<double> gminus, pycuda::complex<double> A, double lambda_abs, double phis, double q, double eta) {
pycuda::complex<double> stuff = pycuda::complex<double>(cos(q*phis),sin(q*phis));
if (q == -1) {
      return A* (gplus + eta*pow(lambda_abs,-1.*q)*stuff*gminus);
  } else {
      return A * eta * (gplus + eta*pow(lambda_abs,-1.*q)*stuff*gminus);
  }
}

__device__ double complex_angular( double helcosthetaK, double helcosthetaL, double helphi, pycuda::complex<double> A_0, pycuda::complex<double> A_S, pycuda::complex<double> A_pa, pycuda::complex<double> A_pe, double Cfact_10){
double ck2 = helcosthetaK*helcosthetaK;
double cl2 = helcosthetaL*helcosthetaL;
double shp = sin(helphi);
double chp = cos(helphi);
double s2hp = sin(2*helphi);
double c2hp = cos(2*helphi);
double mcl2p1 = -cl2 + 1;
double mck2p1 = -ck2 + 1;
double sqmcl2p1 = sqrt(-cl2 + 1);
double sqmck2p1 = sqrt(-ck2 + 1);
double sq2 = sqrt(2.);
double sq3 = sqrt(3.);
double sq6 = sqrt(6.);
double inv_4pi = 1./(4.*M_PI);
double inv_8pi = 1./(8.*M_PI);
double inv_16pi = 1./(16.*M_PI);
pycuda::complex<double> I = pycuda::complex<double>(0,1.); //It misses the CSP!
pycuda::complex<double> A_0_bar = pycuda::conj(A_0);
pycuda::complex<double> A_pa_bar = pycuda::conj(A_pa);
pycuda::complex<double> A_pe_bar = pycuda::conj(A_pe);
pycuda::complex<double> A_S_bar = pycuda::conj(A_S);

return pycuda::real(3.*A_0*ck2*mcl2p1*A_0_bar*inv_4pi - 3.*sq2*I*A_0*helcosthetaK*helcosthetaL*sqmck2p1*sqmcl2p1*shp*A_pe_bar*inv_8pi - 3.*sq2*A_0*helcosthetaK*helcosthetaL*sqmck2p1*sqmcl2p1*chp*A_pa_bar*inv_8pi + sq3*A_0*helcosthetaK*mcl2p1*A_S_bar*Cfact_10*inv_4pi + sq3*Cfact_10*A_S*helcosthetaK*mcl2p1*A_0_bar*inv_4pi - sq6*I*Cfact_10*A_S*helcosthetaL*sqmck2p1*sqmcl2p1*shp*A_pe_bar*inv_8pi - sq6*Cfact_10*A_S*helcosthetaL*sqmck2p1*sqmcl2p1*chp*A_pa_bar*inv_8pi + A_S*mcl2p1*A_S_bar*inv_4pi - 3.*sq2*A_pa*helcosthetaK*helcosthetaL*sqmck2p1*sqmcl2p1*chp*A_0_bar*inv_8pi + 3.*I*A_pa*cl2*mck2p1*s2hp*A_pe_bar*inv_16pi + 3.*A_pa*cl2*mck2p1*c2hp*A_pa_bar*inv_16pi + 3.*A_pa*cl2*mck2p1*A_pa_bar*inv_16pi - sq6*A_pa*helcosthetaL*sqmck2p1*sqmcl2p1*chp*A_S_bar*Cfact_10*inv_8pi - 3.*I*A_pa*mck2p1*s2hp*A_pe_bar*inv_16pi - 3.*A_pa*mck2p1*c2hp*A_pa_bar*inv_16pi + 3.*A_pa*mck2p1*A_pa_bar*inv_16pi + 3.*sq2*I*A_pe*helcosthetaK*helcosthetaL*sqmck2p1*sqmcl2p1*shp*A_0_bar*inv_8pi - 3.*I*A_pe*cl2*mck2p1*s2hp*A_pa_bar*inv_16pi - 3.*A_pe*cl2*mck2p1*c2hp*A_pe_bar*inv_16pi + 3.*A_pe*cl2*mck2p1*A_pe_bar*inv_16pi + sq6*I*A_pe*helcosthetaL*sqmck2p1*sqmcl2p1*shp*Cfact_10*A_S_bar*inv_8pi + 3.*I*A_pe*mck2p1*s2hp*A_pa_bar*inv_16pi + 3.*A_pe*mck2p1*c2hp*A_pe_bar*inv_16pi + 3.*A_pe*mck2p1*A_pe_bar*inv_16pi);
}

__global__ void TimeF( double *data, double *out, double A_0_mod,double A_S_mod,double A_pa_mod,double A_pe_mod,double delta_pa,double delta_pe,double delta_S,double Cfact_10, double Gs, double dG, double deltaMs, double lambda_abs, double phis, int Nevt)
{
int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
if (row >= Nevt) { return;}
int i0 = row*5;// general rule for cuda matrices : index = col + row*N; as it is now, N = 5 (cthk,cthl,cphi,t, q)
int idx = 0 + i0; 
int idy = 1 + i0;
int idz = 2 + i0;
int idt = 3 + i0;
int idq = 4 + i0;
double time = data[idt];
double q = data[idq];

double mt = exp(-(0.5*Gs + 0.25*dG)*time); 
double pt = exp(-(0.5*Gs - 0.25*dG)*time);
pycuda::complex<double> c1 = pycuda::complex<double>(cos(0.5*deltaMs*time),sin( 0.5*deltaMs*time));
pycuda::complex<double> c2 = pycuda::conj(c1);
pycuda::complex<double> gplus  = 0.5*( mt*c1+ pt*c2);
pycuda::complex<double> gminus = 0.5*( mt*c1- pt*c2);
pycuda::complex<double> I = pycuda::complex<double>(0,1.);
pycuda::complex<double> A_0 = pycuda::complex<double>(A_0_mod);
pycuda::complex<double> A_pa = pycuda::complex<double>(A_pa_mod)*exp(I*delta_pa);
pycuda::complex<double> A_pe = pycuda::complex<double>(A_pe_mod)*exp(I*delta_pe);
pycuda::complex<double> A_S = pycuda::complex<double>(A_S_mod)*exp(I*delta_S);
//zurullo;
pycuda::complex<double> A0 = TimeEvolution(gplus, gminus, A_0, lambda_abs, phis, q, 1.); 
pycuda::complex<double> As = TimeEvolution(gplus, gminus, A_S, lambda_abs, phis, q, -1.); 
pycuda::complex<double> Apa = TimeEvolution(gplus, gminus, A_pa, lambda_abs, phis, q, 1.); 
pycuda::complex<double> Ape = TimeEvolution(gplus, gminus, A_pe, lambda_abs, phis, q, -1.); 

out[row] = complex_angular(data[idx],data[idy],data[idz],A0, As, Apa, Ape, Cfact_10);
}

__device__ double integral4piB( double t, double A_0_mod, double A_S_mod, double A_pa_mod, double A_pe_mod, double G,double DG,double DM,double phis )
{
//caca;
double exp_G_t = exp(-t*G);
double cdmt = cos(t*DM);
double sdmt = sin(t*DM);
double cosh_term = cosh(t*DG/2);
double sinh_term = sinh(t*DG/2);
double phi_0 = phis;
double phi_pa = phis;
double phi_pe = phis + M_PI;
double phi_S = phis + M_PI;
double A_0_mod_2 = A_0_mod*A_0_mod;
double A_pa_mod_2 = A_pa_mod*A_pa_mod;
double A_pe_mod_2 = A_pe_mod*A_pe_mod;
double A_S_mod_2 = A_S_mod*A_S_mod;

return 1.0*exp_G_t*(2*cosh_term*A_0_mod_2 + 2*sdmt*A_0_mod_2*sin(phi_0) - 2*sinh_term*A_0_mod_2*cos(phi_0)) + 1.0*exp_G_t*(2*cosh_term*A_S_mod_2 + 2*sdmt*A_S_mod_2*sin(phi_S) - 2*sinh_term*A_S_mod_2*cos(phi_S)) + 1.0*exp_G_t*(2*cosh_term*A_pa_mod_2 + 2*sdmt*A_pa_mod_2*sin(phi_pa) - 2*sinh_term*A_pa_mod_2*cos(phi_pa)) + 1.0*exp_G_t*(2*cosh_term*A_pe_mod_2 + 2*sdmt*A_pe_mod_2*sin(phi_pe) - 2*sinh_term*A_pe_mod_2*cos(phi_pe));

}
__device__ double integral4piBbar( double t, double A_0_mod, double A_S_mod, double A_pa_mod, double A_pe_mod, double G,double DG,double DM,double phis )
{
//caca;
double exp_G_t = exp(-t*G);
double cdmt = cos(t*DM);
double sdmt = sin(t*DM);
double cosh_term = cosh(t*DG/2);
double sinh_term = sinh(t*DG/2);
double phi_0 = phis;
double phi_pa = phis;
double phi_pe = phis + M_PI;
double phi_S = phis + M_PI;
double A_0_mod_2 = A_0_mod*A_0_mod;
double A_pa_mod_2 = A_pa_mod*A_pa_mod;
double A_pe_mod_2 = A_pe_mod*A_pe_mod;
double A_S_mod_2 = A_S_mod*A_S_mod;

return 2.0*cosh_term*exp_G_t*A_0_mod_2 + 2.0*cosh_term*exp_G_t*A_S_mod_2 + 2.0*cosh_term*exp_G_t*A_pa_mod_2 + 2.0*cosh_term*exp_G_t*A_pe_mod_2 - 2.0*exp_G_t*sdmt*A_0_mod_2*sin(phi_0) - 2.0*exp_G_t*sdmt*A_S_mod_2*sin(phi_S) - 2.0*exp_G_t*sdmt*A_pa_mod_2*sin(phi_pa) - 2.0*exp_G_t*sdmt*A_pe_mod_2*sin(phi_pe) - 2.0*exp_G_t*sinh_term*A_0_mod_2*cos(phi_0) - 2.0*exp_G_t*sinh_term*A_S_mod_2*cos(phi_S) - 2.0*exp_G_t*sinh_term*A_pa_mod_2*cos(phi_pa) - 2.0*exp_G_t*sinh_term*A_pe_mod_2*cos(phi_pe);

}
__global__ void binnedTimeIntegralB(double *time, double *out, double A_0_mod, double A_S_mod, double A_pa_mod, double A_pe_mod, double G,double DG,double DM,double phis ){
int idx = threadIdx.x + blockDim.x * blockIdx.x;
out[idx] = integral4piB(time[idx], A_0_mod, A_S_mod, A_pa_mod, A_pe_mod, G, DG, DM, phis);
}
__global__ void binnedTimeIntegralBbar(double *time, double *out, double A_0_mod, double A_S_mod, double A_pa_mod, double A_pe_mod, double G,double DG,double DM,double phis ){
int idx = threadIdx.x + blockDim.x * blockIdx.x;
out[idx] = integral4piBbar(time[idx], A_0_mod, A_S_mod, A_pa_mod, A_pe_mod, G, DG, DM, phis);
}
