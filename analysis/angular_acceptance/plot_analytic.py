import numpy as np
import ipanema
import matplotlib.pyplot as plt











angAccOrderCosThetaK=4
angAccOrderCosThetaL=4
angAccOrderPhi=4



def get_acceptance(cosK, cosL, phi):
  val=0;
  for k2 in range(0,angAccOrderCosThetaK+1):
    for l2 in range(0,angAccOrderCosThetaL+1):
      for m2 in range(0,angAccOrderPhi+1):
        val+=polynomial_coeffs[k2][l2][m2]*(cosK**k2*cosL**l2*phi**m2);
  return val


ipanema.initialize('cuda',1)
import badjanak

fk = ipanema.ristra.allocate(np.zeros(10)).astype(np.float64)
badjanak.__KERNELS__.integral_ijk_fx(
np.float64(-1),np.float64(+1),
np.float64(-1),np.float64(+1),
np.float64(-np.pi),np.float64(+np.pi),
np.int32(+0),np.int32(+0), np.int32(+1), fk, global_size=(1))

fk



1+1

data = np.random.normal(0,1,1000000)
def pdf(x):
  return np.exp(-0.5*x**2)

ref.norm*abs(ref.edges[1]-ref.edges[0])/(y.sum()*abs(x[1]-x[0]))



#%% plot -----------------------------------------------------------------------

nob = 100; nop = 500;

x = np.linspace(-5,5,nop)
y = pdf(x)
pdf_norm = np.trapz(y,x)
print(np.trapz(y/pdf_norm,x))
dx = (x[-1]-x[0])#/len(x)

a = np.histogram(data, bins=nob, range=(-10,0));
xh = (a[1][1:] + a[1][:-1])*0.5
dxh = (xh[-1]-xh[0])#/len(xh)
yh = a[0]
hst_norm = np.trapz(yh, xh)
print(np.trapz(yh/hst_norm,xh))
plt.close()

pdf_hst_norm = np.trapz(pdf(xh),xh)

plt.step(xh,hst_norm*(yh/hst_norm),where='mid')
plt.step(x,hst_norm*( y/pdf_norm),where='mid')
plt.step(x,hst_norm*y/pdf_hst_norm,where='mid')
