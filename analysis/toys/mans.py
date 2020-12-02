DESCRIPTION = """
    blah
"""

from ipanema import Parameters
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import argparse
import numpy as np
from ipanema import plotting

def argument_parser():
  parser = argparse.ArgumentParser(description=DESCRIPTION)
  parser.add_argument('--parameters', help='Bs2JpsiPhi data sample')
  parser.add_argument('--figures', help='Bs2JpsiPhi data sample')
  # parser.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--csp-factors', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  # parser.add_argument('--fitted-params', help='Bs2JpsiPhi MC sample')
  return parser


args = vars(argument_parser().parse_args())

params = args['parameters'].split(',')[:5]
params = 'output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f1_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f2_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f3_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f4_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f5_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f6_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f7_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f8_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f9_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f10_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f11_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f12_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f13_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f14_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f15_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f16_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f17_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f18_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f19_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f20_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f21_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f22_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f23_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f24_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f25_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f26_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f27_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f28_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f29_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f30_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f31_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f32_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f33_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f34_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f35_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f36_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f37_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f38_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f39_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f40_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f41_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f42_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f43_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f44_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f45_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f46_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f47_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f48_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f49_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f50_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f51_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f52_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f53_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f54_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f55_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f56_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f57_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f58_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f59_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f60_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f61_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f62_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f63_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f64_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f65_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f66_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f67_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f68_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f69_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f70_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f71_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f72_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f73_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f74_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f75_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f76_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f77_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f78_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f79_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f80_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f81_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f82_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f83_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f84_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f85_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f86_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f87_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f88_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f89_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f90_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f91_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f92_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f93_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f94_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f95_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f96_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f97_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f98_run2a_run2_simul.json,output/params/physics_params/run2a/TOY_Bs2JpsiPhi/v0r5wrun2wrun2wsimulgen201103f99_run2a_run2_simul.json'.split(',')[:]

figures=

nominal = Parameters.load('output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_simul.json')
params = [Parameters.load(p) for p in params]



G = lambda mu,sigma: np.exp(-0.5*(X-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)
#%% Plot the hisrograms
for par in nominal:
  if nominal[par].free:
    x  = [params[i][par] for i in range(0,len(params))]
    vx = [p.n for p in x]
    ux = [p.s for p in x]
    x = unp.uarray(vx, ux)
    #print("value:", vx)
    #print("stdev:", ux)
    x = unp.uarray(vx, ux)
    mu = np.mean(x)
    sigma = unp.sqrt( np.mean((x - mu) ** 2) )
    sigma = unc.ufloat(unp.nominal_values(sigma), unp.std_devs(sigma))
    #print(par, 'mu', mu, 'sigma', sigma)
    pn = x.mean().n; ps = x.mean().s
    h = np.histogram(vx,10)
    X = np.linspace(0.9*min(vx),1.1*max(vx),100)
    Y = len(vx)*G(mu.n,sigma.n)*(h[1][1]-h[1][0])
    fix, ax = plotting.axes_plot()
    ax.fill_between(0.5*(h[1][1:]+h[1][:-1]),h[0], facecolor='gray', step='mid', alpha=0.5)
    ax.plot(X,Y)

    ax.text(0.8, 0.9,f'$\mu = {mu:.2uL}$',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
    ax.text(0.8, 0.8,f'$\sigma = {sigma:.2uL}$',
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)
    ax.fill_between([mu.n-mu.s, mu.n+mu.s],[max(Y),max(Y)],0,facecolor='C2')
    ax.set_xlabel(f"${nominal[par].latex}$")
    ax.set_ylabel(f"Toys")



#%% Plot the pulls
for par in nominal:
  if nominal[par].free:
    x  = [params[i][par]-nominal[par] for i in range(0,len(params))]
    vx = [p.n for p in x]
    ux = [p.s for p in x]
    x = unp.uarray(vx, ux)
    # Compute mu and sigma with their respective uncertainty
    mu = np.mean(x)
    sigma = unp.sqrt( np.mean((x - mu) ** 2) )
    sigma = unc.ufloat(unp.nominal_values(sigma), unp.std_devs(sigma))

    # Do an histo
    h = np.histogram(vx,10)
    # Create a Gaussian
    X = np.linspace(0.9*min(vx),1.1*max(vx),100)
    Y = len(vx)*G(mu.n,sigma.n)*(h[1][1]-h[1][0])

    # Plot
    fix, ax = plotting.axes_plot()
    ax.fill_between(0.5*(h[1][1:]+h[1][:-1]),h[0], facecolor='gray', step='mid', alpha=0.5)
    ax.plot(X,Y)
    ax.text(0.8, 0.9, f'$\mu = {mu:.2uL}$', transform = ax.transAxes,
            horizontalalignment='center', verticalalignment='center')
    ax.text(0.8, 0.8,f'$\sigma = {sigma:.2uL}$', transform = ax.transAxes,
            horizontalalignment='center', verticalalignment='center')
    ax.fill_between([mu.n-mu.s, mu.n+mu.s],[max(Y),max(Y)], 0, facecolor='C2')
    ax.set_xlabel(f"pull$({nominal[par].latex})$")
    ax.set_ylabel(f"Toys")
    fig.savefig('')
