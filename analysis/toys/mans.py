DESCRIPTION = """
    blah
"""


__all__ = []
__author__ = "Marcos Romero Lamas"



import os
from ipanema import Parameters
import uncertainties as unc
from uncertainties import unumpy as unp
import matplotlib.pyplot as plt
import argparse
import numpy as np
from ipanema import plotting

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--data-parameters', help='Bs2JpsiPhi data sample')
  p.add_argument('--toy-parameters', help='Bs2JpsiPhi data sample')
  p.add_argument('--figures', help='Bs2JpsiPhi data sample')
  p.add_argument('--ntoys', help='Bs2JpsiPhi data sample')
  # p.add_argument('--angacc-biased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--angacc-unbiased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--timeacc-biased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--timeacc-unbiased', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--csp-factors', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--time-resolution', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--flavor-tagging', help='Bs2JpsiPhi MC sample')
  # p.add_argument('--fitted-params', help='Bs2JpsiPhi MC sample')

  args = vars(p.parse_args())

  nomi = args['data_parameters']#[:5]
  #pars = args['toy_parameters'].split(',')#[:5]
  pars = args['toy_parameters']
  print(pars)
  pars = [pars.replace("ntoy", str(i)) for i in range(1,int(args['ntoys'])+1)]
  nomi = Parameters.load(nomi)
  pars = [Parameters.load(p) for p in pars]



  os.makedirs(args['figures']) 



  G = lambda mu,sigma: np.exp(-0.5*(X-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)
  """
  #%% Plot the hisrograms
  for par in nomi:
    if nomi[par].free:
      x  = [pars[i][par].uvalue for i in range(0,len(pars))]
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
      fig, ax = plotting.axes_plot()
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
      ax.set_xlabel(f"${nomi[par].latex}$")
      ax.set_ylabel(f"Toys")
      fig.savefig(f"{args['figures']}/{par}.pdf")
      plt.close('all')
  exit()
  """

  #%% Plot the pulls -0.09848919756781958+/-0 0.9331633463676519+/-0
  for par in nomi:
    if nomi[par].free:
      x = [(pars[i][par]-nomi[par])/pars[i][par].uvalue.s for i in range(0,len(pars))]
      rmse = 1#np.std(np.array([pars[i][par].s for i in range(0,len(pars))]))
      vx = np.array([p.n for p in x])/rmse
      ux = np.array([p.s for p in x])/rmse
      x = unp.uarray(vx, ux)
      
      # Compute mu and sigma with their respective uncertainty
      mu = np.mean(x)
      #sigma = unc.ufloat(np.std(x),0) 
      sigma = unp.sqrt( np.mean((x - mu) ** 2) )
      sigma = unc.ufloat(unp.nominal_values(sigma), unp.std_devs(sigma))
      print(mu, sigma) 
      # Do an histo
      h = np.histogram(vx, 20)
      # Create a Gaussian
      X = np.linspace(-5,5,100)
      Y = len(vx)*G(mu.n,sigma.n)*(h[1][1]-h[1][0])
  
      # Plot
      fig, ax = plotting.axes_plot()
      ax.fill_between(0.5*(h[1][1:]+h[1][:-1]),h[0], facecolor='gray', step='mid', alpha=0.5)
      ax.plot(X,Y)
      ax.text(0.8, 0.9, f'$\mu = {mu:.4uL}$', transform = ax.transAxes,
              horizontalalignment='center', verticalalignment='center')
      ax.text(0.8, 0.8,f'$\sigma = {sigma:.4uL}$', transform = ax.transAxes,
              horizontalalignment='center', verticalalignment='center')
      ax.fill_between([mu.n-mu.s, mu.n+mu.s],[max(Y),max(Y)], 0, facecolor='C2')
      #ax.set_yscale('log')
      ax.set_xlabel(f"pull$({nomi[par].latex})$")
      ax.set_ylabel(f"Toys")
      ax.set_xlim(-5,5)
      fig.savefig(f"{args['figures']}/{par}.pdf")
      plt.close('all')
