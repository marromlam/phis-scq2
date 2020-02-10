import ipanema
BACKEND
ipanema.initialize('opencl',1)

import numpy as np
import importlib
import matplotlib.pyplot as plt
from ipanema import ristra

import bsjpsikk


bsjpsikk.config

bsjpsikk.config['debug'] = 4
bsjpsikk.config['debug_evt'] = 1
bsjpsikk.get_kernels()




#%% Test acceptance function ---------------------------------------------------
if BACKEND == 'cuda':
  time_a = ristra.allocate(np.linspace(0.3,15,200))
  pdf_a  = ristra.allocate(time_a.get()*0)
  time_b = ristra.allocate(np.linspace(0.3,15,200))
  pdf_b  = ristra.allocate(time_b.get()*0)
  time_c = ristra.allocate(np.linspace(0.3,15,200))
  pdf_c  = ristra.allocate(time_c.get()*0)
  bsjpsikk.full_spline_time_acceptance(time_a,time_b,time_c,pdf_a,pdf_b,pdf_c)
  plt.close()
  plt.plot(time_a.get(),pdf_a.get(),
           time_b.get(),pdf_b.get(),
           time_c.get(),pdf_c.get()
          )

#%% Test acceptance function ---------------------------------------------------
if BACKEND == 'cuda':
  bsjpsikk.ratio_spline_time_acceptance(time_a,time_b,pdf_a,pdf_b)
  plt.close()
  plt.plot(time_a.get(),pdf_a.get(),time_b.get(),pdf_b.get())

#%% Test acceptance function ---------------------------------------------------
if BACKEND == 'cuda':
  bsjpsikk.single_spline_time_acceptance(time_a,pdf_a)
  plt.close()
  plt.plot(time_a.get(),pdf_a.get())

#%% Test acceptance function ---------------------------------------------------
if BACKEND == 'cuda':
  bsjpsikk.acceptance_spline(time_a,pdf_a)
  plt.close()
  plt.plot(time_a.get(),pdf_a.get())

#%% Test diff_cross_rate -------------------------------------------------------
data = ristra.allocate(np.array([0,0,-1,0.5,1000,0,-531]))
pdf  = ristra.allocate(np.array([0]))
bsjpsikk.diff_cross_rate(data,pdf)
plt.close()
plt.plot(pdf.get())
