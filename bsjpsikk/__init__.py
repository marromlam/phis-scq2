"""

BSJPSIKK: Python Model for Bs0 -> JPsi K+ K-

BsJpsiKK provides a high-level interface to cuda-C and opencl-C code for the
analysis of the Bs -> Jpsi[mu mu] K+ K- decay. It will return all the kernels
writen in C as a python function called after the given name in C too.

By default this model needs BACKEND, PLATFORM, DEVICE and QUEUE to be defined
beforehand, this is, these should be builtin variables.

Main functionalities:

  * Decay-time acceptance:

  * Angular acceptance:

  * Decay Rate:

Copyright (c) 2020 PHIS_SCQ Developers ; MIT License ; see LICENSE

"""

# import sys
# sys.path.append('BsJpsiKK')
#import builtins
#BACKEND = builtins.BACKEND


# Check backend
#print(f'The backend is set to: {BACKEND}')
try:
  if BACKEND is not None:
    backend = BACKEND
  else:
    backend = 'opencl'
except:
  backend = 'opencl'



# Select code according to BACKEND
#     BsJpsiKK is writen in cuda and in opencl so it can be used both in
#     accelerators and single core cpu.
if backend == 'opencl':
  #print('opencl kernels...')
  from .opencl.Badjanak import *
elif backend == 'cuda':
  #print('cuda kernels...')
  from .cuda.Badjanak import *
else:
  print('Backend was not properly set. No model.')
