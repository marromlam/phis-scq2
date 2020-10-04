# Use OpenCL To Add Two Random Arrays (This Way Hides Details)

import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

a_cpu =   np.random.rand(5000000).astype(np.float32)
a_gpu =   pycl_array.to_device(queue, a_cpu)
b_cpu =   np.random.rand(5000000).astype(np.float32)
b_gpu =   pycl_array.to_device(queue, b_cpu)
c_cpu = 0*np.random.rand(5000000).astype(np.float32)
c_gpu =   pycl_array.to_device(queue, c_cpu)



program = cl.Program(context, """
__kernel void sum(__global const float *a, __global const float *b, __global float *c)
{
  int i = get_global_id(0);
  c[i] = a[i]*a[i] + b[i];
}""").build()  # Create the OpenCL program



from timeit import           default_timer            as timer

t0 = timer()
c_cpu = a_cpu**2+b_cpu
t_cpu = timer()-t0


t0 = timer()
program.sum(queue, a_gpu.shape, None, a_gpu.data, a_gpu.data, a_gpu.data)
t_gpu = timer()-t0

print(t_cpu,t_gpu,t_cpu/t_gpu)
