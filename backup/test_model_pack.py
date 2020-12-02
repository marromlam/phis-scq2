
#import builtins

#%% Select backend
from ipanema import initialize
initialize('cuda',1)


#%% Import modules
from ipanema.core.utils import shit
import numpy as np
import matplotlib.pyplot as plt



#%% Get the model
import bsjpsikk


#%% Testers
time_h = np.linspace(0.3,15,100)
time_d = shit.data_array(time_h)
lkhd_d = shit.data_array(0*time_h)

time_d_a = time_d; lkhd_d_a = lkhd_d
time_d_b = time_d; lkhd_d_b = lkhd_d
time_d_c = time_d; lkhd_d_c = lkhd_d



# acceptance_spline ------------------------------------------------------------
#    Description
plt.close()
lkhd_h = bsjpsikk.acceptance_spline(time_d)
plt.plot(time_h,lkhd_h)



# single_spline_time_acceptance ------------------------------------------------
#    Description
plt.close()
bsjpsikk.single_spline_time_acceptance(
            time_d,lkhd_d
          )
lkhd_h = lkhd_d.get()
plt.plot(time_h,lkhd_h)



# ratio_spline_time_acceptance -------------------------------------------------
#    Description
plt.close()
bsjpsikk.ratio_spline_time_acceptance(
            time_d_a,time_d_b,lkhd_d_a,lkhd_d_b
          )
lkhd_h = lkhd_d_b.get()
plt.plot(time_h,lkhd_h)



# full_spline_time_acceptance --------------------------------------------------
#    Description
plt.close()
bsjpsikk.full_spline_time_acceptance(
            time_d_a,time_d_b,time_d_c,lkhd_d_a,lkhd_d_b,lkhd_d_c
          )
lkhd_h = lkhd_d_c.get()
plt.plot(time_h,lkhd_h)
