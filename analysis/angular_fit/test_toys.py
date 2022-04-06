# import ipanema
# import matplotlib.pyplot as plt
# from ipanema import ristra
# import numpy as np
#
#
# ipanema.initialize('cuda',1)
# import badjanak
#
#
# x = ipanema.ristra.linspace(0.3,15,100)
# y = badjanak.bspline(x,*pars)
#
# plt.plot(ristra.get(x),ristra.get(y),'-')
#
#
# pars = [1,1.3,1.0,4.1,2.3,1.1,5.4,3.4,1.2]
#
#
#
# %--------------------------------------------------
# # EXAMPLE 2: METROPOLIS-HASTINGS
# # COMPONENT-WISE SAMPLING OF BIVARIATE NORMAL
#
# # TARGET DISTRIBUTION
# p = inline('mvnpdf(x,[0 0],[1 0.8;0.8 1])','x');
#
# p = lambda x: ristra.get(badjanak.bspline(ristra.allocate(np.float64([x])),*pars))
#
#
#
#
#
#
# D = 2;
# N = 5000;
# minn = [-3 -3];
# maxx = [3 3];
#
#
#
# np.random.rand
# # INITIALIZE COMPONENT-WISE SAMPLER
# x = np.zeros((N,D));
# x0 = np.random.randn(D)
# x0
#
# xCurrent(1) = randn;
# xCurrent(2) = randn;
# dims = list(range(1,D+1))
#
#
# 1:2; # INDICES INTO EACH DIMENSION
# t = 0;
# x[0,:] = x0
#
# no.
# np.random.normal(3443.324, 1)
# from scipy.stats import multivariate_normal
# q = lambda x, mu: multivariate_normal.pdf(x, mu)
# p = lambda x: multivariate_normal.pdf(x, 3, 5)
#
# #%% RUN SAMPLER - MH sampler
# t = 0
# while t < N:
#     t = t + 1;
#     print(t)
#     # SAMPLE FROM PROPOSAL
#     print(x)
#     xStar = np.random.normal(x[t-1,:],np.eye(D))[0];
#     print(t, xStar)
#     xStar
#     # CORRECTION FACTOR (SHOULD EQUAL 1)
#     print(x[t-1,:],xStar)
#     c = q(x[t-1,:],xStar)/q(xStar,x[t-1,:]);
#
#     # CALCULATE THE M-H ACCEPTANCE PROBABILITY
#     alpha = min(1, c*p(xStar)/p(x[t-1,:]));
#
#     # ACCEPT OR REJECT?
#     u = np.random.rand();
#     if u < alpha:
#       x[t,:] = xStar;
#     else:
#       x[t,:] = x[t-1,:];
