
# баджанак HPC kernel


Badjanak kernel consists in a set of different meta-C90 files which have coded the core functions of this analysis.

## Kernel.cu
The main file containing all the kernels that can be called from python using `pycuda` or `pyopencl` interfaces.


## DifferentialCrossRate.cu
It contains the core of the `DiffRate` function

Let's say the per event pdf values is, `pdf = numerator/denominator`:

* In the `numerator` part, generally the faddeva function is not used, it's only used for a few events. This is a common behavior for __phis-scq__ but also for __bs2jpsiphi-hd__ (and also de old __HD-fitter__). See:https://gitlab.cern.ch/lhcb-b2cc/bs2jpsiphi-hd/-/blob/master/src/bs2jpsiphi/src/bs2jpsiphi_pdf.cc#L318
 or https://gitlab.cern.ch/mromerol/phis-scq/-/blob/master/badjanak/DecayTimeAcceptance.cu#L219
* In the `denominator` part (the pdf integral), `phis-scq` has two options managed by the `FAST_INTEGRAL` flag:
  * `FAST_INTEGRAL= 1`: Since faddeva computation is quite time consuming, (though using gpus is way faster)  __phis-scq__ has implemented an approximation for `t > 0.3 ps`, computing the integral avoiding the faddeva evaluation.
  * `FAST_INTEGRAL= 0`: Then we use the faddeva function as HD-fitter does. Here it's very important the function works fine since for each normalization several calls to faddeva are done.
