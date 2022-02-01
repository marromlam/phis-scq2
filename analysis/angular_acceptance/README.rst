Angular acceptance
------------------

The effects of the angular acceptance are modelled with the *normalization weights*.
These are obtained from fully simulated signal events as will be described next.

The normalization weights are refined as the MC is iteratively weighted to match
the distributions of final-state particle kinematics in the real data and to match
the physics parameters obtained from data, in order to correct for imperfections
in the detector simulation.

The pipeline to run angular aceptance computations is quite simple
.. ::
  snakemake output/params/angular_acceptance/{year}/{mode}/{version}_{angacc}_{timeacc}_{trigger}.json

where:
* `year`: The year corresponding to the set of normalization weights one is looking for.
* `mode`: Refers to the mode of the angular acceptance to be computed. It can be `MC_Bs2JpsiPhi`, `MC_Bs2JpsiPhi_dG0` or `Bs2JpsiPhi`, but be aware not all rules run for all modes. The `Bs2JpsiPhi` corresponds (when MC modes exist) to the weighted average of the MCs.
* `version`: As always, it is the version of the tuples to run with.
* `angacc`: This is the flag to specify different configurations of angular acceptance. Allowed values are: `naive`, `corrected`, `base`, `yearly`, `run2a`
* `timeacc`: This is the flag to specify the time acceptance configuration. See the time accetance README for more information.
* `trigger`: As always, it is the trigger configuration, biased or unbiased. For the naive rule one can also run with combined.

Normalization weights from the full simulation
----------------------------------------------

For each year from 2015 to 2018 there are differente samples of fully simulated events,
generated with the parameters in $`\mathtt{angular\_acceptance/parameters/}`$`year`$`\mathtt{/}`$`mode`$`\mathtt{.json}`$. For
the evatuoation of the generator level p.d.f. of the MC events, true variables are used.
In constras to data, reconstructe helicuty angle are used when evaluating the angular
functions. As matter of fact, the normalization weights are computed as follows

.. math::
  \tilde{w}_k =
  \sum_{i=1}^{\# \mathrm{events}} \omega_i
  f_k({\theta_K}_i^{\mathrm{reco}},{\theta_{\mu}}_i^{\mathrm{reco}},{\phi}_{i}^{\mathrm{reco}})
  \frac
  { \frac{d\Gamma}{dt}(t_i^{\mathrm{true}})}
  { \frac{d\Gamma^4}{dtd\Omega}  (t_i^{\mathrm{true}},{\theta_K}_i^{\mathrm{true}},{\theta_{\mu}}_i^{\mathrm{true}},{\phi}_{i}^{\mathrm{true}})}

where $`\omega_i`$ stands for a per event weight, and finally the *normalization weights* are

```math
{w}_k = \frac{1}{\tilde{w}_0} \tilde{w}_k
```

The nagular acceptance is determined with MC matched events, $`\mathtt{BKGCAT==0|BKGCAT==50}`$
withoud any weights applied. However we also take into account ome events belonging to $`\mathtt{BKGCAT==60}`$ (known as ghost events)
since these are used when computing the sweights. Hence these events will be weighted
with the sweights. Since for the ghost events no true information is avaliable, `true_genlvl`
variables will be used.

One the other hand, MC polarity will be matched with the polarity of the corresponding
data. This is done by using the $`\mathtt{polWeight}`$s, which are precisely computed to take
this into account. So, in this first step $`\omega_i`$ means $`\mathtt{polWeight*sWeight}`$.

To decrease the statistical uncertainty of the normalization weights, not only the
default MC samples but also the MC samples with $`\Delta\Gamma=0`$ are used. The __normalization
weights__ are weight-averaged for the two per year samples by using the covariance matrixes.
The angular acceptance is computed per each year and per trigger category.


To get these
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_naive_{trigger}.json
```


Correcting MC and refining the normalization weights
----------------------------------------------------


We must ensure the distributions of the helicity angles in MC and data
correspond to the same distributions of kinematics in the laboratory frame.
These differences are corrected using GB-weighting in different variables
kinematic variables, namely: $`p_{B_s^0}`$, $`p_{B_s^0}^T`$ and $`m_{KK}`$.

To obtain these corrected files one should run rules like the following
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_corrected_{trigger}.json
```
which will run first the corrected normalization weights for each MC sample and
finally average them to create the `Bs2JpsiPhi` one.


Iterative procedure
-------------------

During the iterative procedure, we will correct the mismodeled acceptance in MC
using GB-weighting in $p_{K^{\pm}}$, $p_{K^{\pm}}^T$ ($p_{\mu^{\pm}}$,
$p_{\mu^{\pm}}^T$, the distribution of the muons agrees already quite well
after the weighting in $B_s^0$ kinematics). This weighting is done splitting
the MC sample by trigger category. The iterative procedure runs till a
convergence is reached, and those angular weights are the ones used in the
final fit where we extract the physics parameters.

One can produce different angular acceptance sets whether using the full set of
years, pairs of years or single-year run. The corresponding rules are, respectively:
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_base_base_{trigger}.json
```
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_run2a_base_{trigger}.json
```
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_yearly_base_{trigger}.json
```
Each of these rules will compute both trigger sets.


Systematics
-----------

...


Every angular acceptance step runs with ta time acceptance flag, as before explained. So, for example, to run the iterative procedure for tthe full Run2 in the first bin of pt,
```
snakemake output/params/angular_acceptance/{year}/Bs2JpsiPhi/{version}_base_baseBinpt1_{trigger}.json
```
