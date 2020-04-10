# Angular acceptance

The effects of the angular acceptance are modelled with the __normalization weights__. 
These are obtained from fully simulated signal events as will be described next.

The __normalization weights__ are refined as the MC is teratively weighted to match
the distributions of final-state particle kinematics in the real data and to mathc 
the physiscs parameters obtained from data, in order to correct for impoerfection 
in the detector simulation.

## Normalization weights from the full simulation

For each year from 2015 to 2018 there are differente samples of fully simulated events,
generated with the parameters in $`\mathtt{angular\_acceptance/parameters/}`$`year`$`\mathtt{/}`$`mode`$`\mathtt{.json}`$. For
the evatuoation of the generator level p.d.f. of the MC events, true variables are used. 
In constras to data, reconstructe helicuty angle are used when evaluating the angular
functions. As matter of fact, the normalization weights are computed as follows

```math
\tilde{w}_k = 
\sum_{i=1}^{\# \mathrm{events}} \omega_i
f_k({\theta_K}_i^{\mathrm{reco}},{\theta_{\mu}}_i^{\mathrm{reco}},{\phi}_{i}^{\mathrm{reco}}) 
\frac
{ \frac{d\Gamma^4}{dtd\Omega}  (t_i^{\mathrm{true}},{\theta_K}_i^{\mathrm{true}},{\theta_{\mu}}_i^{\mathrm{true}},{\phi}_{i}^{\mathrm{true}})}
{ \frac{d\Gamma}{dt}(t_i^{\mathrm{true}})}
```

where $`\omega_i`$ stands for a per event weight, and finally the _normalization weights_ are

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


## Correcting MC and refining the normalization weights

...
