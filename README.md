# Beauty-strange physics

This repository contains the code for running different parts of $`B_s \rightarrow J/\psi K^+K^-`$ analysis. Each folder contains a part that is needed to attain the $`\phi_s`$ value.

<p align="center">
  <img src=".logo/pelegrin_phis.png" width="500" title="Pelegrin and the phis vs DG plot">
</p>



## Fist steps

### 🛠 Set environment
_phis-scq_ relies on basic python libraries and _ipanema3_ (it is not in pypy yet),
and so requires to have a properly working environment to run. One can run
this package on both __linux__ and __macOS__ (also with the new M1 processors) against cpu or gpu,
but if you have a nVidia device it may worth it to install cuda binaries and
libraries and hence speeding up the execution.

The instructions were wrapped under a bash script, so basically you clone this
repository and then
```bash
cd phis-scq
bash .ci/install.sh
```
which will guide the installation by prompting some questions.

After the bash script finishes, you simply need to activate your environment.
The installer will ask you whether you want to write in you bash profile
an `activatephisscq` function too.
> Finally, under `config/user.json`, you should write a proper path under
the sidecar key where tuples will be placed. Make sure there is enough space
there to allocate 20 GiB of files, at least. There is a template under `.ci/user.json`
which would be a good example on how this file should look like. This file
is not tracked by git.

That's all folks! 🎉


### 🐍 The pipeline

All the pipeline can be run with `snakemake`
bla bla bla

The first rules snakemake will run are about downloading locally the tuples that
are placed in `/eos/lhcb/wg/B2CC`. In order to do that, you must be able to
access that place with your CERN credentials which basically involves doing
`scp`, and hence basically `ssh`ing.
In order not to be prompted your password every time a file is being synced,
it is very useful to set up a passwordless login to lxplus (basically involves)
using `kinit user@CERN.CH` therefore bypasing ssh with your kerberos
credentials.

Now one should be able to run all the pipeline without doing nothing but
running a rule. So, for example, if you want to do the final fit, you are
running the following example and that is it!
```bash
activatephisscq
kinit user@CERN.CH
snakemake output/params/physics_params/run2/Bs2JpsiPhi/v0r5_run2_run2_vgc_amsrd_simul3_amsrd_combined.json -j
```


## Run

> Each of the main analysis parts has a _README_ file that I encourage you to
read before running any of the related rules concerning that part

This pipeline can run
- [x] *Tuples syncronization:* automatically download files from eos and correctly sync them locally.
  - [x] Download tuples from eos.
  - [x] Reduction of branches.
  - [ ] Convert them to `.hdf5` format.
- [x] *Background subtaction*: computing sWeights for different samples.
  - [x] $`B_u^+`$, $`B_d^0`$ and $`B_s^0`$ sWeights.
  - [x] Possiblitly to skip sWeight computation in `config/user.json`.
- [x] *Reweighting*: compute different reweightings needed to attain other parts of the analysis (or computed in other parts of the pipeline)
  - [x] Compute `polWeight`, `pdfWeight`, `dg0Weight` and `kinWeight` for time acceptance.
  - [x] Compute `angWeight` and `kkpWeight` for angular acceptance.
  - [x] Plots
- [x] *Time acceptance*: compute the decay-time dependence of the efficiency
  - [x] B-splines method
  - [ ] Histogram method
  - [x] Plots
- [ ] *CSP factors*: compute the interference between S-wave, P-wave and D-wave amplitudes
  - [x] Download directly from git
  - [ ] Compute CSP factors
- [ ] *Time resolution*: ...
  - [x] Download directly from git
- [ ] *Flavor tagging*: ...
  - [x] Download directly from git
- [x] *Angular acceptance*: compute the dependence of the reconstruction and selection efficiency on the three helicity angles
  - [x] Angular weights method
  - [ ] Histogram method
  - [ ] Legendre method
  - [x] Plots
- [x] *Time-dependent angular fit*: extract the physics parameters
  - [ ] Real Run1 data samples
  - [x] Real Run2 data samples
  - [x] Monte Carlo samples
  - [x] Toy samples
  - [ ] Plots
- [ ] *Toy MC generator*: generate toys to estimate fit bias
  - [x] Generate pdf events
  - [x] Generate angular efficiency from angular weights
- [ ] *Cross-checks*: different studies for fit validation
  - [x] J/psiK* lifetime
  - [x] J/psiK+ lifetime
  - [x] Binned studies: `sigmat`, `etaB`, `pTB`, `time`
  - [x] Yearly studies: from 2015 to 2018
  - [x] Time dependence of angular acceptance: check whether angles and time do factorize
  - [x] Magnet studies: dependence on magnet polarity
  - [ ] $`B_u^+`$ angular acceptance
  - [x] $`B_d^0`$ angular acceptance
  - [x] $`B_d^0`$ MC angular acceptance test
  - [ongoing] Fit bias














## Contributing

Contributions are very welcome. Information on contributing is available in the
non existent CONTRIBUTING.md file

## Getting help

Since this is a very alpha version of the repository, and it is not very well documented I guess the best you can do is ask directly to me by mail [mromerol@cern.ch](mailto:mromerol@cern.ch).
