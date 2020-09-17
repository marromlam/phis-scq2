# Beauty-strange physics

This repository contains the code for running different parts of $`B_s \rightarrow J/\psi K^+K^-`$ analysis. Each folder contains a part that is needed to attain the $`\phi_s`$ value.



## Fist steps

### 🛠 Set environment
_phis-scq_ relyes on basic python libraries and _ipanema3_ (it is not in pypy yet),
and so requires to have a properly working environment to run. One can run
this package on both __linux__ and __macos__ against cpu or gpu,
but if you have a nVidia device it may worth it to install cuda binaries and
libraries and hence speeding up the execution.

The instructions were wrapped under a bash script, so basically you clone this
repository and then
```bash
cd phis-scq
bash utils/install.sh
```
which will guide the installation by prompting some questions.

After the bash script finishes, you simply need to activate your environment.
The installer will ask you whether you want to write in you bash profile
an `activatephisscq` function too.
> Finally, under `config.json`, you should write a proper path under
the homonym key where tuples will be placed. Make sure there is enough space
there to allocate 20 GiB of files, at least.

That's it! 🎉


### 🐍 The pipeline

All the pipeline can be run with `snakemake`
bla bla bla

The first rules snakemake will run are about dowloading locally the tuples that
are placed in `/eos/lhcb/wg/B2CC`. In order to do that, you must be able to
access that place with your CERN credentials which basically involves doing
`scp`, and hence basically `ssh`ing.
In order not to be prompted your password every time a file is being synced,
it is very useful to set up a passwordless login to lxplus (basically involves)
using `kinit user@CERN.CH` therefore bypasing ssh with your kerberos
credentials. Sometimes conda harms kinit access to credential cache, hence it is
better to do kinit before activating the environment.

Now one should be able to run all the pipeline without doing nothing but
running a rule. So, for example, if you want to do the final fit, you are
running the following example and that is it!
```bash
kinit user@CERN.CH
activatephisscq
snakemake output/params/angular_fit/run2/Bs2JpsiPhi/v0r5_base_base.json -j
```


## Run

> Each of the main analysis parts has a _README_ file that I encourage you to
read before running any of the related rules concerning that part

This pipeline can run
- [x] Automatically download files from eos and correctly sync them locally
- [ ] *Subtract background*: computing sWeights for different samples
- [x] *Reweighting*: compute _polWeight_, _pdfWeight_, _kinWeight_, _angWeight_ and _kkpWeight_.
- [x] *Time acceptance*: compute the decay-time dependence of the efficiency
- [ ] *CSP factors*: compute the interference between S-wave, P-wave and D-wave amplitudes
- [ ] *Time resolution*: ...
- [ ] *Flavor tagging*: ...
- [x] *Angular acceptance*: compute the dependence of the reconstruction and selection efficiency on the three helicity angles
- [x] *Time-dependent angular fit*: extract the physics parameters
















## Contributing

Contributions are very welcome. Information on contributing is available in the
non existent CONTRIBUTING.md file

## Getting help

Since this is a very alpha version of the repository, and it is not very well documented I guess the best you can do is ask directly to me by mail [mromerol@cern.ch](mailto:mromerol@cern.ch).
