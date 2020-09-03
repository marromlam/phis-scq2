# Beauty-strange physics

This repository contains the code for running different parts of $`B_s \rightarrow J/\psi K^+K^-`$ analysis. Each folder contains a part that is needed to attain the $`\phi_s`$ value.

Each part of the analysis has a (will have) _README.md_ file



bla bla


## Fist steps

### 🛠 Set environment
_phis-scq_ relyes on basic python libraries and _ipanema3_ (it is not in pypy yet),
and so requires to have a properly working environment to run. One can run
this package on both __linux__ and __macos__ against cpu or gpu,
but if you have a nVidia device it may worth it to install cuda binaries and
libraries and hence speeding up the execution.

The instructions were wrapped under a bash script, so basically you clone this
repository and then
```
cd phis-scq
bash utils/install.sh
```
which will guide the installation by prompting some questions.

After the bash script finishes, you simply need to activate your environment.
The installer will ask you whether you want to write in you bash profile
an `activatephisscq` function too.
Finally, under config.json, you should write a proper path under
the homonym key.

That's it! 🎉


### 🐍 The pipeline

All the pipeline can be run with `snakemake`
bla bla bla

a good example to run is

```
snakemake output/params/angular_fit/run2a/Bs2JpsiPhi/v0r5_repo_repo.json -j
```

## Run

--- blah blah blah ---

















## Contributing

Contributions are very welcome. Information on contributing is available in the
non existent CONTRIBUTING.md file

## Getting help

Since this is a very alpha version of the repository, and it is not very well documented I guess the best you can do is ask directly to me at mromerol@cern.ch.
