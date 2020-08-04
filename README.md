This repository contains the code for running different parts of Bs Jpsi KK analysis. Each folder contains a part.

bla bla


# Fist steps
## Set environment

phis-scq relyes on basic python libraries and ipanema3 (it is not in pypy yet),
and so requires to have a properly working environmente to run.
It is recommended to install miniconda and create a clean enviroment. To install conda, just:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Once installed, let's create a proper environment. In order to do that, if you are
using some accelerator, this is the right time to make sure
you have the `PATH` and `LD_LIBRATY_PATH`. Then create the enviroment running:
```
conda  create --name phisscq
conda activate phisscq
conda install pip
conda install -c conda-forge pocl
conda install -c conda-forge pyopencl
conda install -c conda-forge ROOT
```

Then one just needs to install _ipanema3_, to do so first we need to clone the respository
(it's private, ask me if you wish):
```
git clone ssh://git@gitlab.cern.ch:7999/mromerol/ipanema3.git
```
Then we run
```
pip install -e ./ipanema3
```
which will install all python packages that are needed to properly run ipanema3,
but pycuda.  It's not installed by default since, if you do't have a proper
device it will clash at that step. If you have a nVidia device, then you can
install it by using (remember you must add to your PATH the cuda binaries and
libraries before).
```
pip install pycuda
```

phis-scq uses snakemake to pipeline all its outputs, so you must install it
```
pip install snakemake hep_ml py-cpuinfo
```


At this point, you shoud have no problem to run the code in this repository, but
to go faster in future sessions it's recommended you write some piece of code
like the following to quickly set up your environment.

```
function set_phis_scq {
  source /home3/marcos.romero/conda3/bin/activate
  conda activate ipanema3
  export PATH="${PATH//:\/cvmfs\/sft.cern.ch\/lcg\/releases\/gcc\/8.3.0-cebb0\/x86_64-slc6\/bin:/:}"
  export PATH="/usr/local/cuda-10.2/bin:${PATH}"
  export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}"
  export PHIS_SCQ="/home3/marcos.romero/phis-scq/"
  export IPANEMA_BACKEND="cuda"
  export PYOPENCL_COMPILER_OUTPUT=1
  export PYOPENCL_NO_CACHE=1
}
```
