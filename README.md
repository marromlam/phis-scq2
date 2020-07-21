This repository contains the code for running different parts of Bs Jpsi KK analysis. Each folder contains a part.

bla bla

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
