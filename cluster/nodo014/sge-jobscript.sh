#!/bin/bash
# properties = {properties}

source /home3/marcos.romero/conda3/bin/activate
conda activate phisscq
export IPANEMA_BACKEND="opencl"
export PHIS_SCQ="/home3/marcos.romero/phis-scq/"
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_NO_CACHE=1
export PYTHONPATH=/home3/marcos.romero/phis-scq/analysis:$PYTHONPATH
export KRB5_CONFIG="/home3/marcos.romero/phis-scq/krb5.conf"

# exit on first error
set -o errexit

{exec_job}
