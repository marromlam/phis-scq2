#!/bin/bash


source $STORE/conda3/bin/activate
conda activate phisscq
export IPANEMA_BACKEND="opencl"
export OPENBLAS_NUM_THREADS=20
export PHIS_SCQ="$HOME/phis-scq/"
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_NO_CACHE=1
export PYTHONPATH=$HOME/phis-scq/analysis:$PYTHONPATH
export KRB5_CONFIG="/mnt/netapp2/Home_FT2/home/usc/ie/mrl/phis-scq/.ci/krb5.conf"

kinit -k -t /mnt/netapp2/Home_FT2/home/usc/ie/mrl/private/mromerol.keytab mromerol@CERN.CH

# properties = {properties}
{exec_job}
