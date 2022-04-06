#!/bin/bash
# properties = {properties}

source /home3/marcos.romero/conda3/bin/activate
conda activate phisscqNew

export IPANEMA_BACKEND="opencl"
export POCL_MAX_PTHREAD_COUNT=20
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_NO_CACHE=1

export PHISSCQ=/lustre/LHCb/marcos.romero/phis-scq
export PYTHONPATH=$PHISSCQ/:$PYTHONPATH
export KRB5_CONFIG=$PHISSCQ/.ci/krb5.conf

kinit -k -t /home3/marcos.romero/private/mromerol.keytab mromerol@CERN.CH

echo "================================================================================"
echo "RUNNING AT HOSTNAME:"
hostname -f
klist
echo "================================================================================"

{exec_job}
