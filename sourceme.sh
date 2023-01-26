source $HOME/conda3/bin/activate
conda activate phisscqNew

HOST="$(hostname)"
if [[ "${CURRENT_HOST}" == "gpu91" ]]; then
  export IPANEMA_BACKEND="cuda"
  source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.2/x86_64-centos7/setup.sh
  echo "Nice journey with the Ryzen and 3k90"
elif [[ "${CURRENT_HOST}" == "gpu223" ]]; then
  export IPANEMA_BACKEND="cuda"
  source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.2/x86_64-centos7/setup.sh
  echo "Nice journey with the 3k80 Ti"
elif [[ "${CURRENT_HOST:0:3}" == "gpu" ]]; then
  source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.2/x86_64-centos7/setup.sh
  # export PATH="/usr/local/cuda-10.2/bin:${PATH}"
  # export LD_LIBRARY_PATH="/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}"
  export IPANEMA_BACKEND="cuda"
  echo "Nice journey with 1k80 and 2k80 friends"
else
  export IPANEMA_BACKEND="opencl"
  export POCL_MAX_PTHREAD_COUNT=20
fi

CWD=`pwd`
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_NO_CACHE=1
export PHISSCQ=${CWD}
export PYTHONPATH=$PHISSCQ/:$PYTHONPATH
export KRB5_CONFIG=$PHISSCQ/.ci/krb5.conf

export MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
alias cobrega='f(){snakemake $@ --resources mem_mb=$MEM_MB}; f'
