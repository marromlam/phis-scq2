#!/bin/bash

# phis ascii Logo {{{

clear && printf "\e[3J"
echo "
+------------------------------------------------------------------------------+
| ····································**************************************** |
| ···································***************************************** |
| ·····························÷÷÷÷÷÷÷÷÷÷************************************* |
| ························÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷*********************************** |
| ······················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷********************************* |
| ····················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷******************************** |
| ····················÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷******************************* |
| ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%**************************** |
| ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%%%%%************************ |
| ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%%%%%%%%********************* |
| ···················÷÷÷÷÷÷÷÷÷÷÷÷÷//|//÷÷÷÷÷÷÷%%%%%%%%%%%%%%****************** |
| ····················÷÷÷÷÷÷÷÷÷÷÷///|///÷÷÷÷÷%%%%%%%%%%%%%%%%***************** |
| ·····················÷÷÷÷÷÷÷÷÷//##|////÷÷÷%%%%%%%%%%%%%%%%%%%%************** |
| ·················%%%%%%÷÷÷÷÷÷÷÷###|#//÷÷%%%%%%%%%%%%%%%%%%%%%%%************* |
| ··············%%%%%%%%%%%%%%''''##|'%%%%%%%%%%%%%%%%%%%%%%%%%%************** |
| ·········%%%%%%%%%%%%%%%%%''''''''|'''%%%%%%%%%%%%%%%%%%%%%%%%%************* |
| ···%%%%%%%%%%%%%%%%%%%%%%%''''''''''''%%%%%%%%%%%%%%%%%%%%%%%%%************* |
| %%%%%%%%%%%%%%%%%%%%%%%%%%%%'''''''%%%%%%%%%%%%%%%%%%%%%%%%%%%************** |
| %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*************** |
| %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%***************** |
+------------------------------------------------------------------------------+

PHIS-SCQ

This is the phis-scq install wizard ...
"

# }}}


# Conda installation {{{

echo "
[ 1 ] Conda installation -------------------------------------------------------
      You should have a proper conda/miniconda installation on your machine.
      If you have it, please provide its path, otherwise I will create a clean
      installation of miniconda3 in your computer.
"

read -p "      Do you have a conda3/miniconda3 installation (y/[n])? " hasconda
hasconda=${hasconda:-n}
if [ "$hasconda" != "${hasconda#[Yy]}" ] ;then
  hasconda=1
else
  hasconda=0
fi

if [ "$hasconda" == 1 ] ;then
  echo "      Where is that installation?"
  echo -n "      "; read condapath
else
  echo "      Where do you want to place it [$HOME/conda3/]?"
  echo -n "      "; read condapath
  condapath=${condapath:-$HOME/conda3/}
fi

# }}}


# Conda environment creation {{{

echo "
[ 2 ] Conda environment creation -----------------------------------------------
      We are now going to create a new conda enviroment to work. You will be
      prompted to provide a name for the environment and Aall needed
      packages will be installed automatically. First you need to write down
      some paths for ipanema3 package and the cuda path (if you have nVidia
      device).
"

read -p "      Name for new enviroment [phisscq]: " myenv
myenv=${myenv:-phisscq}
echo "      Where do you want to place ipanema3 [$HOME/ipanema3/]?"
read -p "      " ipapath
ipapath=${ipapath:-$HOME/ipanema3/}
read -p "      Does your machine have a nVidia device (y/[n])? " hascuda
hascuda=${hascuda:-n}
if [ "$hascuda" != "${hascuda#[Yy]}" ] ;then
  echo "      Where is the cuda installation [/usr/local/bin/cuda-10.2]?"
  echo -n "      "; read cudapath
  cudapath=${cudapath:-/usr/local/bin/cuda-10.2}
else
  cudapath="None"
fi

# }}}


# Create bash function to activate environment {{{

echo "
[ 3 ] Initialization function ---------------------------------------------------
      In order to properly initialize the required environment to run this
      repository you may want to create a bash function. You will be asked if
      you want to create that function and add it to your bashrc profile.
"

function_string="
function activatephisscq() {
  source $condapath/bin/activate
  conda activate $myenv
  export PYOPENCL_COMPILER_OUTPUT=1
  export PYOPENCL_NO_CACHE=1
  export PHISSCQ=${PWD}
  export PYTHONPATH=\$PHISSCQ/analysis:\$PYTHONPATH
  export KRB5_CONFIG=\$PHISSCQ/krb5.conf"
if [ "${cudapath}" != "None" ];then
  cuda_string="
  export PATH='${cudapath}/bin:\$PATH'
  export LD_LIBRARY_PATH='${cudapath}/lib64:\$LD_LIBRARY_PATH'
  export IPANEMA_BACKEND='cuda'
  }
  "
else
  cuda_string="
  export IPANEMA_BACKEND='opencl'
  }
  "
fi

function_string=$function_string$cuda_string

echo "$function_string"

read -p "      Do you want to dump the previous function to bashrc (y/[n])? " q
q=${q:-n}
if [ "$q" != "${q#[Yy]}" ] ;then
  echo "      Where is your bash profile  installation [$HOME/.bashrc]?"
  echo -n "      "; read bashpath
  bashpath=${bashpath:-$HOME/.bashrc}
else
  bashpath="None"
fi

if [ "${bashpath}" != "None" ];then
  echo "$function_string" >> ${bashpath}
fi

# }}}


# Installation plan {{{

echo "
[ 4 ] Installation plan  --------------------------------------------------------
      The installation plan will be printed below and the installation itself
      will start right after you hit enter.
      A config.json file was created from utils/default.json, and you should
      edit it in order to properly set some variables. It is mandatory to set
      path key.

          Good luck!
          pour yourself a drink, this is going to take a while
"

if [ "${hasconda}" == 0 ];then
  echo "      ! wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh"
  echo "      ! bash miniconda.sh -b -p ${condapath}"
fi

echo "      ! source $condapath/bin/activate"
# echo "      ! conda env create -f .ci/environment.yaml"
echo "      ! conda create --name $myenv --file .ci/environment.txt"
echo "      ! conda activate phisscq"

echo "      ! git clone ssh://git@gitlab.cern.ch:7999/mromerol/ipanema3.git ${ipapath}"
echo "      ! pip install -e ${ipapath}/"
echo "      ! pip install .ci/requirements.txt"

if [ "${cudapath}" != "None" ];then
  echo "      ! export PATH='${cudapath}/bin:\$PATH'"
  echo "      ! export LD_LIBRARY_PATH='${cudapath}/lib64:\$LD_LIBRARY_PATH'"
  echo "      ! pip install pycuda"
fi
echo " "
read -p "      [PRESS ENTER]" dummy

# }}}


# Run the installation {{{

export CONDA_ALWAYS_YES="true"
unset PYTHONPATH
if [ "${hasconda}" == 0 ];then
  if [ "$(uname)" == "Darwin" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  fi
  bash miniconda.sh -b -f -p ${condapath}
fi

if [ "${cudapath}" != "None" ]; then
  export PATH="${cudapath}/bin:${PATH}"
  export LD_LIBRARY_PATH="${cudapath}/lib64:${LD_LIBRARY_PATH}"
fi

source ${condapath}/bin/activate
# conda env create -f .ci/environment.yaml
conda create --name $myenv --file .ci/environment.txt
conda activate $myenv
rm miniconda.sh

git clone ssh://git@gitlab.cern.ch:7999/mromerol/ipanema3.git ${ipapath}
pip install -e ${ipapath}/
pip install -r .ci/requirements.txt

if [ "${cudapath}" != "None" ];then
  pip install pycuda
fi

# }}}


# vim:foldmethod=marker
