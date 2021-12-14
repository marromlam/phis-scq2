# Installation file for the phiscq conda environment to run the 
# Bs2JpsiKK pipeline.


# helper functions {{{
BORDERCHARS="─│─│┌┐┘└"

repeat () {
  str=$(printf "%$2s")
  echo ${str// /$1}
}

min() {
    printf "%s\n" "${@:2}" | sort "$1" | head -n1
}
max() {
    # using sort's -r (reverse) option - using tail instead of head is also possible
    min ${1}r ${@:2}
}

repeat "#" 80

function fill_line () { #(text, width=80, pos='top'):
  _TEXT_=$1
  _LTEXT=`min -g $2 ${#_TEXT_}`
  _TEXT_="${_TEXT_:0:_LTEXT}"
  _TWIDTH=$(expr $2 - $_LTEXT - "4")
  _FILL_=$(repeat "#" $_TWIDTH)
  _TEXT_=$_TEXT_$_FILL_
  echo ${BORDERCHARS:1:1} $_TEXT_ ${BORDERCHARS:3:1} | tr "#" " "
}

function _topbottom_line() {  #(width=80, title=None, pos='top'):
  _TEXT_=$1
  _LTITLE=${#_TEXT_}
  _HWIDTH=$(expr $2 - $_LTITLE)
  _EWIDTH=$(expr $2 - $_LTITLE)
  _HWIDTH=$(expr $_HWIDTH / 2)
  _EWIDTH=$(expr $_EWIDTH % 2)
  _LWIDTH=$(expr $_HWIDTH - "1" + $_EWIDTH )
  _RWIDTH=$(expr $_HWIDTH - "1")
  if [ $3 == 'TOP' ]; then
    _FILL=$(repeat ${BORDERCHARS:0:1} $_LWIDTH)
    _L=${BORDERCHARS:4:1}$_FILL 
    _R=$_FILL${BORDERCHARS:5:1}
  else
    _FILL=$(repeat ${BORDERCHARS:0:1} $_LWIDTH)
    _L=${BORDERCHARS:7:1}$_FILL
    _R=$_FILL${BORDERCHARS:6:1}
  fi
  echo $_L$_TEXT_$_R
}

# }}}


# phis ascii Logo {{{
clear && printf "\e[3J"
echo "
┌──────────────────────────────────────────────────────────────────────────────┐
│ ····································**************************************** │
│ ···································***************************************** │
│ ·····························÷÷÷÷÷÷÷÷÷÷************************************* │
│ ························÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷*********************************** │
│ ······················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷********************************* │
│ ····················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷******************************** │
│ ····················÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷******************************* │
│ ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%**************************** │
│ ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%%%%%************************ │
│ ···················÷÷÷÷÷÷÷÷÷÷÷÷÷÷÷|÷÷÷÷÷÷÷÷÷÷%%%%%%%%%%********************* │
│ ···················÷÷÷÷÷÷÷÷÷÷÷÷÷//|//÷÷÷÷÷÷÷%%%%%%%%%%%%%%****************** │
│ ····················÷÷÷÷÷÷÷÷÷÷÷///|///÷÷÷÷÷%%%%%%%%%%%%%%%%***************** │
│ ·····················÷÷÷÷÷÷÷÷÷//##|////÷÷÷%%%%%%%%%%%%%%%%%%%%************** │
│ ·················%%%%%%÷÷÷÷÷÷÷÷###|#//÷÷%%%%%%%%%%%%%%%%%%%%%%%************* │
│ ··············%%%%%%%%%%%%%%''''##|'%%%%%%%%%%%%%%%%%%%%%%%%%%************** │
│ ·········%%%%%%%%%%%%%%%%%''''''''|'''%%%%%%%%%%%%%%%%%%%%%%%%%************* │
│ ···%%%%%%%%%%%%%%%%%%%%%%%''''''''''''%%%%%%%%%%%%%%%%%%%%%%%%%************* │
│ %%%%%%%%%%%%%%%%%%%%%%%%%%%%'''''''%%%%%%%%%%%%%%%%%%%%%%%%%%%************** │
│ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%*************** │
│ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%***************** │
│                                                                              │"

# fill_line "PHIS-SCQ" 80
# fill_line "This is the install wizard for the phis-scq package" 80
_topbottom_line " PHIS - SCQ " 80 "BOTTOM"
echo " "

# }}}


# Conda installation {{{

echo " "
_topbottom_line " Conda installation " 80 "TOP"
fill_line "You should have a proper conda/miniconda installation on your machine." 80
fill_line "If you have it, please provide its path, otherwise I will create a clean" 80
fill_line "installation of miniconda3 in your computer." 80
fill_line "##NOTE: The path to the conda installation should be a full path." 80
fill_line "########Typical conda path is:" 80
fill_line "########/path/to/my/miniconda3/bin/activate" 80
fill_line "########└── requested path ──┘" 80
fill_line "##NOTE: Please use trailing /." 80
_topbottom_line "" 80 "BOTTOM"
#│─│┌┐┘└"
# try to read conda installatio path
read -p " Do you have a conda3/miniconda3 installation (y/[n])? " HASCONDA
HASCONDA=${HASCONDA:-n}
if [ "$HASCONDA" != "${HASCONDA#[Yy]}" ] ;then
  HASCONDA=1
else
  HASCONDA=0
fi

if [ "$HASCONDA" == 1 ] ;then
  echo " Where is that installation?"
  echo -n " "; read CONDAPATH
else
  echo " Where do you want to place it [$HOME/conda3/]?"
  echo -n " "; read CONDAPATH
  CONDAPATH=${CONDAPATH:-$HOME/conda3/}
fi

# }}}


# Conda environment creation {{{

echo " "
_topbottom_line " Conda environment creation " 80 "TOP"
fill_line "We are going to create a new conda enviroment to work. The new environment" 80
fill_line "is named *phisscq*. It will install all needed conda packages and all PyPI" 80
fill_line "wheels automatically. This environment runs on top of ipanema3 which allows" 80
fill_line "us to compile against openCL or CUDA. You will be prompted if you want to" 80
fill_line "install pycuda (it is now installed by default)." 80
fill_line "## NOTE: pycuda of course needs a working nvcc compiler in your PATH. If you" 80
fill_line "######## don't have currently this binary in your PATH, please abort the" 80
fill_line "######## installation, add it to your PATH and start over." 80
fill_line "## NOTE: pycuda won't be installed if nvcc is not in your PATH" 80
_topbottom_line "" 80 "BOTTOM"

read -p " Do you want to install pycuda wheel (y/[n])? " HASCUDA
HASCUDA=${HASCUDA:-n}
if [ "$HASCUDA" != "${HASCUDA#[Yy]}" ] ;then
  HASCUDA=1
else
  HASCUDA=0
fi

# }}}


# Create bash function to activate environment {{{

echo " "
_topbottom_line " Your initialization function " 80 "TOP"
fill_line "In order to properly initialize the required environment to run this" 80
fill_line "repository you may want to create a bash function. You will be asked if" 80
fill_line "you want to create that function and add it to your bashrc profile." 80
fill_line "" 80
fill_line "##source $CONDAPATH/bin/activate" 80
fill_line "##export PYOPENCL_COMPILER_OUTPUT=1" 80
fill_line "##export PYOPENCL_NO_CACHE=1" 80
fill_line "##export PHISSCQ=${PWD}" 80
fill_line "##export PYTHONPATH=\$PHISSCQ/:\$PYTHONPATH" 80
fill_line "##export KRB5_CONFIG=\$PHISSCQ/krb5.conf" 80
if [ "${HASCUDA}" != 0 ];then
  fill_line "##IPANEMA_BACKEND='cuda'" 80
else
  fill_line "##IPANEMA_BACKEND='opencl'" 80
fi

function_string="
conda activate phisscq
source $CONDAPATH/bin/activate
export PYOPENCL_COMPILER_OUTPUT=1
export PYOPENCL_NO_CACHE=1
export PHISSCQ=${PWD}
export PYTHONPATH=\$PHISSCQ/:\$PYTHONPATH
export KRB5_CONFIG=\$PHISSCQ/krb5.conf"
if [ "${HASCUDA}" != 0 ];then
  cuda_string="
export IPANEMA_BACKEND='cuda'"
else
  cuda_string="
export IPANEMA_BACKEND='opencl'"
fi

fill_line "" 80
_topbottom_line "" 80 "BOTTOM"

function_string=$function_string$cuda_string
read -p " Do you want to dump the previous function to a file ([y]/n)? " q
q=${q:-y}
if [ "$q" != "${q#[Yy]}" ] ;then
  echo -n " Filename [sourceme.sh]: "; read bashpath
  bashpath=${bashpath:-sourceme.sh}
else
  bashpath="None"
fi

if [ "${bashpath}" != "None" ];then
  echo "$function_string" >> ${bashpath}
fi

# }}}


# Installation plan {{{

echo " "
_topbottom_line " Your Installation plan " 80 "TOP"
fill_line "The installation plan will be printed below and the installation itself" 80
fill_line "will start right after you hit enter." 80
fill_line "A config/user.json file was created from .ci/user.json, and you should" 80
fill_line "edit it in order to properly set some variables. It is mandatory to set" 80
fill_line "sidecar key." 80
fill_line "" 80

if [ "${HASCONDA}" == 0 ];then
  fill_line "##wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh" 72
  fill_line "##bash miniconda.sh -b -p ${CONDAPATH}" 80
fi

fill_line "##source $CONDAPATH/bin/activate" 80
fill_line "##conda env create -f .ci/environment.yml" 80
fill_line "##conda activate phisscq" 80
fill_line "##pip install .ci/requirements.txt" 80

if [ "${HASCUDA}" != 0 ]; then
  fill_line "##pip install pycuda" 80
fi
fill_line "" 80

fill_line "######Good luck!" 80
fill_line "######pour yourself a drink, this is going to take a while" 80
fill_line "" 80
_topbottom_line "##hit##enter##" 80 "BOTTOM" | tr "#" " "

echo " "
read -p "" dummy

# }}}


# Run the installation {{{

export CONDA_ALWAYS_YES="true"
unset PYTHONPATH
if [ "${HASCONDA}" == 0 ];then
  if [ "$(uname)" == "Darwin" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
  elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  fi
  bash miniconda.sh -b -f -p ${CONDAPATH}
  rm miniconda.sh
fi

source ${CONDAPATH}/bin/activate
conda env create -f .ci/environment.yml
conda activate phisscq
pip install -r .ci/requirements.txt

if [ "${HASCUDA}" != 0 ];then
  pip install pycuda
fi

cp {.ci,config}/user.json

# }}}


# vim:foldmethod=marker
