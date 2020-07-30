#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.system("""clear && printf "\e[3J" """)
print("""
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

                                 phis-SCQ
""")

print("""
This is the phis-SCQ install wizard
""")


print("""\n\n
[ 1 ] Conda installation -------------------------------------------------------
      You should have a proper conda/miniconda installation on your machine.
      If you have it, please provide its path, otherwise I will create a clean
      installation of miniconda3 in your computer.
""")

condapath = None
while not condapath:
  try:
    condapath = input("Installation of conda at (empty/path): ")
  except SyntaxError:
    condapath = None


  print(f"! wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh")
  print(f"! bash Miniconda3-latest-Linux-x86_64.sh")
  print(f"Miniconda3 was intalled, please tell me where you have installed it")


print(f"! source {condapath}/bin/activate")


print("""\n\n
[ 2 ] Conda environment creation -----------------------------------------------
      We are now going to create a new conda enviroment to work...
""")

try:
  myenv = input("Name for new enviroment (default: phisscq): ")
except SyntaxError:
  myenv = 'phisscq'

print(f"! conda  create --name {myenv}")
print(f"! conda activate {myenv}")
print(f"! conda install pip")
print(f"! conda install pocl")
print(f"! conda install -c conda-forge pyopencl")




print("""\n\n
[ 3 ] Install ipanema3 ---------------------------------------------------------
      We are now going to create a new conda enviroment to work...
""")

try:
  ipapath = input("Path to donwload ipanema3 (default: ./ipanema3): ")
except SyntaxError:
  ipapath = './ipanema3'


print(f"! git clone ssh://git@gitlab.cern.ch:7999/mromerol/ipanema3.git {ipapath}")
print(f"! pip install -e {ipapath}")




print("""\n\n
[ 4 ] Install python packages --------------------------------------------------
      We are now going to create a new conda enviroment to work...
""")

try:
  cudapath = input("Path to cuda (default: None): ")
except SyntaxError:
  cudapath = None

if cudapath:
  print(f"! export PATH=\'f'{cudapath}'/bin:${{PATH}}\'")
  print(f"! export LD_LIBRARY_PATH=\'f'{cudapath}'/lib64:${{LD_LIBRARY_PATH}}\'")
  print(f"! pip install pycuda")

print(f"! pip install snakemake hep_ml py-cpuinfo")



print("""\n\n\n\n
[ 5 ] Create bash function to initialize ---------------------------------------
      We are now going to create a new conda enviroment to work...
""")

print("""
function setup_phisscq {
  source {condapath}/bin/activate
  conda activate {myenv}""")
if cudapath:
  print(""" 
  export PATH=\'f'{cudapath}'/bin:${PATH}\'
  export LD_LIBRARY_PATH=\'f'{cudapath}'/lib64:${LD_LIBRARY_PATH}\'
  """)
print("""
  export IPANEMA_BACKEND="cuda"
  export PYOPENCL_COMPILER_OUTPUT=1
  export PYOPENCL_NO_CACHE=1
}
""")
