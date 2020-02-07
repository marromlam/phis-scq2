# -*- coding: utf-8 -*-
import os
import datetime
import argparse
from subprocess import PIPE, Popen

FNULL = open(os.devnull, 'w')

def ls_lxplus(code):
  proc = Popen(['ssh','-4','lxplus', code],
               stdout=PIPE, stderr=FNULL, universal_newlines=True)
  out, err = proc.communicate()
  return out.split('\n')[:-1]


# Parse arguments --------------------------------------------------------------
def argument_parser():
  parser = argparse.ArgumentParser(
      description='Sync pipeline files to another cluster.'
  )
  # Samples
  parser.add_argument('--version','-v',
                      help='Pipeline version.')
  parser.add_argument('--target-path',
                      default = '/scratch17/marcos.romero/phis_samples',
                      help='Cluster path where files will be copied.')
  parser.add_argument('--origin-path',
                      default = '/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2',
                      help='EOS path where ntuples are stored.')
  return parser

args = vars(argument_parser().parse_args())

# Get version, and exit if it is not provided
VERSION = args['version']
if not VERSION:
  print(argument_parser().print_help())
  exit()

# To add a flag in target files
flag = 'abcdefghijklmnopqrstuvwxyz'
date = str(datetime.date.today()).replace('-','')[2:]

# Full origin and target paths
SCQ_PATH = args['target_path']
EOS_PATH = args['origin_path']
TUPLES_PATH  = os.path.join(EOS_PATH,VERSION)
SAMPLES_PATH = os.path.join(SCQ_PATH,VERSION)



# Getting the job done ---------------------------------------------------------

# Get list of tuples from lxplus
all_tuples = ls_lxplus(f'ls -dR {TUPLES_PATH}/*/*/*.root')
samples_dict = {}
for i, tuple in enumerate(all_tuples):
  key = tuple
  value = tuple[len(TUPLES_PATH)+1:] # remove TUPLES_PATH
  if not value.startswith('PID'): # do not copy PID correction tuples
    value = value.split('/')
    value[2] = value[2][(len(value[0])+len(value[1])+2):] # remove mode_year
    value[2] = value[2][:-5] # remove root
    if value[2].endswith(VERSION):
      value[2] = value[2][:-(len(VERSION)+1)] # remove version tag
    samples_dict[key] = [SCQ_PATH,VERSION,value[1],value[0],f'{value[2]}.root']

# Some prints
print(80*'='+'\n='+33*' '+'SYNC SAMPLES'+33*' '+'=\n'+80*'='+'\n')
print(f'{"Date":>20}: {date}')
print(f'{"Version to sync":>20}: {VERSION}')
print(f'{"Origin path":>20}: {EOS_PATH} (eos)')
print(f'{"Target path":>20}: {SCQ_PATH} (master)')
print('\n')

# Actual work
for eos_file, value in zip(samples_dict.keys(),samples_dict.values()):
  p,v,y,m,s = value
  print(f'{m:>20} {y:>4} > {s:43} @ [{v:>4}]')
  print(80*'-')
  exists = False; counter = 0
  master_path = '/'.join([p,v,y,m])
  os.makedirs(master_path, exist_ok=True)     # create path if it does not exist
  while not exists:
    master_file = '/'.join([master_path,f'{date}{flag[counter]}_'+s])
    if counter >= 26:
      print('Stupid (wo)man! How many files do you want to have!')
      exists = True
    if os.path.isfile(master_file):
      counter += 1
    else:
      print(f'   from: eos:~/{eos_file[len(EOS_PATH)+1:]}')
      print(f'     to: master:~/{master_file[len(SCQ_PATH)+1:]}')
      os.system(f'scp lxplus:{eos_file} {master_file}')
      print(f'   done!\n\n')
      exists = True
