import argparse
import uproot
import yaml
import os
import json
from ipanema.tools.misc import get_vars_from_string




################################################################################
# Define some functions ########################################################

# Load branches.yaml -----------------------------------------------------------
with open(r'samples/branches.yaml') as file:
  # The FullLoader parameter handles the conversion from YAML
  # scalar values to Python the dictionary format
  branches = yaml.load(file, Loader=yaml.FullLoader)


# sWeight setter ---------------------------------------------------------------
#    Doc
def alpha(x, y=1.0):
  z = x/y
  return z*( (z.sum())/((z**2).sum()) )

################################################################################



################################################################################
# Reduce function ##############################################################

def reduce(input_file,output_file,
           input_tree='DecayTree',output_tree='DecayTree',
           uproot_kwargs=None):
  """
  This function reduces a root file
  """
  # load file
  y,m,f = input_file.split('/')[-3:]
  in_file = uproot.open(input_file)[input_tree]

  # get all neeeded branches
  in_branches = [get_vars_from_string(v) for v in list(branches[m].values())]
  in_branches = sum(in_branches, []) # flatten
  in_branches = list(dict.fromkeys(in_branches)) # remove duplicated ones
  all_branches = [file.decode() for file in in_file.keys()]
  needed_branches = [b for b in in_branches if b in all_branches]

  # create df
  if uproot_kwargs:
    df = in_file.pandas.df(branches=needed_branches,**json.loads(uproot_kwargs))
  else:
    df = in_file.pandas.df(branches=needed_branches)

  # loop of branches and add them to df
  for branch, expr in zip(branches[m].keys(),branches[m].values()):
   try:
     df = df.eval(f'{branch}={expr}')
     print(f'Added {expr} as {branch} to output_file.')
   except:
     try:
       df = df.eval(f'{branch}=@{expr}')
       print(f'Added {expr} as {branch} to output_file.')
     except:
       print(f'Cannot add {expr} as {branch} to output_file.')

  # write reduced file
  print(f'\nStarting to write output_file file.')
  if os.path.isfile(output_file):
    print(f'    Deleting previous version of this file.')
    os.remove(output_file)
  with uproot.recreate(output_file,compression=None) as out_file:
   out_file[output_tree] = uproot.newtree({var:'float64' for var in df})
   out_file[output_tree].extend(df.to_dict(orient='list'))
  out_file.close()



################################################################################



################################################################################
# Argument parser ##############################################################

def argument_parser():
  parser = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  # Samples
  parser.add_argument('--input-file',
                      default = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_oldntuple.root',
                      help='Full root file with huge amount of branches.')
  parser.add_argument('--output-file',
                      default = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test.root',
                      help='Reduce root file.')
  parser.add_argument('--input-tree',
                      default = 'DecayTree',
                      help='Input file tree name.')
  parser.add_argument('--output-tree',
                      default = 'DecayTree',
                      help='Output file tree name.')
  parser.add_argument('--uproot-kwargs',
                      #default = '{"entrystart":0, "entrystop":100}',
                      help='Arguments to uproot.pandas.df')

  return parser

################################################################################



################################################################################
# Run on command line ##########################################################

if __name__ == "__main__":
  print(f"{80*'='}\n= {'Reducing root file':77}=\n{80*'='}\n")
  args = vars(argument_parser().parse_args())
  print(f"{'input_file':>15}: {args['input_file'][25:]:<63.63}")
  print(f"{'input_tree':>15}: {args['input_tree']}")
  print(f"{'output_file':>15}: {args['output_file'][25:]:<63.63}")
  print(f"{'output_tree':>15}: {args['output_tree']}")
  print(f"{'uproot_kwargs':>15}: {args['uproot_kwargs']}\n")
  reduce(**args)

################################################################################
