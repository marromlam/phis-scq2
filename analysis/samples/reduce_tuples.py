__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['reduce']


# Libraries needed
import argparse
import uproot3 as uproot     # just while uproot4 does not have writing methods
import yaml
import os
import hjson
from ipanema.tools.misc import get_vars_from_string


# Load branches.yaml
with open(r'analysis/samples/branches.yaml') as file:
    BRANCHES = yaml.load(file, Loader=yaml.FullLoader)


# Reduce function {{{

def reduce(input_file, output_file, input_tree='DecayTree', 
           output_tree='DecayTree', uproot_kwargs=None):
    """
    This function reduces the branches of an input_file root file to output_file.
    uproot library is uses both for loading and writing the input and output
    files.

    Parameters
    ----------
    input_file : str
      Path to input file.
    output_file : str
      Path to output file.
    input_tree : str, default='DecayTree'
      Name of tree to be reduced.
    output_tree : str, default='DecayTree'
      Name of tree to be reduced.
    uproot_kwargs : dict, default=None
      Arguments to uproot loader. Read uproot docs.

    """
    # load file
    y,m,f = input_file.split('/')[-3:]
    in_file = uproot.open(input_file)[input_tree]

    # get all neeeded branches
    in_branches = [get_vars_from_string(v) for v in list(BRANCHES[m].values())]
    in_branches = sum(in_branches, []) # flatten
    in_branches = list(dict.fromkeys(in_branches)) # remove duplicated ones
    all_branches = [file.decode() for file in in_file.keys()]
    needed_branches = [b for b in in_branches if b in all_branches]
    needed_branches = None

    # create df
    if uproot_kwargs:
        df = in_file.pandas.df(branches=needed_branches,
                               **hjson.loads(uproot_kwargs))
    else:
        df = in_file.pandas.df(branches=needed_branches)

    # loop of branches and add them to df
    not_found = []
    for branch, expr in BRANCHES[m].items():
       try:
           df = df.eval(f'{branch}={expr}')
           print(f'Added {expr:>25} as {branch:>25} to ofile.')
       except:
           try:
               df = df.eval(f'{branch}=@{expr}')
               print(f'Added {expr:>25} as {branch:>25} to ofile.')
           except:
               print(f'Cannot add {expr:>25} as {branch:>25} to ofile.')
               not_found.append(branch)
  
    # get only the needed branches, get ride off the rest
    [BRANCHES[m].pop(b,None) for b in not_found]
    df = df[ list( BRANCHES[m].keys() ) ]
  
    # write reduced file
    print(f'\nStarting to write output_file file.')
    if os.path.isfile(output_file):
        print(f'    Deleting previous version of this file.')
        os.remove(output_file)
    with uproot.recreate(output_file,compression=None) as out_file:
        out_file[output_tree] = uproot.newtree({var:'float64' for var in df})
        out_file[output_tree].extend(df.to_dict(orient='list'))

# }}}


# Run on command line {{{

if __name__ == "__main__":
    DESCRIPTION = """
    This script reduces original tuples from eos acording to branches.yaml.
    It plays the role of a translating-layer between the selection pipeline
    and the phis-scq pipeline.
    """
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--input-file', help='Original root file.')
    p.add_argument('--output-file', help='Reduced root file.')
    p.add_argument('--input-tree', help='Input file tree name.')
    p.add_argument('--output-tree', help='Output file tree name.')
    p.add_argument('--uproot-kwargs', help='Arguments to uproot.pandas.df')
    args = vars(p.parse_args())

    # Print information
    print(f"{80*'='}\nReducing root file\n{80*'='}\n")
    print(f"{'input_file':>15}: {args['input_file'][25:]:<63.63}")
    print(f"{'input_tree':>15}: {args['input_tree']}")
    print(f"{'output_file':>15}: {args['output_file'][25:]:<63.63}")
    print(f"{'output_tree':>15}: {args['output_tree']}")
    print(f"{'uproot_kwargs':>15}: {args['uproot_kwargs']}\n")

    # Run the reduction
    reduce(**args)

# }}}


# vim: fdm=marker
