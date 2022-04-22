import argparse
import yaml
from ROOT import (
    vector,
    gInterpreter,
    ROOT,
    RDataFrame,
    TObject,
    TChain
)

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', nargs='+', help='Path to the input files')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--output-tree-name', default='DecayTree', help='Name of the tree')
        
    return parser


def file_merger(input_files, output_file, output_tree_name):
    ch = TChain(output_tree_name)
    
    for entry in input_files:
        ch.Add(entry)

    ch.Merge(output_file)

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    file_merger(**vars(args))
