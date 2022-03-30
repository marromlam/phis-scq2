import sys
import yaml
import argparse
from ROOT import (
    TH2F,
    TFile,
    AddressOf,
    gInterpreter,
    ROOT,
    RDataFrame,
    TChain,
    PyConfig
)
PyConfig.IgnoreCommandLineOptions = True


def read_from_yaml(mode, branches_files):
    selection_dict = dict()
    for file in branches_files:
        with open(file, 'r') as stream:
            selection_dict.update(yaml.safe_load(stream)[mode])
    return selection_dict


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='Path to the input file')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the normal tree')
    parser.add_argument('--input-gen-file', help='Path to the input file')
    parser.add_argument('--input-gen-tree-name', default='MCTuple', help='Name of the tree with generator level info')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--output-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--output-file-tmp', help='Output ROOT file that will be removed')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--branches-files', nargs='+', help='Yaml files with branches')
    return parser


def add_generator_level_info(input_file, input_tree_name, input_gen_file,
    input_gen_tree_name, output_file, output_tree_name, output_file_tmp, mode,
    branches_files):
    # read basic input tree and build new index
    f_in = TFile.Open(input_file)
    norm_tree = f_in.Get(input_tree_name)
    norm_tree.BuildIndex('runNumber', 'eventNumber')
    # read gen level info input tree and build new index
    f_in_gen = TFile.Open(input_gen_file)
    gen_tree = f_in_gen.Get(input_gen_tree_name)
    gen_tree.BuildIndex('runNumber', 'eventNumber')
    # this is a hack since RDataFrame do not support indexed tree for now
    # see https://sft.its.cern.ch/jira/browse/ROOT-9559
    f_out_tmp = TFile.Open(output_file_tmp, 'recreate')
    ordered_gen_tree = gen_tree.CopyTree('0')
    ordered_gen_tree.AutoSave()
    # we need to make a friend with the same number of events and the same order
    # as the original tree
    for ientry in range(norm_tree.GetEntries()):
        norm_tree.GetEntry(ientry)
        runNumber = norm_tree.runNumber
        eventNumber = norm_tree.eventNumber
        gen_tree.GetEntryWithIndex(runNumber, eventNumber)
        ordered_gen_tree.Fill()
    ordered_gen_tree.AutoSave("saveself")
    # add generator level tree as friend
    gen_tree_alias = 'gen_tree'
    norm_tree.AddFriend(ordered_gen_tree, gen_tree_alias)
    # enable multithreading
    # ROOT.EnableImplicitMT()
    # export tree with friend to RDataFrame
    dataframe = RDataFrame(norm_tree)
    # read branches that should be added from all input files
    branches_to_add = read_from_yaml(mode, branches_files)
    # add generator level info branches from MC tree
    for branch in branches_to_add.keys():
        dataframe = dataframe.Define(branch+'_GenLvl', gen_tree_alias+'.'+branches_to_add[branch])
        print('Added ', branch+'_GenLvl')
    # save the output
    dataframe.Snapshot(output_tree_name, output_file)
    # close file with ordered gen tuple
    f_out_tmp.Close()

if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    add_generator_level_info(**vars(args))
