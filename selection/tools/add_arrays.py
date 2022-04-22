import argparse
import yaml
from ROOT import TFile, TTree, RDataFrame, ROOT, gInterpreter


def read_from_yaml(mode, branches_files):
    selection_dict = dict()
    for file in branches_files:
        with open(file, 'r') as stream:
            selection_dict.update(yaml.safe_load(stream)[mode])
    return selection_dict



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='File to add weights to')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the tree in file')
    parser.add_argument('--input-file-arrays', help = 'File from which to take arrays')
    parser.add_argument('--variables-files', nargs='+', help='Path to the file with variable lists')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--output-file', help='File to store the ntuple with weights')
    parser.add_argument('--output-file-tmp', help = 'A temporary output file')
    return parser


def add_arrays(input_file, input_tree_name, input_file_arrays, variables_files, mode, output_file, output_file_tmp):
    # read basic input tree and build new index
    f_in = TFile.Open(input_file)
    norm_tree = f_in.Get(input_tree_name)
    norm_tree.BuildIndex('runNumber', 'eventNumber')
    # read gen level info input tree and build new index
    f_in_gen = TFile.Open(input_file_arrays)
    arr_tree = f_in_gen.Get(input_tree_name)
    arr_tree.BuildIndex('runNumber', 'eventNumber')
    # this is a hack since RDataFrame do not support indexed tree for now
    # see https://sft.its.cern.ch/jira/browse/ROOT-9559
    f_out_tmp = TFile.Open(output_file_tmp, 'recreate')
    ordered_arr_tree = arr_tree.CopyTree('0')
    ordered_arr_tree.AutoSave()

    # we need to make a friend with the same number of events and the same order
    # as the original tree
    for ientry in range(norm_tree.GetEntries()):
        norm_tree.GetEntry(ientry)
        runNumber = norm_tree.runNumber
        eventNumber = norm_tree.eventNumber
        arr_tree.GetEntryWithIndex(runNumber, eventNumber)
        ordered_arr_tree.Fill()

    ordered_arr_tree.AutoSave("saveself")
    # add generator level tree as friend
    arr_tree_alias = 'gen_tree'
    norm_tree.AddFriend(ordered_arr_tree, arr_tree_alias)
    # enable multithreading
    # ROOT.EnableImplicitMT()
    # export tree with friend to RDataFrame
    dataframe = RDataFrame(norm_tree)
    # read branches that should be added from all input files
    gInterpreter.LoadMacro('selection/tools/copy_array.cpp')
 
    branches_to_add = read_from_yaml(mode, variables_files)
    # add generator level info branches from MC tree
    #for branch in branches_to_add.keys():
    #    dataframe = dataframe.Define(branch, arr_tree_alias+'.'+branches_to_add[branch])
    #    print('Added ', branch)
    # save the output
    dataframe.Snapshot(input_tree_name, output_file)
    # close file with ordered gen tuple
    f_out_tmp.Close()



if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    add_arrays(**vars(args))
