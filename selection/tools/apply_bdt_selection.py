import os
import argparse
import yaml
from array import array
from ROOT import (
    ROOT,
    RDataFrame,
    gROOT,
    TMVA,
    TFile,
    TMath,
    AddressOf,
    addressof,
    PyConfig
)
PyConfig.IgnoreCommandLineOptions = True


from XRootD import client
from XRootD.client.flags import DirListFlags, OpenFlags, MkDirFlags, QueryCode

eos = client.FileSystem('root://eoslhcb.cern.ch/')



def read_from_yaml(mode, selection_files):
    bdt_dict = dict()
    for file in selection_files:
        with open(file, 'r') as stream:
            bdt_dict.update(yaml.safe_load(stream)[mode])
    return bdt_dict


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='Path to the input file')
    parser.add_argument('--input-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--output-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--year', help='Year of the selection in yaml')
    parser.add_argument('--bdt-branches', nargs='+', required=True, help='Yaml files with selection')
    parser.add_argument('--bdt-cut-file', required=True, help='Yaml file with bdt cut to be applied')
    parser.add_argument('--tmva-weight-dir', help='File to read TMVA weight from')
    parser.add_argument('--bdt-method-name', default='BDTG3 method', help='Choose which BDT to apply')
    return parser


def apply_bdt_selection(input_file, input_tree_name, output_file, output_tree_name,
    mode, bdt_branches, bdt_cut_file, tmva_weight_dir, bdt_method_name, year):
    # read the cuts and branches from all input files
    bdt_conversion = read_from_yaml(mode, bdt_branches)
    # read separate cut value
    #bdt_cut = read_from_yaml(mode, bdt_cut_file)['cut']
    # print(bdt_cut_file)
    # if 'json' in bdt_cut_file[0]:
    #     bdt_cut = open(bdt_cut_file[0], 'r').read()
    # else:
    #     bdt_cut = read_from_yaml(mode, bdt_cut_file)[year]
    bdt_cut = float(bdt_cut_file)
    print(f"BDT cut is: {bdt_cut}")

    # prepare BDT reader
    TMVA.Tools.Instance()
    reader = TMVA.Reader()
    mva_vars = dict()
    # assign array to each mva variable
    for var in bdt_conversion.keys():
        mva_vars[var] = array('f', [-999])
        reader.AddVariable(var, mva_vars[var])
    
    # ////////////////////////////////////////////
    # status, listing = eos.dirlist(tmva_weight_dir, DirListFlags.STAT)
    # print(listing.parent)
    # for entry in listing:
    #     print(f"{entry.statinfo.modtimestr} {entry.statinfo.size:>10} {entry.name}")
    # filesystem.dirlist('/tmp', DirListFlags.STAT)

    # ////////////////////////////////////////////
    # GENERALIZE ME !!!!

    # tmva_file = output_file.replace('.root', '.xml')
    # status = False
    # for cf in listing:
    #     if cf.name.endswith(".xml"):
    #         tmva_weight_file = os.path.join(tmva_weight_dir, cf.name)
    #         print(tmva_weight_file, tmva_file)
    #         # status = eos.copy(tmva_weight_file, tmva_file, force=True)
    #         status = os.system(f"xrdcp -f 'root://eoslhcb.cern.ch/'{tmva_weight_file} {tmva_file}")
    #         print(status)
    #         status = True if not status else False
    tmva_file = os.path.join(tmva_weight_dir, 'TMVAClassification_BDTG3.weights.xml')


    print(tmva_file)
    # exit()
    reader.BookMVA(bdt_method_name, tmva_file)
    # read input tuple
    input_tuple_file = TFile.Open(input_file)
    input_tree_no_bdt = input_tuple_file.Get(input_tree_name)
    # copy input tree
    output_tuple_file = TFile.Open(output_file, 'recreate')
    output_tuple_file.cd()
    output_tree = input_tree_no_bdt.CopyTree('0')
    # add branch to fill in BDT response
    gROOT.ProcessLine('struct bdt_vars{Double_t bdtg3;};')
    from ROOT import bdt_vars
    bdt_struct = bdt_vars()
    bdt_branch = output_tree.Branch('bdtg3', addressof(bdt_struct, 'bdtg3'), 'bdtg3/D')
    # loop through the tree and add BDT
    for entry in input_tree_no_bdt:

        for var in bdt_conversion.keys():
            mva_vars[var][0] = getattr(entry, var)

        bdt_struct.bdtg3 = reader.EvaluateMVA(bdt_method_name)
        # apply BDT cut
        if bdt_struct.bdtg3 < float(bdt_cut):
            continue
        # save event if it passed the cut
        output_tree.Fill()
    output_tree.AutoSave('saveself')
    output_tuple_file.Close()


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    apply_bdt_selection(**vars(args))
