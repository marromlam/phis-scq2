import os
import sys
import yaml
import argparse


def read_from_yaml(mode, selection_files):
    selection_dict = dict()
    for file in selection_files:
        with open(file, 'r') as stream:
            selection_dict.update(yaml.safe_load(stream)[mode])
    return selection_dict


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', help='Path to the input file')
    parser.add_argument('--input-tree-name', help='Name of the tree')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--data-set', help='Mag and Year, for example MagUp_2016')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--tracks-file', nargs='+', help='Yaml file with tracks')
    parser.add_argument('--tmp1', help='Temporary file to apply PID 1st step')
    parser.add_argument('--tmp2', help='Temporary file to apply PID 2nd step')
    return parser


def PIDGen(input_file, input_tree_name, output_file, data_set, mode,
           tracks_file, tmp1, tmp2):
    ## START OF CONFIG
    # Read comments and check vars
    # at least until end of config section

    # List of input ROOT files with MC ntuples. Format:
    #   (inputfile, outputfile, dataset)
    files = [
        (input_file, output_file, data_set),
    ]

    # Name of the input tree
    # Could also include ROOT directory, e.g. "Dir/Ntuple"
    input_tree = input_tree_name

    # Postfixes of the Pt, Eta and Ntracks variables (ntuple variable name w/o branch name)
    # e.g. if the ntuple contains "pion_PT", it should be just "PT"
    ptvar  = "PT"
    etavar = None
    pvar   = "P"
    ## Could use P variable instead of eta
    # etavar = None
    # pvar   = "p"

    ntrvar = "nTracks"  # This should correspond to the number of "Best tracks", not "Long tracks"!

    #seed = None   # No initial seed
    seed = 100    # Alternatively, could set initial random seed

    # Dictionary of tracks with their PID variables, in the form {branch name}:{pidvars}
    # For each track branch name, {pidvars} is a dictionary in the form {ntuple variable}:{pid config},
    #   where
    #     {ntuple variable} is the name of the corresponding ntuple PID variable without branch name,
    #   and
    #     {pid_config} is the string describing the PID configuration.
    # Run PIDCorr.py without arguments to get the full list of PID configs
    tracks = read_from_yaml(mode, tracks_file)

    # IF ON LXPLUS: if /tmp exists and is accessible, use for faster processing
    # IF NOT: use /tmp if you have enough RAM
    # temp_folder = '/tmp'
    # ELSE: use current folder
    # temp_folder = temp_folder

    ## END OF CONFIG


    # make sure we don't overwrite local files and prefix them with random strings
    import string
    import random
    rand_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))  # get 10 random chars for temp_file prefix

    output_tree = input_tree.split("/")[-1]
    treename = input_tree

    for input_file, output_file, dataset in files :
        tmpinfile = input_file
        tmpoutfile = tmp1
        for track, subst in tracks.iteritems() :
            for var, config in subst.iteritems() :
                command = "python $PIDPERFSCRIPTSROOT/scripts/python/PIDGenUser/PIDGen.py"
                command += " -m %s_%s" % (track, ptvar)
                if etavar:
                    command += " -e %s_%s" % (track, etavar)
                elif pvar:
                    command += " -q %s_%s" % (track, pvar)
                else:
                    print('Specify either ETA or P branch name per track')
                    sys.exit(1)
                command += " -n %s" % ntrvar
                command += " -t %s" % treename
                command += " -p %s_%s_corr" % (track, var)
                command += " -c %s" % config
                command += " -d %s" % dataset
                command += " -i %s" % tmpinfile
                command += " -o %s" % tmpoutfile
                if seed :
                    command += " -s %d" % seed

                treename = output_tree
                tmpinfile = tmpoutfile
                if tmpoutfile==tmp1:
                    tmpoutfile = tmp2
                else:
                    tmpoutfile = tmp1

                print(command)
                os.system(command)

        if "root://" in output_file:
            print("xrdcp %s %s" % (tmpinfile, output_file))
            os.system("xrdcp %s %s" % (tmpinfile, output_file))
        else:
            print("cp %s %s" % (tmpinfile, output_file))
            os.system("cp %s %s" % (tmpinfile, output_file))


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    PIDGen(**vars(args))
