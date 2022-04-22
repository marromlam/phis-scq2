import argparse
import os

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-directory', help='input directory to copy')
    parser.add_argument('--output-directory', help='directory to copy')
    return parser


def copy_files(input_directory, output_directory):

    os.system('mkdir -p ' + output_directory)

    base = ['TMVAClassification_BDTG3.class.C', 'TMVAClassification_BDTG3.weights.xml', 'TMVA.root']
    for file in base:
        input_file = os.path.join(input_directory, file)
        os.system('xrdcp '+input_file+ ' ' + output_directory)
        
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    copy_files(**vars(args))
    

