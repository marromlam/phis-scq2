import argparse
import os

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', help='input file to copy',type=str, nargs='*')
    parser.add_argument('--version',default='', help='specify if version to be added to the name of the file')
    parser.add_argument('--output-directory', help='directory to copy')
    parser.add_argument('--year', default='',help='specify if year to be added to the directory path')
    return parser


def copy_files(input_files, output_directory, version, year):

    os.system('eos mkdir -p' + output_directory)
    print(output_directory)
    head, tail = os.path.split(output_directory)
    if year=='':
        output_directory = head

    print(output_directory)
    os.system('eos mkdir -p' + output_directory)

    if type(input_files) == list: 
        for input_file in input_files:
            base = os.path.basename(input_file)
            file = os.path.splitext(base)[0]
            if version:
                os.system('xrdcp '+input_file+ ' ' + output_directory+'/'+file+'_'+version+'.root')
            else:    
                os.system('xrdcp '+input_file+ ' ' + output_directory+'/'+file+'.root ')
    else:        
        base = os.path.basename(input_files)
        file = os.path.splitext(base)[0]
        if version:
            os.system('xrdcp '+input_files+ ' ' + output_directory+'/'+file+'_'+version+'.root ')
        else:
            os.system('xrdcp '+input_files+ ' ' + output_directory+'/'+file+'.root ')

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    copy_files(**vars(args))
    

