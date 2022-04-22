__all__ = []
import argparse
import yaml
import hjson
import pandas as pd
import uproot3 as uproot

ROOT_PANDAS = False
if ROOT_PANDAS:
  from shutil import copyfile
  import root_numpy
  # import root_pandas


C_LIGHT = hjson.load(open('selection/tools/constants.json'))['C_LIGHT']


def read_from_yaml(mode, selection_files):
    selection_dict = dict()
    for file in selection_files:
        with open(file, 'r') as stream:
            selection_dict.update(yaml.safe_load(stream)[mode])
    return selection_dict


def apply_cuts(cuts, dataframe, year):
    for key in cuts.keys():
        cut = cuts[key].format(C_LIGHT=C_LIGHT, year=year)
        if cut: dataframe = dataframe.Filter(cut, key)
    # check efficiencies
    report = dataframe.Report()
    report.Print()
    return dataframe


def apply_selection(input_files, input_tree_name, output_file, output_tree_name,
                    mode, cut_keys, cut_string, selection_files, branches_files,
                    keep_all_original_branches, year):
  """
  General rule for selecting a set of files.
  TODO: Document me
  """
  # TODO: We need to check this functions works always
  names = [input_files] if type(input_files)!=type([]) else input_files
  print('Specified input files:\n', names)
  dataframe = [uproot.open(n)[input_tree_name].pandas.df() for n in names]
  dataframe = pd.concat(dataframe)
  print('Merged DataFrame look like:')
  print(dataframe)

  # read cuts from all input files
  cuts = read_from_yaml(mode, selection_files) if selection_files else {}
  # if cut keys are specified apply only desired cuts for given mode
  if cut_keys:
    cuts = {cut_key: cuts[cut_key] for cut_key in cut_keys}
  # if cut string is specified create corresponding cuts dictionary
  if cut_string:
    cuts = {'cut': cut_string}
  cuts = "(" + ") & (".join(list(cuts.values())) + ")"
  print('Final selection:')
  print(cuts)

  # read branches from all input files
  branches_to_add = read_from_yaml(mode, branches_files) if branches_files else {}
  print("Add the following branches:")
  print(branches_to_add)

  if branches_to_add:
      # get list of existing branches
      branches_in_df = dataframe.GetColumnNames()
      # define new branches and keep original branches if specified
      branches = vector('string')()
      if keep_all_original_branches:
          branches = branches_in_df

      # in case helicity angles and/or docaz are specified in branches
      gInterpreter.LoadMacro('tools/calculate_helicity_angles.cpp')
      gInterpreter.LoadMacro('tools/calculate_helicity_costheta.cpp')
      gInterpreter.LoadMacro('tools/calculate_docaz.cpp')
      gInterpreter.LoadMacro('tools/copy_array.cpp')
      # add new branches
      for branch in branches_to_add.keys():
          branch_value = branches_to_add[branch].format(C_LIGHT=C_LIGHT, year=year)
          if branch not in branches_in_df:
              if branch==branch_value:
                  print('WARNING: {} branch is not present in the original tree. Setting value to -99999.'.format(branch))
                  dataframe = dataframe.Define(branch, "-99999.0")
              elif not branch_value:
                  print('Skipping branch ', branch)
                  continue
              else:
                  dataframe = dataframe.Define(branch, branch_value)
          elif not branch_value:
              print('Skipping branch ', branch)
              continue
          branches.push_back(branch)
      # apply all cuts
      if cuts: dataframe = dataframe.query(cuts)
      print('Branches kept in the pruned tree:', branches)
  else:
      if cuts: dataframe = dataframe.query(cuts)
      print('All branches are kept in the tree')
  print("Selected dataframe looks like:")
  print(dataframe)
  if ROOT_PANDAS:
    copyfile(ofile, tfile)
    polWeight = np.array(odf['polWeight'].values, dtype=[('polWeight', np.float64)])
    root_numpy.array2root(polWeight, ofile, otree, mode='update')
    # root_pandas.to_root(odf, args['output_file'], key=otree)
  else:
    with uproot.recreate(output_file) as rf:
      rf[output_tree_name] = uproot.newtree({var:'float64' for var in dataframe})
      rf[output_tree_name].extend(dataframe.to_dict(orient='list'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-files', nargs='+', help='Path to the input file')
    parser.add_argument('--input-tree-name', default='DecayTree' , help='Name of the tree')
    parser.add_argument('--output-file', help='Output ROOT file')
    parser.add_argument('--output-tree-name', default='DecayTree', help='Name of the tree')
    parser.add_argument('--mode', help='Name of the selection in yaml')
    parser.add_argument('--cut-keys', default='', nargs='+', help='Specify which cuts for the mode should be applied, if not all')
    parser.add_argument('--cut-string', default=None, help='Alternatively, specify cut string directly')
    parser.add_argument('--selection-files', nargs='+', help='Yaml files with selection')
    parser.add_argument('--branches-files', nargs='+', help='Yaml files with branches')
    parser.add_argument('--keep-all-original-branches', default=False, help='Keeps all original branches if True, only adds specified branches if False')
    parser.add_argument('--year', required=True, help='Year of data taking')
    args = parser.parse_args()
    apply_selection(**vars(args))
