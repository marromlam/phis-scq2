DESCRIPTION = """
  Compute VELO weight.
"""


__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']
__all__ = ['polarity_weighting']
__all__ = []


# Modules {{{

import ipanema
import argparse
import uproot3 as uproot
import numpy as np


ROOT_PANDAS = False
if ROOT_PANDAS:
  from shutil import copyfile
  import root_numpy
  # import root_pandas

# }}}


# CMDline interface {{{

if __name__ == "__main__":
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument("--input-sample")
  p.add_argument("--output-sample")
  p.add_argument("--params")
  p.add_argument("--year")
  p.add_argument("--version")
  p.add_argument("--mode")
  args = vars(p.parse_args())

  ofile = args['input_sample']
  tfile = args['output_sample']
  mode = args['mode']

  eff = ipanema.Parameters.load(args["params"])
  sample = ipanema.Sample.from_root(ofile)
  print(sample.branches)

  # TODO: we need to create some sort of structure to test different VELO
  #       weights parametrizations
  eff_model = lambda x, p: p[0] * (1 + p[1] * x**2)

  # create efficiency evaluator
  eff_eval = lambda x: 1/eff_model(x, list(eff.valuesdict().values()))

  # compute VELO weight
  if 'Bu2JpsiKplus' in mode:
    # skip this thing and put nothing but ones 
    sample.df.eval("veloWeight = time/time", inplace=True)
    # sample.df.eval("veloWeight = @eff_eval(docaz_hplus)", inplace=True)
  elif 'Bs2JpsiPhi' in mode or 'Bd2JpsiKstar' in mode:
    sample.df.eval("veloWeight = @eff_eval(docaz_hplus) * @eff_eval(docaz_hminus)", inplace=True)
  print(sample.df)

  # dump it!
  if ROOT_PANDAS:
    copyfile(ofile, tfile)
    veloWeight = np.array(sample.df['veloWeight'])
    veloWeight = np.array(veloWeight, dtype=[('veloWeight', np.float64)])
    root_numpy.array2root(veloWeight, tfile, "DecayTree", mode='update')
    # root_pandas.to_root(odf, args['output_file'], key=otree)
  else:
    with uproot.recreate(tfile) as rf:
      rf["DecayTree"] = uproot.newtree({var:'float64' for var in sample.df})
      rf["DecayTree"].extend(sample.df.to_dict(orient='list'))
  print('veloWeight was succesfully written.')

# }}}
