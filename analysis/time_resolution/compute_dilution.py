# compute_dilution
#
#


__all__ = ['compute_dilution']
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import argparse
import uproot3 as uproot
import ipanema
import numpy as np
import uncertainties as unc


def compute_dilution(df, data_ll, time_range, sigma_edges, mode, pdf_pars):
  """
  Compute dilution
  """
  print(mode)
  if mode == 'MC_Bs2JpsiPhi':
    time = "time - 1000*B_TRUETAU_GenLvl"
  else:
    time = 'time'

  sigmat = 'sigmat'
  tLL, tUL = time_range
  sLL, sUL = sigma_edges[0], sigma_edges[-1]

  NBin = len(sigma_edges) - 1

  delta_mErr = 0.020
  dM = 17.74

  tbias = []
  fprompt = []
  fphys = []
  fll = []
  for i, p in enumerate(pdf_pars):
    tbias.append(p['mu'].uvalue)
    fprompt.append(p['fprompt'].uvalue)
    fphys.append(p['fphys2'].uvalue)
    fll.append(p['fll'].uvalue)

  # place time cut in the main dataframe
  full_df = df.query(f"{time}>{tLL} & {time}<{tUL}")

  weight = False
  cut = False
  if cut:
    full_df = full_df.query(cut)

  # start loop
  pars = ipanema.Parameters()
  for i, sLL, sUL in zip(range(NBin), sigma_edges[:-1], sigma_edges[1:]):
    cdf = df.query(f"{sigmat}>{sLL} & {sigmat}<{sUL}")
    cdf_ll = df.query(f"{sigmat}>{sLL} & {sigmat}<{sUL}")
    # TODO: use LL sample here!
    # df_ll_input = RDataFrame("b_{}".format(i), data_ll)

    # correct time with the measured bias
    tcorr = f"{time} - {tbias[i].n}"
    # tcorr = f"{time} - -0.0014090904634665863"
    cdf.eval(f"tcorr = {tcorr}", inplace=True)
    cdf_ll.eval(f"tcorr = {tcorr}", inplace=True)

    # get number of events
    if weight:
      cosDm = f"{weight} * cos({dM}*tcorr)"
    else:
      cosDm = f"cos({dM}*tcorr)"

    # create cosine variables
    cdf.eval(f"cosDM = {cosDm}", inplace=True)
    cdf.eval("cosDM2 = cosDM * cosDM", inplace=True)
    cdf_ll.eval(f"cosDM = {cosDm}", inplace=True)
    cdf_ll.eval("cosDM2 = cosDM * cosDM", inplace=True)

    # now we place a cut for tcorr < 0
    tdf = cdf.query("tcorr < 0")
    tdf_ll = cdf_ll.query("tcorr < 0")
    print(tdf)

    # get number of events
    if weight:
      nevts = np.sum(tdf[weight].values)
    else:
      nevts = tdf.shape[0]
    nevts_ll = tdf_ll.shape[0]
    print("Stat power", nevts, nevts_ll)

    # compute dilutions {{{

    x = np.float64(tdf['cosDM'].values)
    x_ll = np.float64(tdf_ll['cosDM'].values)
    w = np.sum(x)
    w_ll = np.sum(x_ll)

    _dilution = w / nevts
    _dilution_ll = w_ll / nevts_ll
    _dilution_unc = np.var(x) / (nevts - 1)
    _dilution_unc_ll = np.var(x_ll) / (nevts_ll - 1)
    dil = unc.ufloat(_dilution, np.sqrt(_dilution_unc))
    dil_ll = unc.ufloat(_dilution_ll, np.sqrt(_dilution_unc_ll))
    print(f"bin {i:2}: dil={dil:2uP}  dil_ll={dil_ll:2uP}")

    # create dilutions as ufloats
    pars.add(dict(name=f'dil{i}', value=dil.n, stdev=dil.s))
    pars.add(dict(name=f'dilLL{i}', value=dil_ll.n, stdev=dil_ll.s))

    # }}}

    # compute corrected dilution {{{

    # wrap function to compute D and propagate its uncertainty
    def compute_d(d, dll, f): return d / (1 - f) - f * dll / (1 - f)
    compute_d = unc.wrap(compute_d)

    dcorr = compute_d(dil, dil_ll, fll[i])
    print(f"bin {i:2}: corredted D={dcorr:.2uP}")
    pars.add(dict(name=f'dcorr{i}', value=dcorr.n, stdev=dcorr.s))

    # }}}

    # compute effective sigma {{{

    # wrap function to compute sigma effective
    def compute_s(d, dm): return np.sqrt(-2 * np.log(d)) / dm
    compute_s = unc.wrap(compute_s)

    seff = compute_s(dcorr, unc.ufloat(dM, delta_mErr))
    print(f"bin {i:2}: sigma effective={seff:.2uP}")
    pars.add(dict(name=f'seff{i}', value=seff.n, stdev=seff.s))

    # }}}

    # compute effctive number of events {{{

    def compute_n(n, f): return n * (1 - f)
    compute_n = unc.wrap(compute_n)
    neff = compute_n(unc.ufloat(nevts, np.sqrt(nevts)), fll[i])
    pars.add(dict(name=f'neff{i}', value=neff.n, stdev=neff.s))
    print(f"bin {i:2}: sigma effective={seff:.2uP}")

    # }}}

  arr_num = np.array([pars[f'neff{i}'].value for i in range(NBin)])
  arr_dil = np.array([pars[f'dcorr{i}'].uvalue for i in range(NBin)])
  __n = [pars[f'neff{i}'].value for i in range(NBin)]
  __d = [pars[f'dcorr{i}'].value for i in range(NBin)]
  __e = [pars[f'dcorr{i}'].stdev for i in range(NBin)]
  from math import sqrt, log, exp
  D_eff = sqrt(sum([i * j * j for i, j in zip(__n, __d)]) / sum(__n))
  D_eff_err = sqrt(pow(1. / (2. * D_eff * sum(__n)), 2) * (pow(D_eff, 2) / sum(__n) + sum([pow(i, 2) / j for i, j in zip(__d, __n)]) + 2 * sum([pow(i * j * k, 2) for i, j, k in zip(__d, __n, __e)])))
  print(D_eff, D_eff_err)
  davg = unc.wrap(np.sqrt)(np.sum(arr_num * arr_dil**2) / np.sum(arr_num))
  print(f"Dilution Effective: {davg:.2uPp}")
  pars.add(dict(name="Deff", value=davg.n, stdev=davg.s))

  return pars


if __name__ == '__main__':
  DESCRIPTION = "Calib"
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--data-in', help='Input prompt data.')
  p.add_argument('--data-ll', help="Long-lived input dataset")
  p.add_argument('--mode', help='Mode')
  p.add_argument('--timeres', help='Time resolution type')
  p.add_argument('--total-bins', help='Time resolution type')
  # p.add_argument('--deltaMs', help='Used deltaMs')
  p.add_argument('--json-in', help='Json input file with time bias')
  p.add_argument('--json-out', help='Json output file')
  args = vars(p.parse_args())

  mode = args['mode']
  branches = ['time', 'sigmat', 'B_PT']
  if "MC" in mode:
    branches.append('B_ID_GenLvl')
    branches.append('B_TRUETAU_GenLvl')

  timeres_binning = [0.010, 0.021, 0.026, 0.032,
                     0.038, 0.044, 0.049, 0.054, 0.059, 0.064, 0.08]
  time_range = [-4, 10]

  # main dataframe
  df = uproot.open(args['data_in'])
  df = df[list(df.keys())[0]].pandas.df(branches=branches)
  # load parameters
  p = [ipanema.Parameters.load(p) for p in args['json_in'].split(',')]
  pars = compute_dilution(df, [df], time_range, timeres_binning, mode, p)
  pars.dump(args['json_out'])


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
