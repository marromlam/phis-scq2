# Samples

The samples this repository is expected to use are the ones produced by the
_Bs2JpsiPhi-FullRun2_ pipeline. It the user wants to use others than those,
then is his/her job to copy them in the correct path-structure.

The samples in phis-scq follow the following structure:
```
/path/to/phis/phis_samples/version/year/mode/flag.root
```
where `version` should be matched with the version of the _Bs2JpsiPhi-FullRun2_ pipeline
user to produce the tuples, `year` and `mode` are the corresponding year and mode of tuple
(which follows the same names as in _Bs2JpsiPhi-FullRun2_) and last `flag` which is a string
that identifies the nature of the tuple.

The `flag` changes in the first steps of the pipeline, namely during the reweighting
and reuctudton of the tuples. Tuples are first named with flag being `dateflag_selected_bdt` or 
`dateflag_selected_bdt_sw` as those are the final steps of _Bs2JpsiPhi-FullRun2_
pipeline. The `dateflag` always consist in 6 digit number and an alphabet letter,
where the numbers correspond tho the date when the tuple was copied  to the host
where phis-scq pipeline is running and the letter's purpose is to avoid 
overwriting tuples if within the same day the user is copying two versions of the same tuple.

## Branches

The `branches.yaml` file contains all branches that will remain after running
`reduce_ntuples.py`, that is, in the soon-to-be analysed `dateflag.root` tuples.
As matter of fact, most of phis-scq relies on ipanema, and ipanema uses as much
as possible pandas.DataFrames. So, those branches are being loaded as a pandas.DataFrame.
The classical operations like adding, product, exp, trigonometric fuctions... are
avaliable by default in pandas. However one can in some circumstances need special
functions, this is also posible within this environment. For example
```sWeight = alpha(sw,gbweights)```
is calling the alpha function which is defined in reduce_ntuples.py. That is the
place new functions to be defined.