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

