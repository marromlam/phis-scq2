Samples
-------

This repository aims to use are the tuples produced by the
*Bs2JpsiPhi-FullRun2* pipeline. It the user wants to use others than those,
then is his/her job to copy them to the correct `sidecar` path.

The main samples in phis-scq follow the following structure:
::
/path/to/sidecar/year/mode/version.root

where `version` should be matched with the version of the *Bs2JpsiPhi-FullRun2*
pipeline user to produce the tuples, `year` and `mode` are the corresponding 
year and mode of tuple
(which follows the same names as in *Bs2JpsiPhi-FullRun2*) 
and last `flag` which is a string
that identifies the nature of the tuple.

However, the first tuples handled by phis-scq have an extra flag which tells us
the reweighting step they were produced with. This changes in the first steps of
the pipeline, namely during the reweighting
and becomes fixed to the nominal one with the reduction of the tuples.

The tuples that are automatically synced from eos do follow the pattern `v#r#(p#)`
so they will of course be properly handled. However, if you want to use your custom 
tuples they must follow two rules:
* They cannot be named as any version of the eos ones. ohis-scq will first try
to download this kind of named tuples from eos. Therefore, custom tuples should
a different name.
* The name cannot have underscores, to avois misparsing of paths with snakemake.

Basically to handle the tuple sync and reduction one needs to run two rules
* samples_sync_tuples: This rules directly dowloads from esos the asked version of tuples
to run it one ask for
:: snakemake /path/to/sidecar/year/mode/version_sWeight.root
* samples_reduce_tuples: This rule takes the last step of the reweightings 
and applies different renamings to the branches (see §1.1 below).
Afterwards, the tuples is saved in disk using less space and previous root files
deleted.
:: snakemake /path/to/sidecar/year/mode/version.root

Tuples are first named with flag being `dateflag_selected_bdt` or 
`dateflag_selected_bdt_sw` as those are the final steps of *Bs2JpsiPhi-FullRun2*
pipeline. The `dateflag` always consist in 6 digit number and an alphabet letter,
where the numbers correspond tho the date when the tuple was copied  to the host
where phis-scq pipeline is running and the letter's purpose is to avoid 
overwriting tuples if within the same day the user is copying two versions of the same tuple.

Branches
^^^^^^^^

The `branches.yaml` file contains all branches that will remain after running
`samples_reduce_tuple`, that is, in the soon-to-be analysed `version.root` tuples.
As matter of fact, most of phis-scq relies on ipanema, and ipanema uses as much
as possible pandas.DataFrames. So, those branches are being loaded as a
pandas.DataFrame.
The classical operations like adding, product, exp, trigonometric functions... are
avaliable by default in pandas. However one can in some circumstances need special
functions, this is also posible within this environment. For example
```sWeight = alpha(sw,gbweights)```
is calling the alpha function which is defined in reduce_ntuples.py. That is the
place new functions to be defined.
