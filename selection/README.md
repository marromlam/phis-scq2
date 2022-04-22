# Snakemake selection pipeline

Selection pipeline is implemented using `Snakemake`. Cuts for all decays under
study are specified in `cuts.yaml` with corresponding key, name of the mode. In
`branches.yaml` branches that should be kept and new ones that should be added to
the tree are specified.

## Rules

Selection pipeline is defined in the `Snakefile` using `rules`. To get
information about available rules and their purpose, run

```
$ pushd selection
$ ../run snakemake help
```

If you want to add new rule, don't forget to add a `docstring` with description
as well.
To run a rule from `Snakefile` do

```
$ pushd selection
$ ../run snakemake name-of-the-rule (or name-of-the-required-output)
```

## Generic rules and wildcards

There are generic rules that can be run for different `wildcards`. For example
rule `apply_selection` can be used for any wildcard `mode` and `year`.

```
rule apply_selection:
    """
    Applies cuts specified in input yaml files on the
    given tree and adds branches specified in the
    corresponding yaml file. Saves new tree in the
    requested location
    """
    input:
        script = 'tools/apply_selection.py',
        ..
    output:
        tuples_path('{mode}/{mode}_{year}_selected.root')
    shell:
        'python {input.script} .. \
                               --mode {wildcards.mode} \
                               .. \
                               --year {wildcards.year}'
```

In order to run the rule you have to provide `Snakemake` with values for these
`wildcards`. So, for `mode='Bs2JpsiPhi'` and `year='2015'`, do

```
$ pushd selection
$ ../run snakemake /path/to/output/tuples/Bs2JpsiPhi/Bs2JpsiPhi_2015_selected.root
```

Alternatively, you can make a dummy rule that will run `apply_selection` for the
wildcards that you want, such as `apply_selection_mc` defined as

```
rule apply_selection_mc:
    input:
        expand(rules.add_generator_level_info.output, year=selconfig['years'],
                                                      mode=selconfig['mc'])
```

Running this rule will run `apply_selection` rule for each pair of `year` and `mc`
specified in `config.yaml`.

## Output files

All the rules from `Snakefile` will run both from `eos/` and from you home institute
cluster. The intermediate output will be persisted in the `output/` folder on
your disk.

## Publishing new tuples

To make the output of selection pipeline available for the group it should be
uploaded to the shared analysis folder `/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2`.

Steps to publish new tuples:

1. Make required changes in selection
2. Make Merge Request with your changes
3. When it is merged, go to `Gitlab` webpage and `tag` repository. **!!!Write a detailed description in "Release notes"!!!**.
4. The version of the repository **must match** the `version` in `config.yaml`
5. Log in to `lxplus`
6. Checkout new tag version of the repository
7. `$ pushd selection`
8. Start `screen` session, since running pipeline can take some time
9. Upload your file with
`../run snakemake /path/to/output/tuples/upload_my_file_to_eos_{mode}_{year}.log`
where {mode} and {year} refer to the sample you'd like to upload
10. If you want to run and upload all data (attention, it will take a long time):
`../run snakemake upload_to_eos_data -F`
or all mc (it will take even longer):
`../run snakemake upload_to_eos_mc -F`
11. You can check new tuples in `/eos/lhcb/wg/B2CC/Bs2JpsiPhi-FullRun2/{version}/`

Good luck.
