# Main Workflow - phis-scq
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors: Marcos Romero

# Main constants ---------------------------------------------------------------
#    VERSION: is the version Bs2JpsiPhi-FullRun2 pipeline was run against, and
#             should be matched with this constant.
#    MAIN_PATH: the path where all eos-samples will be synced, make sure there
#               is enough free space there
#    SAMPLES_PATH: where all ntuples for a given VERSION will be stored
VERSION = 'v0r2'
MAIN_PATH = '/scratch17/marcos.romero/phis_samples/'
SAMPLES_PATH = MAIN_PATH+VERSION+'/'

# Some wildcards options
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_DG0','Bs2JpsiPhi',
         'MC_Bs2JpsiPhi', 'MC_Bd2JpsiKstar'];
years = ['2011','2012','Run1',
         '2015','2016','Run2a','2017','2018','Run2b','Run2'];
trigger = ['combined','biased','unbiased']

# Rule orders
ruleorder: fetch_ntuples > reduce_ntuples



# fetch_ntuples ----------------------------------------------------------------
#    This rule downloads all files corresponding to a given VERSION from EOS
#    to SAMPLES_PATH/VERSION folder. It requires you to make a kinit on your
#    system, and then it work automatically. It takes a bit long.

rule fetch_ntuples:
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_selected_bdt_sw.root'
  shell:
    """
    python samples/get_samples.py -v {VERSION}
    """






# reweightings -----------------------------------------------------------------
#    Reduces the amount of branches in the original ntuples. This rule builds
#    the ntuples that will actually be used for phis-scq analysis package. .split('_')[1],

# rewrite this function int order to reduce code in the rule !!!
def parse_kinw(year,mode,flag):
  if mode.startswith('MC_Bs'):
    return ["Bs2JpsiPhi", "B_PT X_M", "(sw/gb_weights)*polWeight*pdfWeight",
            "Bs2JpsiPhi", "B_PT X_M", "sw"]
  elif mode.startswith('MC_Bd'):
    return mode,'Bd2JpsiKstar'
  elif mode.startswith('Bd'):
    return mode,'Bs2JpsiPhi'
  else:
    return 0


rule polarity_weighting:
  input:
    original = expand(rules.fetch_ntuples.output, year='{year}', mode='MC_{mode}', flag='{flag}'),
    target = expand(rules.fetch_ntuples.output, year='{year}', mode='{mode}', flag='{flag}')
  output:
    sample = SAMPLES_PATH+'{year}/MC_{mode}/{flag}_polWeight.root'
  shell:
    """
    python reweightings/polarity_weighting.py\
           --original-file {input.original}\
           --original-treename DecayTree\
           --target-file {input.target}\
           --target-treename DecayTree\
           --output-file {output.sample}
    """

rule pdf_weighting:
  input:
    sample = expand(rules.polarity_weighting.output, year='{year}', mode='{mode}', flag='{flag}'),
    original = 'reweightings/parameters/tad-2016-both-simon1.json',
    target = 'reweightings/parameters/tad-2016-both-simon2.json'
  output:
    sample = SAMPLES_PATH+'{year}/MC_{mode}/{flag}_pdfWeight.root'
  shell:
    """
    python reweightings/pdf_weighting.py\
           --input-file {input.sample}\
           --tree-name DecayTree\
           --output-file {output.sample}\
           --target-params {input.target}\
           --original-params {input.original}\
           --mode MC_{wildcards.mode}
    """

rule kinematic_weighting:
  input:
    original = lambda wildcards: expand(rules.pdf_weighting.output, year='{year}', mode='{}'.format(parse_kinw(**wildcards)[0]), flag='{flag}'),
    target = lambda wildcards: expand(rules.pdf_weighting.output, year='{year}', mode='{}'.format(parse_kinw(**wildcards)[3]), flag='{flag}'),
  params:
    original_vars = lambda wildcards: "{}".format(parse_kinw(**wildcards)[1]),
    original_weight = lambda wildcards: "{}".format(parse_kinw(**wildcards)[2]),
    target_vars = lambda wildcards: "{}".format(parse_kinw(**wildcards)[4]),
    target_weight = lambda wildcards: "{}".format(parse_kinw(**wildcards)[5])
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_kinWeight.root'
  shell:
    """
    python reweightings/kinematic_weighting.py\
           --original-file {input.original}\
           --original-treename DecayTree\
           --original-vars "{params.original_vars}"\
           --original-weight "{params.original_weight}"\
           --target-file {input.target}\
           --target-treename DecayTree\
           --target-vars "{params.target_vars}"\
           --target-weight "{params.target_weight}"\
           --output-file {output.sample}\
           --n-estimators 20\
           --learning-rate 0.3\
           --max-depth 3\
           --min-samples-leaf 1000\
           --trunc 0
    """


# reduce_ntuples ---------------------------------------------------------------
#    Reduces the amount of branches in the original ntuples. This rule builds
#    the ntuples that will actually be used for phis-scq analysis package.

rule rename_ntuples:
  params:
    sample = lambda wildcards: expand(rules.kinematic_weighting.output,
                   year=f'{wildcards.year}', mode=f'{wildcards.mode}', flag=f'{wildcards.flag}')
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_full.root'
  run:
    import os
    input_file = params.sample[0]
    if os.path.isfile(input_file):
      print(f"Renaming {input_file} to {output.sample}.")
      os.rename(f"{input_file}",f"{output.sample}")
    else:
      try:
        shell(f"snakemake {input_file}")
      except:
        print(f"There are no rules for {input_file}.")
      finally:
        print(f"Renaming {input_file} to {output.sample}.")
        os.rename(f"{input_file}",f"{output.sample}")

rule reduce_ntuples:
  input:
    sample = expand(rules.rename_ntuples.output,
                   year='{year}', mode='{mode}', flag='{flag}')
    # sample = SAMPLES_PATH+'{year}/{mode}/{flag}_selected_bdt_sw.root'
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag,[A-Za-z0-9]+}.root'
  shell:
    """
    python samples/reduce_ntuples.py\
           --input-file {input.sample}\
           --output-file {output.sample}\
           --input-tree DecayTree\
           --output-tree DecayTree
    """


# decay_time_acceptance --------------------------------------------------------
#    These are several rules related to decay-time acceptance. The main one is
#    decay_time_acceptance, which computes the spline coefficients of the
#    Bs2JpsiPhi acceptance.

rule decay_time_acceptance:
  input:
    sample_BsMC = expand(rules.reduce_ntuples.output,
                         mode='MC_Bs2JpsiPhi_dG0',year='{year}',flag='{flag}'),
    sample_BdMC = expand(rules.reduce_ntuples.output,
                         mode='MC_Bd2JpsiKstar',year='{year}',flag='{flag}'),
    sample_BdDT = expand(rules.reduce_ntuples.output,
                         mode='Bd2JpsiKstar',year='{year}',flag='{flag}'),
    script = 'time_acceptance/baseline.py'
  output:
    params_BsMC = 'output/time_acceptance/parameters/{year}/MC_Bs2JpsiPhi_DG0/{flag}_{trigger}.json',
    params_BdMC = 'output/time_acceptance/parameters/{year}/MC_Bs_Bd_ratio/{flag}_{trigger}.json',
    params_BdDT = 'output/time_acceptance/parameters/{year}/Bs2JpsiPhi/{flag}_{trigger}.json'
  shell:
    """
    python {input.script}\
           --BsMC-sample {input.sample_BsMC}\
           --BdMC-sample {input.sample_BdMC}\
           --BdDT-sample {input.sample_BdDT}\
           --BsMC-params {output.params_BsMC}\
           --BdMC-params {output.params_BdMC}\
           --BdDT-params {output.params_BdDT}\
           --mode Bs2JpsiPhi\
           --year {wildcards.year}\
           --flag {wildcards.flag}\
           --trigger {wildcards.trigger}\
           --pycode {input.script}
    """


rule decay_time_acceptance_single:
  input:
    sample = expand(rules.reduce_ntuples.output,
                    mode='{mode}',year='{year}',flag='{flag}'),
    script = 'time_acceptance/single.py'
  output:
    params = 'output/time_acceptance/parameters/{year}/{mode}/{flag}_{trigger}_single.json'
  shell:
    """
    python {input.script}\
           --sample {input.sample}\
           --params {output.params}\
           --mode {wildcards.mode}\
           --year {wildcards.year}\
           --flag {wildcards.flag}\
           --trigger {wildcards.trigger}\
           --pycode {input.script}
    """


# rule decay_time_acceptance_ratio:
#   input:
#     sample_BsMC = 'samples/MC_Bs2JpsiPhi_DG0_{year}.json',
#     sample_BdMC = 'samples/MC_Bd2JpsiKstar_{year}.json',
#     script = 'time_acceptance/{pycode}.py'
#   output:
#     shit = 'output/time_acceptance/{year}/{mode}/{pycode}/{flag}.json'
#   shell:
#     """
#     python {input.script}
#            --BsMC-sample {input.sample_BsMC}
#            --BdMC-sample {input.sample_BdMC}
#            --year        {input.year}
#            --model       {input.mode}
#            --flag        {input.flag}
#     """
#
#
# rule decay_time_acceptance_single:
#   input:
#     sample = 'samples/{mode}_{year}.json',
#     script = 'time_acceptance/{pycode}.py'
#   output:
#     shit = 'output/time_acceptance/{year}/{mode}/{pycode}/{flag}.json'
#   shell:
#     """
#     python {input.script}
#            --BsMC-sample {input.sample}
#            --year        {input.year}
#            --model       {input.mode}
#            --flag        {input.flag}
#     """
