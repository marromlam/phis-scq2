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
SAMPLES_PATH = MAIN_PATH + VERSION + '/'

# Some wildcards options ( this is not actually used )
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
  run:
    import os
    if not os.path.exists(f"output.sample"):
      shell("""
        python samples/get_samples.py -v {VERSION}
      """)




# reweightings -----------------------------------------------------------------
#    Reduces the amount of branches in the original ntuples. This rule builds
#    the ntuples that will actually be used for phis-scq analysis package. .split('_')[1],

# rewrite this function int order to reduce code in the rule !!!
def parse_kinw(year,mode,flag):
  print(year,mode,flag)
  if mode.startswith('MC_Bs'):
    return ["Bs2JpsiPhi", "B_PT X_M", "(sw/gb_weights)*polWeight*pdfWeight",
            "Bs2JpsiPhi", "B_PT X_M", "sw"]
  elif mode.startswith('MC_Bd'):
    return ["Bd2JpsiKstar", "B_PT X_M", "sw*polWeight*pdfWeight",
            "Bd2JpsiKstar", "B_PT X_M", "sw"]
  elif mode.startswith('Bd'):
    return ["Bd2JpsiKstar", "B_PT B_P", "sw",
            "Bs2JpsiPhi", "B_PT B_P", "sw"]


rule polarity_weighting:
  params:
    original = lambda wildcards: SAMPLES_PATH+f"{wildcards.year}/{wildcards.mode}/{wildcards.flag}.root",
    target = lambda wildcards: SAMPLES_PATH+f"{wildcards.year}/{wildcards.mode[3:]}/{wildcards.flag}.root",
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_polWeight.root'
  run:
    if f'{wildcards.mode}' in ('MC_Bs2JpsiPhi_dG0',
                               'MC_Bs2JpsiPhi',
                               'MC_Bd2JpsiKstar'):
      shell(f"""
        python reweightings/polarity_weighting.py\
             --original-file {params.original}\
             --original-treename DecayTree\
             --target-file {params.target}\
             --target-treename DecayTree\
             --output-file {output.sample}
      """)
    else:
      shell(f"""
        cp {params.original} {output.sample}
      """)


rule pdf_weighting:
  input:
    sample = expand(rules.polarity_weighting.output,
                    year='{year}', mode='{mode}', flag='{flag}'),
  params:
    original = lambda wildcards: f'{wildcards.mode[3:]}_2016.json',
    target = '{mode}_2016.json'
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_pdfWeight.root'
  run:
    if f'{wildcards.mode}' in ('MC_Bs2JpsiPhi_dG0',
                               'MC_Bs2JpsiPhi',
                               'MC_Bd2JpsiKstar'):
      shell(f"""
      python reweightings/pdf_weighting.py\
             --input-file {input.sample}\
             --tree-name DecayTree\
             --output-file {output.sample}\
             --target-params reweightings/parameters/{params.target}\
             --original-params reweightings/parameters/{params.original}\
             --mode {wildcards.mode}
      """)
    else:
      shell(f"""
        cp {input.sample} {output.sample}
      """)


def super_shit(mode,year,flag):
  file = f"{SAMPLES_PATH}{year}/"
  if mode.startswith('MC_'):
    file += f"{mode[3:]}/{flag}"
    if mode == 'MC_Bd2JpsiKstar':
      file += '_kinWeight.root'
    else:
      file += '_pdfWeight.root'
  else:
    file += f"{'Bs2JpsiPhi'}/{flag}_pdfWeight.root"
  return file



rule kinematic_weighting:
  input:
    original = SAMPLES_PATH+'{year}/{mode}/{flag}_pdfWeight.root',
    target = lambda wildcards: super_shit(wildcards.mode,wildcards.year,wildcards.flag)
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag}_kinWeight.root'
  run:
    import os
    year = f'{wildcards.year}'
    mode = f'{wildcards.mode}'
    flag = f'{wildcards.flag}'
    end  = 'selected_bdt_sw'
    if mode.startswith('MC_Bs2JpsiPhi'):
      shell(f"""
        python reweightings/kinematic_weighting.py\
          --original-file {input.original}\
          --original-treename DecayTree\
          --original-vars "B_PT X_M" \
          --original-weight "(sw/gb_weights)*polWeight*pdfWeight"\
          --target-file {input.target}\
          --target-treename DecayTree\
          --target-vars "B_PT X_M"\
          --target-weight "sw"\
          --output-file {output.sample}\
          --n-estimators 20\
          --learning-rate 0.3\
          --max-depth 3\
          --min-samples-leaf 1000
      """)
    elif mode.startswith('MC_Bd2JpsiKstar'):
      shell(f"""
        python reweightings/kinematic_weighting.py\
          --original-file {input.original}\
          --original-treename DecayTree\
          --original-vars "B_PT X_M" \
          --original-weight "sw*polWeight*pdfWeight"\
          --target-file {input.target}\
          --target-treename DecayTree\
          --target-vars "B_PT X_M"\
          --target-weight "sw*kinWeight"\
          --output-file {output.sample}\
          --n-estimators 20\
          --learning-rate 0.3\
          --max-depth 3\
          --min-samples-leaf 1000
      """)
    elif mode.startswith('Bd2JpsiKstar'):
      shell(f"""
        python reweightings/kinematic_weighting.py\
          --original-file {input.original}\
          --original-treename DecayTree\
          --original-vars "B_PT B_P" \
          --original-weight "sw"\
          --target-file {input.target}\
          --target-treename DecayTree\
          --target-vars "B_PT B_P"\
          --target-weight "sw"\
          --output-file {output.sample}\
          --n-estimators 20\
          --learning-rate 0.3\
          --max-depth 3\
          --min-samples-leaf 1000
      """)
    else:
      shell(f"""
        cp {input.original} {output.sample}
      """)



# reduce_ntuples ---------------------------------------------------------------
#    Reduces the amount of branches in the original ntuples. This rule builds
#    the ntuples that will actually be used for phis-scq analysis package.

def caca(mode):
  if mode.startswith('MC_Bs2JpsiPhi') | mode.startswith('MC_Bd2JpsiKstar') | mode.startswith('Bd2JpsiKstar'):
    return '_kinWeight.root'
  else:
    return '_selected_bdt_sw.root'

rule reduce_ntuples:
  input:
    sample = lambda wildcards: SAMPLES_PATH+"{year}/{mode}/{flag}_kinWeight.root"
  output:
    sample = SAMPLES_PATH+'{year}/{mode}/{flag,[A-Za-z0-9]+}.root'
  run:
    shell(f"""
      python samples/reduce_ntuples.py\
             --input-file {input.sample}\
             --output-file {output.sample}\
             --input-tree DecayTree\
             --output-tree DecayTree
    """)
    shell(f"""
      rm {SAMPLES_PATH}{wildcards.year}/{wildcards.mode}/{wildcards.flag}_*Weight.root
    """)

# rule reduce_ntuples:
#   params:
#     sample = SAMPLES_PATH+'{year}/{mode}/{flag}_selected_bdt_sw.root'
#   output:
#     sample = SAMPLES_PATH+'{year}/{mode}/{flag,[A-Za-z0-9]+}.root'
#   run:
#     import os
#     year = f'{wildcards.year}'
#     mode = f'{wildcards.mode}'
#     flag = f'{wildcards.flag}'
#     end  = 'selected_bdt_sw'
#     filename = f"{SAMPLES_PATH}{year}/{mode}/{flag}
#     if not os.path.isfile(params.sample):
#       shell(f"""
#         python samples/get_samples.py -v {VERSION}
#       """)"
#     if mode[:5] == 'MC_Bs' | mode[:5] == 'MC_Bd' | modemode[:4] == 'Bd2J'):
#       input_file = f"{filename}_kinWeight.root"
#       if not os.path.isfile(f"{input_file}"):
#         print(f"snakemake {input_file}")
#       print(f"""
#         python samples/reduce_ntuples.py\
#                --input-file {input.sample}\
#                --output-file {output.sample}\
#                --input-tree DecayTree\
#                --output-tree DecayTree
#       """)
#     else:
#
#     """
#     python samples/reduce_ntuples.py\
#            --input-file {input.sample}\
#            --output-file {output.sample}\
#            --input-tree DecayTree\
#            --output-tree DecayTree
#     """



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
