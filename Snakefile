# -*- coding: utf-8 -*-

# Main Workflow - phis-scq
#
# Contributors: YOUR NAME(S)


MAIN_PATH = '/scratch17/marcos.romero/phis_samples/'
MAIN_PATH = 'samples/'
VERSION = 'v0r2/'
SAMPLES_PATH = MAIN_PATH+VERSION

# rule rule_name:
#     input:
#         input_name1 = "PATH/TO/input_one",
#         input_name2 = "PATH/TO/input_two"
#     output:
#         output_name1 = "PATH/TO/SAVE/output_one",
#         output_name2 = "PATH/TO/SAVE/output_two"
#     shell:
#         "HOW TO MIX IT ALL TOGETHER"

modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi','MC_Bs2JpsiPhi_DG0','Bs2JpsiPhi','MC_Bd2JpsiKstar'];
years = ['2011','2012','Run1','2015','2016','Run2a','2017','2018','Run2b','Run2'];
#flags = ['baseline']
trigger = ['combined','biased','unbiased']
# expand("output/decay-time-acceptance/{mode}_{year}__.json", mode=modes, year=years);


# def dta_parser(mode,year,flag):
#   if mode == 'Bs2JpsiPhi':
#     modes = ['MC_Bs2JpsiPhi', 'MC_Bd2JpsiKstar', 'Bd2JpsiKstar']
#   elif mode == 'MC_Bd2JpsiKstar':
#     modes = ['MC_Bs2JpsiPhi', 'MC_Bd2JpsiKstar', None]
#   else:
#     print('For testing please use decay_time_acceptance_test')
#   files_to_load = []
#   for file in modes:
#     if file:
#     #files_to_load.append(os.path.abspath('samples/'+file+'_'+str(year)+'__'+flag))
#       files_to_load.append('samples/'+file+'_'+str(year)+'__'+flag+'.json')
#     else:
#       files_to_load.append(None)
#   return *files_to_load


# name = 'MC_Bs2JpsiPhi_2017__baseline'
# file, flags = name.split('__')
# file






rule decay_time_acceptance:
  input:
    # sample_BsMC = SAMPLES_PATH+'{year}/MC_Bs2JpsiPhi_dG0/{flag}.root',
    # sample_BdMC = SAMPLES_PATH+'{year}/MC_Bd2JpsiKstar/{flag}.root',
    # sample_BdDT = SAMPLES_PATH+'{year}/Bd2JpsiKstar/{flag}.root',
    sample_BsMC = SAMPLES_PATH+'{year}/MC_Bs2JpsiPhi_dG0/{flag}.json',
    sample_BdMC = SAMPLES_PATH+'{year}/MC_Bd2JpsiKstar/{flag}.json',
    sample_BdDT = SAMPLES_PATH+'{year}/Bd2JpsiKstar/{flag}.json',
    script = 'time_acceptance/baseline.py'
  output:
    #params_BdDT = 'output/time_acceptance/{year}/{mode}/{flag}_{trigger}.json'
    params_BsMC = 'output/time_acceptance/{year}/MC_Bs2JpsiPhi_DG0/{flag}_{trigger}.json',
    params_BdMC = 'output/time_acceptance/{year}/MC_Bs_Bd_ratio/{flag}_{trigger}.json',
    params_BdDT = 'output/time_acceptance/{year}/Bs2JpsiPhi/{flag}_{trigger}.json'
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
