
import os

version = 'v0r4'
folders = ['params', 'figures', 'tables']
disciplines = ['time_acceptance']
years = ['2016', '2018', '2017', '2015']
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar']

Bs2JpsiPhi,MC_Bs2JpsiPhi_dG0, MC_Bs2JpsiPhi, Bd2JpsiKstar, MC_Bd2JpsiKstar

look_file =

[for i in os.listdir('output/{version}/time_acceptance/params')]
os.listdir('output/v0r4/time_acceptance/params/2016')
os.listdir('output/v0r4/*/params/2016')


f"{SAMPLES_PATH}/{v0r4}"

SAMPLES_PATH = "/scratch17/marcos.romero/phis_samples/"


find_flags('v0r4')


meh = ['v0r4/angular_acceptance/2016/Bs2JpsiPhi/200506a_corrected_unbiased.json',
'v0r4/angular_acceptance/2016/Bs2JpsiPhi/200506a_corrected_biased.json',
'v0r4/angular_acceptance/2018/Bs2JpsiPhi/200506a_corrected_unbiased.json',
'v0r4/angular_acceptance/2018/Bs2JpsiPhi/200506a_corrected_biased.json',
'v0r4/angular_acceptance/2017/Bs2JpsiPhi/200506a_corrected_unbiased.json',
'v0r4/angular_acceptance/2017/Bs2JpsiPhi/200506a_corrected_biased.json',
'v0r4/angular_acceptance/2015/Bs2JpsiPhi/200506a_corrected_unbiased.json',
'v0r4/angular_acceptance/2015/Bs2JpsiPhi/200506a_corrected_biased.json',
'v0r4/time_acceptance/2016/Bd2JpsiKstar/200506a_baseline_unbiased.json',
'v0r4/time_acceptance/2016/Bd2JpsiKstar/200506a_baseline_biased.json',
'v0r4/time_acceptance/2018/Bd2JpsiKstar/200506a_baseline_biased.json',
'v0r4/time_acceptance/2018/Bd2JpsiKstar/200506a_baseline_unbiased.json',
'v0r4/time_acceptance/2017/Bd2JpsiKstar/200506a_baseline_unbiased.json',
'v0r4/time_acceptance/2017/Bd2JpsiKstar/200506a_baseline_biased.json',
'v0r4/time_acceptance/2015/Bd2JpsiKstar/200506a_baseline_unbiased.json',
'v0r4/time_acceptance/2015/Bd2JpsiKstar/200506a_baseline_biased.json']



for item in meh:
  print(f'mv {item} {item.replace("200506a_","").replace("Bd2JpsiKstar","Bs2JpsiPhi")}')
mv v0r4/angular_acceptance/2016/Bs2JpsiPhi/200506a_corrected_unbiased.json v0r4/angular_acceptance/2016/Bs2JpsiPhi/corrected_unbiased.json
mv v0r4/angular_acceptance/2016/Bs2JpsiPhi/200506a_corrected_biased.json v0r4/angular_acceptance/2016/Bs2JpsiPhi/corrected_biased.json
mv v0r4/angular_acceptance/2018/Bs2JpsiPhi/200506a_corrected_unbiased.json v0r4/angular_acceptance/2018/Bs2JpsiPhi/corrected_unbiased.json
mv v0r4/angular_acceptance/2018/Bs2JpsiPhi/200506a_corrected_biased.json v0r4/angular_acceptance/2018/Bs2JpsiPhi/corrected_biased.json
mv v0r4/angular_acceptance/2017/Bs2JpsiPhi/200506a_corrected_unbiased.json v0r4/angular_acceptance/2017/Bs2JpsiPhi/corrected_unbiased.json
mv v0r4/angular_acceptance/2017/Bs2JpsiPhi/200506a_corrected_biased.json v0r4/angular_acceptance/2017/Bs2JpsiPhi/corrected_biased.json
mv v0r4/angular_acceptance/2015/Bs2JpsiPhi/200506a_corrected_unbiased.json v0r4/angular_acceptance/2015/Bs2JpsiPhi/corrected_unbiased.json
mv v0r4/angular_acceptance/2015/Bs2JpsiPhi/200506a_corrected_biased.json v0r4/angular_acceptance/2015/Bs2JpsiPhi/corrected_biased.json
mv v0r4/time_acceptance/2016/Bd2JpsiKstar/200506a_baseline_unbiased.json v0r4/time_acceptance/2016/Bs2JpsiPhi/baseline_unbiased.json
mv v0r4/time_acceptance/2016/Bd2JpsiKstar/200506a_baseline_biased.json v0r4/time_acceptance/2016/Bs2JpsiPhi/baseline_biased.json
mv v0r4/time_acceptance/2018/Bd2JpsiKstar/200506a_baseline_biased.json v0r4/time_acceptance/2018/Bs2JpsiPhi/baseline_biased.json
mv v0r4/time_acceptance/2018/Bd2JpsiKstar/200506a_baseline_unbiased.json v0r4/time_acceptance/2018/Bs2JpsiPhi/baseline_unbiased.json
mv v0r4/time_acceptance/2017/Bd2JpsiKstar/200506a_baseline_unbiased.json v0r4/time_acceptance/2017/Bs2JpsiPhi/baseline_unbiased.json
mv v0r4/time_acceptance/2017/Bd2JpsiKstar/200506a_baseline_biased.json v0r4/time_acceptance/2017/Bs2JpsiPhi/baseline_biased.json
mv v0r4/time_acceptance/2015/Bd2JpsiKstar/200506a_baseline_unbiased.json v0r4/time_acceptance/2015/Bs2JpsiPhi/baseline_unbiased.json
mv v0r4/time_acceptance/2015/Bd2JpsiKstar/200506a_baseline_biased.json v0r4/time_acceptance/2015/Bs2JpsiPhi/baseline_biased.json
