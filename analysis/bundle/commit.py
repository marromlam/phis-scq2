__all__ = []
import os
import sys

import numpy as np
import ipanema
import uproot
import matplotlib.pyplot as plt
from git import Repo

import shutil


# From, where and at
git_url = "ssh://git@gitlab.cern.ch:7999/lhcb-b2cc/Bs2JpsiPhi-FullRun2.git"
repo_dir = "tmp/bs2jpsiphi"
branche = "phis-scq-parameters"




# Remove dir if it exists
try:
  shutil.rmtree(f'{repo_dir}')
  print(f'\nRemoving {repo_dir} to have a fresh and clean repo')
except:
  0

# Clone repository
print('Cloning Bs2JpsiPhi-FullRun2.git')
repo = Repo.clone_from(git_url, repo_dir)
repo.git.checkout(f'origin/master')
print(repo.git.status())

# Remove remote branch if it exists
remote = repo.remote(name='origin')
try:
  remote.push(refspec=(f':{branche}'))
  print(f'Previous {branche} branch was deleted')
except:
  0
# Create
repo.git.checkout(f'origin/master')
print(repo.git.status())
print(f'Creating {branche} branch')
repo.git.checkout(b=f'{branche}')
print(repo.git.status())

version = 'v0r5'
pars_path = f'output_new/bundle/params/{version}'
output_dir = f"{repo_dir}/fitinputs/"

# List all files in the bundle
bundle = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{pars_path}")) for f in fn]
print(bundle)

# Prepare file paths in repo tree
output = []
for file in bundle:
  if file.endswith('.json'):
    fdir = os.path.abspath(os.path.dirname(file))
    fdir = fdir.replace('output_new/bundle/params/',output_dir)
    fbase = os.path.basename(file)
    output.append([fdir, fbase, file])
print(output)

# Copy files into repo
print(f'\nCopy all files to {output_dir}')
for fdir, fname, origin in output:
  os.makedirs(fdir, exist_ok=True)
  if fname in os.listdir(fdir):
    os.remove( os.path.join(fdir, fname) )
  shutil.copy(origin, os.path.join(fdir,fname) )
  repo.git.add( os.path.join(fdir,fname) )

print('\nAll files added')
#print(repo.git.status())

print('\nCreate commit')
commit_message  = f"[PHIS-SCQ auto-commit] Adding {version} acceptances"
print(repo.git.commit( '-m', commit_message ))
#print(repo.git.status())

print(f'\nPush to {branche}')
# git push origin phis-scq-parameters
print(repo.git.push("--set-upstream",repo.remote(name='origin'),repo.head.ref))

exit()








def find_flags(v):
  all_flags = []
  all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(f"{SAMPLES_PATH}")) for f in fn]
  print
  for file in all_files:
    filename = os.path.splitext(os.path.basename(file))[0]
    all_flags.append(filename.split('_')[0])
  all_flags = list(dict.fromkeys(all_flags)) # remove duplicates
  return all_flags




# Clone a repository
repo_path = '/tmp/gittle_bare'
repo_url = 'git://github.com/FriendCode/gittle.git'
repo = Gittle.clone(repo_url, repo_path)

# Stage multiple files
repo.stage(['other1.txt', 'other2.txt'])

# Do the commit
repo.commit(name="Samy Pesse", email="samy@friendco.de", message="This is a commit")

# Authentication with RSA private key
key_file = open('/Users/Me/keys/rsa/private_rsa')
repo.auth(pkey=key_file)

# Do push
repo.push()



os.makedirs("./tmp/",exist_ok=True)

print('Cloning Bs2JpsiPhi-FullRun2.git')
os.system('git clone ssh://git@gitlab.cern.ch:7999/lhcb-b2cc/Bs2JpsiPhi-FullRun2.git ./tmp/bs2jpsiphi_repo')

print('Checking out phis-scq-parameters branch')





exit()

sidecar = '/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/'

import uproot
import ipanema

for MODE in ['MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar','Bd2JpsiKstar']:
  if MODE == 'MC_Bs2JpsiPhi_dG0':
    path = '/scratch17/marcos.romero/phis_samples/2015/MC_Bs2JpsiPhi_dG0/'
    polWeight = uproot.open(sidecar+'BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].array('PolWeight')
    pdfWeight = uproot.open(sidecar+'BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights'].array('PDFWeight')
    kinWeight = uproot.open(sidecar+'BsJpsiPhi_DG0_MC_2015_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_BsMCToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
  elif MODE == 'MC_Bd2JpsiKstar':
    path = '/scratch17/marcos.romero/phis_samples/2015/MC_Bd2JpsiKstar/'
    polWeight = uproot.open(sidecar+'BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PolWeight.root')['PolWeight'].array('PolWeight')
    pdfWeight = uproot.open(sidecar+'BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_PDFWeightsSetRun1Bd.root')['PDFWeights'].array('PDFWeight')
    kinWeight = uproot.open(sidecar+'BdJpsiKstar_MC_2015_UpDown_CombDSTLDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_trigCat_BdMCToBdData_BaselineDef_15102018.root')['weights'].array('kinWeight')
  elif MODE == 'Bd2JpsiKstar':
    path = '/scratch17/marcos.romero/phis_samples/2015/Bd2JpsiKstar/'
    kinWeight = uproot.open(sidecar+'BdJpsiKstar_Data_2015_UpDown_20180821_tmva_cut58_sel_sw_trigCat_BdDataToBsData_BaselineDef_15102018.root')['weights'].array('kinWeight')
  if MODE in ('MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar'):
    print('\n\n')
    # print(f'v0r0 tuples: {MODE}')
    # old = ipanema.Sample.from_root(path+'v0r1.root')
    # arrold = old.df[['time','polWeight','pdfWeight','kinWeight']].values
    # print(f"{'time':>10} | {'polWeight':>10} {'polWeight':>10} | {'pdfWeight':>10} {'pdfWeight':>10} | {'kinWeight':>10} {'kinWeight':>10}")
    # print(f"{'':>10} | {'me':>10} {'simon':>10} | {'me':>10} {'simon':>10} | {'me':>10} {'simon':>10}")
    # print(f"{82*'-'}")
    # for i in (0,1,2,3,4,-5,-4,-3,-2,-1):
    #   if i == -5:
    #     print(f"{'...':>10} | {'...':>10} {'...':>10} | {'...':>10} {'...':>10} | {'...':>10} {'...':>10}")
    #   print(f"{arrold[i,0]:1.8f} | {arrold[i,1]:1.8f} {polWeight[i]:1.8f} | {arrold[i,2]:1.8f} {pdfWeight[i]:1.8f} | {arrold[i,3]:1.8f} {kinWeight[i]:1.8f}")
    # print('\n')
    print(f'v0r5 tuples: {MODE}')
    print(f"{'time':>17} | {'sw':>17} | {'polWeight':>17} | {'pdfWeight':>17} | {'kinWeight':>17}")
    print(f"{97*'-'}")
    new = ipanema.Sample.from_root(path+'v0r5.root')
    arrnew = new.df[['time', 'sw', 'polWeight','pdfWeight','kinWeight']].values
    for i in (0,1,2,3,4,-5,-4,-3,-2,-1):
      if i == -5:
        print(f"{'...':>17} | {'...':>17} | {'...':>17} | {'...':>17}")
      print(f"{arrnew[i,0]:+1.14f} | {arrnew[i,1]:+1.14f} | {arrnew[i,2]:+1.14f} | {arrnew[i,3]:+1.14f} | {arrnew[i,4]:+1.14f}")
  else:
    print('\n\n')
    # print(f'v0r0 tuples: {MODE}')
    # old = ipanema.Sample.from_root(path+'v0r1.root')
    # arrold = old.df[['time','kinWeight']].values
    # print(f"{'time':>10} | {'kinWeight':>10} {'kinWeight':>10}")
    # print(f"{'':>10} | {'me':>10} {'simon':>10}")
    # print(f"{33*'-'}")
    # for i in (0,1,2,3,4,-5,-4,-3,-2,-1):
    #   if i == -5:
    #     print(f"{'...':>10} | {'...':>10} {'...':>10}")
    #   print(f"{arrold[i,0]:1.8f} | {arrold[i,1]:1.8f} {kinWeight[i]:1.8f}")
    # print('\n')
    print(f'v0r5 tuples: {MODE}')
    print(f"{'time':>17} | {'sw':>17} | {'kinWeight':>17}")
    print(f"{57*'-'}")
    new = ipanema.Sample.from_root(path+'v0r5.root')
    arrnew = new.df[['time','sw','kinWeight']].values
    for i in (0,1,2,3,4,-5,-4,-3,-2,-1):
      if i == -5:
        print(f"{'...':>17} | {'...':>17} | {'...':>17}")
      print(f"{arrnew[i,0]:+1.14f} | {arrnew[i,1]:+1.14f} | {arrnew[i,2]:+1.14f}")


#%%

plt.plot(arrnew[i,1])


import json
