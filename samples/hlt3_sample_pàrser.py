import uproot
import os
import pandas as pd



# Untriggered sample

input_file = '/scratch17/diego/lhcb/marcosito.root'
input_tree = 'T'

output_file = '/scratch17/marcos.romero/phis_samples/v0r4/2021/TOY_Bs2JpsiPhi/200512a.root'
output_tree = 'DecayTree'

df = uproot.open(input_file)[input_tree].pandas.df()

original_branches = list(df.keys())

parser = {
'B_ID' : 'MCID',
'B_ID_GenLvl' : 'MCID',
'time' : 'MCtime',
'cosK' : 'MCcthk',
'cosL' : 'MCcthmu',
'hphi' : 'MCphi',
'truetime' : 'MCtime',
'truecosK' : 'MCcthk',
'truecosL' : 'MCcthmu',
'truehphi' : 'MCphi',
'truetime_GenLvl' : 'MCtime',
'truecosK_GenLvl' : 'MCcthk',
'truecosL_GenLvl' : 'MCcthmu',
'truehphi_GenLvl' : 'MCphi',
'hlt1b' : "1",
}

for k,v in parser.items():
  df.eval(f'{k} = {v}', inplace=True)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with uproot.recreate(output_file,compression=None) as out_file:
 out_file[output_tree] = uproot.newtree({var:'float64' for var in df})
 out_file[output_tree].extend(df.to_dict(orient='list'))
out_file.close()



# Triggeres samples

input_file1 = '/scratch17/diego/lhcb/marcositoTrigIncJpsi.root'
input_file2 = '/scratch17/diego/lhcb/marcositoTrig.root'
input_tree1 = 'T'
input_tree2 = 'T'

output_file = '/scratch17/marcos.romero/phis_samples/v0r4/2021/TOY_Bs2JpsiPhi/200518a.root'
output_tree = 'DecayTree'

df1 = uproot.open(input_file1)[input_tree1].pandas.df()
df2 = uproot.open(input_file2)[input_tree2].pandas.df()
df1.eval('hlt1b=1',inplace=True)
df2.eval('hlt1b=0',inplace=True)
df1.eval('Jpsi_Hlt1DiMuonHighMassDecision_TOS=0',inplace=True)
df2.eval('Jpsi_Hlt1DiMuonHighMassDecision_TOS=1',inplace=True)

parser = {
'B_ID' : 'MCID',
'B_ID_GenLvl' : 'MCID',
'time' : 'MCtime',
'cosK' : 'MCcthk',
'cosL' : 'MCcthmu',
'hphi' : 'MCphi',
'truetime' : 'MCtime',
'truecosK' : 'MCcthk',
'truecosL' : 'MCcthmu',
'truehphi' : 'MCphi',
'truetime_GenLvl' : 'MCtime',
'truecosK_GenLvl' : 'MCcthk',
'truecosL_GenLvl' : 'MCcthmu',
'truehphi_GenLvl' : 'MCphi',
'hlt1b' : "hlt1b",
'Jpsi_Hlt1DiMuonHighMassDecision_TOS' : "Jpsi_Hlt1DiMuonHighMassDecision_TOS",
}

df = pd.concat([df1,df2])

original_branches = list(df.keys())
for k,v in parser.items():
  df.eval(f'{k} = {v}', inplace=True)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with uproot.recreate(output_file,compression=None) as out_file:
 out_file[output_tree] = uproot.newtree({var:'float64' for var in df})
 out_file[output_tree].extend(df.to_dict(orient='list'))
out_file.close()

import numpy as np
np.zeros((10,10,10000000))

+1.55334E+05/2505893.325670605
-5.60576E+04/-887100.0040488661
