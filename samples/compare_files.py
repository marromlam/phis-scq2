from ipanema import hist

from ipanema import samples
import uproot
import matplotlib.pyplot as plt
import numpy as np
branches = ['time','sw']


f1 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/Bd2JpsiKstar/test_kinWeight.root'
f2 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bd2JpsiKstar/test_pdfWeight.root'
f3 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_kinWeight.root'
f4 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/Bs2JpsiPhi/test_selected_bdt_sw.root'

t1 = '/scratch03/marcos.romero/phisRun2/cooked_test_files/Bd2JpsiKstar/test_kinWeight.root'
t4 = '/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_Data_2016_UpDown_20190123_tmva_cut58_sel_comb_sw_corrLb.root'
t2 = '/scratch03/marcos.romero/phisRun2/cooked_test_files/MC_Bd2JpsiKstar/test_kinWeight.root'
t3 = '/scratch03/marcos.romero/phisRun2/cooked_test_files/MC_Bs2JpsiPhi_dG0/test_kinWeight.root'

#me1 = uproot.open(f1)['DecayTree'].pandas.df(flatten=False)
me2 = uproot.open(f2)['DecayTree'].pandas.df(flatten=False)
#me3 = uproot.open(f3)['DecayTree'].pandas.df(flatten=False)
#me4 = uproot.open(f4)['DecayTree'].pandas.df(flatten=False)
#simon1 = uproot.open(t1)['DecayTree'].pandas.df(flatten=False)
simon2 = uproot.open(t2)['DecayTree'].pandas.df(flatten=False)
#simon3 = uproot.open(t3)['DecayTree'].pandas.df(flatten=False)
#simon4 = uproot.open(t4)['DecayTree'].pandas.df(branches=list(me4.keys()),flatten=False)

#np.where(np.abs(   (simon2.query('X_M > 826 & X_M > 861')['pdfWeight']-me2.query('X_M > 826 & X_M > 861')['pdfWeight'])  )>1e-2,0,1).sum()
#me2['pdfWeight']


#simon2.query('X_M > 826 & X_M < 861')['pdfWeight']-me2.query('X_M > 826 & X_M < 861')['pdfWeight']
#simon2.query('X_M > 861 & X_M < 896')['pdfWeight']-me2.query('X_M > 861 & X_M < 896')['pdfWeight']
#np.where(simon2['truehelcosthetaL']-simon2['truehelcosthetaL_GenLvl']>1e-8,1,0).sum()
#np.where(simon2['truehelcosthetaK']-simon2['truehelcosthetaK_GenLvl']>1e-8,1,0).sum()
#np.where(simon2['truehelphi']-simon2['truehelphi_GenLvl']>1e-8,1,0).sum()

#(simon2.iloc[1:20]['pdfWeight']-me2.iloc[1:20]['pdfWeight']) / (simon2.iloc[1:20]['pdfWeight'])

#simon2.iloc[8]


#simon2.iloc[0]['B_ID'], simon2.iloc[0]-me2.iloc[0]
#simon2.iloc[2]['B_ID'], simon2.iloc[2]-me2.iloc[2]


me2['shit'] = simon2['pdfWeight']

a['simon'] = b['pdfWeight']
del a; del b
a = me2[['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl','truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat','B_ID_GenLvl','pdfWeight']]
b= simon2[['truehelcosthetaK_GenLvl','truehelcosthetaL_GenLvl','truehelphi_GenLvl','B_TRUETAU_GenLvl', 'X_M','sigmat','B_ID_GenLvl','pdfWeight']]

a-b
a.query('(B_TRUETAU_GenLvl)>0.0003 & (B_TRUETAU_GenLvl)<0.015')
np.amin(a.query('(B_TRUETAU_GenLvl)>0.0003 & (B_TRUETAU_GenLvl)<0.015')-b.query('(B_TRUETAU_GenLvl)>0.0003 & (B_TRUETAU_GenLvl)<0.015'))
a.query('abs(pdfWeight-simon)>0.1').query('X_M>826 & X_M<861')


a.iloc[10],b.iloc[10]


me2.query('abs(pdfWeight-shit)<0.1')[['B_TRUETAU_GenLvl']][0:10]
problems = me2.query('abs(pdfWeight-shit)>0.1')
problems
problems.query('B_ID>0')







ilist = shit.index[shit['pdfWeight']>0.001].tolist()
me2.iloc[ilist][['X_M','pdfWeight']]


for i in range(0,100):
  print(shit.query('pdfWeight>0.001')['pdfWeight'][i])
shit.query('pdfWeight>0.01')

np.amax(shit['pdfWeight'])

simon2.query('B_ID<0')['pdfWeight']-me2.query('B_ID<0')['pdfWeight']






simon2.query('X_M>861 & X_M<896')['pdfWeight'], me2.query('X_M>861 & X_M<896')['pdfWeight']
simon2.query('X_M>896 & X_M<931')['pdfWeight'], me2.query('X_M>896 & X_M<931')['pdfWeight']








# polWeight
np.amax(np.abs(me2['polWeight']-simon2['polWeight']))
np.amax(np.abs(me3['polWeight']-simon3['polWeight']))

# pdfWeight
np.amax(np.abs(me2['pdfWeight']-simon2['pdfWeight']))
np.amax(np.abs(me3['pdfWeight']-simon3['pdfWeight']))




# %% ---------

# kinWeight
np.amax(np.abs(me1['kinWeight']-simon1['kinWeight']))
np.amax(np.abs(me2['kinWeight']-simon2['kinWeight']))
np.amax(np.abs(me3['kinWeight']-simon3['kinWeight']))



BREAK


w1 = uproot.open('/scratch03/marcos.romero/phisRun2/cooked_test_files/MC_Bs2JpsiPhi_dG0/test.root')['DecayTree'].pandas.df()
w2 = uproot.open('/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/test_selected_bdt_sw.root')['DecayTree'].pandas.df()
w1-w2



me3['kinWeight']
simon3['kinWeight']
plt.plot(np.abs(me3['kinWeight']-simon3['kinWeight']))


simon2['pdfWeight']













#%% data Bd

f1 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/Bd2JpsiKstar/200212a.root'
f2 = '/scratch17/marcos.romero/phis_samples/v0r2/2017/Bd2JpsiKstar/200212a.root'
f3 = '/scratch17/marcos.romero/phis_samples/v0r2/2018/Bd2JpsiKstar/200212a.root'
t1 = 'DecayTree'
t2 = 'DecayTree'
t3 = 'DecayTree'


s1 = uproot.open(f1)[t1].pandas.df()
s2 = uproot.open(f2)[t2].pandas.df()
s3 = uproot.open(f3)[t3].pandas.df()

len(s1)-len(s2)
s1['sWeight'].values
sw1 = hist(s1.query('hlt1b==0')['sWeight'],bins=60,density=True)
sw2 = hist(s2.query('hlt1b==0')['sWeight'],bins=sw1.edges,density=True)
sw3 = hist(s3.query('hlt1b==0')['sWeight'],bins=sw1.edges,density=True)
plt.step(sw1.bins,sw1.counts,label='2016',linewidth=2)
plt.step(sw2.bins,sw2.counts,label='2017',linewidth=2)
plt.step(sw3.bins,sw3.counts,label='2018',linewidth=2)
plt.legend()
plt.ylabel(r'${}_s\mathrm{Weights} - {B_d} \mathrm{data}$')
plt.xlabel('Events')

#%% MC Bd

f1 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bd2JpsiKstar/200212a.root'
f2 = '/scratch17/marcos.romero/phis_samples/v0r2/2017/MC_Bd2JpsiKstar/200212a.root'
f3 = '/scratch17/marcos.romero/phis_samples/v0r2/2018/MC_Bd2JpsiKstar/200212a.root'
t1 = 'DecayTree'
t2 = 'DecayTree'
t3 = 'DecayTree'


s1 = uproot.open(f1)[t1].pandas.df()
s2 = uproot.open(f2)[t2].pandas.df()
s3 = uproot.open(f3)[t3].pandas.df()

sw1 = hist(s1.query('hlt1b==0')['sWeight'],bins=350,density=True)
sw2 = hist(s2.query('hlt1b==0')['sWeight'],bins=sw1.edges,density=True)
sw3 = hist(s3.query('hlt1b==0')['sWeight'],bins=sw1.edges,density=True)
plt.step(sw1.bins,sw1.counts,label='2016',linewidth=2)
plt.step(sw2.bins,sw2.counts,label='2017',linewidth=2)
plt.step(sw3.bins,sw3.counts,label='2018',linewidth=2)
plt.xlim(-0.1,3)
plt.legend()
plt.ylabel(r'${}_s\mathrm{Weights} - {B_d} \mathrm{MC}$')
plt.xlabel('Events')


#%% MC Bs dG0

f1 = '/scratch17/marcos.romero/phis_samples/v0r2/2016/MC_Bs2JpsiPhi_dG0/200212a.root'
f2 = '/scratch17/marcos.romero/phis_samples/v0r2/2017/MC_Bs2JpsiPhi_dG0/200212a.root'
f3 = '/scratch17/marcos.romero/phis_samples/v0r2/2018/MC_Bs2JpsiPhi_dG0/200212a.root'
t1 = 'DecayTree'
t2 = 'DecayTree'
t3 = 'DecayTree'


s1 = uproot.open(f1)[t1].pandas.df()
s2 = uproot.open(f2)[t2].pandas.df()
s3 = uproot.open(f3)[t3].pandas.df()

sw1 = hist(s1.query('hlt1b==1')['sWeight'],bins=300,density=True,range=(0.99,0.995))
sw2 = hist(s2.query('hlt1b==1')['sWeight'],bins=sw1.edges,density=True)
sw3 = hist(s3.query('hlt1b==1')['sWeight'],bins=sw1.edges,density=True)
plt.plot(sw1.bins,sw1.counts,label='2016',linewidth=2, drawstyle='steps-mid')
plt.plot(sw2.bins,sw2.counts,label='2017',linewidth=2, drawstyle='steps-mid')
plt.plot(sw3.bins,sw3.counts,label='2018',linewidth=2, linestyle=':', drawstyle='steps-mid')
plt.legend()






me = uproot.open('/scratch03/marcos.romero/phisRun2/testers/MC_JpsiPhi_sample2016_kinWeight.root')['DecayTree']
s2 = uproot.open('/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_PDFWeightsSetRun1BsdG0.root')['PDFWeights']
s2 = uproot.open('/scratch03/marcos.romero/phisRun2/test-files/BsJpsiPhi_DG0_MC_2016_UpDown_MDST_20181101_Sim09b_tmva_cut58_sel_sw_kinematicweight.root')['weights']

plt.plot(me.array('kinWeight')-s2.array('kinWeight'))
me.array('pdfWeight')-s2.array('kinWeight')
