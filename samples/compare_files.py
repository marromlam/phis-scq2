from ipanema import hist

from ipanema import samples
import uproot
import matplotlib.pyplot as plt
branches = ['time','sw']


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
