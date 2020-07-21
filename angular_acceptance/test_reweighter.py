#!/usr/bin/env python
# -*- coding: utf-8 -*-


################################################################################
# %% Modules ###################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import os
import sys
import hjson
import uncertainties as unc
from uncertainties import unumpy as unp

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# load ipanema
from ipanema import initialize
initialize('cuda',1)
from ipanema import ristra, Sample, Parameters, histogram
from utils.plot import get_range, watermark, mode_tex, get_var_in_latex



# reweighting config
from hep_ml import reweight
# reweighter = reweight.GBReweighter(n_estimators     = 40,
#                                    learning_rate    = 0.25,
#                                    max_depth        = 5,
#                                    min_samples_leaf = 500,
#                                    gb_args          = {'subsample': 1})

#30:0.3:4:500
#20:0.3:3:1000

################################################################################



################################################################################

# YEAR = 2017
# VERSION = 'v0r5'
# MODE = 'MC_Bs2JpsiPhi_dG0'
# TRIGGER = 'unbiased'
# input_params_path = f'angular_acceptance/params/{2016}/MC_Bs2JpsiPhi.json'
#
# sample_mc_path = f'/scratch17/marcos.romero/phis_samples/{YEAR}/{MODE}/{VERSION}.root'
# sample_data_path = f'/scratch17/marcos.romero/phis_samples/{YEAR}/Bs2JpsiPhi/{VERSION}.root'
# output_tables_path = f'output_new/tables/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
# output_params_path = f'output_new/params/angular_acceptance/{YEAR}/{MODE}/{VERSION}_corrected_{TRIGGER}.json'
# output_weight_file = f'/scratch17/marcos.romero/phis_samples/{VERSION}/{YEAR}/{MODE}/{VERSION}_angWeight.root'



################################################################################
################################################################################
################################################################################

VERSION = 'v0r5'

# %% Load samples --------------------------------------------------------------

# Load Monte Carlo samples
mc = {}; data = {}
for y in [2015,2016,2017,2018]:
  mc[f"{y}_new"] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/{y}/MC_Bs2JpsiPhi_dG0/{VERSION}.root')#, cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
  data[f"{y}_new"] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/{y}/Bs2JpsiPhi/{VERSION}.root')#, cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
  try:
    mc[f"{y}_old"] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/{y}/MC_Bs2JpsiPhi_dG0/v0r1.root')#, cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
    data[f"{y}_old"] = Sample.from_root(f'/scratch17/marcos.romero/phis_samples/{y}/Bs2JpsiPhi/v0r1.root')#, cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1')
  except:
    0




##############################################################################
##############################################################################
##############################################################################
"""
for var in ['B_P','B_PT','X_M','hplus_P','hplus_PT','hminus_P','hminus_PT']:
  plt.close()
  y = 2015
  old, new = histogram.compare_hist([mc[f'{y}_old'].df[f'{var}'],mc[f'{y}_new'].df[f'{var}']], [mc[f'{y}_old'].df.eval('sw/gb_weights'),mc[f'{y}_new'].df.eval('sw/gb_weights')],density=True)
  plt.step(old.cmbins,old.counts,label='2015 v0r0')
  plt.step(new.cmbins,new.counts,label='2015 v0r5')
  y = 2016
  old, new = histogram.compare_hist([mc[f'{y}_old'].df[f'{var}'],mc[f'{y}_new'].df[f'{var}']], [mc[f'{y}_old'].df.eval('sw/gb_weights'),mc[f'{y}_new'].df.eval('sw/gb_weights')],density=True)
  plt.step(old.cmbins,old.counts,label='2016 v0r0')
  plt.step(new.cmbins,new.counts,label='2016 v0r5')
  y = 2017
  old, new = histogram.compare_hist([mc[f'2016_old'].df[f'{var}'],mc[f'{y}_new'].df[f'{var}']], [mc[f'2016_old'].df.eval('sw/gb_weights'),mc[f'{y}_new'].df.eval('sw/gb_weights')],density=True)
  plt.step(new.cmbins,new.counts,label=f'{y} v0r5')
  y = 2018
  old, new = histogram.compare_hist([mc[f'2016_old'].df[f'{var}'],mc[f'{y}_new'].df[f'{var}']], [mc[f'2016_old'].df.eval('sw/gb_weights'),mc[f'{y}_new'].df.eval('sw/gb_weights')],density=True)
  plt.step(new.cmbins,new.counts,label=f'{y} v0r5')
  #plt.yscale('log')
  plt.xlabel(f'${get_var_in_latex(var)}$')
  plt.ylabel('sWeighted candidates')
  plt.legend()
  plt.savefig(f'tmp/{var}.pdf')
"""
##############################################################################
##############################################################################
##############################################################################


#%% -----

def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score
from hep_ml.metrics_utils import ks_2samp_weighted


def KS_test(original, target, original_weight, target_weight):
  vars = ['hplus_P','hplus_PT','hminus_P','hminus_PT']
  for i in range(0,4):
    xlim = np.percentile(np.hstack([target[:,i]]), [0.01, 99.99])
    print(f'KS over {vars[i]} ', ' = ', ks_2samp_weighted(original[:,i], target[:,i],
                                     weights1=original_weight, weights2=target_weight))


conf = dict(n_estimators=50, learning_rate=0.1,
            max_depth=3, min_samples_leaf=1000, gb_args={'subsample': 1})

# Kinematic
Xmc = mc['2017_new'].df[['X_M','B_P','B_PT']].values
Xrd = data['2017_new'].df[['X_M','B_P','B_PT']].values
Wmc = mc['2017_new'].df.eval('(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*polWeight*sw/gb_weights').values
Wrd = data['2017_new'].df.eval('(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*sw').values

r = reweight.GBReweighter(**conf)
kinWeight = r.fit(original=Xmc,target=Xrd,original_weight=Wmc,target_weight=Wrd).predict_weights(Xmc)



# Kaon momenta
Xmc = mc['2017_new'].df[['hplus_P','hplus_PT','hminus_P','hminus_PT']].values
Xrd = data['2017_new'].df[['hplus_P','hplus_PT','hminus_P','hminus_PT']].values
Wmc = kinWeight*mc['2017_new'].df.eval('(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*polWeight*pdfWeight*sw/gb_weights').values
Wrd = data['2017_new'].df.eval('(Jpsi_Hlt1DiMuonHighMassDecision_TOS==1)*sw').values

r = reweight.GBReweighter(**conf)
kkpWeight = r.fit(original=Xmc,target=Xrd,original_weight=Wmc,target_weight=Wrd).predict_weights(Xmc)


KS_test(Xmc, Xrd, np.ones_like(Wmc)*kkpWeight, np.ones_like(Wrd))
KS_test(Xmc, Xrd, Wmc*kkpWeight, Wrd)

kkpWeight


for name, new_weights in weights.items():
    W = numpy.concatenate([new_weights / new_weights.sum() * len(target), [1] * len(target)])
    Xtr, Xts, Ytr, Yts, Wtr, Wts = train_test_split(data, labels, W, random_state=42, train_size=0.51)
    clf = GradientBoostingClassifier(subsample=0.3, n_estimators=30).fit(Xtr, Ytr, sample_weight=Wtr)

    print(name, roc_auc_score(Yts, clf.predict_proba(Xts)[:, 1], sample_weight=Wts))






















#%%------------

X = np.r_[Xmc,Xrd]
y = np.hstack((0*Xmc[:,0],0*Xrd[:,0]+1))

# y.shape
# X.shape
# plt.plot(Xmc[:,0],Xmc[:,1],'.',label='MC')
# plt.plot(Xrd[:,0],Xrd[:,1],'.',label='RD')
# plt.legend()
# plt.xlabel( f"${get_var_in_latex('hplus_P')}$" )
# plt.ylabel( f"${get_var_in_latex('hplus_PT')}$" )
# make_square_axes(plt.gca())







#%% -----

#lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
#y_pred = lda.fit(X, y).predict(Xmc)



##############################################################################
##############################################################################
##############################################################################


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

original = mc['2015_new'].df[['X_M','B_P','B_PT']]
target = data['2015_new'].df[['X_M','B_P','B_PT']]
original_weight = mc['2015_new'].df.eval('polWeight*sw/gb_weights')
target_weight = data['2015_new'].df.eval('sw')

X = np.vstack([original, target])
y = np.array([1] * len(original) + [0] * len(target))
w = np.hstack([original_weight, target_weight])

clf = LinearDiscriminantAnalysis()
clf.fit(X, y)





  rdvar = srd.df.eval(strvar)
  mcvar = smc.df.eval(strvar)
  rdwei = srd.df.eval('sWeight')
  mckin = smc.df.eval('sWeight')*kin.df.eval('kinWeight')
  mckkp = mckin*kkp.df.eval(f'pdfWeight{niter}*kkpWeight{niter}')
  #mckkp = mckin
  #%% ---
  hrd, hmckin = ipanema.histogram.compare_hist(
                    [rdvar,mcvar], weights=[rdwei,mckin],
                    density=True, range=get_range(strvar)
                )
  hrd, hmckkp = ipanema.histogram.compare_hist(
                    [rdvar,mcvar], weights=[rdwei,mckkp],
                    density=True, range=get_range(strvar)
                )

  fig, axplot, axpull = ipanema.plotting.axes_plotpull();
  axplot.fill_between(hrd.cmbins,hrd.counts,
                      step="mid",color='k',alpha=0.2,
                      label=f"${mode_tex('Bs2JpsiPhi')}$")
  axplot.fill_between(hmckkp.cmbins,hmckkp.counts,
                      step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
                      label=f"${mode_tex('MC_Bs2JpsiPhi')}$")
  axpull.fill_between(hrd.bins,hmckkp.counts/hrd.counts,1,color='C0')
  axpull.set_ylabel(f"$\\frac{{N( {mode_tex('MC_Bs2JpsiPhi')} )}}{{N( {mode_tex('Bs2JpsiPhi')} )}}$")
  axpull.set_ylim(-0.8,3.2)
  axpull.set_yticks([-0.5, 1, 2.5])
  axplot.set_ylabel('Weighted candidates')
  axpull.set_xlabel(f"${get_var_in_latex(strvar)}$")















y6, y5 = histogram.compare_hist([mc['2016'].df['B_P'],mc['2015'].df['B_P']],[mc['2016'].df.eval('sw/gb_weights'),mc['2015'].df.eval('sw/gb_weights')],density=True)
y6, y7 = histogram.compare_hist([mc['2016'].df['B_P'],mc['2017'].df['B_P']],[mc['2016'].df.eval('sw/gb_weights'),mc['2017'].df.eval('sw/gb_weights')], density=True)
y6, y8 = histogram.compare_hist([mc['2016'].df['B_P'],mc['2018'].df['B_P']],[mc['2016'].df.eval('sw/gb_weights'),mc['2018'].df.eval('sw/gb_weights')], density=True)

plt.step(y5.cmbins,y5.counts,label='2015')
plt.step(y6.cmbins,y6.counts,label='2016')
plt.step(y7.cmbins,y7.counts,label='2017')
plt.step(y8.cmbins,y8.counts,label='2018')
plt.xlim(0,5e5)
plt.legend()

help(plt.hist)
plt.hist(mc['2016'].df['B_PT'], weights=mc['2016'].df.eval('X_M'))
plt.hist(mc['2016'].df['B_PT'])

y6, y5 = histogram.compare_hist([mc['2016'].df['B_PT'],mc['2015'].df['B_PT']],weights=[10*mc['2016'].df.eval('sw/gb_weights'),mc['2015'].df.eval('sw/gb_weights')], density=True)
plt.step(y5.cmbins,y5.counts,label='2015')
plt.step(y6.cmbins,y6.counts,label='2016')
y6, y5 = histogram.compare_hist([mc['2016'].df['B_PT'],mc['2015'].df['B_PT']],density=True)
plt.step(y5.cmbins,y5.counts,label='2015')
plt.step(y6.cmbins,y6.counts,label='2016')

y6, y5 = histogram.compare_hist([mc['2016'].df['B_PT'],mc['2015'].df['B_PT']],[mc['2016'].df.eval('sw/gb_weights'),mc['2015'].df.eval('sw/gb_weights')],density=True)
y6, y7 = histogram.compare_hist([mc['2016'].df['B_PT'],mc['2017'].df['B_PT']],[mc['2016'].df.eval('sw/gb_weights'),mc['2017'].df.eval('sw/gb_weights')], density=True)
y6, y8 = histogram.compare_hist([mc['2016'].df['B_PT'],mc['2018'].df['B_PT']],[mc['2016'].df.eval('sw/gb_weights'),mc['2018'].df.eval('sw/gb_weights')], density=True)

plt.step(y5.cmbins,y5.counts,label='2015')
plt.step(y6.cmbins,y6.counts,label='2016')
plt.step(y7.cmbins,y7.counts,label='2017')
plt.step(y8.cmbins,y8.counts,label='2018')
plt.xlim(0,4e4)
plt.legend()

y6, y5 = histogram.compare_hist([mc['2016'].df['X_M'],mc['2015'].df['X_M']],density=True)
y6, y7 = histogram.compare_hist([mc['2016'].df['X_M'],mc['2017'].df['X_M']],density=True)
y6, y8 = histogram.compare_hist([mc['2016'].df['X_M'],mc['2018'].df['X_M']],density=True)


plt.step(y5.cmbins,y5.counts,label='2015')
plt.step(y6.cmbins,y6.counts,label='2016')
plt.step(y7.cmbins,y7.counts,label='2017')
plt.step(y8.cmbins,y8.counts,label='2018')
plt.legend()


y6, y5 = histogram.compare_hist([mc['2016'].df.eval('sw/gb_weights'),mc['2015'].df.eval('sw/gb_weights')],density=True)
y6, y7 = histogram.compare_hist([mc['2016'].df.eval('sw/gb_weights'),mc['2017'].df.eval('sw/gb_weights')],density=True)
y6, y8 = histogram.compare_hist([mc['2016'].df.eval('sw/gb_weights'),mc['2018'].df.eval('sw/gb_weights')],density=True)


plt.hist(mc['2016'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2016');
#plt.hist(mc['2015'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2015');
plt.hist(mc['2017'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2017');
#plt.hist(mc['2018'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2018');
plt.legend()

plt.hist(Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2015/{MODE}/v0r1.root', cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1').df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2015');
plt.hist(Sample.from_root(f'/scratch17/marcos.romero/phis_samples/2016/{MODE}/v0r1.root', cuts='Jpsi_Hlt1DiMuonHighMassDecision_TOS==1').df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2016');
plt.hist(mc['2015'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2015 v0r5');
plt.hist(mc['2016'].df.eval('sw/gb_weights').values,100,range=(0.9,1.1),label='2016 v0r5');
plt.legend()








plt.xlim(.8,1.2)
plt.legend()


plt.hist2d(mc['2016'].df['B_P'].values, mc['2016'].df['B_PT'].values,1000)
plt.xlim(2e4,5e5)
plt.ylim(0,4e4)

plt.hist2d(mc['2017'].df['B_P'].values, mc['2017'].df['B_PT'].values,1000)
plt.xlim(2e4,5e5)
plt.ylim(0,4e4)

plt.hist2d(mc['2018'].df['B_P'].values, mc['2018'].df['B_PT'].values,1000)
plt.xlim(2e4,5e5)
plt.ylim(0,4e4)

plt.hist2d(mc['2015'].df['B_P'].values, mc['2015'].df['B_PT'].values,1000)
plt.xlim(2e4,5e5)
plt.ylim(0,4e4)






#%% Compute standard kinematic weights -------------------------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables

print(f"\n{80*'='}\n",
      "Weight the MC samples to match data ones",
      f"\n{80*'='}\n")

reweighter = reweight.GBReweighter(n_estimators     = 50,
                                   learning_rate    = 0.2,
                                   max_depth        = 10,
                                   min_samples_leaf = 1000,
                                   gb_args          = {'subsample': 1})

reweighter.fit(original        = mc['2015'].df[['X_M','B_P','B_PT']],
               target          = data['2015'].df[['X_M','B_P','B_PT']],
               original_weight = mc['2015'].df.eval('polWeight*sw/gb_weights'),
               target_weight   = data['2015'].df.eval('sw')
                  );

kinWeight = reweighter.predict_weights(mc['2015'].df[['X_M','B_P','B_PT']])
kinWeight = np.where(mc['2015'].df.eval('polWeight*sw/gb_weights')!=0, kinWeight, 0)
print(f"The kinematic-weighting in B_PT, B_P and X_M is done for {MODE}-{TRIGGER}")
print(f"kinWeight: {kinWeight}")



#%%

import hep_ml



def _normalize_input(data, weights, normalize=True, n_features_=None):
    """ Normalize input of reweighter
    :param data: array like of shape [n_samples] or [n_samples, n_features]
    :param weights: array-like of shape [n_samples] or None
    :return: tuple with
        data - numpy.array of shape [n_samples, n_features]
        weights - numpy.array of shape [n_samples] with mean = 1.
    """
    weights = hep_ml.commonutils.check_sample_weight(data, sample_weight=weights, normalize=normalize)
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    if n_features_ is None:
        n_features_ = data.shape[1]
    assert n_features_ == data.shape[1], \
        'number of features is wrong: {} {}'.format(n_features_, data.shape[1])
    return data, weights



['/home3/marcos.romero/conda3/envs/ipanema3/lib/python3.8/site-packages/hep_ml']import hep_ml
dir(hep_ml)







def my_fit(original, target, original_weight=None, target_weight=None,
           n_estimators=50, learning_rate=0.2, subsample=1., min_samples_split=2, min_samples_leaf=1000, max_features=None, max_leaf_nodes=None, max_depth=2,loss_regularization=5., splitter='best', update_tree=True, train_features=None, random_state=None):
    n_features_ = None
    original, original_weight = _normalize_input(original, original_weight)
    target, target_weight = _normalize_input(target, target_weight)

    loss = hep_ml.losses.ReweightLossFunction(loss_regularization)
    gb = hep_ml.gradientboosting.UGradientBoostingClassifier(loss=loss,
                                             n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             min_samples_leaf=min_samples_leaf,
                                             learning_rate=learning_rate)
    data = np.vstack([original, target])
    target = np.array([1] * len(original) + [0] * len(target))
    weights = np.hstack([original_weight, target_weight])
    gb.fit(data, target, sample_weight=weights)
    return gb


shit = my_fit(original        = mc['2015'].df[['X_M','B_P','B_PT']],
                 target          = data['2015'].df[['X_M','B_P','B_PT']],
                 original_weight = mc['2015'].df.eval('polWeight*sw/gb_weights'),
                 target_weight   = data['2015'].df.eval('sw')
                  );

kinWeight = shit.predict_weights(mc['2015'].df[['X_M','B_P','B_PT']])

    def predict_proba(self, X):
        """Predicted probabilities for each event

        :param X: pandas.DataFrame with all train_features
        :return: numpy.array of shape [n_samples, n_classes]
        """
        return


def staged_decision_function(self, X):
    """Raw output, sum of trees' predictions after each iteration.

    :param X: data
    :return: sequence of numpy.array of shape [n_samples]
    """
    X = SklearnClusteringTree.prepare_data(self._get_train_features(X))
    y_pred = numpy.zeros(len(X)) + self.initial_step
    for tree, leaf_values in self.estimators:
        y_pred += self.learning_rate * self._estimate_tree(tree, leaf_values=leaf_values, X=X)
        yield y_pred

def decision_function(X):
    result = None
    for score in staged_decision_function(X):
        result = score
    return result

def predict( X):
  return numpy.argmax( hep_ml.commonutils.score_to_proba( decision_function(X)) , axis=1)



from sklearn.utils.random import check_random_state




def UGradientBoostingClassifier_fit(self, X, y, sample_weight=None):
  X, y, sample_weight = hep_ml.commonutils.check_xyw(X, y, sample_weight=sample_weight, classification=True)
  return UGradientBoostingBase.fit(self, X, y, sample_weight=sample_weight)



class UGradientBoostingBase(BaseEstimator):
    """ Base class for gradient boosting estimators """

    def __init__(self, loss=None,
                 n_estimators=100,
                 learning_rate=0.1,
                 subsample=1.,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 max_leaf_nodes=None,
                 max_depth=3,
                 splitter='best',
                 update_tree=True,
                 train_features=None,
                 random_state=None):
        """
        `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `min_samples_split`, `max_features` are parameters
        of regression tree, which is used as base estimator.

        :param loss: any descendant of AbstractLossFunction, those are very various.
            See :class:`hep_ml.losses` for available losses.
        :type loss: AbstractLossFunction
        :param int n_estimators: number of trained trees.
        :param float subsample: fraction of data to use on each stage
        :param float learning_rate: size of step.
        :param bool update_tree: True by default. If False, 'improvement' step after fitting tree will be skipped.
        :param train_features: features used by tree.
            Note that algorithm may require also variables used by loss function, but not listed here.
        """
        self.loss = loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.update_tree = update_tree
        self.train_features = train_features
        self.random_state = random_state
        self.splitter = splitter
        self.classes_ = [0, 1]

def _check_params():
    """Checking parameters of classifier set in __init__"""
    assert isinstance(self.loss, AbstractLossFunction), \
        'LossFunction should be derived from AbstractLossFunction'
    assert self.n_estimators > 0, 'n_estimators should be positive'
    assert 0 < self.subsample <= 1., 'subsample should be in (0, 1]'
    self.random_state = check_random_state(self.random_state)

def UGradientBoostingBase_fit(X, y, sample_weight=None, loss=None,
                 n_estimators=100, learning_rate=0.1, subsample=1., min_samples_split=2, min_samples_leaf=1, max_features=None, max_leaf_nodes=None, max_depth=3, splitter='best', update_tree=True, train_features=None, random_state=None):
  #_check_params()

  estimators = []
  scores = []

  n_samples = len(X)
  n_inbag = int(self.subsample * len(X))

  # preparing for loss function
  X, y, sample_weight = check_xyw(X, y, sample_weight=sample_weight)

  assert isinstance(self.loss, AbstractLossFunction), 'loss function should be derived from AbstractLossFunction'
  self.loss = copy.deepcopy(self.loss)
  self.loss.fit(X, y, sample_weight=sample_weight)

  # preparing for fitting in trees, setting appropriate DTYPE
  X = self._get_train_features(X)
  X = SklearnClusteringTree.prepare_data(X)
  self.n_features = X.shape[1]

  y_pred = numpy.zeros(len(X), dtype=float)
  self.initial_step = self.loss.compute_optimal_step(y_pred=y_pred)
  y_pred += self.initial_step

  for stage in range(self.n_estimators):
      # tree creation
      tree = SklearnClusteringTree(
          criterion='mse',
          splitter=self.splitter,
          max_depth=self.max_depth,
          min_samples_split=self.min_samples_split,
          min_samples_leaf=self.min_samples_leaf,
          max_features=self.max_features,
          random_state=self.random_state,
          max_leaf_nodes=self.max_leaf_nodes)

      # tree learning
      residual, weights = self.loss.prepare_tree_params(y_pred)
      train_indices = self.random_state.choice(n_samples, size=n_inbag, replace=False)

      tree.fit(X[train_indices], residual[train_indices],
               sample_weight=weights[train_indices], check_input=False)
      # update tree leaves
      leaf_values = tree.get_leaf_values()
      if self.update_tree:
          terminal_regions = tree.transform(X)
          leaf_values = self.loss.prepare_new_leaves_values(terminal_regions, leaf_values=leaf_values,
                                                            y_pred=y_pred)

      y_pred += self.learning_rate * self._estimate_tree(tree, leaf_values=leaf_values, X=X)
      self.estimators.append([tree, leaf_values])
      self.scores.append(self.loss(y_pred))
  return self




a = [Parameters.load('output_new/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_iteration6_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_iteration6_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2017/Bs2JpsiPhi/v0r5_iteration6_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2018/Bs2JpsiPhi/v0r5_iteration6_unbiased.json')]
b = [Parameters.load('output_new/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_iteration3_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_iteration3_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2017/Bs2JpsiPhi/v0r5_iteration3_unbiased.json'),
     Parameters.load('output_new/params/angular_acceptance/2018/Bs2JpsiPhi/v0r5_iteration3_unbiased.json')]
c = [Parameters.load('output_new/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_iteration6_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_iteration6_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2017/Bs2JpsiPhi/v0r5_iteration6_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2018/Bs2JpsiPhi/v0r5_iteration6_biased.json')]
d = [Parameters.load('output_new/params/angular_acceptance/2015/Bs2JpsiPhi/v0r5_iteration3_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2016/Bs2JpsiPhi/v0r5_iteration3_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2017/Bs2JpsiPhi/v0r5_iteration3_biased.json'),
     Parameters.load('output_new/params/angular_acceptance/2018/Bs2JpsiPhi/v0r5_iteration3_biased.json')]


def print_side_by_side(a, b, size=60, space=4):
    while a or b:
        print(a[:size].ljust(size) + " " * space + b[:size])
        a = a[size:]
        b = b[size:]

print_side_by_side(a[1].print(['value']),a[2].print(['value']))

player = [0, 1, 2, 3]
lines = [ [y]+a[i].__str__(['value']).splitlines() for i,y in enumerate([2015,2016,2017,2018]) ]
for l in zip(*lines):
  print(*l, sep=' | ')
a[1].__str__(['value']).splitlines()


for l in zip(*lines):
    print(*l, sep='')



q1 = [check_for_convergence(a[0],b[0]), check_for_convergence(a[1],b[1]), check_for_convergence(a[2],b[2]), check_for_convergence(a[3],b[3])]
all(q1)

q2 = [check_for_convergence(c[0],d[0]), check_for_convergence(c[1],d[1]), check_for_convergence(c[2],d[2]), check_for_convergence(c[3],d[3])]
all(q2)
a[-2:]




#%%
1+1

plt.hist(biased_kinWeight)

biased_kinWeight = np.copy(kinWeight)
unbiased_kinWeight = np.copy(kinWeight)











if os.path.exists(output_weight_file):
  try:
    oldWeight = uproot.open(output_weight_file)['DecayTree'].array('kinWeight')
    kinWeight = np.where(kinWeight!=0,kinWeight,oldWeight).astype(np.float64)
  except:
    kinWeight = np.where(kinWeight!=0,kinWeight,0*kinWeight).astype(np.float64)
else:
  os.makedirs(os.path.dirname(output_weight_file), exist_ok=True)
  kinWeight = np.where(kinWeight!=0,kinWeight,0).astype(np.float64)

print(f"kinWeight: {kinWeight}")


print(f'Saving this kinWeight to {output_weight_file}')
with uproot.recreate(output_weight_file,compression=None) as out_file:
  out_file['DecayTree'] = uproot.newtree({'kinWeight':np.float64})
  out_file['DecayTree'].extend({'kinWeight':kinWeight})
  out_file.close()


#%% Compute angWeights correcting with kinematic weights -------------------------
#     This means compute the kinematic weights using 'X_M','B_P' and 'B_PT'
#     variables
print(f"\n{80*'='}\n",
      "Compute angWeights correcting MC sample in kinematics",
      f"\n{80*'='}\n")

print('Computing angular weights')
ang_acc = bsjpsikk.get_angular_cov(mc.true, mc.reco, mc.weight*ristra.allocate(kinWeight), **mc.params.valuesdict() )
w, uw, cov, corr = ang_acc
mc.w_corrected = Parameters()

for i in range(0,len(w)):
  correl = {f'w{j}':cov[i][j] for j in range(0,len(w)) if i>0 and j>0}
  mc.w_corrected.add({'name': f'w{i}',
                        'value': w[i],
                        'stdev': uw[i],
                        'free': False,
                        'latex': f'w_{i}',
                        'correl': correl
                      })

# Dump the parameters
print('Dumping parameters')
mc.w_corrected.dump(mc.params_path)
# Export parameters in tex tables
print('Saving table of params in tex')
with open(mc.tables_path, "w") as tex_file:
  tex_file.write(
    mc.w_corrected.dump_latex( caption="""
    Kinematically corrected angular weights for \\textbf{%s} \\texttt{\\textbf{%s}} \\textbf{%s}
    category.""" % (YEAR,TRIGGER,MODE.replace('_', ' ') )
    )
  )
tex_file.close()
print(f"Corrected angular weights for {MODE}{YEAR}-{TRIGGER} sample are:")
print(f"{mc.w_corrected}")


exit()



BREAK





#%% Iterative procedure computing angWeights with corrections --------------------
#     dfdfdf Weight MC to match data in the iterative variables namely
#              p and pT of K+ and K-



print('STEP 4: Launching iterative procedure, pdf and kinematic-weighting')
for i in range(1,5):
  for k, v in mc.items():
    # fit MC  ------------------------------------------------------------------
    print('\tFitting %s in %s iteration' % (k,str(i)))
    tparams_pdf = hjson.load(
                    open('angular_acceptance/params/2016/iter/MC_Bs2JpsiPhi_'+str(i-1)+'.json')
                  )

    # do the pdf-weighting -----------------------------------------------------
    print('\tCalculating pdfWeight in %s iteration' % str(i))
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **v['sample'].params.valuesdict())
    original_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **v['sample'].params.valuesdict())
    original_pdf_h /= v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=1, **tparams_pdf)
    target_pdf_h = v['sample'].pdf.get()
    bsjpsikk.diff_cross_rate(v['sample'].true, v['sample'].pdf, use_fk=0, **tparams_pdf)
    target_pdf_h /= v['sample'].pdf.get()
    v[f'pdfWeight{i}'] = np.nan_to_num(target_pdf_h/original_pdf_h)
    print(f"\tpdfWeight{i}:",v[f'pdfWeight{i}'])

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f'\tCalculating p and pT of K+ and K- weight in {i} iteration')
    reweighter.fit(original        = v['sample'].df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   target          = data.df[['hminus_PT','hplus_PT','hminus_P','hplus_P']],
                   original_weight = v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{i}']*v['kinWeight'].get(),
                   target_weight   = data.df.eval(trig_cut+'sw')
                  );
    kkpWeight = reweighter.predict_weights(v['sample'].df[kin_vars])
    v[f'kkpWeight{i}'] = ristra.allocate(np.where(oweight!=0, kkpWeight, 0))
    print(f"\tkkpWeight{i} = {v[f'kkpWeight{i}']}")

    # kinematic-weighting over P and PT of K+ and K- ---------------------------
    print(f"\tAngular weights for {k} category in {i} iteration")
    v[f'w_kkpweighted{i}'] = bsjpsikk.get_angular_weights(
                v['sample'].true,
                v['sample'].reco,
                v['sample'].weight*v['kinWeight']*v[f'kkpWeight{i}'],
                v['sample'].params.valuesdict()
                )
    v[f'w_kkpweighted{i}'] /= v[f'w_kkpweighted{i}'][0]
    print(10*"\t%+.8lf\n" % tuple(v[f'w_kkpweighted{i}']) )



foo = uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')





"""
i = 1
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03782 +- 0.000949792 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03762 +- 0.000926573 (unbinned)
int f4 dcostheta dphi dcospsi = -0.00103561 +- 0.000740913 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000329971 +- 0.000447396 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000272403 +- 0.000457952 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01036 +- 0.000640444 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000336074 +- 0.000567024 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000553363 +- 0.000584319 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00322722 +- 0.00121307 (unbinned)

i = 2
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949104 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03758 +- 0.000926344 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966926 +- 0.00073987 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383365 +- 0.000446895 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456986 (unbinned)
int f7 dcostheta dphi dcospsi = 1.0103 +- 0.000640177 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000283837 +- 0.000566376 (unbinned)
int f9 dcostheta dphi dcospsi = 0.00058618 +- 0.000583348 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00280243 +- 0.00121186 (unbinned)

i = 3
int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03781 +- 0.000949098 (unbinned)
int f3 dcostheta dphi dcospsi = 1.03757 +- 0.000926338 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000966919 +- 0.000739864 (unbinned)
int f5 dcostheta dphi dcospsi = 0.000383704 +- 0.000446893 (unbinned)
int f6 dcostheta dphi dcospsi = 0.0002568 +- 0.000456984 (unbinned)
int f7 dcostheta dphi dcospsi = 1.01029 +- 0.000640172 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000284358 +- 0.000566372 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000586183 +- 0.000583344 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00278764 +- 0.00121186 (unbinned)








v['sample'].df.eval(trig_cut+'polWeight*sw/gb_weights') * v['kinWeight'].get() * v[f'kkpWeight{i}'].get()

v['sample'].weight*v[f'kkpWeight{1}']*v['kinWeight']
*

0.47/2


data = uproot.open('/scratch08/marcos.romero/tuples/mc/new1/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw.root')['DecayTree'].arrays(['sw','gb_weights'])

swg = data[b'sw']/data[b'gb_weights']
pol = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_PolWeight.root')['PolWeight'].array('PolWeight')

pdf_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step0'].array('pdfWeight')
pdf_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')['PdfWeight_Step1'].array('pdfWeight')
kin_0 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step0'].array('kinWeight')
kin_1 = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')['kinWeight_Unbiased_Step1'].array('kinWeight')


kin_0

kin_1

swg*pol*kin_0*kin_1

mc['MC_Bs2JpsiPhi'].df.eval(trig_cut+'polWeight*sw/gb_weights') *
mc['MC_Bs2JpsiPhi']['kinWeight'].get()
v[f'kkpWeight{1}'].get()




swg*pol
mc_std.df.eval('polWeight*(sw/gb_weights)')




























int f1 dcostheta dphi dcospsi = 1 +- 0 (unbinned)
int f2 dcostheta dphi dcospsi = 1.03714 +- 0.000948856 (unbinned)
int f3 dcostheta dphi dcospsi = 1.0369 +- 0.000926164 (unbinned)
int f4 dcostheta dphi dcospsi = -0.000945802 +- 0.000738975 (unbinned)
int f5 dcostheta dphi dcospsi = 0.00035262 +- 0.000447158 (unbinned)
int f6 dcostheta dphi dcospsi = 0.000285595 +- 0.000457371 (unbinned)
int f7 dcostheta dphi dcospsi = 1.00986 +- 0.00063955 (unbinned)
int f8 dcostheta dphi dcospsi = 0.000366054 +- 0.000566205 (unbinned)
int f9 dcostheta dphi dcospsi = 0.000567317 +- 0.000583275 (unbinned)
int f10 dcostheta dphi dcospsi = -0.00220241 +- 0.00121173 (unbinned)

"""





v[f'kkpWeight{1}']


mc_std.df.eval(trig_cut+'polWeight*sw/gb_weights')*v[f'pdfWeight{1}']*v['kinWeight'].get()


tweight


v['kkpWeight1']
v['kinWeight']

simon['MC_Bs2JpsiPhi']['w_kkpweighted1']



simon['MC_Bs2JpsiPhi']['kkpWeight1']



v['sample'].weight*v['kinWeight']*v[f'kkpWeight1']

os.listdir('/scratch08/marcos.romero/SideCar/')

a.keys()



################################################################################
################################################################################
################################################################################



################################################################################
# compare with Simon ###########################################################
################################################################################
simon = {}
for mode in ['MC_Bs2JpsiPhi']:
  d = {}
  for i in range(-1,10):
    # Get angular weights
    f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{i}.root')['fi']
    if i==-1:
      d[f'w_uncorrected'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    elif i==0:
      d[f'w_kinweighted'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    else:
      d[f'w_kkpweighted{i}'] = np.array([f.array(f'f{i}')[0] for i in range(1,11)])
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018_Unbiased.root')[f'kinWeight_Unbiased_Step{i}']
    if i==0:
      d[f'kinWeight'] = f.array('kinWeight')
    elif i>0:
      d[f'kkpWeight{i}'] = f.array('kinWeight')
    # Get kinWeight and kppWeights
    if i >=0:
      f = uproot.open('/scratch08/marcos.romero/SideCar/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{i}"]
      d[f'pdfWeight{i+1}'] = f.array('pdfWeight')
  simon[mode] = d
s = simon['MC_Bs2JpsiPhi']

s['w_uncorrected']

f = uproot.open(f'/scratch08/marcos.romero/Bs2JpsiPhi-Run2/ANALYSIS/analysis/HD-fitter/output/acceptances/unbinned_2016_UnbiasedTrig_AngAcc_BaselineDef_15102018_Iteration{-1}.root')['fi']
mat = np.zeros((10,10))
for j1 in range(1,10):
  for j2 in range(1,10):
    mat[j1,j2] = f.array(f'cf{j1+1}{j2+1}')[0]

mat

f.arrays('cf*')
scale = mc['MC_Bs2JpsiPhi']['unbiased']['weight'].get().sum()
scale
0.8*mc['MC_Bs2JpsiPhi']['unbiased']['cov']/(scale*scale)
uproot.open('/scratch03/marcos.romero/phisRun2/UNTOUCHED_SIMON_SIDECAR/BsJpsiPhi_MC_2016_UpDown_CombDSTMDST_20181101_CombSim09bSim09c_tmva_cut58_sel_sw_AngAcc_BaselineDef_15102018.root')[f"PdfWeight_Step{1}"].array('pdfWeight')
