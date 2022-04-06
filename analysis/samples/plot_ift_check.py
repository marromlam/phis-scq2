__all__ = []
# Plot branches with noweight, sweight and rweight (all)
import ipanema
import uproot
import yaml
import matplotlib.pyplot as plt
import matplotlib
from ipanema import plotting, Sample, Parameter, Parameters
import numpy as np
import argparse
from utils.plot import watermark, mode_tex

PHISSCQ = os.environ['PHISSCQ']
with open(rf"{PHISSCQ}/analysis/samples/branches_latex.yaml") as file:
    BRANCHES = yaml.load(file, Loader=yaml.FullLoader)

def argument_parser():
  p = argparse.ArgumentParser(description='Plot branches of samples')
  p.add_argument('--tuple', help='MC sample')
  p.add_argument('--control', help='RD sample')
  p.add_argument('--year', help='Year of data-taking')
  p.add_argument('--version', help='version of the tuple')
  p.add_argument('--mode', help='MODE')
  p.add_argument('--branch', help='branch which want to plot')
  p.add_argument('--output', help='output')
  return p

if __name__ == '__main__':
  #WARNING: for the moment sw and noweight is the same as we dont have the tuples with weights
  args = vars(argument_parser().parse_args())
  YEARS = args['year'].split(',')
  if YEARS==['run2']:
    YEARS = ['2015', '2016', '2017', '2018']
  input_tuples = args['tuple'].split(',')
  input_control = args['control'].split(',')
  output = args['output'].split(',')
  mode = args['mode']
  branch = args['branch']
  version = args['version']
  w = ['time/time', 'time/time'] #second element =sw once we have the tuples
  for j in range(len(output)):
    asym = []
    fig, axplot, axpull = plotting.axes_plotpull()
    for i, year in enumerate(YEARS):
      #Load tuple and control samples
      tuple = Sample.from_root(input_tuples[i])
      control = Sample.from_root(input_control[i])
      #Histogram
      range = BRANCHES[mode][branch].get('range')
      bins = 30
      ht = ipanema.hist(tuple.df[branch], bins=np.linspace(range[0],range[1],bins),
                    weights=tuple.df.eval(w[j]), density=True)
      hc = ipanema.hist(control.df[branch], bins=ht.edges,
                    weights=control.df.eval(w[j]), density=True)
      axplot.step(ht.bins, ht.counts, where='mid', label=rf'{year} Bs {version}',
            linestyle='-', color=f'C{i}')
      axplot.step(hc.bins, hc.counts, where='mid', label=rf'{year} MC Bs {version}',
          linestyle='-.', color=f'C{i}')
      ht.counts = np.where(ht.counts==0, 1, ht.counts)
      hc.counts = np.where(ht.counts==1, 1, hc.counts)
      print(ht.counts)
      print(hc.counts)
      bins = ht.counts[ht.counts!=1.]
      binsc = hc.counts[hc.counts!=1.]
      print(bins)
      print(binsc)
      asym.append(np.round(binsc/bins,2))
      if branch=='B_SSKaonLatest_TAGDEC':
        if (year=='2018' and YEARS=='run2'):
          bin1 = [item[0] for item in asym]
          r1 = np.round(np.mean(np.array(bin1)), 2)
          bin2 = [item[1] for item in asym]
          r2 = np.round(np.mean(np.array(bin2)), 2)
          bin3 = [item[2] for item in asym]
          r3 = np.round(np.mean(np.array(bin3)),2)
          axplot.text(-1.2, 0.5, f'r={r1}')
          axplot.text(-0.2, 0.5, f'r={r2}')
          axplot.text(0.8, 0.5, f'r={r3}')
      if branch=='B_IFT_InclusiveTagger_TAGDEC':
        if (year=='2018' and YEARS=='run2'):
          bin1 = [item[0] for item in asym]
          r1 = np.round(np.mean(np.array(bin1)), 2)
          bin3 = [item[2] for item in asym]
          r3 = np.round(np.mean(np.array(bin3)),2)
          axplot.text(-1.2, 0.55, f'r={r1}')
          axplot.text(0.8, 0.55, f'r={r3}')
      axpull.fill_between(ht.bins,hc.counts/ht.counts,1,
                        facecolor=f'C{i}', alpha=0.4)
      axplot.set_ylabel(f"Candidates")
      axpull.set_xlabel(BRANCHES[mode][branch].get('latex_name'))
      if mode=='MC_Bs2JpsiPhi':
        axpull.set_ylabel(f"$\\frac{{N(MC-BsJpsiPhi-{version})}}{{N(RD-Bs2JpsiPhi-{version})}}$")
      else:
        axpull.set_ylabel(f"$\\frac{{N(RD-DsPi-{version})}}{{N(RD-DsPi-{version})}}$")
      axpull.set_ylim(0.0,2.0)
      axpull.set_xlim(range[0], range[1])
      axpull.set_yticks([0.5, 1, 1.5])
      axplot.legend(fontsize='small', loc='upper right')
      watermark(axplot, version=f'$v1r0$', scale=1.2)
      fig.savefig(f'{output[j]}')
