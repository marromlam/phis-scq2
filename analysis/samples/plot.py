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
  args = vars(argument_parser().parse_args())
  YEARS = args['year'].split(',')
  input_tuples = args['tuple'].split(',')
  input_control = args['control'].split(',')
  output = args['output'].split(',')
  mode = args['mode']
  branch = args['branch']
  version = args['version']
  comp_version = False
  #w = weights for tuples, wc=weights for control channels
  w = ['time/time', 'sw', 'sw']
  if (mode=='MC_Bs2JpsiPhi') or (mode=='MC_Bs2JpsiPhi_dG0'):
      wc = ['time/time', 'sw/gb_weights', 'sw*polWeight*pdfWeight*kinWeight/gb_weights'] #Control MC Bs (both)
  elif (mode=='Bd2JpsiKstar') or (mode=='Bs2JpsiPhi'):
      wc = ['time/time', 'sw', 'sw*kinWeight'] #control Bd RD
  else:
      wc = ['time/time', 'sw', 'sw*polWeight*pdfWeight*kinWeight'] #Control MC Bd
  if input_control[0].endswith('v0r0.root'): #comparacion v0r5 vs v0r0
      w = wc
      comp_version = True
  for j in range(len(output)):
    fig, axplot, axpull = plotting.axes_plotpull()
    for i, year in enumerate(YEARS):
      #Load tuple and control samples
      tuple = Sample.from_root(input_tuples[i])
      control = Sample.from_root(input_control[i])
      #histogram
      range = BRANCHES[mode][branch].get('range')
      ht = ipanema.hist(tuple.df[branch], bins=np.linspace(range[0],range[1],70),
                    weights=tuple.df.eval(w[j]), density=True)
      hc = ipanema.hist(control.df[branch], bins=ht.edges,
                    weights=control.df.eval(wc[j]), density=True)
      if comp_version==True:
          if mode=='MC_Bd2JpsiKstar':
            axplot.step(ht.bins, ht.counts, where='mid', label=f'{year} MC Bd {version}',
                      linestyle='-', color=f'C{i}')
            axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} MC Bd v0r0',
                    linestyle='-.', color=f'C{i+3}')
          else:
            axplot.step(ht.bins, ht.counts, where='mid', label=f'{year} RD Bd {version}',
                      linestyle='-', color=f'C{i}')
            axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} RD Bd v0r0',
                    linestyle='-.', color=f'C{i+3}')
      else:
        if (mode=='Bd2JpsiKstar') or (mode=='Bs2JpsiPhi'):
          axplot.step(ht.bins, ht.counts, where='mid', label=f'{year} ${mode_tex("Bs2JpsiPhi")}$',
                    linestyle='-', color=f'C{i}')
          axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} ${mode_tex("Bd2JpsiKstar")}$',
                    linestyle='-.', color=f'C{i}')
        else:
          axplot.step(ht.bins, ht.counts, where='mid', label=f'{year} data',
                    linestyle='-', color=f'C{i}')
          axplot.step(hc.bins, hc.counts, where='mid', label=f'{year} MC',
                  linestyle='-.', color=f'C{i}')
      axpull.fill_between(ht.bins,hc.counts/ht.counts,1,
                        facecolor=f'C{i+3}', alpha=0.4)
    axplot.set_ylabel(f"Candidates")
    axpull.set_xlabel(BRANCHES[mode][branch].get('latex_name'))
    if comp_version==True:
      if mode=='MC_Bd2JpsiKstar':
        axpull.set_ylabel(f"$\\frac{{N(MC-Bd-v0r0)}}{{N(MC-Bd-v0r5)}}$")
      else:
        axpull.set_ylabel(f"$\\frac{{N(RD-Bd-v0r0)}}{{N(RD-Bd-v0r5)}}$")
    else:
      if (mode=='Bd2JpsiKstar') or (mode=='Bs2JpsiPhi'):
          axpull.set_ylabel(f"$\\frac{{N({mode_tex('Bd2JpsiKstar')})}}{{N({mode_tex('Bs2JpsiPhi')}}}$")
      else:
          axpull.set_ylabel(f"$\\frac{{N(MC)}}{{N(data)}}$")
    axpull.set_ylim(0.0,2.0)
    axpull.set_xlim(range[0], range[1])
    axpull.set_yticks([0.5, 1, 1.5])
    if branch[0:3] == 'log':
        axplot.set_yscale('log')
        axplot.legend(loc='center left', fontsize='small')
        watermark(axplot, version=f'$v0r5$', scale=100.0)
    else:
        axplot.legend(fontsize='small')
        watermark(axplot, version=f'$v0r5$', scale=1.2)
    fig.savefig(f'{output[j]}')
