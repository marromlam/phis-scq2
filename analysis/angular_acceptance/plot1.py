__all__ = []
import matplotlib.pyplot as plt
import ipanema
from utils.plot import get_range, watermark, mode_tex, get_var_in_latex
import argparse
from matplotlib.backends.backend_pdf import PdfPages

from ipanema import plotting, Sample, Parameter, Parameters
from utils.plot import watermark, mode_tex


def argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--original', default = f'/scratch17/marcos.romero/phis_samples/2015/MC_Bs2JpsiPhi/v0r5.root',
                      help='File to correct')
  parser.add_argument('--weights', default = f'/scratch17/marcos.romero/phis_samples/2015/MC_Bs2JpsiPhi/v0r5_angWeight.root',
                      help='File to correct')
  parser.add_argument('--target', default = f'/scratch17/marcos.romero/phis_samples/2015/Bs2JpsiPhi/v0r5.root',
                      help='File to correct')
  parser.add_argument('--year', default = '2015',
                      help='File to reweight to')
  parser.add_argument('--mode', default='MC_Bs2JpsiPhi',
                      help='Name of the target tree')
  parser.add_argument('--version', default='v0r5',
                      help='Name of the target tree')
  parser.add_argument('--branch', default='B_P',
                      help='Name of the target tree')
  parser.add_argument('--weight', default='corrected or iterations of iterative',
                      help='Name of the target tree')
  parser.add_argument('--kkpweighted', default='output/figures/reweightings/2015/MC_Bs2JpsiPhi/v0r5_B_P_yearly_repo_sWeight.pdf',
                      help='File to store the ntuple with weights')
  return parser


def plot_angular_acceptance_reweightings(srd, smc, kkp,strvar, weight):
  niter = 1
  if weight=='kkpWeight':
    niter = len(kkp.find('kkp.*')) # get last iteration number

  rdvar = srd.df.eval(strvar)
  mcvar = smc.df.eval(strvar)
  rdwei = srd.df.eval('sWeight')
  mcwei = smc.df.eval('sWeight/gb_weights*polWeight')
  correction = kkp.df.eval('angWeight/angWeight')
  if weight in ('angWeight' , 'kkpWeight'):
    correction = kkp.df.eval('angWeight')

  for i in range(1, niter+1):
    if niter > 1:
      kkpit = kkp.df.eval(f'pdfWeight{i}*kkpWeight{i}')
    else:
      kkpit = kkp.df.eval(f'angWeight/angWeight')
    fig, axplot, axpull = plotting.axes_plotpull()
    hrd, hmckkp = ipanema.histogram.compare_hist(
                      [rdvar,mcvar], weights=[rdwei,mcwei*correction*kkpit],
                      density=True, range=get_range(strvar)
                  )
    fig, axplot, axpull = ipanema.plotting.axes_plotpull();
    axplot.fill_between(hrd.cmbins,hrd.counts,
                        step="mid",color='k',alpha=0.2,
                        label=f"${mode_tex('Bs2JpsiPhi')}$")
    axplot.fill_between(hmckkp.cmbins,hmckkp.counts,
                        step="mid",facecolor='none',edgecolor='C0',hatch='xxx',
                        label=f"${mode_tex('MC_Bs2JpsiPhi')}$")
    axpull.fill_between(hrd.bins, hmckkp.counts/hrd.counts, 1, color='C0')
    axpull.set_ylabel(f"$\\frac{{N( {mode_tex('MC_Bs2JpsiPhi')} )}}{{N( {mode_tex('Bs2JpsiPhi')} )}}$")
    axpull.set_ylim(0.3,1.9)
    axpull.set_yticks([0.5, 1.0, 1.5])
    axplot.set_ylabel('Weighted candidates')
    if var=='mHH':
      axpull.set_xlabel(rf"${get_var_in_latex(strvar)}$ $[MeV/c^2]$")
    else:
      axpull.set_xlabel(rf"${get_var_in_latex(strvar)}$ $[MeV/c]$")
    axplot.legend()
    watermark(axplot,version=f"${args['version']}$",scale=1.25)
    pdf_pages.savefig(fig)
  pdf_pages.close()
  return pdf_pages


if __name__ == '__main__':
  args = vars( argument_parser().parse_args() )
  srd = ipanema.Sample.from_root(args['target'])
  smc = ipanema.Sample.from_root(args['original'])
  kkp = ipanema.Sample.from_root(args['weights'])
  weight = args['weight']
  print(kkp.df.keys())
  var = args['branch']
  path = args['kkpweighted']
  pdf_pages = PdfPages(path)
  pdf_pages = plot_angular_acceptance_reweightings(srd, smc, kkp, var, weight)
