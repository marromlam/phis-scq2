import matplotlib.pyplot as plt
import ipanema
from utils.plot import get_range, watermark, mode_tex, get_var_in_latex
import argparse



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
  parser.add_argument('--treename', default='yearly_repo',
                      help='Name of the target tree')
  parser.add_argument('--kkpweighted', default='output/figures/reweightings/2015/MC_Bs2JpsiPhi/v0r5_B_P_yearly_repo_sWeight.pdf',
                      help='File to store the ntuple with weights')
  return parser


def plot_angular_acceptance_reweightings(srd, smc, kin, kkp, strvar):
  niter = len(kkp.find('kkp.*')) # get last iteration number
  print(niter)
  rdvar = srd.df.eval(strvar)
  mcvar = smc.df.eval(strvar)
  rdwei = srd.df.eval('sWeight')
  try:
    mckin = smc.df.eval('sWeight*polWeight')*kin.df.eval('kinWeight')
  except:
    mckin = smc.df.eval('sWeight*polWeight')
  if niter > 0:
    mckkp = mckin*kkp.df.eval(f'pdfWeight{niter}*kkpWeight{niter}')
  else:
    mckkp = mckin
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
  axplot.legend()
  return fig, axplot, axpull


if __name__ == '__main__':
  args = vars( argument_parser().parse_args() )
  srd = ipanema.Sample.from_root(args['target'])
  smc = ipanema.Sample.from_root(args['original'])
  kin = ipanema.Sample.from_root(args['weights'])
  kkp = ipanema.Sample.from_root(args['weights'],treename=args['treename'])
  var = args['branch']
  path = args['kkpweighted']
  if 'kkpWeight' in path:
    fig, axplot, axpull = plot_angular_acceptance_reweightings(srd, smc, kin, kkp, var)
  elif 'kinWeight' in path:
    fig, axplot, axpull = plot_angular_acceptance_reweightings(srd, smc, kin, kin, var)
  else: #sWeight
    fig, axplot, axpull = plot_angular_acceptance_reweightings(srd, smc, smc, smc, var)
  fig.show()
  watermark(axplot,version=f"${args['version']}$",scale=1.25)
  fig.savefig(path)
