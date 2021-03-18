DESCRIPTION = """
    This file contains 3 fcn functions to be minimized under ipanema3 framework
    those functions are, actually functions of badjanak kernels.
"""

__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']



################################################################################
# Modules ######################################################################

import argparse
import os
import hjson

# load ipanema
from ipanema import initialize, plotting
from ipanema import ristra, Parameters, Sample, plot_conf2d, Optimizer

# import some phis-scq utils
from utils.plot import mode_tex
from utils.strings import cuts_and, printsec
from utils.helpers import version_guesser, timeacc_guesser
from utils.helpers import swnorm, trigger_scissors

# binned variables
bin_vars = hjson.load(open('config.json'))['binned_variables_cuts']
resolutions = hjson.load(open('config.json'))['time_acceptance_resolutions']
all_knots = hjson.load(open('config.json'))['time_acceptance_knots']
Gdvalue = hjson.load(open('config.json'))['Gd_value']
tLL = hjson.load(open('config.json'))['tLL']
tUL = hjson.load(open('config.json'))['tUL']

# Parse arguments for this script
def argument_parser():
  p = argparse.ArgumentParser(description=DESCRIPTION)
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit') 
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--contour', help='Different flag to ... ')
  return p

if __name__ != '__main__':
  import badjanak

################################################################################



################################################################################
# Run and get the job done #####################################################
if __name__ == '__main__':

  # Parse arguments ------------------------------------------------------------
  args = vars(argument_parser().parse_args())
  VERSION, SHARE, MAG, FULLCUT, VAR, BIN = version_guesser(args['version'])
  YEAR = args['year']
  TRIGGER = args['trigger']
  MODE = 'Bs2JpsiPhi'
  TIMEACC, NKNOTS, CORR, FLAT, LIFECUT, MINER = timeacc_guesser(args['timeacc'])

  # Get badjanak model and configure it
  initialize(os.environ['IPANEMA_BACKEND'], 1 if YEAR in (2015,2017) else 1)
  import time_acceptance.fcn_functions as fcns

  # Prepare the cuts
  CUT = bin_vars[VAR][BIN] if FULLCUT else ''   # place cut attending to version
  CUT = trigger_scissors(TRIGGER, CUT)          # place cut attending to trigger
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # Print settings
  printsec("Settings")
  print(f"{'backend':>15}: {os.environ['IPANEMA_BACKEND']:50}")
  print(f"{'trigger':>15}: {TRIGGER:50}")
  print(f"{'cuts':>15}: {CUT:50}")
  print(f"{'timeacc':>15}: {TIMEACC:50}")
  print(f"{'minimizer':>15}: {MINER:50}")
  print(f"{'contour':>15}: {args['contour']:50}\n")

  # List samples, params and tables
  samples = args['samples'].split(',')
  oparams = args['params'].split(',')
  otables = args['tables'].split(',')

  # Check timeacc flag to set knots and weights and place the final cut
  knots = all_knots[str(NKNOTS)]
  kinWeight = f'kinWeight_{VAR}*' if VAR else 'kinWeight*'
  sw = f'sw_{VAR}' if VAR else 'sw'



  # Get data into categories ---------------------------------------------------
  printsec(f"Loading categories")

  cats = {}
  for i,m in enumerate(['MC_Bs2JpsiPhi_dG0','MC_Bd2JpsiKstar','Bd2JpsiKstar']):
    # Correctly apply weight and name for diffent samples
    if (m=='MC_Bs2JpsiPhi_dG0') and not ('MC_Bs2JpsiPhi_dG0' in samples[i]):
      if CORR:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      else:
        weight = f'dg0Weight*{sw}/gb_weights'
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bs2JpsiPhi_dG0':
      if CORR:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}/gb_weights'
      else:
        weight = f'{sw}/gb_weights'
      mode = 'BsMC'; c = 'a'
    elif m=='MC_Bd2JpsiKstar':
      if CORR:
        weight = f'{kinWeight}polWeight*pdfWeight*{sw}'
      else:
        weight = f'{sw}'
      mode = 'BdMC'; c = 'b'
    elif m=='Bd2JpsiKstar':
      if CORR:
        weight = f'{kinWeight}{sw}'
      else:
        weight = f'{sw}'
      mode = 'BdRD'; c = 'c'
    print(weight)

    # Load the sample
    cats[mode] = Sample.from_root(samples[i], cuts=CUT, share=SHARE, name=mode)
    cats[mode].allocate(time='time',lkhd='0*time')
    cats[mode].allocate(weight=weight)
    cats[mode].weight = swnorm(cats[mode].weight)

    # Add knots
    cats[mode].knots = Parameters()
    cats[mode].knots.add(*[
                {'name':f'k{j}', 'value':v, 'latex':f'k_{j}', 'free':False}
                 for j,v in enumerate(knots[:-1])
               ])
    cats[mode].knots.add({'name':f'tLL', 'value':tLL,
                          'latex':'t_{ll}', 'free':False})
    cats[mode].knots.add({'name':f'tUL', 'value':tUL,
                          'latex':'t_{ul}', 'free':False})

    # Add coeffs parameters
    cats[mode].params = Parameters()
    cats[mode].params.add(*[
                    {'name':f'{c}{j}{TRIGGER[0]}', 'value':1.0,
                     'latex':f'{c}_{j}^{TRIGGER[0]}',
                     'free':False if j==0 else True, #'min':0.10, 'max':5.0
                    } for j in range(len(knots[:-1])+2)
    ])
    cats[mode].params.add({'name':f'gamma_{c}',
                           'value':Gdvalue+resolutions[m]['DGsd'],
                           'latex':f'\Gamma_{c}', 'free':False})
    cats[mode].params.add({'name':f'mu_{c}',
                           'value':resolutions[m]['mu'],
                           'latex':f'\mu_{c}', 'free':False})
    cats[mode].params.add({'name':f'sigma_{c}',
                           'value':resolutions[m]['sigma'],
                           'latex':f'\sigma_{c}', 'free':False})
    print(cats[mode].knots)
    print(cats[mode].params)

    # Attach labels and paths
    cats[mode].label = mode_tex(mode)
    cats[mode].pars_path = oparams[i]
    cats[mode].tabs_path = otables[i]



  # Configure kernel -----------------------------------------------------------
  fcns.badjanak.config['knots'] = knots[:-1]
  fcns.badjanak.get_kernels(True)



  # Time to fit ----------------------------------------------------------------
  printsec(f"Simultaneous minimization procedure")
  fcn_call = fcns.saxsbxscxerf
  fcn_pars = cats['BsMC'].params+cats['BdMC'].params+cats['BdRD'].params
  fcn_kwgs={
    'data': [cats['BsMC'].time, cats['BdMC'].time, cats['BdRD'].time],
    'prob': [cats['BsMC'].lkhd, cats['BdMC'].lkhd, cats['BdRD'].lkhd],
    'weight': [cats['BsMC'].weight, cats['BdMC'].weight, cats['BdRD'].weight],
    'flatend': FLAT
  }
  mini = Optimizer(fcn_call=fcn_call, params=fcn_pars, fcn_kwgs=fcn_kwgs)

  if MINER.lower() in ("minuit","minos"):
    result = mini.optimize(method='minuit', verbose=False, tol=0.1);
  elif MINER.lower() in ('bfgs', 'lbfgsb'):
    _res = mini.optimize(method='nelder', verbose=False);
    result = mini.optimize(method=MINER, params=_res.params, verbose=False);
  elif MINER.lower() in ('nelder'):
    result = mini.optimize(method='nelder', verbose=False)
  elif MINER.lower() in ('emcee'):
    _res = mini.optimize(method='minuit', verbose=False, tol=0.05)
    result = mini.optimize(method='emcee', verbose=False, params=_res.params,
                           steps=1000, nwalkers=100, behavior='chi2')
  print(result)

  # Do contours or scans if asked ----------------------------------------------
  if args['contour'] != "0":
    if len(args['contour'].split('vs')) > 1:
      fig, ax = plot_conf2d(
          mini, result, args['contour'].split('vs'), size=(50, 50))
      fig.savefig(cats[mode].tabs_path.replace('tables', 'figures').replace(
          '.tex', f"_scan{args['contour']}.pdf"))
    else:
      import matplotlib.pyplot as plt
      # x, y = result._minuit.profile(args['contour'], bins=100, bound=5, subtract_min=True)
      # fig, ax = plotting.axes_plot()
      # ax.plot(x,y,'-')
      # ax.set_xlabel(f"${result.params[ args['contour'] ].latex}$")
      # ax.set_ylabel(r"$L-L_{\mathrm{opt}}$")
      # fig.savefig(cats[mode].tabs_path.replace('tables', 'figures').replace('.tex', f"_contour{args['contour']}.pdf"))
      result._minuit.draw_mnprofile(
          args['contour'], bins=20, bound=3, subtract_min=True, band=True, text=True)
      plt.savefig(cats[mode].tabs_path.replace('tables', 'figures').replace('.tex', f"_contour{args['contour']}.pdf"))



  # Writing results ------------------------------------------------------------
  printsec(f"Dumping parameters")

  for name, cat in zip(cats.keys(),cats.values()):
    list_params = cat.params.find('(a|b|c)(\d{1})(u|b)')
    print(list_params)
    cat.params.add(*[result.params.get(par) for par in list_params])

    print(f"Dumping tex table to {cats[name].tabs_path}")
    with open(cat.tabs_path, "w") as text:
      text.write(cat.params.dump_latex(caption=f"Time acceptance for the $\
      {mode_tex(f'{MODE}')}$ ${YEAR}$ {TRIGGER} category in simultaneous fit."))
    text.close()

    print(f"Dumping json parameters to {cats[name].pars_path}")
    cat.params = cat.knots + cat.params
    cat.params.dump(cats[name].pars_path)

################################################################################
# that's all folks!
