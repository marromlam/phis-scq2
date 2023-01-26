__all__ = []
__author__ = ['Marcos Romero Lamas']
__email__ = ['mromerol@cern.ch']


# Modules {{{

import argparse
import numpy as np

from ipanema import initialize
from ipanema import Parameters, optimize, Sample

from utils.strings import cuts_and, printsec, printsubsec
from utils.helpers import swnorm
from analysis.time_acceptance.simultaneousBdasBs import wildcards_parser
from utils.helpers import trigger_scissors

import config
resolutions = config.timeacc['constants']
bdtconfig = config.timeacc['bdtconfig']
Gdvalue = config.general['Gd']

# }}}


# command line interface {{{

if __name__ == '__main__':

  # Parse arguments {{{

  DESCRIPTION = """
  Computes the lifetime of half a Bd RD sample using spline coefficients 
  taken from the other halve. Runs over YEARS variable tuples.
  """
  p = argparse.ArgumentParser(description='Compute decay-time acceptance.')
  p.add_argument('--samples', help='Bs2JpsiPhi MC sample')
  p.add_argument('--resolutions', help='Bs2JpsiPhi MC sample')
  p.add_argument('--biased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--unbiased-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-params', help='Bs2JpsiPhi MC sample')
  p.add_argument('--output-tables', help='Bs2JpsiPhi MC sample')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--timeacc', help='Different flag to ... ')
  p.add_argument('--trigger', help='Different flag to ... ')
  p.add_argument('--minimizer', default='minuit', help='Different flag to ... ')
  args = vars(p.parse_args())

  this_minimizer = args['minimizer']

  this_conf = wildcards_parser(version=args['version'], year=args['year'],
                               trigger=args['trigger'], timeacc=args['timeacc'],
                               # mode=args['mode']
                               )

  # }}}


  # Get badjanak model and configure it
  initialize(config.user['backend'], 1)
  import analysis.time_acceptance.fcn_functions as fcns


  # Prepare the cuts
  CUT = this_conf['trigger']['cut']
  tLL = this_conf['timeacc']['time_lower_limit']
  tUL = this_conf['timeacc']['time_upper_limit']
  CUT = cuts_and(CUT, f'time>={tLL} & time<={tUL}')

  # set how to split data
  splitter = '(evtN%2)==0' # this is Bd as Bs
  if this_conf['timeacc']['cuts'] == 'mKstar':
    splitter = cuts_and(splitter, "mHH>890")
    splitter = "mHH>890"
  elif this_conf['timeacc']['cuts'] == 'alpha':
    splitter = "alpha<0.025"
  elif this_conf['timeacc']['cuts'] == 'deltat':
    splitter = cuts_and(splitter, "sigmat<0.04")

  # final arrangemets
  samples = args['samples'].split(',')
  # oparams = args['params'].split(',')
  # oparams = [Parameters.load(i) for i in oparams]
  time_offset = args['resolutions'].split(',')
  time_offset = [Parameters.load(i) for i in time_offset]


  # Get data into categories {{{

  printsec("Loading categories")

  cats = {}
  for i, m in enumerate(this_conf['year']['categories']):
    cats[m] = {}
    for t in this_conf['trigger']['categories']:
      cats[m][t] = Sample.from_root(samples[i], share=this_conf['version']['share'])
      tmp = Sample.from_root(samples[i].replace('.root', '_sWeight.root'), share=this_conf['version']['share']).df
      angle_str = "arccos((hplus_PX*hminus_PX+hplus_PY*hminus_PY+hplus_PZ*hminus_PZ)/(hplus_P*hminus_P))"
      angle_arr = tmp.eval(angle_str)
      cats[m][t].df['alpha'] = angle_arr
      cats[m][t].name = f"Bd2JpsiJKstar-{m}-{t}"

      cats[m][t].chop(cuts_and(CUT, f'({splitter}) == 1', trigger_scissors(t)))

      # allocate arrays
      print(cats[m][t])
      cats[m][t].allocate(time='time', lkhd='0*time')
      cats[m][t].allocate(weight='sWeight')
      cats[m][t].weight = swnorm(cats[m][t].weight)

      # Add coeffs parameters
      c = Parameters.load(args[f'{t}_params'].split(',')[i])
      knots = Parameters.build(c, c.find('k.*'))
      cats[m][t].params = Parameters.build(c,c.find('c.*')+['mu_Bc','sigma_Bc'])

      # Update kernel with the corresponding knots
      fcns.badjanak.config['knots'] = np.array(knots).tolist()

  # recompile kernel (just in case)
  fcns.badjanak.get_kernels()

  # }}}


  # Time to fit lifetime {{{

  printsec(f"Minimization procedure")

  # create a common gamma parameter for biased and unbiased
  lfpars = Parameters()
  lfpars.add(dict(name='gamma', value=0.6, min=0.0, max=1.0, latex=r"\Gamma_d"))

  # join and print parameters before the lifetime fit
  for y in cats:
    for t in cats[y]:
      for p, par in cats[y][t].params.items():
        if p[0] == 'c':
          lfpars.add({"name": f"{p.replace('cB', 'cA')}_{y[2:]}",
                      "value": par.value, "stdev": par.stdev,
                      "latex": f"{par.latex.replace('cB', 'cA')}{{}}^{y[2:]}",
                      "min": par.min, "max": par.max, "free": par.free
                    })
        else:
          lfpars.add({"name": p.replace('Bc', 'Ac')})
          lfpars[p.replace('Bc', 'Ac')] = par
  print("Parameters before fit")
  #lfpars.lock(); lfpars.unlock('gamma')
  print(lfpars)

  # lifetime fit
  printsubsec("Simultaneous minimization procedure")

  if this_minimizer.lower() in ("minuit", "minos"):
    lifefit = optimize(fcn_call=fcns.splinexerfconstr,
                       params=lfpars,
                       fcn_kwgs={'cats':cats, 'weight':True},
                       method=this_minimizer,
                       verbose=False, timeit=True, strategy=1, tol=0.05);
  else:
    lifefit = optimize(fcn_call=fcns.splinexerfconstr,
                       params=lfpars,
                       fcn_kwgs={'cats':cats, 'weight':True},
                       method=this_minimizer,
                       verbose=False, timeit=True)
  print(lifefit)
  printsubsec("Lifetime estimation")
  print(f"\\tau(B_d^0) = {1/lifefit.params['gamma'].uvalue:.2uL}")

  # }}}


  # Writing results {{{

  printsubsec("Dumping parameters")
  print(f"Dumping json parameters to {args['output_params']}")
  lifefit.params.dump(args['output_params'])

  # }}}


# }}}


# vim: fdm=marker
