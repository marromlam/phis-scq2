from ipanema import (Parameters, ristra, plotting)
import argparse
import numpy as np

DOCAz_BINS = 8
# 10
docaz = [0.000e+00, 8.700e-03, 1.770e-02, 2.730e-02, 3.800e-02, 5.050e-02,
         6.660e-02, 8.990e-02, 1.317e-01, 2.402e-01, 1.000e+01]
# 5
docaz = [ 0.,     0.0291, 0.0651, 0.1265, 0.3032,10.    ]
# 8
docaz = [ 0.0, 0.0179, 0.0372, 0.0598, 0.0903, 0.1386, 0.2359, 0.4714, 5.0 ]

docaz = np.array(docaz)

if __name__ == '__main__':
  # Parse command line arguments ----------------------------------------------
  p = argparse.ArgumentParser(description='Get efficiency in DOCAz bin.')
  p.add_argument('--params', help='Mass fit parameters')
  p.add_argument('--params-match', help='Mass fit parameters VELO matching')
  p.add_argument('--plot', help='Plot of the mass fit')
  p.add_argument('--plot-log', help='Plot of the log mass fit')
  p.add_argument('--year', help='Year to fit')
  p.add_argument('--mode', help='Year to fit', default='Bd2JpsiKstar')
  p.add_argument('--version', help='Version of the tuples to use')
  p.add_argument('--trigger', help='Trigger to fit')
  p.add_argument('--mass-model', help='Different flag to ... ')
  args = vars(p.parse_args())


  # 
  TRIGGER = args["trigger"]
  MODEL = args["mass_model"]

  # Create lists for all parameters
  velo_unmatch = {k: Parameters.load(p) 
                  for k, p in enumerate(args['params'].split(','))}
  velo_match = {k: Parameters.load(p)
                for k, p in enumerate(args['params_match'].split(','))}

  # We compute the efficiency doca bins
  # eff = {k: velo_match[k]['nsig'].uvalue/velo_unmatch[k]['nsig'].uvalue
  #        for k in velo_match.keys()}
  # print("The efficiency is: ")
  # for db in range(0,len(docaz)-1):
  #   print(f"[{docaz[db]},{docaz[db+1]}) : {velo_match[db]['nsig'].uvalue} / {velo_unmatch[db]['nsig'].uvalue} = {eff[db]}")

  eff = {k: velo_match[k]['nsig'].uvalue/(velo_match[k]['nsig'].uvalue+velo_unmatch[k]['nsig'].uvalue)
         for k in velo_match.keys()}
  print("The efficiency a/a+b is: ")
  for db in range(0,len(docaz)-1):
    print(f"[{docaz[db]},{docaz[db+1]}) : {velo_match[db]['nsig'].uvalue} / {velo_unmatch[db]['nsig'].uvalue} = {eff[db]}")

  # %% plot eff vs. docaz -----------------------------------------------------
  fig, ax = plotting.axes_plot()
  x = 0.5*(docaz[1:]+docaz[:-1])
  y = np.array([eff[i].n for i in eff.keys()])
  uy = np.array([eff[i].s for i in eff.keys()])
  ax.errorbar(x, y, yerr=[uy, uy], xerr=[x-docaz[:-1], docaz[1:]-x], fmt='.',
              label="+".join(MODEL.split('_')))
  #ax.set_xlim(0, max(docaz))
  ax.set_ylim(0.5, 1.25)
  #ax.set_ylim(0.4, 0.6)
  ax.set_xlabel("DOCAz [mm]")
  ax.set_ylabel("Efficiency")
  ax.legend()
  fig.savefig(args["plot"])
  ax.set_xscale('log')
  fig.savefig(args["plot_log"])
