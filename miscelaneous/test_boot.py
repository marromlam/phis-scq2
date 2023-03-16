# test_boot
#
#

__all__ = []
__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@CERN.CH"]



import numpy as np
import pandas as pd
import ipanema
import matplotlib.pyplot as plt
from scipy.stats import norm
from ROOT import TH1F

#For the moment the closest this i got in comparison with ROOT
def RMS(values):
  mean = np.mean(values)
  variance = np.mean((values-mean)**2)
  rms = np.sqrt(variance)
  rmserr = np.sqrt(variance/(2*(len(values))))
  return (rms, rmserr)

n_pseudo = 100


_nominal = "output/params/physics_params/run2/Bs2JpsiPhi/v5r1@LcosK_run2_run2Dual_vgc_peilian_simul3_amsrd_combined.json"
nominal = ipanema.Parameters.load(_nominal)
paths = [ f"output/params/physics_params/run2/Bs2JpsiPhi/v5r1@LcosKboot{i}_run2_run2Dual_vgc_peilian_simul3_amsrd_combined.json" for i in range(n_pseudo)]

_names = list(nominal[key].name for key in nominal.keys() if nominal[key].free==True)
pars = {}
for _n in _names:
  pars[_n] = []


for i in range(0, n_pseudo):
  _pars = ipanema.Parameters.load(paths[i])
  for _n in _names:
    pars[_n].append(_pars[_n].uvalue)


_names = ['fPlon', 'fPper', 
                'dPpar', 'dPper',
                'pPlon',
                'lPlon',
                'DGsd', 'DGs', 'DM',
                'fSlon1', 'fSlon2', 'fSlon3', 'fSlon4', 'fSlon5', 'fSlon6', 
                'dSlon1', 'dSlon2', 'dSlon3', 'dSlon4', 'dSlon5', 'dSlon6', 
]

latex = {
	          "fPlon" : "F_0",
	          "fPper" : "F_{\perp}",
		        "dPpar": "\delta_{\parallel} - \delta_0",
		        "dPper": "\delta_{\perp} - \delta_0",
	          "pPlon" : "\phi_s",
		        "lPlon": "|\lambda|",
	          "DGsd" : "\Gamma_s - \Gamma_d",
	          "DGs" : "\Delta \Gamma_s",
		        "DM"  : "\Delta M",
	          "fSlon1" : "FS_{1}",
	          "fSlon2" : "FS_{2}",
	          "fSlon3" : "FS_{3}",
	          "fSlon4" : "FS_{4}",
	          "fSlon5" : "FS_{5}",
	          "fSlon6" : "FS_{6}",
		        "dSlon1": "\delta_{S1} - \delta_0",
		        "dSlon2": "\delta_{S2} - \delta_0",
		        "dSlon3": "\delta_{S3} - \delta_0",
		        "dSlon4": "\delta_{S4} - \delta_0",
		        "dSlon5": "\delta_{S5} - \delta_0",
		        "dSlon6": "\delta_{S6} - \delta_0",
}

table = []
table.append(r"\begin{tabular}{l|c|c}")
_table = [
    f"{'Parameter':>50}",
    f"{'Fit':>20}",
    f"{'Bootstrapping':>30}"
  ]
table.append( " & ".join(_table) + r" \\")
table.append("\hline")

for n in _names:
  print(f"{n}")

	#Plot distribution
  plt.close()
  _vals = np.array([v.n for v in pars[n]])
  mean, std = norm.fit(_vals)
  rms, rms_err = RMS(_vals)
  # h = TH1F("h", "hist", 10, min(_vals), max(_vals))
  h = TH1F("h", "hist", 10, min(_vals), max(_vals))
  for entry in _vals:
    h.Fill(entry)
  rms_root = h.GetRMS()
  rms_error_root = h.GetRMSError()
  
  print(f"sigma fit {std}") 
  print(f"Nominal {nominal[n].stdev}")
  print(f"Monchito {rms}, {rms_err}")
  print(f"ROOT: {rms_root}, {rms_error_root}")

	#Plot distributions:
  fig, ax = plt.subplots(figsize=(8,6))
  ax.hist(_vals, bins=15, density = True)
  plt.xlabel(rf"${latex[n]}$")
  plt.ylabel(f"Counts")

  text = rf"Fit: {np.round(nominal[n].value,4)} $\pm$ {np.round(nominal[n].stdev,4)}"+f"\nMean: {np.round(np.mean(_vals),4)} \n RMS: {rms:.4f} \n RMS err: {rms_err:.4f}"
  props = dict(boxstyle='round', facecolor='white', alpha=0.5)
  ax.text(0.95, 0.95, text, transform=ax.transAxes,
        ha='right', va='top', bbox=props, fontsize=14)
	
	# Add some labels and a title
  ax.set_xlabel('Values', fontsize=13)
  ax.set_ylabel('Frequency', fontsize=13)
  ax.set_title(rf"${latex[n]}$", fontsize=15)

  # Add some gridlines
  ax.grid(alpha=0.3)
  plt.savefig(f"boot_plots/distro/{n}.pdf")

	#Generate table and printed it out!
  _table = [
    f"{'Parameter':>50}",
    f"{'Fit':>20}",
    f"{'Bootstrap.':>30}"]
  _table = [
        rf"${nominal[n].latex:>50}$",
		    rf"${np.round(nominal[n].stdev,4):>20}$",
        # rf"{np.round(rms_root, 5)} $\pm$ {np.round(rms_error_root, 5)}",
        rf"{np.round(rms, 5)} $\pm$ {np.round(rms_err, 5)}",
      ]

  table.append( " & ".join(_table) + r" \\")

table.append(r"\end{tabular}")
print("\n".join(table))







#





# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
