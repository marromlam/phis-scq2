# generate_table
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


from argparse import ArgumentParser
from string import Template
import ipanema
import argparse


syst_row_top = [
    "{systname:50}",
    "${fPlon:8}$",
    "${fPper:8}$",
    "${pPlon:8}$",
    "${lPlon:8}$",
    "${dPper:8}$",
    "${dPpar:8}$",
    "${DGsd:8}$",
    "${DGs:8}$",
    "${DM:8}$",
]
syst_row_bottom = [
    "{systname:50}",
    "${dSlon1:8}$",
    "${dSlon2:8}$",
    "${dSlon3:8}$",
    "${dSlon4:8}$",
    "${dSlon5:8}$",
    "${dSlon6:8}$",
    "${fSlon1:8}$",
    "${fSlon2:8}$",
    "${fSlon3:8}$",
    "${fSlon4:8}$",
    "${fSlon5:8}$",
    "${fSlon6:8}$",
]
syst_row_top = " & ".join(syst_row_top) + r"  \\"
syst_row_bottom = " & ".join(syst_row_bottom) + r"  \\"


DESCRIPTION = """
Creates the final systematic table
"""
p = argparse.ArgumentParser(description=DESCRIPTION)
p.add_argument('--input-pars', help='Bs2JpsiPhi MC sample')
p.add_argument('--input-systs', help='Bs2JpsiPhi MC sample')
p.add_argument('--output-table', help='Bs2JpsiPhi MC sample')
args = vars(p.parse_args())


# load parameters
pars = ipanema.Parameters.load(args['input_pars'])
_systs = args['input_systs'].split(',')
_systs = [ipanema.Parameters.load(s) for s in _systs]

systematics_list = {
    "massFactorization": "Mass factorization",
    "angaccStat": "Angular acceptance statistical",
    # "angaccGBconf",
}

systs = {}
for i, name in enumerate(systematics_list.keys()):
  systs[name] = _systs[i]

_values = {k: f"{v.uvalue:.2uL}".split(r'\pm')[0] for k, v in pars.items()}
_errors = {k: f"{v.uvalue:.2uL}".split(r'\pm')[1] for k, v in pars.items()}
_values.update({'systname': "Central value"})
_errors.update({'systname': "Stat. error"})

top_table, bottom_table = [], []
top_table.append(syst_row_top.format(**_values))
top_table.append(r"\hline")
top_table.append(syst_row_top.format(**_errors))
top_table.append(r"\hline")
bottom_table.append(syst_row_bottom.format(**_values))
bottom_table.append(r"\hline")
bottom_table.append(syst_row_bottom.format(**_errors))
bottom_table.append(r"\hline")


for systname, systpars in systs.items():
  _syst = {k: v.casket[list(v.casket.keys())[0]] for k, v in systpars.items()}
  _syst = {k: f"{abs(v):.4f}".replace("0.0000", "-") for k, v in _syst.items()}
  _syst.update({'systname': systematics_list[systname]})
  top_table.append(syst_row_top.format(**_syst))
  bottom_table.append(syst_row_bottom.format(**_syst))


table_template = Template(open('analysis/systematics/table.tex').read())
with open(args['output_table'], "w") as latex_table:
  latex_table.write(table_template.substitute(dict(
      PLACEHOLDER_TOP_TABLE="\n".join(top_table),
      PLACEHOLDER_BOTTOM_TABLE="\n".join(bottom_table)
  )))


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
