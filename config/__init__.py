import hjson
import os

# config folder path
__PATH = os.path.dirname(os.path.abspath(__file__))

# years is a dict translating years or groups of years to a list of years
years = hjson.load(open(f"{__PATH}/years.json", 'r'))

# timeacc contains the configuration for time acceptance used in the whole 
# pipeline
timeacc = hjson.load(open(f"{__PATH}/timeacc.json", 'r'))

# angacc contains the configuration for the angular acceptance used in the whole
# pipeline
angacc = hjson.load(open(f"{__PATH}/angacc.json", 'r'))

# User {{{

#  path : Sidecar is the folder where all tuples will be placed, and 
#         from rules are fed from.
#  
#  Sidecar path --------------------------------------------------------------
#     Sidecar is the folder where all tuples will be placed, and from rules
#     are fed from.
#  
#  EOS path ------------------------------------------------------------------
#     This is the EOS folder where all tuples for the Run2 phis analysis are
#     placed.
#  
#  Mail list -----------------------------------------------------------------
#     If a rule runs succesfully, its log will be forwarded to all the emails
#     in the following list (by default it is empty).

user = hjson.load(open(f"{__PATH}/user.json", 'r'))
user['backend'] = os.environ['IPANEMA_BACKEND'] if os.environ["IPANEMA_BACKEND"] else user['backend']

# }}}

base = hjson.load(open(f"{__PATH}/base.json", 'r'))
blinding = hjson.load(open(f"{__PATH}/blinding.json", 'r'))
general = hjson.load(open(f"{__PATH}/globals.json", 'r'))

"""

  // Gd value ------------------------------------------------------------------
  //    This value is shared by all steps of this analysis, so a change here
  //    would imply a change in the whole set of results. By default this value
  //    is set to the World Average
  "Gd_value": 0.65789, //1/ps
  //
  // Toys ----------------------------------------------------------------------
  //    Number of toys to generate
  "ntoys": 4000,
  //
  // Cuts in decay-time --------------------------------------------------------
  //    When doing computations, some cuts for decay-time are applied, both to
  //    low decay-times (removing prompt background) and to higher decay-times
  //    (since there there is no statistics)
  "tLL": 0.3, // ps
  "tUL": 15.0, // ps
"""
