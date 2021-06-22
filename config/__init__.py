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


user = hjson.load(open(f"{__PATH}/user.json", 'r'))


base = hjson.load(open(f"{__PATH}/base.json", 'r'))


