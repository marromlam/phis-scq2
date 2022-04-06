__all__ = []
import json
from string import Template
from ipanema import Parameters

def physics_params_translator(pars, fp, flag="base"):
  # Load json as string
  json_str = open('analysis/bundle/templates/physics_params.json').read()
  json_template = Template(json_str)

  # Create dict to parse json-string
  parser = {"FLAG":flag}
  parser.update({f"{p}_value":float(v.unc_round[0]) for p,v in pars.items()})
  parser.update({f"{p}_error":float(v.unc_round[1]) for p,v in pars.items()})

  # Parse
  json_parsed = json_template.substitute(**parser)

  # Dump
  if isinstance(fp,str):
    #json.dump(json.loads(json_parsed), open(fp,'w'), indent=4)
    with open(fp,'w') as file:
      file.write(json_parsed)
  else:
    #json.dump(json.loads(json_parsed), fp, indent=4)
    fp.write(json_parsed)



#physics_params_translator(Parameters.load('output/params/angular_fit/run2/Bs2JpsiPhi/v0r5_base_base.json') , 'shit.json', flag="base")
