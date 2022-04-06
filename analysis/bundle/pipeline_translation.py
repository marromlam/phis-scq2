__all__ = []
import ipanema
import json
import os
import numpy as np
import argparse


def argument_parser():
  parser = argparse.ArgumentParser(description='Cook parameters to share.')
  parser.add_argument('--params-biased',
                      default = 'output_new/params/time_acceptance/2016/Bd2JpsiKstar/v0r1_baseline_biased.json',
                      help='Set of biased acceptance params')
  parser.add_argument('--params-unbiased',
                      default = 'output_new/params/time_acceptance/2016/Bd2JpsiKstar/v0r1_baseline_unbiased.json',
                      help='Set of unbiased acceptance params')
  parser.add_argument('--params-output',
                      default = 'output_new/bundle/params/v0r1/time_acceptance/2016/time_acceptance_2016.json',
                      help='Set of unbiased acceptance params')
  parser.add_argument('--year',
                      default = '2016',
                      help='Year of data-taking')
  parser.add_argument('--version',
                      default = 'v0r1',
                      help='Version of tuples your are working with')

  return parser



if __name__ == '__main__':
  # Parse arguments
  try:
    args = vars(argument_parser().parse_args())
  except:
    args = vars(argument_parser().parse_args(''))

  # Load parameters
  uinfo = args['params_biased']
  uinfo = uinfo.split('/')
  acceptance = uinfo[2]
  year = int(args['year'])
  version = args['version']

  parsb = ipanema.Parameters.load(args['params_biased'])
  parsu = ipanema.Parameters.load(args['params_unbiased'])
  print(parsb)
  print(parsu)

  # Load template
  json_str = json.dumps(json.load(open(f'analysis/bundle/templates/{acceptance}.json')))

  # Convert to lists
  if acceptance == 'time_acceptance':
    listb = [ f"{parsb[p].uvalue:.2u}".split('+/-') for p in parsb.find('c.*')]
    listu = [ f"{parsu[p].uvalue:.2u}".split('+/-') for p in parsu.find('c.*')]
    listk = [ [float(f"{parsb[p].value:.2f}")] for p in parsu.find('k.*')]
    listb = list(map(lambda x: [float(x[0]),float(x[1])] ,listb))
    listu = list(map(lambda x: [float(x[0]),float(x[1])] ,listu))
    ristra = [[year]] + listb + listk + listu
  elif acceptance == 'angular_acceptance':
    listb = [ f"{parsb[p].uvalue:.2u}".split('+/-') for p in parsb.find('w.*')]
    listu = [ f"{parsu[p].uvalue:.2u}".split('+/-') for p in parsu.find('w.*')]
    listb = list(map(lambda x: [float(x[0]),float(x[1])] ,listb))
    listu = list(map(lambda x: [float(x[0]),float(x[1])] ,listu))
    ristra = [[year]] + listb + listu

  ristra = sum(ristra, [])        # a little bit risky, maybe one should change it

  # Fill template and build dict
  pars_out = json.loads( json_str % tuple(ristra) )

  # Dump json
  fpath = os.path.join(uinfo[0],'bundle',uinfo[1],f'{version}',uinfo[2],f'{year}')
  fname = f"{uinfo[2]}_{year}.json"
  #os.makedirs(fpath, exist_ok=True)
  print(os.path.join(fpath,fname))
  json.dump( pars_out, open(args['params_output'],'w') , indent=4)
