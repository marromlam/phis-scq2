# Main Workflow - phis-scq
#
#    This is the Snakefile for the phis analysis within Santiago framework
#
# Contributors: Marcos Romero

import hjson
CONFIG = hjson.load(open('config.json'))

# Main constants ---------------------------------------------------------------
#    VERSION: is the version Bs2JpsiPhi-FullRun2 pipeline was run against, and
#             should be matched with this constant.
#    MAIN_PATH: the path where all eos-samples will be synced, make sure there
#               is enough free space there
#    SAMPLES_PATH: where all ntuples for a given VERSION will be stored

SAMPLES_PATH = CONFIG['path']
MAILS = CONFIG['mail']

MINERS = "(Minos|BFGS|LBFGSB|CG|Nelder)"

def send_mail(subject, body, files=None):
  import os
  body = os.path.abspath(body)
  attachments = None
  # Check if there are attachments
  if files:
    attachments = []
    for f in files:
      attachments.append( os.path.abspath(f) )
  # Loop over mails
  for mail in MAILS:
    # If there are attachments, then we use mail directly
    if attachments:
      shell(f"""ssh master 'echo "{attachments}" | mail -s"{subject}" -a {" -a ".join(attachments)} {mail}'""")
    else:
      # If it's a log file, then we try to force monospace fonts in mails
      # maybe not all of the email apps can read it as monospaced
      # WIP
      start = f'Content-Type: text/html\nSubject: {subject}\n<pre style="font: monospace">'
      end = '</pre>'
      cmd = f'echo "{start}\n`cat {body}`\n{end}"'
      shell(f"""ssh master '{cmd} | /usr/sbin/sendmail {mail}'""")


# Some wildcards options ( this is not actually used )
modes = ['Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0',#'MC_Bs2JpsiPhi',
         'Bd2JpsiKstar', 'MC_Bd2JpsiKstar'];
years = ['2011','2012','Run1',
         '2015','2016','Run2a','2017','2018','Run2b','Run2'];
yd = {'2011':'2011',
      '2012':'2012',
      'Run1':['2011','2012'],
      '2015':'2015',
      '2016':'2016',
      'Run2a':['2015','2016'],
      '2017':'2017',
      '2018':'2018',
      'Run2b':['2017','2018'],
      'Run2':['2015','2016','2017','2018']};



# Rule orders
ruleorder: sync_ntuples > reduce_ntuples


def tuples(wcs,version=False,year=None,mode=None,weight=None):
  # Check if user is giving version
  v = f'{wcs.version}'
  try:
    m = f'{wcs.mode}'
  except:
    m = '{mode}'

  #print(f'input: {v}-{m}-{y}\nmode={mode} year={year} weight={weight}')
  #print(f'input: {v}-{m}\nmode={mode} year={year} weight={weight}')

  # Folder swicher
  """
  if   v == 'v0r0':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  elif v == 'v0r1':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  elif v == 'v0r2':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  elif v == 'v0r3':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  elif v == 'v0r4':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  elif v == 'v0r5':
    samples_path = '/scratch17/marcos.romero/phis_samples/'
  """
  samples_path = SAMPLES_PATH

  # If version, this function only returns samples_path
  if version:
    return samples_path

  # Check modifiers for mode
  if mode:
    if mode == 'data':
      if m.startswith('MC_'):
        m = m[3:]
        if m.endswith('_dG0'):
          m = m[:-4]
    elif mode == 'cdata':
      if m.startswith('MC_'):
        m = m[3:]
        if m.endswith('_dG0'):
          m = m[:-4]
      elif m == 'Bs2JpsiPhi':
        m = 'Bd2JpsiKstar'
      elif m == 'Bd2JpsiKstar':
        m = 'Bs2JpsiPhi'
    elif mode in ('Bs2JpsiPhi','MC_Bs2JpsiPhi_dG0','MC_Bs2JpsiPhi',
                  'Bd2JpsiKstar', 'MC_Bd2JpsiKstar'):
      m = mode


  # Model handler when asking for weights
  if m == 'Bs2JpsiPhi':
    if weight:
      weight = 'sWeight'
  elif m == 'Bd2JpsiKstar':
    if weight:
      if weight != 'kinWeight':
        weight = 'sWeight'

  # Year
  if year:
    years = yd[year]
    path = []
    for y in years:
      terms = [y, m, v]
      if weight:
        path.append( samples_path + '/'.join(terms) + f'_{weight}.root' )
      else:
        path.append( samples_path + '/'.join(terms) + f'.root' )
  else:
    y = f'{wcs.year}'
    terms = [y, m, v]
    if weight:
      path = samples_path + '/'.join(terms) + f'_{weight}.root'
    else:
      path = samples_path + '/'.join(terms) + f'.root'
  #print(path)
  return path

# Set pipeline-wide constraints ------------------------------------------------
#     Some wilcards will only have some well defined values.
wildcard_constraints:
  trigger = "(biased|unbiased|combined)"


# Including Snakefiles
include: 'samples/Snakefile'
include: 'reweightings/Snakefile'
include: 'time_acceptance/Snakefile'
include: 'flavor_tagging/Snakefile'
include: 'csp_factors/Snakefile'
include: 'time_resolution/Snakefile'
include: 'angular_acceptance/Snakefile'
include: 'angular_fit/Snakefile'
include: 'bundle/Snakefile'
include: 'params/Snakefile'
