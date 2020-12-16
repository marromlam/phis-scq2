import re
import os
import hjson
import numpy as np
from ipanema import ristra
# %%
CONFIG = hjson.load(open('config.json'))
SAMPLES_PATH = CONFIG['path']
MAILS = CONFIG['mail']

YEARS = {#
  '2011'  : ['2011'],
  '2012'  : ['2012'],
  'Run1'  : ['2011','2012'],
  'run1'  : ['2011','2012'],
  '2015'  : ['2015'],
  '2016'  : ['2016'],
  'Run2a' : ['2015','2016'],
  'run2a' : ['2015','2016'],
  '2017'  : ['2017'],
  '2018'  : ['2018'],
  'Run2b' : ['2017','2018'],
  'run2b' : ['2017','2018'],
  'Run2'  : ['2015','2016','2017','2018'],
  'run2'  : ['2015','2016','2017','2018']
};

from utils.strings import cammel_case_split, cuts_and
import numpy as np




def timeacc_guesser(timeacc):
  # Check if the tuple will be modified
  #pattern = r'\A(single|simul|lifeBd|lifeBu)(1[0-2]|[3-9]knots)?(Noncorr)?(deltat|alpha|mKstar)?(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  pattern = r'\A(single|simul|lifeBd|lifeBu)(1[0-2]|[3-9])?(Noncorr)?(deltat|alpha|mKstar)?(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  p = re.compile(pattern)
  try:
    acc, knots, corr, lifecut, mini = p.search(timeacc).groups()
    corr = False if corr=='Noncorr' else True
    mini = mini.lower() if mini else 'minuit'
    knots = int(knots) if knots else 6
    #knots = int(knots[0]) if knots else 6
    return acc, knots, corr, lifecut, mini
  except:
    raise ValueError(f'Cannot interpret {timeacc} as a timeacc modifier')



def physpar_guesser(physics):
  # Check if the tuple will be modified
  pattern = r'\A(0)(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  p = re.compile(pattern)
  try:
    timeacc, kind, lifecut, mini = p.search(timeacc).groups()
    return timeacc, kind, lifecut, mini.lower() if mini else 'minuit'
  except:
    raise ValueError(f'Cannot interpret {timeacc} as a timeacc modifier')




def version_guesser(version):
  # Check if the tuple will be modified
  # print(version)
  version = version.split('@')
  v, mod = version if len(version)>1 else [version[0],None]
  # Dig in mod
  if mod:
    #pattern = r'\A(\d+)?(magUp|magDown)?(cut(B_PT|B_ETA|sigmat)(\d{1}))?\Z'
    pattern = r'\A(\d+)?(magUp|magDown)?(cut(pTB|etaB|sigmat)(\d{1}))?\Z'
    p = re.compile(pattern)
    try:
      share, mag, fullcut, var, bin = p.search(mod).groups()
      return v, int(share) if share else None, mag, fullcut, var, int(bin)-1 if bin else None
    except:
      raise ValueError(f'Cannot interpret {mod} as a version modifier')
  else:
    return v, int(100), None, None, None, None


#version_guesser('v0r5@10')






def tuples(wcs, version=False, year=None, mode=None, weight=None):
  # Get version withoud modifier
  if not version:
    version = f"{wcs.version}"
  #print(f'{version}')
  v, share, mag, fullcut, var, bin = version_guesser(f'{version}')
  #print(wcs)
  # Try to extract mode from wcs then from mode arg
  try:
    m = f'{wcs.mode}'
  except:
    m = '{mode}'

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
    elif mode in ('Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar', 'Bu2JpsiKplus', 'MC_Bu2JpsiKplus'):
      m = mode


  # Model handler when asking for weights
  if weight:
    if m == 'Bs2JpsiPhi':
      weight = 'sWeight'
    elif m == 'Bu2JpsiKplus':
      weight = 'sWeight'
    elif m == 'Bd2JpsiKstar':
      if weight not in ('kinWeight', 'kbuWeight'):
        weight = 'sWeight'
    elif m == 'MC_Bu2JpsiKplus':
      if weight not in ('sWeight', 'polWeight', 'kinWeight'):
        weight = 'polWeight'
    elif m == 'MC_Bd2JpsiKstar':
      if weight not in ('sWeight', 'polWeight', 'pdfWeight', 'kbuWeight', 'kinWeight'):
        weight = 'polWeight'
    elif m == 'MC_Bs2JpsiPhi':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight not in ('sWeight', 'polWeight', 'dg0Weight', 'pdfWeight', 'kinWeight', 'angWeight'):
        weight = 'polWeight'
    elif m == 'MC_Bs2JpsiPhi_dG0':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight not in ('sWeight', 'polWeight', 'pdfWeight', 'kinWeight', 'angWeight'):
        weight = 'polWeight'

  
  # Year
  if year:
    years = YEARS[year]
    path = []
    for y in years:
      terms = [y, m, v]
      if weight:
        if weight=='angWeight':
          path.append( SAMPLES_PATH + '/'.join([y,m,f'{version}']) + f'_{weight}.root' )
        elif weight=='kkpWeight':
          path.append( SAMPLES_PATH + '/'.join([y,m,f'{version}']) + f'_{wcs.angacc}_{wcs.timeacc}_{weight}.root' )
        else:
          path.append( SAMPLES_PATH + '/'.join(terms) + f'_{weight}.root' )
      else:
        path.append( SAMPLES_PATH + '/'.join(terms) + f'.root' )
  else:
    y = f'{wcs.year}'
    terms = [y, m, v]
    if weight:
        if weight=='angWeight':
          path =  SAMPLES_PATH + '/'.join([y,m,f'{version}']) + f'_{weight}.root'
        elif weight=='kkpWeight':
          path =  SAMPLES_PATH + '/'.join([y,m,f'{version}']) + f'_{wcs.angacc}_{wcs.timeacc}_{weight}.root'
        else:
          path =  SAMPLES_PATH + '/'.join(terms) + f'_{weight}.root'
    else:
      path = SAMPLES_PATH + '/'.join(terms) + f'.root'
  #print(path)
  return path



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
      os.system(f"""ssh master 'echo "{attachments}" | mail -s"{subject}" -a {" -a ".join(attachments)} {mail}'""")
    else:
      # If it's a log file, then we try to force monospace fonts in mails
      # maybe not all of the email apps can read it as monospaced
      # WIP
      start = f'Content-Type: text/html\nSubject: {subject}\n<pre style="font: monospace">'
      end = '</pre>'
      cmd = f'echo "{start}\n`cat {body}`\n{end}"'
      os.system(f"""ssh master '{cmd} | /usr/sbin/sendmail {mail}'""")



def trigger_scissors(trigger, CUT=""):
  if trigger == 'biased':
    CUT = cuts_and("hlt1b==1",CUT)
  elif trigger == 'unbiased':
    CUT = cuts_and("hlt1b==0",CUT)
  return CUT



def swnorm(sw):
  sw_ = ristra.get(sw)
  return ristra.allocate(sw_*(np.sum(sw_)/np.sum(sw_**2)))

