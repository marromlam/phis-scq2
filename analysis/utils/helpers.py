# HELPERS
#
#
#


# Modules {{{

import re
import hjson
import numpy as np
from utils.strings import cammel_case_split, cuts_and
from ipanema import ristra
import config

SAMPLES_PATH = config.user['path']+'/'
MAILS = config.user['mail']

YEARS = config.years

if config.user['velo_weights']:
  vw8s = 'veloWeight'
else:
  vw8s = 'sWeight'

# }}}


# Guessers {{{

def timeacc_guesser(timeacc):
  """
  Parse decay-time acceptance string and derive its components.

  Parameters
  ----------
  timeacc : str
    Decay-time acceptance string.

  Returns
  -------
  tuple
    Components of the decay-time acceptance.
  """
  # Check if the tuple will be modified
  #pattern = r'\A(single|simul|lifeBd|lifeBu)(1[0-2]|[3-9]knots)?(Noncorr)?(deltat|alpha|mKstar)?(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  pattern = r'\A(single|simul|lifeBd|lifeBu)(1[0-2]|[2-9])?(BuasBd)?(Noncorr)?(Odd)?(Flatend)?(deltat|alpha|mKstar)?(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  p = re.compile(pattern)
  try:
    acc, knots, pipas, corr, oddity, flat, lifecut, mini = p.search(timeacc).groups()
    corr = False if corr=='Noncorr' else True
    mini = mini.lower() if mini else 'minuit'
    knots = int(knots) if knots else 3
    flat = True if flat=='Flatend' else False
    return acc, knots, corr, flat, lifecut, mini
  except:
    raise ValueError(f'Cannot interpret {timeacc} as a timeacc modifier')


def parse_angacc(angacc):
  """
  Parse angular acceptance string and derive its components.

  Parameters
  ----------
  angacc : str
    Angular acceptance string.

  Returns
  -------
  tuple
    Components of the angular acceptance.
  """
  pattern = r'\A(yearly|run2|run2a|run2b)(Odd)?\Z'
  p = re.compile(pattern)
  try:
    acc, oddity = p.search(angacc).groups()
    oddity = True if oddity=='Odd' else False
    return acc, oddity
  except:
    raise ValueError(f'Cannot interpret {angacc} as a angacc modifier')


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
  """
  From a version string, return the configuration of the tuple
  """
  # Check if the tuple will be modified
  # print(version)
  version = version.split('@')
  v, mod = version if len(version)>1 else [version[0],None]
  # Dig in mod
  if mod:
    #pattern = r'\A(\d+)?(magUp|magDown)?(cut(B_PT|B_ETA|sigmat)(\d{1}))?\Z'
    pattern = [
        # percentage of tuple to be loaded
        r"(\d+)?",
        # split tuple by event number: useful for MC tests and crosschecks
        # Odd is plays data role and MC is Even
        r"(evtOdd|evtEven)?",
        # background category  
        r"(bkgcat60)?",
        # split in runNumber  
        r"(l210300|g210300)?",
        # split by magnet Up or Down: useful for crosschecks
        r"(magUp|magDown)?",
        # split in pTB, etaB and sigmat bins: for systematics
        r"((pTB|etaB|sigmat)(\d{1}))?"
        ]
    pattern = rf"\A{''.join(pattern)}\Z"
    # print(pattern)
    p = re.compile(pattern)
    try:
      share, evt, shit, runN, mag, fullcut, var, nbin = p.search(mod).groups()
      share = int(share) if share else 100
      evt = evt if evt else None
      nbin = int(nbin)-1 if nbin else None
      return v, share, evt, mag, fullcut, var, nbin
    except:
      raise ValueError(f'Cannot interpret {mod} as a version modifier')
  else:
    return v, int(100), None, None, None, None, None

# }}}


# Cuts and @ modifiers {{{

def cut_translate(version_substring):
  vsub_dict = {
    # "evtOdd": "( (evt % 2) != 0 ) & logIPchi2B>=0 & log(BDTFchi2)>=0",
    "evtOdd": "(evt % 2) != 0",
    "evtEven": "(evt % 2) == 0",
    "magUp": "magnet == 1",
    "magDown": "magnet == 0",
    "bkgcat60": "bkgcat != 60",
    "g210300": "runN > 210300",
    "l210300": "runN < 210300",
    "pTB1": "pTB >= 0 & pTB < 3.8e3",
    "pTB2": "pTB >= 3.8e3 & pTB < 6e3",
    "pTB3": "pTB >= 6e3 & pTB <= 9e3",
    "pTB4": "pTB >= 9e3",
    "etaB1": "etaB >= 0 & etaB <= 3.3",
    "etaB2": "etaB >= 3.3 & etaB <= 3.9",
    "etaB3": "etaB >= 3.9 & etaB <= 6",
    "sigmat1": "sigmat >= 0 & sigmat <= 0.031",
    "sigmat2": "sigmat >= 0.031 & sigmat <= 0.042",
    "sigmat3": "sigmat >= 0.042 & sigmat <= 0.15"
  }
  list_of_cuts = []
  for k,v in vsub_dict.items():
    if k in version_substring:
      list_of_cuts.append(v)
  return f"( {' ) & ( '.join(list_of_cuts)} )"

# }}}


# Snakemake helpers {{{

def tuples(wcs, version=False, year=None, mode=None, angacc=False, csp=False, weight=None):
  # Get version withoud modifier
  if not version:
    version = f"{wcs.version}"

  # print(f'{version}')
  v, share, evt, mag, fullcut, var, bin = version_guesser(f'{version}')
  v = f'{version}'
  # print(v, share, evt, mag, fullcut, var, bin)

  # Try to extract mode from wcs then from mode arg
  try:
    m = f'{wcs.mode}'
  except:
    m = '{mode}'

  if not csp:
    csp = 'none'

  if not angacc:
    angacc = 'corrected'

  # Check modifiers for mode
  if mode:
    if "evtEven" in v:
      if mode == 'data':
        if m.startswith('MC_'):
          v = v.replace('evtEven', 'evtOdd')
      elif mode == 'cdata':
        if m.startswith('MC_'):
          m = m[3:]
          if m.endswith('_dG0'):
            m = m[:-4]
          elif m.endswith('_Swave'):
            m = m[:-8] + 'Phi'
    elif "evtOdd" in v:
      if mode == 'data':
        if m.startswith('MC_'):
          v = v.replace('evtOdd', 'evtEven')
      elif mode == 'cdata':
        # v = v.replace('evtOdd', '')
        if m.startswith('MC_'):
          m = m[3:]
          if m.endswith('_dG0'):
            m = m[:-4]
          elif m.endswith('_Swave'):
            m = m[:-8] + 'Phi'
    else:
      if mode == 'data':
        if m.startswith('MC_'):
          m = m[3:]
          if m.endswith('_dG0'):
            m = m[:-4]
          elif m.endswith('_Swave'):
            m = m[:-8] + 'Phi'
      elif mode == 'cdata':
        if m.startswith('MC_'):
          m = m[3:]
          if m.endswith('_dG0'):
            m = m[:-4]
          elif m.endswith('_Swave'):
            m = m[:-8] + 'Phi'
        elif m == 'Bs2JpsiPhi':
          m = 'Bd2JpsiKstar'
        elif m == 'Bd2JpsiKstar':
          m = 'Bs2JpsiPhi'
        elif m == 'Bu2JpsiKplus':
          m = 'Bs2JpsiPhi'
      elif mode in ('Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar', 'Bu2JpsiKplus', 'MC_Bu2JpsiKplus', 'MC_Bs2JpsiKK_Swave'):
        m = mode
  
  # Model handler when asking for weights {{{
  __weight = weight
  if weight:
    # Bs2JpsiPhi {{{
    if m == 'Bs2JpsiPhi':
      if weight not in ('sWeight', vw8s):
        weight = vw8s
    # }}}
    # Bu2JpsiKplus {{{
    elif m == 'Bu2JpsiKplus':
      if weight not in ('sWeight', vw8s, 'kinWeight'):
        weight = vw8s
    # }}}
    # Bd2JpsiKstar {{{
    elif m == 'Bd2JpsiKstar':
      if weight == 'oddWeight':
        weight = 'kbuWeight'
      elif weight not in ('sWeight', vw8s, 'kbuWeight', 'kinWeight'):
        weight = vw8s
    # }}}
    # MC_Bu2JpsiKplus {{{
    elif m == 'MC_Bu2JpsiKplus':
      if weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'polWeight', 'kinWeight'):
        weight = 'polWeight'
    # }}}
    # MC_Bd2JpsiKstar {{{
    elif m == 'MC_Bd2JpsiKstar':
      if weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'polWeight', 'pdfWeight',
                        'kbuWeight', 'oddWeight', 'kinWeight'):
        weight = 'polWeight'
    # }}}
    # MC_Bs2JpsiPhi {{{
    elif m == 'MC_Bs2JpsiPhi':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'dg0Weight', 'polWeight',
                          'pdfWeight', 'oddWeight', 'kinWeight', 'angWeight'):
        weight = 'dg0Weight'
    # }}}
    # MC_Bs2JpsiKK_Swave {{{
    elif m == 'MC_Bs2JpsiKK_Swave':
      if weight == 'kbuWeight':
        weight = 'polWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'dg0Weight', 'polWeight',
                          'pdfWeight', 'oddWeight', 'kinWeight', 'angWeight'):
        weight = 'dg0Weight'
    # }}}
    # MC_Bs2JpsiPhi_dG0 {{{
    elif m == 'MC_Bs2JpsiPhi_dG0':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'polWeight', 'pdfWeight',
                          'oddWeight', 'kinWeight', 'angWeight'):
        weight = 'polWeight'
    # }}}
  # print("Weight was transformed {__weight}->{weight}")
  # }}}


  # Return list of tuples for YEARS[year] {{{
  if year:
    # print(year)
    years = YEARS[year]
    path = []
    for y in years:
      terms = [y, m, v]
      if weight:
        if weight=='angWeight':
          path.append( SAMPLES_PATH + '/'.join([y,m,f'{v}']) + f'_{angacc}_{csp}_{weight}.root' )
        elif weight=='kkpWeight':
          path.append( SAMPLES_PATH + '/'.join([y,m,f'{v}']) + f'_{angacc}_{wcs.csp}_{wcs.flavor}_{wcs.timeacc}_{wcs.timeres}_{weight}.root' )
        else:
          path.append( SAMPLES_PATH + '/'.join(terms) + f'_{weight}.root' )
      else:
        path.append( SAMPLES_PATH + '/'.join(terms) + f'.root' )
  else:
    y = f'{wcs.year}'
    terms = [y, m, v]
    if weight:
        if weight=='angWeight':
          path =  SAMPLES_PATH + '/'.join([y,m,f'{v}']) + f'_{angacc}_{csp}_{weight}.root'
        elif weight=='kkpWeight':
          path =  SAMPLES_PATH + '/'.join([y,m,f'{v}']) + f'_{angacc}_{wcs.csp}_{wcs.flavor}_{wcs.timeacc}_{wcs.timeres}_{weight}.root'
        else:
          path =  SAMPLES_PATH + '/'.join(terms) + f'_{weight}.root'
    else:
      path = SAMPLES_PATH + '/'.join(terms) + f'.root'
  #print(path)
  # }}}
  return path


def timeress(wcs, version=False, year=False, mode=False, timeres=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not timeres:
    timeres = f"{wcs.timeres}"
  if not mode:
    mode = f"{wcs.mode}"

  # loop over years and return list of time acceptances
  ans = []
  for y in YEARS[year]:
    ans.append(f'output/params/time_resolution/{y}/{mode}/{version}_{timeres}.json')
  return ans


def csps(wcs, version=False, year=False, mode=False, csp=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not csp:
    csp = f"{wcs.csp}"
  if not mode:
    mode = f"{wcs.mode}"

  # loop over years and return list of time acceptances
  ans = []
  for y in YEARS[year]:
    ans.append(f'output/params/csp_factors/{y}/{mode}/{version}_{csp}.json')
  return ans


def flavors(wcs, version=False, year=False, mode=False, flavor=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not flavor:
    flavor = f"{wcs.flavor}"
  if not mode:
    mode = f"{wcs.mode}"

  # loop over years and return list of time acceptances
  ans = []
  for y in YEARS[year]:
    ans.append(f'output/params/flavor_tagging/{y}/{mode}/{version}_{flavor}.json')
  return ans


def timeaccs(wcs, version=False, year=False, mode=False, timeacc=False, trigger=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not timeacc:
    timeacc = f"{wcs.timeacc}"
  if not trigger:
    trigger = f"{wcs.trigger}"
  if not mode:
    mode = f"{wcs.mode}"

  # select mode
  if mode=='Bs2JpsiPhi':
    if "BuasBd" in timeacc:
      m = 'Bu2JpsiKplus'
    elif timeacc.startswith('simul'):
      m = 'Bd2JpsiKstar'
    elif timeacc.startswith('single'):
      if timeacc.endswith('DGn0'):
        m = 'MC_Bs2JpsiPhi'
      else:
        m = 'MC_Bs2JpsiPhi_dG0'

  # loop over years and return list of time acceptances
  ans = []
  for y in YEARS[year]:
    ans.append(f'output/params/time_acceptance/{y}/{m}/{version}_{timeacc}_{trigger}.json')
  return ans


def angaccs(wcs, version=False, year=False, mode=False, timeacc=False,
            angacc=False, csp=False, timeres=False, flavor=False,
            trigger=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not timeacc:
    timeacc = f"{wcs.timeacc}"
  if not angacc:
    angacc = f"{wcs.angacc}"
  if not flavor:
    flavor = f"{wcs.flavor}"
  if not csp:
    csp = f"{wcs.csp}"
  if not timeres:
    timeres = f"{wcs.timeres}"
  if not trigger:
    trigger = f"{wcs.trigger}"
  if not mode:
    mode = f"{wcs.mode}"

  # select mode
  if angacc.startswith('naive') or angacc.startswith('corrected'):
    # loop over years and return list of time acceptances
    m = mode
    ans = []
    for y in YEARS[year]:
      ans.append(f'output/params/angular_acceptance/{y}/{m}/{version}_{angacc}_{csp}_{trigger}.json')
  elif angacc.startswith('analytic'):
    print("To be implemented!!")
  else:
    m = mode
    ans = []
    for y in YEARS[year]:
      ans.append(f'output/params/angular_acceptance/{y}/{m}/{version}_{angacc}_{csp}_{flavor}_{timeacc}_{timeres}_{trigger}.json')
  return ans

def sphyspar(wcs, version=False, year=False, mode=False, timeacc=False,
            angacc=False, fit=False, csp=False, timeres=False, flavor=False,
            trigger=False):
  if not version:
    version = f"{wcs.version}"
  if not year:
    year = f"{wcs.year}"
  if not timeacc:
    timeacc = f"{wcs.timeacc}"
  if not angacc:
    angacc = f"{wcs.angacc}"
  if not flavor:
    flavor = f"{wcs.flavor}"
  if not csp:
    csp = f"{wcs.csp}"
  if not fit:
    csp = f"{wcs.fit}"
  if not timeres:
    timeres = f"{wcs.timeres}"
  if not trigger:
    trigger = f"{wcs.trigger}"
  if not mode:
    mode = f"{wcs.mode}"

  # select mode
  m = mode
  ans = []
  for y in YEARS[year]:
    ans.append(f'output/params/physics_params/{y}/{m}/{version}_{fit}_{angacc}_{csp}_{flavor}_{timeacc}_{timeres}_{trigger}.json')
  return ans
# }}}


# Send mail {{{

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
      # WIP:
      start = f'Content-Type: text/html\nSubject: {subject}\n<pre style="font: monospace">'
      end = '</pre>'
      cmd = f'echo "{start}\n`cat {body}`\n{end}"'
      os.system(f"""ssh master '{cmd} | /usr/sbin/sendmail {mail}'""")

# }}}


# Other utils {{{

def trigger_scissors(trigger, CUT=""):
  if trigger == 'biased':
    # CUT = cuts_and("Jpsi_Hlt1DiMuonHighMassDecision_TOS==0",CUT)
    CUT = cuts_and("hlt1b==1",CUT)
  elif trigger == 'unbiased':
    # CUT = cuts_and("Jpsi_Hlt1DiMuonHighMassDecision_TOS==1",CUT)
    CUT = cuts_and("hlt1b==0",CUT)
  return CUT


def swnorm(sw):
  sw_ = ristra.get(sw)
  return ristra.allocate(sw_*(np.sum(sw_)/np.sum(sw_**2)))

# }}}


# vim: foldmethod=marker
