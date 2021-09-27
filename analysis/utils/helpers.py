# HELPERS
#
#
#    Marcos Romero Lamas


# Modules {{{

import re
import hjson
import numpy as np
from utils.strings import cuts_and
from ipanema import ristra
import config

# }}}


# Some common settings {{{

SAMPLES = config.user['path']
MAILS = config.user['mail']

YEARS = config.years

if config.user['velo_weights']:
  vw8s = 'veloWeight'
else:
  vw8s = 'sWeight'

# }}}


# Wildcard parsers {{{

# Parse time acceptance wildcard (aka timeacc) {{{

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
  # Define the pattern we use to regex-parse the time acceptance wildcard
  pattern = [
    r"(single|simul|lifeBd|lifeBu)",
    # number of knots
    r"(1[0-2]|[2-9])?",
    # whether to use Velo Weights or not
    r"(VW)?",
    # whether to use Velo Weights or not
    r"(LT|UT)?",
    # whether to use reweightings or not
    r"(Noncorr)?",
    # wether to use resolution in time
    r"(timeT)?",
    # wether to use resolution in time
    r"(Nores)?",
    # whether to use oddWeight or not
    r"(Odd)?",
    # wether to create a pT weight or not
    r"(pT)?",
    # whether to impose flat condition at upper decay times
    r"(Flatend)?",
    # custom variable cuts for lifetime test
    r"(deltat|alpha|mKstar)?",
    # use samples as others
    r"(BuasBs|BdasBs|BuasBd)?"
  ]
  pattern = rf"\A{''.join(pattern)}\Z"
  p = re.compile(pattern)
  try:
    acc, nknots, vw8, lut, corr, timeT, res, oddW, pTW, flat, cuts, swap = p.search(timeacc).groups()
    ans = {
      "acc": acc,
      "nknots": int(nknots) if nknots else 3,
      "use_truetime": True if res=='Nores' else False,
      "use_transvers_time": True if timeT=='timeT' else False,
      "use_oddWeight": True if oddW=='Odd' else False,
      "use_lowTime": True if lut=='LT' else False,
      "use_upTime": True if lut=='UT' else False,
      "use_veloWeight": True if vw8=='VW' else False,
      "use_pTWeight": True if pTW=='pT' else False,
      "corr": False if corr=='Noncorr' else True,
      "use_flatend": True if flat=='Flatend' else False,
      "swap": swap if swap else False,
      "cuts": cuts if cuts else False
    }
    return ans
  except:
    raise ValueError(f'Cannot interpret {timeacc} as a timeacc modifier.')

# }}}


# Parse angular acceptance wildcard (aka angacc) {{{

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
  # Define the pattern we use to regex-parse the time acceptance wildcard
  pattern = [
    r"(naive|corrected|analytic|yearly|run2a|run2b|run2)",
    # dual MC or single
    r"(Dual)?",
    # wether to use resolution time or not
    r"(Nores)?",
    # whether to use oddWeight or not
    r"(Odd)?",
    # create a pT weight
    r"(pT)?",
  ]
  pattern = rf"\A{''.join(pattern)}\Z"
  p = re.compile(pattern)
  try:
    acc, dual, res, oddity, ptW = p.search(angacc).groups()
    ans = {
      "acc": acc,
      "dual": True if dual else False,
      "use_truetime": True if res=='Nores' else False,
      "use_oddWeight": True if oddity=='Odd' else False,
      "use_pTWeight": True if ptW=='pT' else False
    }
    return ans
  except:
    raise ValueError(f'Cannot interpret {angacc} as a angacc modifier')

# }}}


# Parse physics parameters wildcard (aka fit) {{{

def physpar_guesser(physics):
  # Check if the tuple will be modified
  pattern = r'\A(0)(Minos|BFGS|LBFGSB|CG|Nelder|EMCEE)?\Z'
  p = re.compile(pattern)
  try:
    timeacc, kind, lifecut, mini = p.search(timeacc).groups()
    return timeacc, kind, lifecut, mini.lower() if mini else 'minuit'
  except:
    raise ValueError(f'Cannot interpret {timeacc} as a timeacc modifier')

# }}}


# Parse version wildcard (aka version) {{{

def version_guesser(version):
  """
  From a version string, return the configuration of the tuple
  """
  # Check if the tuple will be modified
  # print(version)
  version = version.split('@')
  v, mod = version if len(version)>1 else [version[0], None]
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
        # upper / lower times
        r"(LT|UT)?",
        # split in runNumber  
        r"(l210300|g210300)?",
        # split by magnet Up or Down: useful for crosschecks
        r"(magUp|magDown)?",
        # cut in cosK
        r"(LcosK|UcosK)?",
        # split in pTB, etaB and sigmat bins: for systematics
        r"((pTB|etaB|sigmat)(\d{1}))?"
        ]
    pattern = rf"\A{''.join(pattern)}\Z"
    p = re.compile(pattern)
    try:
      share, evt, shit, time, runN, mag, cosk, fullcut, var, nbin = p.search(mod).groups()
      share = int(share) if share else 100
      evt = evt if evt else None
      nbin = int(nbin)-1 if nbin else None
      return v, share, evt, mag, fullcut, var, nbin
    except:
      raise ValueError(f'Cannot interpret {mod} as a version modifier')
  else:
    return v, int(100), None, None, None, None, None

# }}}

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
    "LT": "time < 2",
    "UT": "time > 2",
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

def tuples(wcs, version=False, year=False, mode=False, weight=False,
           angacc=False, csp=False, flavor=False, timeacc=False, timeres=False):
  """
  Snakemake tuple helper returns tuple path once some set of wildcards is
  properly provided.

  IMPROVE
  """

  # parse version {{{

  if not version:
    version = f"{wcs.version}"
  # print(f'{version}')
  v, share, evt, mag, fullcut, var, bin = version_guesser(f'{version}')
  v = f'{version}'
  # print(v, share, evt, mag, fullcut, var, bin)

  # }}}

  # parse disciplines {{{

  if not angacc:
    try:
      angacc = f'{wcs.angacc}'
    except:
      angacc = '{angacc}'

  if not csp:
      try:
        csp = f'{wcs.csp}'
      except:
        csp = f'{csp}'

  if not flavor:
    try:
      flavor = f'{wcs.flavor}'
    except:
      flavor = 'none'

  if not timeacc:
    try:
      timeacc = f'{wcs.timeacc}'
    except:
      timeacc = 'none'

  if not timeres:
    try:
      timeres = f'{wcs.timeres}'
    except:
      timeres = 'none'

  # }}}

  # Try to extract mode from wcs then from mode arg {{{
  if not mode:
    try:
      m = f'{wcs.mode}'
    except:
      m = 'none'
  elif mode in ('data', 'cdata'):
    try:
      m = f'{wcs.mode}'
    except:
      m = 'none'
  else:
    m = mode
  # }}}

  # Check modifiers for mode {{{

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
        elif m == 'Bs2JpsiPhi':
          m = 'Bd2JpsiKstar'
        elif m == 'Bd2JpsiKstar':
          m = 'Bs2JpsiPhi'
        elif m == 'Bu2JpsiKplus':
          m = 'Bs2JpsiPhi'
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
        elif m == 'Bs2JpsiPhi':
          m = 'Bd2JpsiKstar'
        elif m == 'Bd2JpsiKstar':
          m = 'Bs2JpsiPhi'
        elif m == 'Bu2JpsiKplus':
          m = 'Bs2JpsiPhi'
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

  # }}}
  # print("wcs.mode, m, mode = ", f"{wcs.mode}", m, mode)
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
      if weight == 'oddWeight': # kbuWeight needed for Bd RD
        weight = 'kbuWeight'
      if weight not in ('sWeight', vw8s, 'kbuWeight', 'kinWeight'):
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
                          'kbuWeight', 'oddWeight', 'angWeight', 'kinWeight', 'kkpWeight'):
        weight = 'polWeight'
    # }}}
    # MC_Bs2JpsiPhi {{{
    elif m == 'MC_Bs2JpsiPhi':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'dg0Weight', 'polWeight',
                          'pdfWeight', 'oddWeight', 'kinWeight', 'angWeight',
                          'kkpWeight'):
        weight = 'dg0Weight'
    # }}}
    # MC_Bs2JpsiKK_Swave {{{
    elif m == 'MC_Bs2JpsiKK_Swave':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'dg0Weight', 'polWeight',
                          'pdfWeight', 'oddWeight', 'kinWeight', 'angWeight',
                          'kkpWeight'):
        weight = 'dg0Weight'
    # }}}
    # MC_Bs2JpsiPhi_dG0 {{{
    elif m == 'MC_Bs2JpsiPhi_dG0':
      if weight == 'kbuWeight':
        weight = 'pdfWeight'
      elif weight == 'veloWeight':
        weight = vw8s
      elif weight not in ('sWeight', vw8s, 'polWeight', 'pdfWeight',
                          'oddWeight', 'kinWeight', 'angWeight', 'kkpWeight'):
        weight = 'polWeight'
    # }}}
  # print(f"Weight was transformed {__weight}->{weight}")
  # }}}

  # Return list of tuples for YEARS[year] {{{

  if year:
    years = YEARS[year]
  else:
    years = YEARS[f'{wcs.year}']

  path = []
  for y in years:
    if weight:
      if weight == 'angWeight':
        path.append(f"{SAMPLES}/{y}/{m}/{v}_{angacc}_{csp}_{weight}.root")
      elif weight == 'kkpWeight':
        path.append(f"{SAMPLES}/{y}/{m}/{v}_{angacc}_{csp}_{flavor}_{timeacc}_{timeres}_{weight}.root")
      else:
        path.append(f"{SAMPLES}/{y}/{m}/{v}_{weight}.root")
    else:
        path.append(f"{SAMPLES}/{y}/{m}/{v}.root")

  # }}}

  return path[0] if len(path) == 1 else path


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
    if timeacc.startswith('simul'):
      if "BuasBd" in timeacc:
        m = 'Bu2JpsiKplus'
      else:
        m = 'Bd2JpsiKstar'
    elif timeacc.startswith('single'):
      if timeacc.endswith('DGn0'):
        m = 'MC_Bs2JpsiPhi'
      else:
        m = 'MC_Bs2JpsiPhi_dG0'
  elif mode=='Bd2JpsiKstar':
    if timeacc.startswith('simul'):
      if 'BdasBs' in timeacc:
        if 'evtEven' in version:
          version = version.replace('evtEven', 'evtOdd')
        elif 'evtOdd' in version:
          version = version.replace('evtOdd', 'evtEven')
        else:
          0#print('i need to understand this..')
      m = 'Bd2JpsiKstar'
    else:
      m = 'Bd2JpsiKstar'
  elif mode=='Bu2JpsiKplus':
    if timeacc.startswith('simul'):
      if 'BuasBs' in timeacc:
        if 'evtEven' in version:
          version = version.replace('evtEven', 'evtOdd')
        elif 'evtOdd' in version:
          version = version.replace('evtOdd', 'evtEven')
        else:
          0#print('i need to understand this..')
      m = 'Bd2JpsiKstar'
    else:
      m = 'Bu2JpsiKplus'
  else:
    m = mode

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


def smkphyspar(wcs, version=False, year=False, mode=False, timeacc=False,
               angacc=False, fit=False, csp=False, timeres=False, flavor=False,
               trigger=False):
  """
  Snakemake physics params helper returns a list of paths to parameters once
  a set of wildcards is properly provided.
  """

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
