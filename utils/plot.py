# STRINGS
#
#
#


# Modules {{{

import yaml
import os

# }}}


# Load branches {{{

PHISSCQ = os.environ['PHISSCQ']
with open(rf"{PHISSCQ}/analysis/samples/branches_latex.yaml") as file:
  BRANCHES = yaml.load(file, Loader=yaml.FullLoader)

# }}}


# Get mode in TeX {{{

def guess_mode(mode, modifier=False, version=False):
  m = mode
  v = version if version else ''
  if modifier:
    if "evtEven" in v:
      if modifier == 'data':
        if m.startswith('MC_'):
          v = v.replace('evtEven', 'evtOdd')
      elif modifier == 'cdata':
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
      if modifier == 'data':
        if m.startswith('MC_'):
          v = v.replace('evtOdd', 'evtEven')
      elif modifier == 'cdata':
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
      if modifier == 'data':
        if m.startswith('MC_'):
          m = m[3:]
          if m.endswith('_dG0'):
            m = m[:-4]
          elif m.endswith('_Swave'):
            m = m[:-8] + 'Phi'
      elif modifier == 'cdata':
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
      elif modifier in ('Bs2JpsiPhi', 'MC_Bs2JpsiPhi_dG0', 'MC_Bs2JpsiPhi', 'Bd2JpsiKstar', 'MC_Bd2JpsiKstar', 'Bu2JpsiKplus', 'MC_Bu2JpsiKplus', 'MC_Bs2JpsiKK_Swave'):
        m = modifier
  return m


def mode2tex(mode, modifier=False, version=False):
  m = guess_mode(mode, modifier, version)
  print(mode, modifier, version, m)
  ans = ['', '', '']
  if 'Bs' in m:
    ans[1] = 'B_s^0'
  elif 'Bd' in m:
    ans[1] = 'B_d^0'
  elif 'Bu' in m:
    ans[1] = 'B_u^+'
  else:
    print('ERROR: I cannot convert this mode')

  # which kind of samples is it?
  if 'MC' in m:
    ans[0] = 'MC'
  elif 'TOY' in m:
    ans[0] = 'TOY'
  else:
    ans[0] = 'RD'

  # special MC handlers
  if 'dG0' in m:
    ans[2] = r'\mathrm{w}\,\Delta\Gamma=0'
  if 'Swave' in m:
    ans[2] = r'\mathrm{w\,S-wave}'

  return ans


def mode_tex(mode, modifier=False, version=False):
  _tex_list = mode2tex(mode, modifier, version)
  _tex_str = rf"{_tex_list[1]}\,{_tex_list[0]}\,{_tex_list[2]}"
  return _tex_str

# }}}


# Get range and get bins {{{

def get_range(var, mode='Bs2JpsiPhi', modifier=False, version=False):
  _mode = guess_mode(mode, modifier, version)
  return BRANCHES[_mode][var].get('range')


def get_nbins(var, mode='Bs'):
  if mode in ('Bs', 'Bs2JpsiPhi'):
    mode = 'Bs'
  if mode in ('Bd', 'Bd2JpsiKstar'):
    mode = 'Bd'
  ranges_dict = dict(
      B_PT=(70, 0, 4e4),
      B_P=(70, 0, 8e5),
      X_M=(70, 840, 960) if 'Bd2JpsiKstar' in mode else (70, 990, 1050),
      hplus_P=(70, 0, 1.5e5),
      hplus_PT=(70, 0, 1.0e4),
      hminus_P=(70, 0, 1.5e5),
      hminus_PT=(70, 0, 1.0e4),
  )
  return ranges_dict[var][0]

# }}}


# Get branch and units in TeX {{{

def name_from_lhcb_to_scq(branch, mode='Bs2JpsiPhi'):
  for b in BRANCHES[mode]:
    if BRANCHES[mode][b]['peval'] == branch:
      return b
  raise ValueError(f"Unknown branch {branch}")


def get_var_in_latex(var, mode='Bs2JpsiPhi'):
  return BRANCHES[mode][var].get('latex')


def get_units(var, mode='Bs2JpsiPhi'):
  return BRANCHES[mode][var].get('units')

# }}}


# Force square axes / Watermark {{{

def make_square_axes(ax):
  """Make an axes square in screen units.

  Should be called after plotting.
  """
  ax.set_aspect(1 / ax.get_data_ratio())


def watermark(ax, version='final', tag='', size=(20, 8.25), scale=1.2, pos=(0.05, 0.9), color='black'):
  if 'final' in version:
    version = 'LHCb'
    tag = 'THIS THESIS'
    size = [23, 9.8]
  if scale:
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * scale)
  # ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.03, ax.get_ylim()[1]*0.95,
  #         f'{version}', fontsize=size[0], color='black',
  #         ha='left', va='top', alpha=1.0)
  ax.text(pos[0] - 0.02, pos[1], f'{version}', fontsize=size[0], color=color,
          ha='left', va='center', transform=ax.transAxes, alpha=1.0)
  ax.text(pos[0] - 0.02, pos[1] - 0.06, f'{tag}', fontsize=size[1], color=color,
          ha='left', va='center', transform=ax.transAxes, alpha=1.0)
  # ax.text(ax.get_xlim()[1]*0.025, ax.get_ylim()[1]*0.85,
  #         f'{tag}', fontsize=size[1], color='black',
  #         ha='left', va='top', alpha=1.0)

# }}}


# vim: fdm=marker
