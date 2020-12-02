def mode_tex(mode, mod=None, verbose=False):
  fmode = mode
  if mode.startswith('MC_') and mod=='comp':
    if mode in ('MC_BsJpsiPhi','MC_BsJpsiPhi_dG0','MC_Bs2JpsiKK_Swave'):
      fmode = 'Bs2JpsiPhi'
    elif mode in ('MC_BdJpsiKstar'):
      fmode = 'Bd2JpsiKstar'
  if mode=='Bd2JpsiKstar' and mod=='comp':
    fmode = 'Bs2JpsiPhi'
  if verbose:
    print(f'{mode}@{mod} => {fmode}')
  tex_str = ''
  # Particle in latex form
  if 'Bd2JpsiKstar' in fmode:
    tex_str += 'B_d'
  elif 'Bs2JpsiPhi' in fmode:
    tex_str += 'B_s^0'
  elif 'Bs2JpsiKK' in fmode:
    tex_str += 'B_s^0'
  # Add sim, toy or data info
  tex_str += r'\,\mathrm{'
  if 'MC' in fmode:
    tex_str += 'sim'
  elif 'TOY' in fmode:
    tex_str += 'toy'
  else:
    tex_str += 'data'
  tex_str += '}'
  # Add other info
  if 'dG0' in fmode:
    tex_str += r' \mathrm{w } \Delta \Gamma = 0'
  if 'Swave' in fmode:
    tex_str += r' \mathrm{w S-wave}'
  #print(tex_str)
  return tex_str

def get_range(var, mode='Bs2JpsiPhi'):
  ranges_dict = dict(
  B_PT = (70,0,4e4),
  B_P = (70,0,8e5),
  X_M = (70,840,960) if 'Bd2JpsiKstar' in mode else (70,990,1050),
  hplus_P = (70,0,1.5e5),
  hplus_PT = (70,0,1.0e4),
  hminus_P = (70,0,1.5e5),
  hminus_PT = (70,0,1.0e4),
  )
  return ranges_dict[var][1:]


def get_nbins(var, mode='Bs2JpsiPhi'):
  ranges_dict = dict(
  B_PT = (70,0,4e4),
  B_P = (70,0,8e5),
  X_M = (70,840,960) if 'Bd2JpsiKstar' in mode else (70,990,1050),
  hplus_P = (70,0,1.5e5),
  hplus_PT = (70,0,1.0e4),
  hminus_P = (70,0,1.5e5),
  hminus_PT = (70,0,1.0e4),
  )
  return ranges_dict[var][0]


def get_var_in_latex(var, mode='Bs2JpsiPhi'):
  ranges_dict = dict(
  B_PT = r"p_T(B)",
  B_ETA = r"\eta(B)",
  pt = r"p_T(B)",
  eta = r"\eta(B)",
  sigmat = r"\sigma_t(B)",
  B_P = r"p(B)",
  X_M = r"m(K^+\pi^-)" if 'Bd2JpsiKstar' in mode else r"m(K^+K^-)",
  hplus_P = r"m(K^+)" if 'Bd2JpsiKstar' in mode else r"p(K^+)",
  hplus_PT = r"m(K^+)" if 'Bd2JpsiKstar' in mode else r"p_T(K^+)",
  hminus_P = r"m(\pi^-)" if 'Bd2JpsiKstar' in mode else r"p(K^-)",
  hminus_PT = r"m(\pi^-)" if 'Bd2JpsiKstar' in mode else r"p_T(K^-)",
  )
  return ranges_dict[var]



def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())



def watermark(ax, version='final', tag='', size=(20,8.25), scale=1.2, pos=(0.05,0.9)):
  if version == 'final':
    version = 'LHC$b$'
    tag = ' COLLABORATION'
    size = [20,5.8]
  if scale:
    ax.set_ylim(ax.get_ylim()[0],ax.get_ylim()[1]*scale)
  # ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.03, ax.get_ylim()[1]*0.95,
  #         f'{version}', fontsize=size[0], color='black',
  #         ha='left', va='top', alpha=1.0)
  ax.text(pos[0], pos[1], f'{version}', fontsize=size[0], color='black',
          ha='left', va='center', transform = ax.transAxes, alpha=1.0)
  ax.text(pos[0], pos[1]-0.05, f'{tag}', fontsize=size[1], color='black',
          ha='left', va='center', transform = ax.transAxes, alpha=1.0)
  # ax.text(ax.get_xlim()[1]*0.025, ax.get_ylim()[1]*0.85,
  #         f'{tag}', fontsize=size[1], color='black',
  #         ha='left', va='top', alpha=1.0)
