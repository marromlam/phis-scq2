import ipanema
import numpy as np
import complot

from analysis.angular_acceptance import bdtconf_tester
import matplotlib.pyplot as plt

from utils.plot import watermark, make_square_axes


def plot_bdtres(bdtvar, param):
  fig, ax = complot.axes_plot()
  idx1 = names[bdtvar]; idx2 = names[param]
  print(idx1, idx2)
  u = pars[param].uvalue
  minX = bdtpars[:,idx1][ 0]*0.9
  maxX = max(bdtpars[:,idx1])*1.1
  minX = min(bdtpars[:,idx1]) - max(bdtpars[:,idx1])*0.1
  _alp = 1
  ax.fill_between([minX, maxX], # bdt var range
                   [u.n+_alp*u.s,u.n+_alp*u.s], [u.n-_alp*u.s,u.n-_alp*u.s], # nominal conf interv
                   alpha=0.2, label=f'nominal ${_alp}\sigma$ c.b.')
  ax.plot([minX, maxX],[u.n,u.n],color='C0', label='nominal')
  ax.fill_between([minX, maxX], # bdt var range
                   [np.mean(bdtpars[:,idx2])+np.std(bdtpars[:,idx2]),
                    np.mean(bdtpars[:,idx2])+np.std(bdtpars[:,idx2])],
                   [np.mean(bdtpars[:,idx2])-np.std(bdtpars[:,idx2]),
                    np.mean(bdtpars[:,idx2])-np.std(bdtpars[:,idx2])],
                   alpha=0.4, facecolor = 'C2', label='bdt dispersion')
  ax.plot(bdtpars[:,idx1],bdtpars[:,idx2], 'C2.', label='bdt tests')
  ax.set_ylabel(f"${pars[param].latex}$")
  ax.set_xlabel(f"$\\verb|{bdtvar}|$")
  ax.legend()
  watermark(ax, version=f"${version}$", scale=1.0)
  make_square_axes(ax)

  return fig, ax



if __name__ == '__main__':

  # parse_args
  version = 'v0r5'
  num_of_tests = 10

  # load nominal parameters
  _general_parameters  = "output/params/physics_params/run2/Bs2JpsiPhi/v2r0@LcosK_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.json"
  pars = ipanema.Parameters.load("output/params/physics_params/run2/Bs2JpsiPhi/v2r0@LcosK_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.json")

  # load all bdt parameters
  pp = "output/params/physics_params/run2/Bs2JpsiPhi/v2r0bdt{i}@LcosK_auto_run2Dual_vgc_amsrd_simul3_amsrd_combined.json"
  bdtpars = []
  # for i in range(0,109):
  for i in range(1+num_of_tests+1):
    # try:
      _item = list(bdtconf_tester.bdtmesh(i, num_of_tests, verbose=False).values())[:-1]
      _pars = ipanema.Parameters.load(pp.format(i=i))
      _item += list(_pars.valuesdict().values())
      if abs(_item[31]) < 0.0078:
        print('yes!')
        print(_item[31])
        bdtpars.append(_item)
    # except:
    # 0
  bdtpars = np.array(bdtpars)

  # create naming protocol
  _names = list(bdtconf_tester.bdtmesh(1, num_of_tests, verbose=False).keys())[:-1]
  _names += list(map(lambda x: pars[x].name, pars.valuesdict().keys()))

  names = {}
  for k,v in enumerate(_names):
    names[v] = k

  for bdtvar in list(names.keys())[:4]:
    for param in list(names.keys())[4:]:
      if pars[param].free:
         _bdtvar = "".join(bdtvar.split('_'))
         _param = "".join(param.split('_'))
         fig, ax = plot_bdtres(bdtvar, param)
         _filename = _general_parameters.replace('output/params', 'output/figures')
         _filename = _filename.replace('.json', f'_{_bdtvar}_{_param}.pdf')
         fig.savefig(_filename)
