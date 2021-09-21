import argparse
import os
import ipanema
import matplotlib.pyplot as plt
import uncertainties as unc
import numpy as np
from scipy.stats import chi2

from ipanema import Parameters


import matplotlib as mpl
def mergecells(table, cells):
    '''
    Merge N matplotlib.Table cells

    Parameters
    -----------
    table: matplotlib.Table
        the table
    cells: list[set]
        list of sets od the table coordinates
        - example: [(0,1), (0,0), (0,2)]

    Notes
    ------
    https://stackoverflow.com/a/53819765/12684122
    '''
    cells_array = [np.asarray(c) for c in cells]
    h = np.array([cells_array[i+1][0] - cells_array[i][0] for i in range(len(cells_array) - 1)])
    v = np.array([cells_array[i+1][1] - cells_array[i][1] for i in range(len(cells_array) - 1)])

    # if it's a horizontal merge, all values for `h` are 0
    if not np.any(h):
        # sort by horizontal coord
        cells = np.array(sorted(list(cells), key=lambda v: v[1]))
        edges = ['BTL'] + ['BT' for i in range(len(cells) - 2)] + ['BTR']
    elif not np.any(v):
        cells = np.array(sorted(list(cells), key=lambda h: h[0]))
        edges = ['TRL'] + ['RL' for i in range(len(cells) - 2)] + ['BRL']
    else:
        raise ValueError("Only horizontal and vertical merges allowed")

    for cell, e in zip(cells, edges):
        table[cell[0], cell[1]].visible_edges = e
        
    txts = [table[cell[0], cell[1]].get_text() for cell in cells]
    tpos = [np.array(t.get_position()) for t in txts]

    # transpose the text of the left cell
    trans = (tpos[-1] - tpos[0])/2
    # didn't had to check for ha because I only want ha='center'
    txts[0].set_transform(mpl.transforms.Affine2D().translate(*trans))
    for txt in txts:
        txt.set_visible(True)


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--params')
  p.add_argument('--param')
  p.add_argument('--figure')
  p.add_argument('--mode')
  p.add_argument('--version')
  args = vars(p.parse_args())

  __pars = args['params'].split(',')
  _par = args['param']

  # create a dict with all information
  fd = {}; pval = {}
  splitter = 0; ntests = -1
  d = {}; _ntest = 0
  for i, path in enumerate(__pars):
    if 'splitter' not in path:
      _parameter = Parameters.load(path)[_par]
      _tex = _parameter.latex
      _parameter = [_parameter.uvalue.n, _parameter.uvalue.s]
      _label = os.path.basename(path).split('.')[0].replace('_',' ')
      if 'yearly' in _label:
        if '2015' in path:
          _label = _label.replace('yearly', '2015')
        elif '2016' in path:
          _label = _label.replace('yearly', '2016')
        elif '2017' in path:
          _label = _label.replace('yearly', '2017')
        elif '2018' in path:
          _label = _label.replace('yearly', '2018')
      _color = f"C{splitter}"
      d[_ntest] = dict(tex=_tex, y=ntests+1, value=_parameter, label=_label, color=_color)
      _ntest += 1
      ntests += 1
    else:
      d = {}; _ntest = 0
      splitter +=1
    fd[splitter] = d
    pval[splitter] = 0
  
  # print(fd)
  # print(fd)
  fd_pvals = {}
  base = unc.ufloat(*fd[0][0]['value'])
  for k in range(1,splitter+1):
    _test = [unc.ufloat(*v['value']) for v in fd[k].values()]
    _xi = [v.n for v in _test]
    _wi = [v.s**(-2) for v in _test]
    # weighted average
    avg = 0; uavg = 0
    for i in range(len(_xi)):
      avg += _xi[i] * _wi[i] 
      uavg += _wi[i]
    avg /= uavg; uavg = 1/np.sqrt(uavg)
    # print(_test)
    # print(avg, uavg)
    # compute chi2 and pvalue
    chi2val = (base.n - avg)**2 / (base.s**2 + uavg**2)
    pval = chi2.sf(chi2val, 1)
    fd_pvals[k] = f"x2 = {chi2val:.4f}  pval = {pval:.2f}"
    # print("chi2, pval = ", chi2val, pval)


  # create figure
  fig = plt.figure(figsize=(6,12))
  ax = fig.add_subplot(111)
  ax.set_xlabel(rf"${d[0]['tex']}$")
  ax.set_xlim([fd[0][0]['value'][0]-30*fd[0][0]['value'][1],
               fd[0][0]['value'][0]+30*fd[0][0]['value'][1]])
  ax.set_ylim(0.5, ntests+0.5)
  # plot nominal confidence band
  ax.fill_between(
    [fd[0][0]['value'][0]-fd[0][0]['value'][1], fd[0][0]['value'][0]+fd[0][0]['value'][1]], # x
    [0, 0], # y_low
    [ntests+1, ntests+1], # y_up
    alpha = 0.2
  )

  # plot all errobars
  for ik, iv in fd.items():
    if ik != 0:
      for jk, jv in iv.items():
        # if jk == 0: # label first test with the pval and chi2
        #   ax.text(
        #     # position text relative to data
        #     fd[0][0]['value'][0] + 30*fd[0][0]['value'][1], jv['y'],
        #     fd_pvals[ik],  # x, y, text,
        #     ha='left', va='bottom',   # text alignment,
        #     fontsize = 4,
        #     transform=ax.transData      # coordinate system transformation
        #   )
        ax.errorbar(jv['value'][0], jv['y'], xerr=jv['value'][1],
                    color=jv['color'], fmt='.')

  # ticks
  # plt.set_yticks(
  #   list(range(0,ntests+1)),
  #   [''] + [d[i]['label'] for i in d][1:] + ['']
  # )

  all_labels = []
  all_data = []
  for ik, iv in fd.items():
    for jk, jv in iv.items():
      all_labels.append(jv['label'])
      all_data.append(jv['label'].split(' '))
      if jk == 0 and ik != 0: # label first test with the pval and chi2
        all_data[-1].append(fd_pvals[ik])
      else:
        all_data[-1].append('')
  all_labels[0] = ''
  # all_data[0] = ['', '', '', '', '', '', '', '']
  all_labels.append('')
  all_data = all_data[1:]
  # all_data.append(['', '', '', '', '', '', '', ''])
  # ax.set_yticks(list(range(len(all_labels))))
  # ax.set_yticklabels(all_labels, fontsize=7)
  
  print(all_data)

  # table form
  the_table = ax.table(cellText=all_data[::-1],
                       rowLabels=list(range(len(all_data))),
                       colLabels=['     version     ', '   fit   ', '    AA    ', '   CSP   ', '     FT     ', '     TA     ', '     TR    ', '    trig    ', '       stat.       '],
                       # colWidths=[0.1,2.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],
                       cellLoc='center',
                       # edges='horizontal',
                       bbox=(1.2, 0.0, -0.2, (len(all_data)+1) / len(all_data)))
  mergecells(the_table, [(1,8), (2,8), (3,8), (4,8)])
  the_table.auto_set_font_size(False)
  the_table.auto_set_column_width(-1)
  the_table.set_fontsize(8)

  # print(  list(range(0,ntests+1)))
  # print(all_labels)
  # ax.set_yticks(list(range(0,ntests+1)))
  # ax.set_yticklabels(all_labels, fontsize=7)

  # enforce aspect ratio
  # extent =  ax.get_images().get_extent()
  # ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/(2))
  # ax.set_aspect(1 / ax.get_data_ratio())

  ax.tick_params(
      axis='y',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      left=False,      # ticks along the bottom edge are off
      right=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off
  ax.axes.get_yaxis().set_visible(False)
  # save 
  fig.savefig(args['figure'])

