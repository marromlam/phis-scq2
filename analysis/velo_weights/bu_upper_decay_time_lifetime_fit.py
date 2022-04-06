from ipanema import (initialize, Parameters, optimize, Sample)
import uproot3 as uproot


__all__ = []

if __name__ == "__main__":
  # ---
  initialize('cuda',1)
  # load sample and allocate 
  branches = ['time', 'docaz_hplus']
  s = Sample.from_root("/scratch17/marcos.romero/Bs2JpsiPhi-FullRun2/selection/output/tuples/MC_Bu2JpsiKplus/MC_Bu2JpsiKplus_2016_selected_bdt.root", branches=branches)
  print(s)

  # let's get 
  pars_velo = Parameters.load("output/params/time_acceptance/2016/MC_Bu2JpsiKplus/v0r5_francesca_cb.json")
  pars_velo = pars_velo.valuesdict()
  print(pars_velo)
   
  def francesca(x, p):
    return p['a'] * (1 + p['c'] * x**2)





  # allocate 
  s.df.eval("weight=@get_vw8s(docaz_hplus)", inplace=True)
  s.allocate(time="time", weight="weight", prob="0*time")

  # create fit paramters set
  fpars = Parameters()
  fpars.add(dict(name='a', value=1, min=-3, max=3, free=True))
  fpars.add(dict(name='b', value=1, min=-3, max=3, free=True))
  fpars.add(dict(name='gamma', value=0.610500610500610, free=False))


  def fcn(pars, data):
    p = pars.valuesdict()
    pdf = ristra.exp(-p['gamma']*data.time) 
    pdf *= (1 + p['a']*data.time + p['b']*data.time*data.time)
    lkhd = 0
    return ristra.get(-2*lkhd)
