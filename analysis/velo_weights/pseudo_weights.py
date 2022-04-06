__all__ = []
# from bisect import *
import os
from ipanema import initialize, ristra, IPANEMALIB
initialize('opencl', 1)
from timeit import default_timer as timer
import numpy as np
import pickle as cPickle
from scipy.special import erfinv
#from multiprocessing import Pool
from matplotlib import pyplot as plt
os.environ['PYOPENCL_NO_CACHE'] = '1'

# KolmogorovProb from scipy
from scipy.special import kolmogorov

def get_sizes(size, BLOCK_SIZE=256):
    '''
    i need to check if this worls for 3d size and 3d block
    '''
    a = size % BLOCK_SIZE
    if a == 0:
      gs, ls = size, BLOCK_SIZE
    elif size < BLOCK_SIZE:
      gs, ls = size, 1
    else:
      a = np.ceil(size/BLOCK_SIZE)
      gs, ls = a*BLOCK_SIZE, BLOCK_SIZE
    return int(gs), int(ls)


Gs = 0.6614 
DGs = 0.08543 
Gd = 1./ 1.519
Gddat = 1./1.520
Gudat = 1./1.638
Gu = 1./1.638
sf = 5367./5280
SAMPLE = "Bu"
#SW = "cor_sWeights_Bd"
SW = "truth_match"
if "dat" in SAMPLE: SW = "cor_sWeights_Bd"

dataDic = {}
k_TRUE = {"Bd":-Gd, "Bs": - Gs , "Bu": -Gu , "Bddat": -Gddat, "Budat": -Gudat}
# dataDic["Bd"] = cPickle.load(open("/scratch28/diego/b2cc/Bd_MC_helcut_" + SW + "_0.3ps.crap", 'rb'))
dataDic["Bu"] = cPickle.load(open("/scratch28/diego/b2cc/Bu_MC" + SW + "_0.3ps.crap", 'rb'))
print("/scratch28/diego/b2cc/Bu_MC" + SW + "_0.3ps.crap")
# dataDic["Bs"] = cPickle.load(open("/scratch28/diego/b2cc/Bs_MC" + SW + "_0.3ps.crap", 'rb'))
print("Data was loaded correctly")


BIN = 4
PTbins = [0, 3.8, 6, 9, 100]
tmin = 1
tmax = 15.
PTmin = 0#1000*PTbins[BIN-1]
PTmax = 1e9#1000*PTbins[BIN]

#BREAK





def fit_gpu(epsh_d, epsl_d, mup_doca, mum_doca, hp_doca, hm_doca, sw, pred):
    number_of_events = sw.shape[0]
    # create the kernel
    prog = THREAD.compile(
    f"#define DOCA_SIZE {number_of_events}" +
    """
    #define USE_DOUBLE 1
    #include <ipanema/core.c>
    #include <ipanema/stats.c>
    
    KERNEL
    void shit( GLOBAL_MEM float *out,
               GLOBAL_MEM float *epsh,     GLOBAL_MEM float *epsl,
               GLOBAL_MEM float *lp_docaz, GLOBAL_MEM float *lm_docaz,
               GLOBAL_MEM float *hp_docaz, GLOBAL_MEM float *hm_docaz,
               GLOBAL_MEM float *sw,       GLOBAL_MEM float *pred )
    {
      int idx = get_global_id(0);

      // compute efficiency
      float ws[DOCA_SIZE] = {0.};
      for (int k = 0; k<DOCA_SIZE; k++)
      {
        ws[k]  = (1. + epsl[idx] * lp_docaz[k] * lp_docaz[k]);
        ws[k] *= (1. + epsl[idx] * lm_docaz[k] * lm_docaz[k]);
        ws[k] *= (1. + epsh[idx] * hp_docaz[k] * hp_docaz[k]);
        ws[k] *= (1. + epsh[idx] * hm_docaz[k] * hm_docaz[k]);
      }

    
    
      // create weights
      for (int k = 0; k<DOCA_SIZE; k++) { ws[k] = 1.0 / ws[k] * sw[k]; }
     
      // cumsum
      for (int k = 1; k<DOCA_SIZE; k++) { ws[k] = ws[k-1] + ws[k]; }
      const float N0 = ws[DOCA_SIZE-1];
      for (int k = 0; k<DOCA_SIZE; k++) { ws[k] *= 1.0/N0; }
      
      // calculate distances
      double max_distance = 0.0;
      for (int k = 0; k<DOCA_SIZE; k++)
      {
        max_distance = max(max_distance, (double)fabs(ws[k] - pred[k]));
      }
      
      // build the probalbility 
      const double kstat = sqrt((double)DOCA_SIZE) * max_distance;
      const double kprob = kolmogorov_prob(kstat);

      // translate kprob into sort of chi2
      double ans = 0.0;
      if (kprob < 1e-7) { ans = 25. + kstat; }
      else { ans = 2. * rpow(erf_inverse(1-kprob), 2); }
      
      // save it
      out[idx] = (float)ans;

    }
    """, compiler_options=[f"-I{IPANEMALIB}"])
    
    # create output array
    out = epsl_d * 0.0
    # prepare block sizes and launch the kernel
    g_size, l_size = get_sizes(out.shape[0],1)
    prog.shit(out, epsh_d, epsl_d, mup_doca, mum_doca, hp_doca, hm_doca, sw, pred,
              # global_size=(1,))
              global_size=g_size, local_size=l_size)
    chi2_array = out.get().reshape(len(x), len(y)).T
    print("GPU kernel min chi2 =", np.min(out.get()))
    return chi2_array




# merdisima {{{
def do(SAMPLE):
    print("Selected sample:", SAMPLE)
    data = dataDic[SAMPLE]
    data.sort()
    data = np.float32(data)
    print("Data allocated")
    mask = (( data[:,0] > tmin ) * ( data[:,0] < tmax)*( data[:,6] > PTmin ) *( data[:,6] < PTmax ) )
    t_gpu = THREAD.to_device(np.float32(np.extract(mask,data[:,0])))
    mup_DZ_gpu = THREAD.to_device(np.float32(np.extract(mask, data[:,1])))
    mum_DZ_gpu = THREAD.to_device(np.float32(np.extract(mask,data[:,2])))
    hp_DZ_gpu = THREAD.to_device(np.float32(np.extract(mask,data[:,3])))
    hm_DZ_gpu = THREAD.to_device(np.float32(np.extract(mask, data[:,4])))

    # theoretical prediction
    G = k_TRUE[SAMPLE]
    sw_gpu = THREAD.to_device(np.float32(np.extract(mask, data[:,5])))
    expo_int = np.exp(G*t_gpu.get()) - np.exp(G*tmin)
    expo_int *= 1./(np.exp(G*tmax)-np.exp(G*tmin))
    expo_int = THREAD.to_device(expo_int)
    # print(expo_int)
    # print(mup_DZ_gpu, mum_DZ_gpu, hp_DZ_gpu, hm_DZ_gpu, sw_gpu, expo_int)

    Ntot = sum(mask)
    nevt = Ntot
    print("doca size", nevt)
    
    prog = THREAD.compile(
    f"#define DOCA_SIZE {nevt}" +
    """
    #define USE_DOUBLE 0
    #include <ipanema/core.c>
    #include <ipanema/stats.c>
    
    KERNEL
    void shit( GLOBAL_MEM ftype *out,
    GLOBAL_MEM const ftype *epsh,     GLOBAL_MEM const ftype *epsl,
    GLOBAL_MEM const ftype *lp_docaz, GLOBAL_MEM const ftype *lm_docaz,
    GLOBAL_MEM const ftype *hp_docaz, GLOBAL_MEM const ftype *hm_docaz,
    GLOBAL_MEM const ftype *sw,       GLOBAL_MEM const ftype *pred
    )
    {
      const int idx = get_global_id(0);

      // compute efficiency
      ftype eff[DOCA_SIZE];
      for (unsigned int k = 0; k<DOCA_SIZE; k++)
      {
        eff[k]  = (1. + epsl[idx] * lp_docaz[k] * lp_docaz[k]);
        eff[k] *= (1. + epsl[idx] * lm_docaz[k] * lm_docaz[k]);
        eff[k] *= (1. + epsh[idx] * hp_docaz[k] * hp_docaz[k]);
        eff[k] *= (1. + epsh[idx] * hm_docaz[k] * hm_docaz[k]);
      }
    
    
      // create weights
      ftype ws[DOCA_SIZE] = {0.};
      ftype cws[DOCA_SIZE] = {0.};
      for (unsigned int k = 0; k<DOCA_SIZE; k++)
      {
        ws[k] = 1.0 / eff[k] * sw[k];
        // printf("ws[%d] = %f\\n", k, ws[k]);
        if (idx == 0) printf("ws[%d] = %f\\n", k, ws[k]);
      }
    
      
      // cumsum
      cws[0] = ws[0];
      for (int k = 1; k<DOCA_SIZE; k++)
      {
        cws[k] = cws[k-1] + ws[k];
      }
      const ftype N0 = cws[DOCA_SIZE-1];
      // printf("N0 = %f\\n",N0);
      for (unsigned int k = 0; k<DOCA_SIZE; k++)
      {
        cws[k] *= 1.0/N0;
      }
    
      // calculate distances
      ftype distances[DOCA_SIZE];
      ftype max_distance = 0.0;
      for (unsigned int k = 0; k<DOCA_SIZE; k++)
      {
        distances[k] = fabs(cws[k] - pred[k]);
        max_distance = max(max_distance, fabs(cws[k] - pred[k]));
        if (idx == 0) printf("dist[%d] = %f\\n", k, distances[k]);
      }
    
    
      const ftype kstat = sqrt((float)DOCA_SIZE) * max_distance;
      const ftype kprob = kolmogorov_prob(kstat);
    
      // translate kprob into sort of chi2
      if (kprob < 1e-7)
      {
        out[idx] = (float) (25 + kstat);
      }
      else
      {
        out[idx] = (float) 2 * pow(erf_inverse(1-kprob), 2);
      }
    }
    """, compiler_options=[f"-I{IPANEMALIB}"]
    )

    print("ready to go")

    # @np.vectorize
    def FCN(epsmu, epsh):
      # compute efficiency
      eff = (1. + epsmu*mup_DZ_gpu*mup_DZ_gpu) 
      eff *= (1. + epsmu*mum_DZ_gpu*mum_DZ_gpu)
      eff *= (1. + epsh*hp_DZ_gpu*hp_DZ_gpu)
      eff *= (1. + epsh*hm_DZ_gpu*hm_DZ_gpu)
      # print(eff)
      print(eff)
      ws = 1./eff*sw_gpu
      print(ws)
      # ////
      cws =  ristra.cumsum(ws)
      N0 = np.float32(cws[-1].get())
      cws *= 1./N0

      # build a KS stat
      distances = cws - expo_int
      print(distances)
      max_distance = np.max(np.abs(ristra.get(distances)))
      kstat = np.sqrt(Ntot) * max_distance
      kprob = kolmogorov(kstat)

      # translate kprob into sort of chi2
      if kprob < 1e-7:
        chi2 = 25 + kstat 
      else:
        chi2 = 2 * erfinv(1-kprob)**2
      print("chi2 =", chi2)
      prog.shit(pipas, epsh_d, epsl_d, mup_DZ_gpu, mum_DZ_gpu, hp_DZ_gpu, hm_DZ_gpu, sw_gpu, expo_int, 
              global_size=(len(epsl_d),))
      print("pipas = ", pipas)
      exit()
      return chi2

    def contFCN(pair):
      return FCN(pair[0], pair[1])

    # return chi2_array 
    chi2list = list(map(contFCN, pairs))
    # chi2list = FCN(pairs[:,0], pairs[:,1])
    chi2_array = np.array(chi2list).reshape(len(x), len(y)).T
    prog.shit(pipas, epsh_d, epsl_d, mup_DZ_gpu, mum_DZ_gpu, hp_DZ_gpu, hm_DZ_gpu, sw_gpu, expo_int, 
              global_size=(len(mup_DZ_gpu),))
    print(chi2_array)
    print(pipas.get().reshape(len(x), len(y)).T)
    print("min gpu chi2 =", np.min(pipas.get()))
    return chi2_array
# }}}





if __name__ == '__main__':
  start = timer()
  
  SAMPLE = 'Bu'
  print("Selected sample:", SAMPLE)
  # samples from diego {{{
  data = cPickle.load(open("/scratch28/diego/b2cc/Bu_MC" + SW + "_0.3ps.crap", 'rb'))
  data.sort()
  data = np.float32(data)
  print("Data allocated")
  mask = (( data[:,0] > tmin ) * ( data[:,0] < tmax)*( data[:,6] > PTmin ) *( data[:,6] < PTmax ) )
  time     = THREAD.to_device(np.float32(np.extract(mask,data[:,0])))
  lp_docaz = THREAD.to_device(np.float32(np.extract(mask, data[:,1])))
  lm_docaz = THREAD.to_device(np.float32(np.extract(mask, data[:,2])))
  hp_docaz = THREAD.to_device(np.float32(np.extract(mask, data[:,3])))
  hm_docaz = THREAD.to_device(np.float32(np.extract(mask, data[:,4])))
  sw       = THREAD.to_device(np.float32(np.extract(mask, data[:,5])))
  # }}}
  
  # create meshgrid {{{

  delta = 1e-2
  x = np.arange(-0.02, 0.02, delta)
  y = np.arange(-0.02, 0.02, delta)
  X,Y = np.meshgrid(x,y)
  
  pairs = np.array([Y.ravel(), X.ravel()]).T
  epsl_d = THREAD.to_device(np.float32(pairs[:,0]))
  epsh_d = THREAD.to_device(np.float32(pairs[:,1]))
  chi2_d = THREAD.to_device(np.float32(pairs[:,1])) * 0.0
  print("Meshgrid was created:", pairs.shape)
  # }}}


  # theoretical prediction {{{
  G = k_TRUE[SAMPLE]
  predict = np.exp(G*time.get()) - np.exp(G*tmin)
  predict *= 1./(np.exp(G*tmax)-np.exp(G*tmin))
  predict = THREAD.to_device(np.float32(predict))
  
  # Get number of events
  nevt = len(time) #1000000
  print("Event size:", nevt)
  # }}}
  
  
  Z = fit_gpu(epsh_d, epsl_d, lp_docaz, lm_docaz, hp_docaz, hm_docaz, sw, predict)
  # print(Z)
  # Z = do("Bu") #+ do("Bd") + do("Bs")
  # ZZ = np.array(Z)
  print("time:" , timer() - start)
  #BREAK
  # Zlist = list(Z)
  # print("shit =", Zlist)
  # v = []
  # for shit in Zlist:
      # v.append(min(shit))
  # print("v =", v)
  # minchi2 = min(v)
  # print(minchi2)
  minchi2 = np.min(Z)
  print("Minimum chi2 =", minchi2)
  
  if minchi2 > 4:
      print("WARNING: Look at minchi2")
  
  # Contour plot
  cs = plt.contour(X,Y,Z, levels = [minchi2 +  2*1.15, minchi2 + 5.99,minchi2 + 11.8 ]) ## 4 is 1.5 sigma
  plt.clabel(cs, inline=1, fontsize=10)
  plt.xlabel("epsmu")
  plt.ylabel("epsh")
  plt.savefig("banana_plot_pseudo_velo_weights.png")
  plt.show()
  
