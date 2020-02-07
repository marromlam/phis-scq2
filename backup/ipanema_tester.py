from ipanema import Parameter, Parameters


a = Parameter('a',23.423,stdev=3.6545)
b = Parameter('b',3.240,stdev=0.876)

c = Parameters()
c.add(a,b)
meh = a.uvalue
a.uvalue-a.uvalue

a.dumps_latex()

a-a
meh-meh
c = Parameters.load('decay_time_acceptance/input/params-full-baseline.json')
meh._nominal_value = 2
meh._std_dev = 0.2
meh
dir(meh)

def f(x,a,b):
  return a*x+b

f(1,a,b)

c.latex_dumps()
c.print()

a-a

a.uvalue-a.uvalue
a.dumps_latex()

a.unc_round
a.set(expr='b**2')

a.expr



dir(a)
a.set(3.2345)
a.latex+'='+a.dumps_latex()
