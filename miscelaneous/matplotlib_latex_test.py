# matplotlib_latex_test
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


from pylab import *
rcParams["text.latex.preamble"] = r"\usepackage{concmath}"
figtext(.5, .5, r"\textrm{hello, world}", usetex=True)
savefig("/tmp/test.pdf")


# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
