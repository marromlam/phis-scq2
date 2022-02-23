As against the real data sets, the simulated samples do not used to contain S-wave
components, which means that all their final state hadrons originate from an
intermediate $`\phi`$ or $`K^{*}`$ meson. Furthermore, the relative phases and
fractions of the different polarization states do not necessarily agree with the
ones observed in real data. These two effects can cause differences in the angular
distributions of the final state particles, which might affect the decay-time acceptance. 
After running these three scripts one should be able to determine the decay-time acceptance. 

To run all reweigthings for all tuples in version v0r2, just type

.. code-block:: bash

    snakemake /path/to/phis_samples/v0r2/{2015,2016,2017,2018}/{MC_Bs2JpsiPhi,MC_Bs2JpsiPhi_dG0,MC_Bd2JpsiKstar,Bs2JpsiPhi,Bd2JpsiKstar}/DATEFLAG_kinWeight.root



PDF weighting
-------------

In order to correct these differences, the simulated samples are weighted to
match the S-wave and polarization fractions and phases measured in Run 1.
Every simulated event gets a weight that is determined by evaluating the
p.d.f. for this event once with the physics parameters set to the ones used
in the simulation (gen) and once with the parameters observed in data (obs).
The ratio of these two values is then used as the weight ($`w`$):

.. math::

    w = \frac{ p.d.f.( p_{\mathrm{obs}}) }{ p.d.f.( p_{\mathrm{gen}}) }.

The code that computes this weight, ``pdf-weighting.py``, requires some json
files where parameters of the simulated and observed samples. These parameter
files are stored in input folder, and their names do have to follow a
particular syntax that is
``discipline``-``mode``-``year``-``trigger``-`other``.json
where these wildcards mean:
* ``discipline``: `tad` (time-angular distribution), `dta` (decay-time acceptance) or `ang` (angular acceptance).
* ``mode``      : `Bs2JpsiPhi`, `MC_Bs2JpsiPhi`, `Bs2JpsiKstar` or `MC_Bs2JpsiKstar`
* ``year``      : `2011`, `2012`, `Run1`, `2015`, `2016` or `Run2`
* ``trigger``   : `biased`, `unbiased` or `both`.
* ``other``     : other flag (default one should be named baseline)

This script writes a new root file with all the input file branches plus a
new one named `pdfWeight`.



Polarity weighting
------------------

The MC samples are weighted to have the same fraction of the two magnet
polarities as the corresponding data sample. This script, 
`polarity-weighting.py`, writes a new root file with all the input file
branches plus a new one named `polWeight`.



Kinematic weighting
-------------------

Each of the samples (in bold) that will be used to compute the decay-time
acceptance has a kinematic weight aplied. To compute this kinematic weights
all the above must be avaliable beforehand.
The kinematic weighting relies on a *gb* reweighting technique where we use two
variables and a weight, that are sumarized in the table below.

+----------------------+------------+-----------------+------------------+
| Reweighting          | Branches   | Original        | Target           |
|                      |            | weight          | weight           |
+======================+============+=================+==================+
| ``BdMC`` to ``BsRD`` | ``pTB``,   | ``sWeight``     | ``sWeight``      |
|                      | ``pT``,    |                 |                  |
+----------------------+------------+-----------------+------------------+
| ``BsMC`` to ``BsRD`` | ``pT``,    | ``sWeight`` x   | ``sWeight``      |
|                      | ``mX``,    | ``polWeight`` x |                  |
|                      |            | ``pdfWeight`` x |                  |
+----------------------+------------+-----------------+------------------+
| ``BdMC`` to ``BdRD`` | ``pT``,    | ``sWeight`` x   | ``sWeight`` x    |
|                      | ``mX``,    | ``polWeight`` x | ``kinWeight``    |
|                      |            | ``pdfWeight`` x |                  |
+----------------------+------------+-----------------+------------------+

