Installation
============


Environment
-----------


.. code-block:: bash

   wget conda -o miniconda.sh
   bash miniconda.sh
   source conda3/bin/activate
   conda env create -f installation/environment_linux.sh
   conda activate phisscq
   pip install -r installation/requirements.txt


If there is an nVidia device installed, 

.. code-block:: bash

   pip install pycuda==2020.1



.. code-block:: bash

   ./installation/environment.sh
   ./installation/espresso.sh


Espresso Monitor
----------------


.. code-block:: bash

   bash ./Installation/espresso.sh
