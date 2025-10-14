Quick Start Guide
=================

Basic Usage
-----------

LiLit is designed to be easy to use with Cobaya. Here's a basic example:

.. code-block:: python

   from lilit import LiLit

   # Define the fields you want to use
   fields = ["t", "e", "b"]

   # Set multipole ranges for each field
   lmax = [1500, 1200, 900]  # [lmaxTT, lmaxEE, lmaxBB]
   lmin = [20, 2, 2]         # [lminTT, lminEE, lminBB]
   fsky = [1.0, 0.8, 0.6]    # [fskyTT, fskyEE, fskyBB]

   # Create the likelihood
   likelihood = LiLit(fields=fields, lmax=lmax, lmin=lmin, fsky=fsky)

Working with Different Field Combinations
-----------------------------------------

LiLit is flexible and supports different field combinations:

Temperature Only
~~~~~~~~~~~~~~~~

.. code-block:: python

   from lilit import LiLit

   likelihood = LiLit(
       fields=["t"],
       lmax=[1500],
       lmin=[20],
       fsky=[1.0]
   )

Temperature + Polarization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lilit import LiLit

   likelihood = LiLit(
       fields=["t", "e"],
       lmax=[1500, 1200],
       lmin=[20, 2],
       fsky=[1.0, 0.8]
   )

B-modes Only
~~~~~~~~~~~~

.. code-block:: python

   from lilit import LiLit

   likelihood = LiLit(
       fields=["b"],
       lmax=[900],
       lmin=[2],
       fsky=[0.6],
       r=0.01  # tensor-to-scalar ratio for B-modes
   )

Using Custom Fiducial Spectra
------------------------------

If you want to provide your own fiducial power spectra instead of using the default Planck 2018 values:

.. code-block:: python

   import numpy as np
   from lilit import LiLit

   # Generate your custom spectra (example)
   ells = np.arange(2, 1001)
   cl_tt = your_tt_spectrum(ells)
   cl_ee = your_ee_spectrum(ells)
   cl_bb = your_bb_spectrum(ells)

   fiducial_spectra = {
       'tt': cl_tt,
       'ee': cl_ee,
       'bb': cl_bb
   }

   likelihood = LiLit(
       fields=["t", "e", "b"],
       lmax=[1500, 1200, 900],
       lmin=[20, 2, 2],
       fsky=[1.0, 0.8, 0.6],
       fiducial_spectra=fiducial_spectra
   )

Integration with Cobaya
-----------------------

LiLit is designed to work seamlessly with Cobaya. Here's an example configuration:

.. code-block:: yaml

   likelihood:
     lilit.LiLit:
       fields: ["t", "e", "b"]
       lmax: [1500, 1200, 900]
       lmin: [20, 2, 2]
       fsky: [1.0, 0.8, 0.6]
       r: 0.01

   params:
     # Your cosmological parameters here
     H0:
       prior:
         min: 60
         max: 80
       ref:
         dist: norm
         loc: 67.4
         scale: 0.5
     # ... other parameters

Utility Functions
-----------------

LiLit also provides utility functions for working with CAMB results:

.. code-block:: python

   from lilit import CAMBres2dict
   import camb

   # Get CAMB results
   pars = camb.CAMBparams()
   # ... set up parameters
   results = camb.get_results(pars)

   # Convert to dictionary format
   cl_dict = CAMBres2dict(results)

This function converts CAMB results into a dictionary format that's easy to work with in your analysis.