Examples
========

This section provides practical examples of using LiLit for different types of analyses.

Basic Example: Temperature and Polarization
--------------------------------------------

Here's a simple example using temperature and E-mode polarization:

.. code-block:: python

   from lilit import LiLit

   # Define the configuration
   info = {
       'likelihood': {
           'lilit.LiLit': {
               'fields': ['t', 'e'],
               'lmax': [2500, 2000],
               'lmin': [2, 2],
               'fsky': [0.7, 0.7],
           }
       },
       'params': {
           # Cosmological parameters
           'H0': {'prior': {'min': 60, 'max': 80}, 'ref': 67.4},
           'omega_b': {'prior': {'min': 0.019, 'max': 0.025}, 'ref': 0.02237},
           'omega_cdm': {'prior': {'min': 0.09, 'max': 0.15}, 'ref': 0.1200},
           'tau_reio': {'prior': {'min': 0.01, 'max': 0.10}, 'ref': 0.0544},
           'A_s': {'prior': {'min': 1.7e-9, 'max': 3.5e-9}, 'ref': 2.1e-9},
           'n_s': {'prior': {'min': 0.92, 'max': 1.02}, 'ref': 0.9649},
       },
       'theory': {'camb': None},
       'sampler': {'evaluate': None}
   }

   from cobaya.model import get_model
   model = get_model(info)

B-mode Analysis with Tensor Modes
----------------------------------

Example for analyzing B-mode polarization with primordial tensor perturbations:

.. code-block:: python

   from lilit import LiLit

   # Configuration for B-mode analysis
   info = {
       'likelihood': {
           'lilit.LiLit': {
               'fields': ['t', 'e', 'b'],
               'lmax': [2500, 2000, 500],
               'lmin': [2, 2, 2],
               'fsky': [0.7, 0.7, 0.6],
               'r': 0.01,  # tensor-to-scalar ratio
           }
       },
       'params': {
           # Standard cosmological parameters
           'H0': {'prior': {'min': 60, 'max': 80}, 'ref': 67.4},
           'omega_b': {'prior': {'min': 0.019, 'max': 0.025}, 'ref': 0.02237},
           'omega_cdm': {'prior': {'min': 0.09, 'max': 0.15}, 'ref': 0.1200},
           'tau_reio': {'prior': {'min': 0.01, 'max': 0.10}, 'ref': 0.0544},
           'A_s': {'prior': {'min': 1.7e-9, 'max': 3.5e-9}, 'ref': 2.1e-9},
           'n_s': {'prior': {'min': 0.92, 'max': 1.02}, 'ref': 0.9649},
           # Tensor mode parameters
           'r': {'prior': {'min': 0, 'max': 0.1}, 'ref': 0.01},
           'n_t': {'derived': 'lambda r: -r/8'},  # consistency relation
       },
       'theory': {'camb': None},
       'sampler': {'evaluate': None}
   }

Custom Fiducial Spectra
------------------------

Using your own fiducial power spectra instead of Planck 2018 defaults:

.. code-block:: python

   import numpy as np
   from lilit import LiLit, CAMBres2dict
   import camb

   # Generate custom fiducial spectra with CAMB
   pars = camb.CAMBparams()
   pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.12)
   pars.InitPower.set_params(As=2.2e-9, ns=0.96)
   pars.set_for_lmax(3000, lens_potential_accuracy=0)

   results = camb.get_results(pars)
   powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
   
   # Convert to dictionary format
   fiducial_spectra = CAMBres2dict(results)

   # Use in LiLit
   info = {
       'likelihood': {
           'lilit.LiLit': {
               'fields': ['t', 'e'],
               'lmax': [2500, 2000],
               'lmin': [2, 2],
               'fsky': [0.7, 0.7],
               'fiducial_spectra': fiducial_spectra,
           }
       },
       # ... rest of configuration
   }

Running an MCMC Analysis
-------------------------

Complete example of running an MCMC chain:

.. code-block:: python

   info = {
       'likelihood': {
           'lilit.LiLit': {
               'fields': ['t', 'e', 'b'],
               'lmax': [2500, 2000, 500],
               'lmin': [2, 2, 2],
               'fsky': [0.7, 0.7, 0.6],
           }
       },
       'params': {
           'H0': {'prior': {'min': 60, 'max': 80}, 'ref': {'dist': 'norm', 'loc': 67.4, 'scale': 0.5}},
           'omega_b': {'prior': {'min': 0.019, 'max': 0.025}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.00037}},
           'omega_cdm': {'prior': {'min': 0.09, 'max': 0.15}, 'ref': {'dist': 'norm', 'loc': 0.1200, 'scale': 0.0036}},
           'tau_reio': {'prior': {'min': 0.01, 'max': 0.10}, 'ref': {'dist': 'norm', 'loc': 0.0544, 'scale': 0.0074}},
           'A_s': {'prior': {'min': 1.7e-9, 'max': 3.5e-9}, 'ref': {'dist': 'norm', 'loc': 2.1e-9, 'scale': 3e-11}},
           'n_s': {'prior': {'min': 0.92, 'max': 1.02}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.0042}},
           'r': {'prior': {'min': 0, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.01, 'scale': 0.01}},
       },
       'theory': {'camb': None},
       'sampler': {
           'mcmc': {
               'max_tries': 10000,
               'Rminus1_stop': 0.01,
               'Rminus1_cl_stop': 0.2,
           }
       },
       'output': 'chains/lilit_example'
   }

   from cobaya.run import run
   updated_info, sampler = run(info)

Using Different Sky Fractions per Field
----------------------------------------

LiLit allows you to specify different sky fractions for each field:

.. code-block:: python

   # Different sky coverage for T, E, and B modes
   info = {
       'likelihood': {
           'lilit.LiLit': {
               'fields': ['t', 'e', 'b'],
               'lmax': [3000, 2500, 600],
               'lmin': [2, 2, 2],
               'fsky': [0.8, 0.7, 0.4],  # T has larger sky coverage than E, B is most limited
           }
       },
       # ... rest of configuration
   }

This flexibility allows you to model realistic observational scenarios where different observational modes may have different systematic limitations or observing strategies.