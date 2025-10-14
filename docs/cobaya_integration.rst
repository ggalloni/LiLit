Integration with Cobaya
=======================

LiLit is designed to work seamlessly with `Cobaya <https://cobaya.readthedocs.io/>`_, the code for Bayesian analysis in cosmology. This section explains how to integrate LiLit into your Cobaya workflows.

Basic Cobaya Configuration
---------------------------

LiLit can be used directly in Cobaya input files. Here's the basic structure:

.. code-block:: yaml

   likelihood:
     lilit.LiLit:
       fields: ["t", "e", "b"]
       lmax: [2500, 2000, 500]
       lmin: [2, 2, 2]
       fsky: [0.7, 0.7, 0.6]

   params:
     # Your cosmological parameters
     H0:
       prior:
         min: 60
         max: 80
       ref:
         dist: norm
         loc: 67.4
         scale: 0.5

   theory:
     camb: null

   sampler:
     mcmc:
       max_tries: 10000

Working with CAMB
-----------------

LiLit integrates with CAMB through Cobaya's theory interface:

.. code-block:: yaml

   theory:
     camb:
       extra_args:
         halofit_version: mead
         AccuracyBoost: 1.2
         lSampleBoost: 1.2
         lAccuracyBoost: 1.2

   likelihood:
     lilit.LiLit:
       fields: ["t", "e"]
       lmax: [2500, 2000]
       lmin: [2, 2]
       fsky: [0.7, 0.7]

Working with CLASS
------------------

You can also use CLASS as your theory code:

.. code-block:: yaml

   theory:
     classy:
       extra_args:
         non_linear: halofit

   likelihood:
     lilit.LiLit:
       fields: ["t", "e", "b"]
       lmax: [2500, 2000, 500]
       lmin: [2, 2, 2]
       fsky: [0.7, 0.7, 0.6]

Parameter Definitions
---------------------

When working with tensor modes (B-mode analysis), you need to include the appropriate parameters:

.. code-block:: yaml

   params:
     # Standard cosmological parameters
     H0:
       prior: {min: 60, max: 80}
       ref: {dist: norm, loc: 67.4, scale: 0.5}
     
     omega_b:
       prior: {min: 0.019, max: 0.025}
       ref: {dist: norm, loc: 0.02237, scale: 0.00037}
     
     omega_cdm:
       prior: {min: 0.09, max: 0.15}
       ref: {dist: norm, loc: 0.1200, scale: 0.0036}
     
     tau_reio:
       prior: {min: 0.01, max: 0.10}
       ref: {dist: norm, loc: 0.0544, scale: 0.0074}
     
     A_s:
       prior: {min: 1.7e-9, max: 3.5e-9}
       ref: {dist: norm, loc: 2.1e-9, scale: 3e-11}
     
     n_s:
       prior: {min: 0.92, max: 1.02}
       ref: {dist: norm, loc: 0.9649, scale: 0.0042}
     
     # Tensor parameters (for B-mode analysis)
     r:
       prior: {min: 0, max: 0.1}
       ref: {dist: norm, loc: 0.01, scale: 0.01}
     
     n_t:
       derived: 'lambda r: -r/8'  # Consistency relation

Combining Multiple Likelihoods
-------------------------------

You can combine LiLit with other cosmological likelihoods:

.. code-block:: yaml

   likelihood:
     # LiteBIRD forecast
     lilit.LiLit:
       fields: ["t", "e", "b"]
       lmax: [2500, 2000, 500]
       lmin: [2, 2, 2]
       fsky: [0.7, 0.7, 0.6]
     
     # Add BAO measurements
     bao.sixdf_2011_bao: null
     bao.sdss_dr7_mgs: null
     
     # Add supernovae
     sn.pantheon: null

Custom Sampler Settings
-----------------------

Configure your sampler appropriately for the problem size:

.. code-block:: yaml

   sampler:
     mcmc:
       # Number of samples after burn-in
       max_tries: 100000
       
       # Convergence criteria
       Rminus1_stop: 0.01
       Rminus1_cl_stop: 0.2
       
       # Chain settings
       learn_every: 20
       temperature: 1
       
       # Proposal covariance
       covmat: auto
       proposal_scale: 2.4

Output Configuration
--------------------

Set up output appropriately:

.. code-block:: yaml

   output: chains/lilit_run

   # Optional: save additional info
   debug: false
   resume: true

Running from Python
--------------------

You can also run Cobaya with LiLit from Python:

.. code-block:: python

   from cobaya.run import run

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
           # ... your parameters
       },
       'theory': {'camb': None},
       'sampler': {'mcmc': {'max_tries': 10000}},
       'output': 'chains/my_run'
   }

   updated_info, sampler = run(info)

Using the Model Interface
-------------------------

For more control, use Cobaya's model interface:

.. code-block:: python

   from cobaya.model import get_model

   info = {
       # ... your configuration
   }

   model = get_model(info)

   # Evaluate likelihood at a point
   point = {
       'H0': 67.4,
       'omega_b': 0.02237,
       'omega_cdm': 0.1200,
       # ... other parameters
   }

   loglike = model.loglike(point)
   print(f"Log-likelihood: {loglike}")

   # Get derived parameters
   derived = model.loglike(point, return_derived=True)
   print(f"Derived parameters: {model.provider.get_param_dict()}")

This flexibility allows you to integrate LiLit into complex analysis pipelines and combine it with other cosmological data as needed.