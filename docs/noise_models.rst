Noise Models and Fiducial Spectra
=================================

LiLit requires noise power spectra to compute realistic forecasts. This section explains how noise is handled and how to provide custom fiducial spectra.

Default Noise Placeholder
--------------------------

As mentioned in the README, LiLit includes a placeholder noise implementation using inverse-weighted noise over LiteBIRD channels, based on `Campeti et al. (2020) <https://arxiv.org/abs/2007.04241>`_ and channel specifications from `Allys et al. (2022) <https://arxiv.org/abs/2202.02773>`_.

**Important:** This is only a placeholder. For realistic forecasting, you should provide proper LiteBIRD noise power spectra.

Providing Custom Noise Spectra
-------------------------------

You should provide realistic noise power spectra through the ``nl_file`` parameter:

.. code-block:: python

   from lilit import LiLit

   # Provide your own noise file
   likelihood = LiLit(
       name="realistic_forecast",
       fields=["t", "e", "b"], 
       like="exact",
       nl_file="/path/to/litebird_noise.pkl",  # Your noise file
       lmax=[2500, 2000, 500],
       lmin=[2, 2, 2],
       fsky=[0.7, 0.7, 0.6]
   )

Expected Noise File Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The noise file should contain the noise power spectra :math:`N_\ell` for the fields you're using. The exact format depends on your implementation, but typically includes:

- Temperature noise: :math:`N_\ell^{TT}`
- E-mode polarization noise: :math:`N_\ell^{EE}`  
- B-mode polarization noise: :math:`N_\ell^{BB}`
- Cross-correlation noise (if any): :math:`N_\ell^{TE}`

Default Fiducial Spectra
-------------------------

If you don't provide custom fiducial power spectra, LiLit will internally compute spectra using **Planck 2018 best-fit cosmological parameters**.

For B-mode analysis, you must specify the tensor-to-scalar ratio:

.. code-block:: python

   # B-mode analysis requires specifying r
   likelihood = LiLit(
       name="tensor_forecast",
       fields=["t", "e", "b"],
       like="exact", 
       r=0.01,  # Tensor-to-scalar ratio
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

The tensor spectral index :math:`n_t` will follow the standard consistency relation :math:`n_t = -r/8` if not specified.

Custom Fiducial Spectra
------------------------

You can provide your own fiducial spectra instead of using Planck 2018 values:

.. code-block:: python

   import numpy as np
   from lilit import LiLit, CAMBres2dict
   import camb

   # Generate custom fiducial spectra with CAMB
   pars = camb.CAMBparams()
   pars.set_cosmology(H0=70, ombh2=0.022, omch2=0.12)
   pars.InitPower.set_params(As=2.2e-9, ns=0.96, r=0.01)
   pars.set_for_lmax(3000, lens_potential_accuracy=0)

   results = camb.get_results(pars)
   
   # Convert CAMB results to the format expected by LiLit
   fiducial_spectra = CAMBres2dict(results)

   # Use custom fiducial spectra
   likelihood = LiLit(
       name="custom_fiducial",
       fields=["t", "e", "b"],
       like="exact",
       fiducial_spectra=fiducial_spectra,  # Your custom spectra
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

The CAMBres2dict Function
~~~~~~~~~~~~~~~~~~~~~~~~~

LiLit provides the ``CAMBres2dict`` utility function to convert CAMB results into the dictionary format used by LiLit:

.. code-block:: python

   from lilit import CAMBres2dict
   
   # Convert CAMB results
   cl_dict = CAMBres2dict(camb_results)
   
   # cl_dict now contains the power spectra in LiLit format

Noise Requirements for LiteBIRD
--------------------------------

For accurate LiteBIRD forecasts, your noise model should account for:

1. **Instrumental noise** from detector sensitivity
2. **Beam effects** and resolution 
3. **Atmospheric noise** (for ground-based observations)
4. **Systematic uncertainties** 
5. **Foreground residuals** after component separation

**Recommendation:** Contact the LiteBIRD collaboration for official noise specifications and recommended noise models for forecasting studies.

Multipole-Dependent Sky Coverage
---------------------------------

LiLit supports different sky fractions for different fields, which can represent:

- Different systematic limitations per observational mode
- Varying foreground contamination levels
- Different observing strategies

.. code-block:: python

   # Different sky coverage per field
   likelihood = LiLit(
       fields=["t", "e", "b"],
       lmax=[3000, 2500, 600],
       fsky=[0.8, 0.7, 0.4],  # T > E > B in sky coverage
       # ... other parameters
   )

Note on Sky Cut Approximation
-----------------------------

LiLit approximates partial sky effects by rescaling the effective number of modes with :math:`f_{sky}`. This approach:

- **Does account for:** Reduced statistical power due to smaller sky coverage
- **Does not account for:** Mode coupling between different multipoles introduced by the sky cut

For more sophisticated treatments of partial sky effects, consider using specialized codes that handle the full mode coupling.

Validation
----------

Always validate that your noise model produces reasonable results:

.. code-block:: python

   # Enable debug mode to check setup
   likelihood = LiLit(
       name="validation_run",
       fields=["t", "e", "b"],
       debug=True,  # Shows diagnostic information
       lmax=[1000, 800, 200],
       fsky=[0.7, 0.7, 0.6]
   )

The debug output will help you verify that the noise levels and multipole ranges are set up correctly.