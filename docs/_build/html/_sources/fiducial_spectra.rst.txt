Fiducial Spectra
================

LiLit needs fiducial (theoretical) power spectra to compute forecasts. This section explains how these are handled and how to customize them.

Default: Planck 2018 Best-Fit
------------------------------

If you don't specify custom fiducial spectra, LiLit automatically computes them using **Planck 2018 best-fit cosmological parameters**. This provides a reasonable baseline for most forecasting studies.

The default cosmological parameters used are:

- :math:`H_0 = 67.36` km/s/Mpc
- :math:`\Omega_b h^2 = 0.02237`
- :math:`\Omega_{cdm} h^2 = 0.1200`
- :math:`\tau_{reio} = 0.0544`
- :math:`A_s = 2.1 \times 10^{-9}` (at :math:`k_0 = 0.05` Mpc⁻¹)
- :math:`n_s = 0.9649`

For B-mode Analysis
~~~~~~~~~~~~~~~~~~~

When analyzing B-mode polarization, you must specify the tensor-to-scalar ratio:

.. code-block:: python

   from lilit import LiLit

   likelihood = LiLit(
       fields=["t", "e", "b"],
       r=0.01,  # Must specify r for B-modes
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

The tensor spectral index :math:`n_t` follows the consistency relation :math:`n_t = -r/8` unless specified otherwise.

Using Custom Fiducial Spectra
------------------------------

You can provide your own fiducial power spectra using the ``fiducial_spectra`` parameter:

.. code-block:: python

   # Your custom spectra (example format)
   custom_spectra = {
       'tt': cl_tt_array,    # Temperature auto-correlation
       'ee': cl_ee_array,    # E-mode auto-correlation  
       'bb': cl_bb_array,    # B-mode auto-correlation
       'te': cl_te_array,    # Temperature-E cross-correlation
       # Add other spectra as needed
   }

   likelihood = LiLit(
       fields=["t", "e", "b"],
       fiducial_spectra=custom_spectra,
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

Generating Spectra with CAMB
-----------------------------

The recommended way to generate custom fiducial spectra is using CAMB with the ``CAMBres2dict`` utility:

.. code-block:: python

   import camb
   from lilit import CAMBres2dict

   # Set up CAMB parameters
   pars = camb.CAMBparams()
   
   # Basic cosmology
   pars.set_cosmology(
       H0=70.0,           # Hubble constant
       ombh2=0.022,       # Baryon density
       omch2=0.12,        # Cold dark matter density  
       tau=0.06,          # Reionization optical depth
       mnu=0.06           # Neutrino mass (eV)
   )
   
   # Primordial power spectrum
   pars.InitPower.set_params(
       As=2.2e-9,         # Amplitude of scalar perturbations
       ns=0.96,           # Scalar spectral index
       r=0.01,            # Tensor-to-scalar ratio
       nt=-0.00125        # Tensor spectral index (or use consistency relation)
   )
   
   # Set maximum multipole
   pars.set_for_lmax(3000, lens_potential_accuracy=0)
   
   # Generate results
   results = camb.get_results(pars)
   
   # Convert to LiLit format
   fiducial_spectra = CAMBres2dict(results)
   
   # Use in LiLit
   likelihood = LiLit(
       fields=["t", "e", "b"],
       fiducial_spectra=fiducial_spectra,
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

The CAMBres2dict Function
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``CAMBres2dict`` function converts CAMB results into the dictionary format expected by LiLit:

.. code-block:: python

   from lilit import CAMBres2dict
   
   # After running CAMB
   cl_dict = CAMBres2dict(camb_results)
   
   # cl_dict contains:
   # - 'tt': Temperature power spectrum
   # - 'ee': E-mode power spectrum  
   # - 'bb': B-mode power spectrum
   # - 'te': T-E cross-correlation
   # etc.

This function handles the unit conversions and formatting needed by LiLit.

Modifying Default Parameters
----------------------------

You can modify specific cosmological parameters while keeping others at Planck 2018 values:

.. code-block:: python

   # Example: Different Hubble constant
   pars = camb.CAMBparams()
   pars.set_cosmology(H0=73.0)  # Higher H0, other parameters default to Planck
   pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0.01)
   pars.set_for_lmax(3000)
   
   results = camb.get_results(pars)
   custom_spectra = CAMBres2dict(results)

Lensed vs Unlensed Spectra
---------------------------

By default, LiLit uses lensed CMB power spectra, which include the effects of gravitational lensing by large-scale structure. This is appropriate for most analyses.

If you need unlensed spectra for specific studies:

.. code-block:: python

   # Get unlensed spectra from CAMB
   powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
   unlensed_spectra = powers['unlensed_scalar']
   
   # Convert to LiLit format manually or modify CAMBres2dict as needed

Field Combinations
------------------

LiLit automatically includes only the power spectra needed for your chosen field combination:

- ``fields = ["t"]``: Only TT spectrum
- ``fields = ["t", "e"]``: TT, EE, TE spectra  
- ``fields = ["t", "e", "b"]``: TT, EE, BB, TE spectra
- ``fields = ["b"]``: Only BB spectrum

Cross-correlations between fields are automatically included when multiple fields are specified.

Validation
----------

Always validate your fiducial spectra before running forecasts:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot your fiducial spectra
   ells = np.arange(2, len(fiducial_spectra['tt']) + 2)
   
   plt.figure(figsize=(12, 8))
   
   plt.subplot(2, 2, 1)
   plt.loglog(ells, ells*(ells+1)*fiducial_spectra['tt']/(2*np.pi), 'b-')
   plt.xlabel(r'$\ell$')
   plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu$K$^2$]')
   plt.title('Temperature')
   
   plt.subplot(2, 2, 2)
   plt.loglog(ells, ells*(ells+1)*fiducial_spectra['ee']/(2*np.pi), 'r-')
   plt.xlabel(r'$\ell$')
   plt.ylabel(r'$\ell(\ell+1)C_\ell^{EE}/2\pi$ [$\mu$K$^2$]')
   plt.title('E-mode')
   
   if 'bb' in fiducial_spectra:
       plt.subplot(2, 2, 3)
       plt.loglog(ells, ells*(ells+1)*fiducial_spectra['bb']/(2*np.pi), 'g-')
       plt.xlabel(r'$\ell$')
       plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/2\pi$ [$\mu$K$^2$]')
       plt.title('B-mode')
   
   plt.tight_layout()
   plt.show()

This helps ensure your spectra have reasonable amplitudes and shapes before using them in forecasts.