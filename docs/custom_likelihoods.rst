Likelihood Approximations
=========================

LiLit implements two likelihood approximations for CMB power spectrum analysis: the exact likelihood and the Gaussian likelihood. This section details how they work and when to use each.

Exact Likelihood
----------------

The exact likelihood is based on the Hamimeche & Lewis (2008) formulation. For a single field, the log-likelihood is:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)\left[\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}-\log\left(\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}\right)-1\right]

For multiple fields (N fields), the formula generalizes to:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)\left[\text{Tr}\left(\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right) - \log\left|\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right| - N\right]

where :math:`\mathcal{C}_{\rm obs}` and :math:`\mathcal{C}_{\rm th}` are the covariance matrices containing the observed and theoretical power spectra.

Using the Exact Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lilit import LiLit

   # Create likelihood with exact approximation
   likelihood = LiLit(
       name="example_exact",
       fields=["t", "e", "b"],
       like="exact",
       nl_file="/path/to/noise.pkl",
       lmax=[2500, 2000, 500],
       lmin=[2, 2, 2],
       fsky=[0.7, 0.7, 0.6],
       debug=False
   )

Gaussian Likelihood
-------------------

The Gaussian approximation assumes that the power spectra are Gaussian distributed around their theoretical values:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)\left[\frac{(C_{\ell}^{\rm obs} - C_{\ell}^{\rm th})^2}{\sigma^{2}_{\ell}}\right]

For a single field, the variance is:

.. math::
   \sigma^{2}_{\ell} = \frac{2}{(2\ell+1)f_{\rm sky}}(C_{\ell}^{\rm obs})^2

For multiple fields, the data vector is formed from the upper triangular part of the covariance matrix, and the covariance between different power spectrum elements is:

.. math::
   \text{Cov}^{\rm ABCD}_{\ell} = \frac{1}{(2\ell+1)f_{\rm sky}^{AB}f_{\rm sky}^{CD}}\left( \sqrt{f_{\rm sky}^{AC}f_{\rm sky}^{BD}}C_\ell^{AC}C_\ell^{BD} + \sqrt{f_{\rm sky}^{AD}f_{\rm sky}^{BC}}C_\ell^{AD}C_\ell^{BC} \right)

Using the Gaussian Likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lilit import LiLit

   # Create likelihood with Gaussian approximation
   likelihood = LiLit(
       name="example_gaussian",
       fields=["t", "e", "b"],
       like="gaussian",
       nl_file="/path/to/noise.pkl",
       lmax=[2500, 2000, 500],
       lmin=[2, 2, 2],
       fsky=[0.7, 0.7, 0.6],
       debug=False
   )

Field-Specific Multipole Ranges
--------------------------------

LiLit supports different multipole ranges for different fields. The handling differs between the two likelihood approximations:

**Exact Likelihood:**
- Cross-correlation ranges are determined by the intersection of individual field ranges
- Entries outside the valid range for each field are set to zero in the covariance matrix
- Singular matrices are handled by removing null diagonal entries and corresponding rows/columns

**Gaussian Likelihood:**
- Cross-correlation ranges use the geometric mean: :math:`\ell_{\rm max}^{XY} = \sqrt{\ell_{\rm max}^{XX}\ell_{\rm max}^{YY}}`
- A mask is applied to exclude invalid multipoles before matrix inversion

Excluding Specific Probes
--------------------------

You can exclude specific cross-correlations from the analysis:

.. code-block:: python

   # Exclude T-B cross-correlation
   likelihood = LiLit(
       name="example_no_tb",
       fields=["t", "e", "b"],
       like="exact",
       excluded_probes=["tb"],  # Exclude T-B cross-correlation
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

Sky Coverage Effects
--------------------

LiLit approximates partial sky coverage effects by rescaling the effective number of modes:

- Single :math:`f_{\rm sky}` value: Applied uniformly to all fields
- Multiple values: Geometric mean is computed and applied
- Cross-correlations: :math:`f_{\rm sky}^{XY} = \sqrt{f_{\rm sky}^{XX}f_{\rm sky}^{YY}}`

**Note:** This approach does not account for mode coupling introduced by the sky cut.

Debug Mode
----------

Enable debug mode to check the likelihood setup:

.. code-block:: python

   likelihood = LiLit(
       name="debug_example",
       fields=["t", "e"],
       like="exact",
       debug=True,  # Enable debugging output
       lmax=[2500, 2000],
       fsky=[0.7, 0.7]
   )

When to Use Each Approximation
------------------------------

**Use Exact Likelihood when:**
- You need the most accurate likelihood for parameter inference
- Working with low signal-to-noise data
- Precise error estimation is critical

**Use Gaussian Likelihood when:**
- Computational speed is important
- Signal-to-noise is high (Gaussian approximation is more accurate)
- Working with large-scale surveys or many multipoles

**Performance Considerations:**
The exact likelihood requires matrix operations at each multipole, making it computationally more expensive than the Gaussian approximation, especially for many fields or high :math:`\ell_{\rm max}` values.