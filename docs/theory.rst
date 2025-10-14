Likelihood Theory
=================

LiLit implements multiple likelihood approximations for CMB analysis. This section provides detailed formulations and implementation notes.

Overview
--------

LiLit supports both single-field and multi-field likelihood calculations with the following approximations:

- **Exact likelihood**: Based on Hamimeche & Lewis (2008)
- **Gaussian likelihood**: Standard Gaussian approximation with proper multi-field covariance
- **Correlated Gaussian likelihood**: Accounts for multipole correlations (single-field only, under development)

Each approximation handles field-specific multipole ranges (:math:`\ell_{\rm min}`, :math:`\ell_{\rm max}`) and sky fractions (:math:`f_{\rm sky}`) appropriately.

Exact Likelihood
----------------

The exact likelihood approximation follows Hamimeche & Lewis (2008).

Single-field Case
~~~~~~~~~~~~~~~~~

For a single field, the log-likelihood is:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)f_{\rm sky}\left[\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}-\log\left(\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}\right)-1\right]

Multi-field Case
~~~~~~~~~~~~~~~~

For :math:`N` fields, the formula becomes:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)f_{\rm sky}^{\rm eff}\left[\text{Tr}\left(\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right) - \log\left|\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right| - N\right]

where the covariance matrix has the structure:

.. math::
   \mathcal{C}_{\rm obs} = \left(\begin{array}{ccc}
                           C_{\ell}^{XX} & C_{\ell}^{XY} & C_{\ell}^{XZ}\\
                           C_{\ell}^{YX} & C_{\ell}^{YY} & C_{\ell}^{YZ}\\
                           C_{\ell}^{ZX} & C_{\ell}^{ZY} & C_{\ell}^{ZZ}
                           \end{array}\right)

Each entry includes signal and noise contributions:

.. math::
   C_{\ell}^{XX} = C_{\ell}^{\rm CMB} + C_{\ell}^{\rm FGs} + N_{\ell}^{X} + \ldots

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

**Multipole Ranges**: The multipole range for cross-correlations between two fields is determined by the intersection of their individual ranges.

**Sky Fraction**: For multiple :math:`f_{\rm sky}` values, an effective value is computed as the geometric mean: :math:`f_{\rm sky}^{\rm eff} = \sqrt[N]{\prod_i f_{\rm sky}^i}`.

**Singular Matrices**: When multipole cuts or excluded probes make the covariance matrix singular, LiLit automatically identifies and removes null diagonal entries along with their corresponding rows and columns.

**Excluded Probes**: Specific cross-correlations can be excluded (e.g., ``excluded_probes = ["xz"]``) by setting the corresponding covariance entries to zero.

Gaussian Likelihood
-------------------

The Gaussian approximation assumes normally distributed power spectra with known covariance.

Single-field Case
~~~~~~~~~~~~~~~~~

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)f_{\rm sky}\left[\frac{\left(C_{\ell}^{\rm obs} - C_{\ell}^{\rm th}\right)^2}{\sigma^{2}_{\ell}}\right]

where the variance is:

.. math::
   \sigma^{2}_{\ell} = \frac{2}{(2\ell+1)f_{\rm sky}}\left(C_{\ell}^{\rm obs}\right)^2

Multi-field Case
~~~~~~~~~~~~~~~~

The data vector is formed from the upper triangular part of the covariance matrix:

.. math::
   X_\ell = \left(C_{\ell}^{XX}, C_{\ell}^{XY}, C_{\ell}^{XZ}, C_{\ell}^{YY}, C_{\ell}^{YZ}, C_{\ell}^{ZZ}\right)

The covariance matrix for this data vector is:

.. math::
   \text{Cov}^{\rm ABCD}_{\ell} = \frac{1}{(2\ell+1)f_{\rm sky}^{AB}f_{\rm sky}^{CD}}\left( \sqrt{f_{\rm sky}^{AC}f_{\rm sky}^{BD}}C_\ell^{AC}C_\ell^{BD} + \sqrt{f_{\rm sky}^{AD}f_{\rm sky}^{BC}}C_\ell^{AD}C_\ell^{BC} \right)

where cross-correlation sky fractions are: :math:`f_{\rm sky}^{XY} = \sqrt{f_{\rm sky}^{XX}f_{\rm sky}^{YY}}`.

The likelihood is then:

.. math::
   \log\mathcal{L} = -\frac{1}{2}\sum_{\ell}(2\ell+1)\left[X_\ell \cdot \text{Cov}^{-1}_{\ell} \cdot X_\ell^{\rm T}\right]

Implementation Differences from Exact Case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Multipole Ranges**: Cross-correlation ranges use geometric mean: :math:`\ell_{\rm max}^{XY} = \sqrt{\ell_{\rm max}^{XX}\ell_{\rm max}^{YY}}`.

**Masking**: The covariance matrix is computed for the full range, then masked entries are removed before inversion. The data vector is masked consistently.

**Sky Fractions**: Each field retains its individual :math:`f_{\rm sky}` factor, unlike the exact case which uses an effective value.

Correlated Gaussian Likelihood
------------------------------

This approximation accounts for correlations between different multipoles (currently single-field only).

Formulation
~~~~~~~~~~~

.. math::
   \log\mathcal{L} = -\frac{1}{2}\left[\left(\vec{C^{\rm obs}} - \vec{C^{\rm th}}\right) \cdot \text{Ext}^{-1} \cdot \left(\vec{C^{\rm obs}} - \vec{C^{\rm th}}\right)^{\rm T}\right]

where :math:`\vec{C^{\rm obs}}` and :math:`\vec{C^{\rm th}}` are vectors over the multipole range, and :math:`\text{Ext}` is an externally provided covariance matrix.

Requirements
~~~~~~~~~~~~

- External covariance matrix must exclude :math:`\ell = 0, 1`
- Covariance matrix should already account for :math:`f_{\rm sky}` effects
- Multi-field extension is under development

Sky Cut Approximations
----------------------

All likelihood implementations make approximations regarding sky cuts:

1. **Mode Counting**: The factor :math:`(2\ell+1)f_{\rm sky}` approximates the reduction in available modes due to masking
2. **No Mode Coupling**: Correlations between different multipoles induced by the mask are neglected
3. **Effective Sky Fraction**: Multi-field cases use geometric averaging of individual sky fractions

These approximations are valid when:

- Sky cuts are not too severe (:math:`f_{\rm sky} \gtrsim 0.3`)
- Mask boundaries are not too complex
- Cross-correlation regions have substantial overlap

For more accurate treatment of sky cuts, external covariance matrices accounting for mode coupling should be used with the correlated Gaussian likelihood.

References
----------

- Hamimeche, S. & Lewis, A., 2008, *Likelihood analysis of CMB temperature and polarization power spectra*, `arXiv:0801.0554 <https://arxiv.org/abs/0801.0554>`_
- Campeti, P. et al., 2020, *Measuring the spectrum of primordial gravitational waves with CMB, PTA and Laser Interferometers*, `arXiv:2007.04241 <https://arxiv.org/abs/2007.04241>`_
- Allys, E. et al., 2022, *Probing cosmic inflation with the LiteBIRD cosmic microwave background polarization survey*, `arXiv:2202.02773 <https://arxiv.org/abs/2202.02773>`_