LiLit: Likelihood for LiteBIRD
===============================

- **Author:** `Giacomo Galloni <mailto:giacomo.galloni@unife.it>`_
- **Source:** `Source code at GitHub <https://github.com/ggalloni/LiLit>`_
- **Documentation:** `Documentation at Readthedocs <https://lilit.readthedocs.io/>`_
- **Licence:** `GNU General Public License v3.0 <https://www.gnu.org/licenses/gpl-3.0.html>`_
- **Installation:** ``pip install lilit --upgrade`` (see the :doc:`installation instructions <installation>`)

.. image:: https://github.com/ggalloni/LiLit/actions/workflows/testing.yml/badge.svg?branch=main
    :target: https://github.com/ggalloni/LiLit/actions
    :alt: Build Status

.. image:: https://readthedocs.org/projects/lilit/badge/?version=latest
    :target: https://lilit.readthedocs.io/en/latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/lilit.svg?style=flat
    :target: https://pypi.python.org/pypi/lilit/
    :alt: PyPI version

LiLit (Likelihood for LiteBIRD) is a framework for forecasting likelihoods for the LiteBIRD CMB polarization satellite. It is designed to work seamlessly with `Cobaya <https://cobaya.readthedocs.io/>`_ (code for bayesian analysis), providing a common framework for LiteBIRD researchers to homogenize post-PTEP cosmological analyses.

The main product of this package is the LiLit likelihood class, which supports various combinations of CMB temperature, E-mode polarization, B-mode polarization, and lensing observations. LiLit is highly flexible and can be dynamically configured at declaration time to work with different field combinations, multipole ranges, and sky fractions.

LiLit has been designed from the beginning to be highly and effortlessly extensible: without touching LiLit's source code, you can define your own noise models, custom fiducial spectra, and integrate it with your existing analysis pipeline.

Though LiLit is specialized for LiteBIRD forecasting, it includes interfaces to cosmological theory codes (`CAMB <https://camb.info/>`_ and `CLASS <https://class-code.net/>`_) through Cobaya, and can be easily extended to work with other CMB experiments.

How to cite us
==============

If you use LiLit, please cite its repository and the relevant cosmological codes that you use. You can generate appropriate citations using Cobaya's ``cobaya-bib`` script with your input files.

Table of contents
=================

.. toctree::
   :caption: Installation and quickstart
   :maxdepth: 1

   installation
   quickstart
   examples

.. toctree::
   :caption: Theory and Implementation
   :maxdepth: 1

   theory

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   api/modules

.. toctree::
   :caption: Integration with Cobaya
   :maxdepth: 1

   cobaya_integration

.. toctree::
   :caption: Advanced Topics  
   :maxdepth: 1

   custom_likelihoods
   noise_models
   fiducial_spectra
   troubleshooting

Indices and tables
==================

* :ref:`modindex`
* :ref:`search`