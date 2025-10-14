# LiLit: Likelihood for LiteBIRD

[![Build Status](https://github.com/ggalloni/LiLit/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/ggalloni/LiLit/actions)
[![Documentation Status](https://readthedocs.org/projects/lilit/badge/?version=latest)](https://lilit.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/lilit.svg?style=flat)](https://pypi.python.org/pypi/lilit/)

**Author:** Giacomo Galloni

LiLit (Likelihood for LiteBIRD) is a framework for forecasting likelihoods for the LiteBIRD CMB polarization satellite. It provides a common framework for LiteBIRD researchers working within the [Cobaya](https://cobaya.readthedocs.io/) cosmological analysis ecosystem.

## Quick Start

Install LiLit from PyPI:

```bash
pip install lilit
```

Basic usage:

```python
from lilit import LiLit

# Create a likelihood for temperature and polarization
fields = ["t", "e", "b"]
likelihood = LiLit(
    fields=fields,
    lmax=[1500, 1200, 900],
    lmin=[20, 2, 2], 
    fsky=[1.0, 0.8, 0.6]
)
```

## Key Features

- **Multiple field support**: Temperature, E-mode, B-mode polarization, and lensing
- **Flexible configuration**: Field-specific multipole ranges and sky fractions
- **Multiple likelihood approximations**: Exact, Gaussian, and correlated Gaussian
- **Seamless Cobaya integration**: Drop-in replacement for existing likelihood codes
- **Extensible design**: Easy integration of custom noise models and fiducial spectra

## Documentation

📖 **Complete documentation is available at [https://lilit.readthedocs.io/](https://lilit.readthedocs.io/)**

The documentation includes:

- **Installation guide** and quick start tutorial
- **Detailed examples** for common LiteBIRD use cases  
- **Theoretical background** on likelihood approximations
- **API reference** with full class and function documentation
- **Cobaya integration** guide for parameter estimation and model comparison

## Examples

See the [examples](examples/) directory and the [online documentation](https://lilit.readthedocs.io/en/latest/examples.html) for working examples including:

- Basic temperature and polarization analysis
- Multi-field likelihood configurations
- Integration with Cobaya sampling chains
- Custom noise model implementations

## Contributing

Contributions are welcome! Please see our [documentation](https://lilit.readthedocs.io/) for development guidelines, or open an issue to discuss major changes.

## Citation

If you use LiLit in your research, please cite this repository and the relevant cosmological codes. Use Cobaya's `cobaya-bib` script to generate appropriate citations for your specific analysis.

## Support

- **Documentation**: [https://lilit.readthedocs.io/](https://lilit.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/ggalloni/LiLit/issues)
- **Cobaya documentation**: [https://cobaya.readthedocs.io/](https://cobaya.readthedocs.io/)
