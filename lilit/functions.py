"""
Backward compatibility module for lilit.functions.

This module maintains backward compatibility with the old functions.py interface.
All functions have been moved to more appropriate modules in the new structure:

- Data loading functions: lilit.data
- Mathematical operations: lilit.math
- Chi-square calculations: lilit.core

.. deprecated:: 1.3.0
    This module is deprecated. Please import functions directly from their
    new locations (lilit.data, lilit.math, lilit.core) instead.
"""

import warnings

# Chi-square calculations (moved to lilit.core)
from .core import (
    get_chi_correlated_gaussian,
    get_chi_exact,
    get_chi_gaussian,
    get_chi_HL,
    get_chi_LoLLiPoP,
)

# Data loading functions (moved to lilit.data)
from .data import CAMBres2dict, txt2dict

# Mathematical operations (moved to lilit.math)
from .math import (
    cov_filling,
    find_spectrum,
    get_Gauss_keys,
    get_keys,
    get_masked_sigma,
    get_reduced_covariances,
    get_reduced_data_vectors,
    inv_sigma,
    sigma,
)

__all__ = [
    # Data functions
    "CAMBres2dict",
    "txt2dict",
    # Math functions
    "get_keys",
    "get_Gauss_keys",
    "get_reduced_covariances",
    "get_reduced_data_vectors",
    "cov_filling",
    "find_spectrum",
    "sigma",
    "get_masked_sigma",
    "inv_sigma",
    # Chi-square functions
    "get_chi_exact",
    "get_chi_gaussian",
    "get_chi_correlated_gaussian",
    "get_chi_HL",
    "get_chi_LoLLiPoP",
]


def __getattr__(name):
    """
    Provide backward compatibility with deprecation warnings.

    This function is called when an attribute is not found in the module.
    It issues a deprecation warning and tries to import the function from
    the appropriate new module location.
    """
    if name in __all__:
        warnings.warn(
            f"Importing '{name}' from lilit.functions is deprecated. "
            f"Please import directly from the appropriate module "
            f"(lilit.data, lilit.math, or lilit.core) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Return the already imported function
        return globals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
