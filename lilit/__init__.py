"""# Welcome to LiLit!

A Python package encoding the likelihood for LiteBIRD.

LiLit provides forecasting likelihoods for LiteBIRD, implemented to be used in a
Cobaya context. This package aims to ease the creation of a common framework among
different LiteBIRD researchers.
"""

# Import from new modular structure while maintaining backward compatibility
from .core import *
from .data import *
from .likelihood import LiLit
from .math import *

__author__ = "Giacomo Galloni"
__docformat__ = "numpy"

# Dynamic version from setuptools_scm
try:
    from importlib.metadata import version

    __version__ = version("lilit")
except ImportError:
    # Fallback for older Python versions
    try:
        import pkg_resources

        __version__ = pkg_resources.get_distribution("lilit").version
    except Exception:
        __version__ = "unknown"
