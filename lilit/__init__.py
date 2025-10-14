"""# Welcome to LiLit!

A Python package encoding the likelihood for LiteBIRD.

LiLit provides forecasting likelihoods for LiteBIRD, implemented to be used in a 
Cobaya context. This package aims to ease the creation of a common framework among 
different LiteBIRD researchers.
"""

from .functions import *
from .likelihood import LiLit

__author__ = "Giacomo Galloni"
__version__ = "1.2.9"
__docformat__ = "numpy"
