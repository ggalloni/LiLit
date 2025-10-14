"""
Key generation utilities for spectra indexing.

This module provides functions to generate proper keys for different
types of likelihood computations based on the fields being analyzed.
"""

from typing import List

import numpy as np


def get_keys(fields: List[str], *, debug: bool = False) -> List[str]:
    """Extracts the keys that has to be used as a function of the requested fields. These
    will be the usual 2-points, e.g., tt, te, ee, etc.

    Parameters:
        fields (List[str]):
            List of fields
        debug (bool, optional):
            If True, print out the requested keys

    Returns:
        List[str]: List of 2-point correlation keys
    """
    n = len(fields)
    res = [fields[i] + fields[j] for i in range(n) for j in range(i, n)]
    if debug:
        print(f"\nThe requested keys are {res}")
    return res


def get_Gauss_keys(n: int, keys: List[str], *, debug: bool = False) -> np.ndarray:
    """Find the proper dictionary keys for the requested fields.

    Extracts the keys that has to be used as a function of the requested fields for the
    Gaussian likelihood. Indeed, the Gaussian likelihood is computed using 4-points, so
    the keys are different. E.g., there will be keys such as tttt, ttee, tete, etc.

    Parameters:
        n (int):
            Number of fields.
        keys (List[str]):
            List of keys to use for the computation.
        debug (bool, optional):
            If set, print the keys that are used, by default False.

    Returns:
        np.ndarray: Array of 4-point correlation keys with shape (n, n, 4)
    """
    n = int(n * (n + 1) / 2)
    res = np.zeros((n, n, 4), dtype=str)
    for i in range(n):
        for j in range(i, n):
            elem = keys[i] + keys[j]
            for k in range(4):
                res[i, j, k] = np.asarray(list(elem)[k])
                res[j, i, k] = res[i, j, k]
    if debug:
        print(f"\nThe requested keys are {res}")
    return res


__all__ = ["get_keys", "get_Gauss_keys"]
