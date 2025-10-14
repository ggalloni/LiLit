"""
Spectrum operations and utilities.

This module provides functions for spectrum operations including
covariance matrix filling and spectrum lookup operations.
"""

from typing import Dict, List, Optional

import numpy as np


def find_spectrum(lmin: int, lmax: int, input_dict: Dict, key: str) -> np.ndarray:
    """Find a spectrum in a given dictionary.

    Returns the corresponding power sepctrum for a given key. If the key is not found,
    it will try to find the reverse key. Otherwise it will fill the array with zeros.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        input_dict (Dict):
            Dictionary where you want to search for keys.
        key (str):
            Key to search for.

    Returns:
        np.ndarray: Power spectrum array
    """
    res = np.zeros(lmax + 1)

    if key in input_dict:
        cov = input_dict[key]
    # if the key is not found, try the reverse key, otherwise fill with zeros
    else:
        cov = input_dict.get(key[::-1], np.zeros(lmax + 1))

    res[lmin : lmax + 1] = cov[lmin : lmax + 1]

    return res


def cov_filling(
    fields: List[str],
    excluded_probes: Optional[List[str]],
    absolute_lmin: int,
    absolute_lmax: int,
    cov_dict: Dict,
    lmins: Dict[str, int] = None,
    lmaxs: Dict[str, int] = None,
) -> np.ndarray:
    """Fill covariance matrix with appropriate spectra.

    Computes the covariance matrix once given a dictionary. Returns the covariance
    matrix of the considered fields, in a shape equal to (num_fields x num_fields x
    lmax). Note that if more than one lmax, or lmin, is specified, there will be null
    values in the matrices, making them singular. This will be handled in another
    method.

    Parameters:
        fields (List[str]):
            The list of fields to consider.
        excluded_probes (Optional[List[str]]):
            The list of probes to exclude.
        absolute_lmin (int):
            The minimum multipole to consider.
        absolute_lmax (int):
            The maximum multipole to consider.
        cov_dict (Dict):
            The input dictionary of spectra.
        lmins (Dict[str, int], optional):
            The dictionary of minimum multipole to consider for each field pair.
        lmaxs (Dict[str, int], optional):
            The dictionary of maximum multipole to consider for each field pair.

    Returns:
        np.ndarray: Covariance matrix with shape (n_fields, n_fields, lmax+1)
    """
    if lmins is None:
        lmins = {}
    if lmaxs is None:
        lmaxs = {}

    n = len(fields)
    res = np.zeros((n, n, absolute_lmax + 1))

    for i, field1 in enumerate(fields):
        for j, field2 in enumerate(fields[i:]):
            j += i

            key = field1 + field2

            lmin = lmins.get(key, absolute_lmin)
            lmax = lmaxs.get(key, absolute_lmax)

            cov = cov_dict.get(key, np.zeros(lmax + 1))

            if excluded_probes is not None and key in excluded_probes:
                cov = np.zeros(lmax + 1)

            res[i, j, lmin : lmax + 1] = cov[lmin : lmax + 1]
            res[j, i] = res[i, j]

    return res


__all__ = ["find_spectrum", "cov_filling"]
