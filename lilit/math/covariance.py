"""
Covariance matrix operations for Gaussian likelihood computations.

This module provides functions for covariance matrix computation, masking,
and inversion operations used in Gaussian likelihood approximations.
"""

import numpy as np
from numpy.ma import MaskedArray

from .spectra import find_spectrum


def sigma(
    n: int,
    lmin: int,
    lmax: int,
    gauss_keys: np.ndarray,
    fiduDICT: dict,
    noiseDICT: dict,
    fsky: float | None = None,
    fskies: dict[str, float] = None,
) -> np.ndarray:
    """Define the covariance matrix for the Gaussian case.

    In case of Gaussian likelihood, this returns the covariance matrix needed for the
    computation of the chi2. Note that the inversion is done in a separate funciton.

    Parameters:
        n (int):
            Number of fields.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        gauss_keys (np.ndarray):
            Keys for the covariance elements.
        fiduDICT (Dict):
            Dictionary with the fiducial spectra.
        noiseDICT (Dict):
            Dictionary with the noise spectra.
        fsky (float, optional):
            The fraction of sky to consider. If not specified, it means that the
            fraction of sky is different for each field pair.
        fskies (Dict[str, float], optional):
            The dictionary of fraction of sky to consider for each field pair.

    Returns:
        np.ndarray: Covariance matrix with shape (n_probes, n_probes, lmax+1)
    """
    if fskies is None:
        fskies = {}

    n = int(n * (n + 1) / 2)  # Num. of probes from num. of fields
    res = np.zeros((n, n, lmax + 1))
    for i in range(n):
        for j in range(i, n):
            C_AC = find_spectrum(
                lmin, lmax, fiduDICT, gauss_keys[i, j, 0] + gauss_keys[i, j, 2]
            )
            C_BD = find_spectrum(
                lmin, lmax, fiduDICT, gauss_keys[i, j, 1] + gauss_keys[i, j, 3]
            )
            C_AD = find_spectrum(
                lmin, lmax, fiduDICT, gauss_keys[i, j, 0] + gauss_keys[i, j, 3]
            )
            C_BC = find_spectrum(
                lmin, lmax, fiduDICT, gauss_keys[i, j, 1] + gauss_keys[i, j, 2]
            )
            N_AC = find_spectrum(
                lmin, lmax, noiseDICT, gauss_keys[i, j, 0] + gauss_keys[i, j, 2]
            )
            N_BD = find_spectrum(
                lmin, lmax, noiseDICT, gauss_keys[i, j, 1] + gauss_keys[i, j, 3]
            )
            N_AD = find_spectrum(
                lmin, lmax, noiseDICT, gauss_keys[i, j, 0] + gauss_keys[i, j, 3]
            )
            N_BC = find_spectrum(
                lmin, lmax, noiseDICT, gauss_keys[i, j, 1] + gauss_keys[i, j, 2]
            )

            ell = np.arange(len(C_AC))
            if fsky is not None:
                res[i, j] = (
                    (C_AC + N_AC) * (C_BD + N_BD) + (C_AD + N_AD) * (C_BC + N_BC)
                ) / fsky
            else:
                AC = gauss_keys[i, j, 0] + gauss_keys[i, j, 2]
                BD = gauss_keys[i, j, 1] + gauss_keys[i, j, 3]
                AD = gauss_keys[i, j, 0] + gauss_keys[i, j, 3]
                BC = gauss_keys[i, j, 1] + gauss_keys[i, j, 2]
                AB = gauss_keys[i, j, 0] + gauss_keys[i, j, 1]
                CD = gauss_keys[i, j, 2] + gauss_keys[i, j, 3]
                res[i, j] = (
                    np.sqrt(fskies[AC] * fskies[BD]) * (C_AC + N_AC) * (C_BD + N_BD)
                    + np.sqrt(fskies[AD] * fskies[BC]) * (C_AD + N_AD) * (C_BC + N_BC)
                ) / (fskies[AB] * fskies[CD])
            res[i, j, 2:] /= 2 * ell[2:] + 1
            res[j, i] = res[i, j]
    return res


def get_masked_sigma(
    n: int,
    absolute_lmin: int,
    absolute_lmax: int,
    gauss_keys: np.ndarray,
    sigma_array: np.ndarray,
    excluded_probes: list[str] | None,
    lmins: dict[str, int] = None,
    lmaxs: dict[str, int] = None,
) -> np.ndarray:
    """Mask the covariance matrix for the Gaussian case in certain ranges of multipoles.

    The covariance matrix is correctly built between lmin and lmax by the function
    "sigma". However, some observables might be missing in some multipole ranges, so we
    need to fill the matrix with zeros.

    Parameters:
        n (int):
            Number of fields.
        absolute_lmin (int):
            The minimum multipole to consider.
        absolute_lmax (int):
            The maximum multipole to consider.
        gauss_keys (np.ndarray):
            Keys for the covariance elements.
        sigma_array (np.ndarray):
            The covariance matrix.
        excluded_probes (Optional[List[str]]):
            List of probes to exclude.
        lmins (Dict[str, int], optional):
            The dictionary of minimum multipole to consider for each field pair.
        lmaxs (Dict[str, int], optional):
            The dictionary of maximum multipole to consider for each field pair.

    Returns:
        np.ndarray: Masked covariance matrix
    """
    if lmins is None:
        lmins = {}
    if lmaxs is None:
        lmaxs = {}

    n = int(n * (n + 1) / 2)
    mask = np.zeros(sigma_array.shape)

    for i in range(n):
        key = gauss_keys[i, i, 0] + gauss_keys[i, i, 1]

        lmin = lmins.get(key, absolute_lmin)
        lmax = lmaxs.get(key, absolute_lmax)

        for ell in range(absolute_lmax + 1):
            if ell < lmin or ell > lmax:
                mask[i, :, ell] = 1
                mask[:, i, ell] = 1
            if excluded_probes is not None and key in excluded_probes:
                mask[i, :, ell] = 1
                mask[:, i, ell] = 1

    return np.ma.masked_array(sigma_array, mask)


def inv_sigma(
    lmin: int, lmax: int, masked_sigma: MaskedArray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Invert the covariance matrix of the Gaussian case.

    Inverts the previously calculated sigma ndarray. Note that some elements may be
    null, thus the covariance may be singular. If so, this also reduces the dimension
    of the matrix by deleting the corresponding row and column.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        masked_sigma (MaskedArray):
            Previously computed and masked covariance matrix (not inverted).

    Returns:
        Tuple[List[np.ndarray], np.ndarray]:
            Inverted covariance matrices and mask array
    """
    res = []
    for ell in range(lmax + 1):
        # Here we need to remove the masked elements to get the non null covariance matrix
        new_dimension = np.count_nonzero(np.diag(masked_sigma.mask[:, :, ell]) == False)
        COV = masked_sigma[:, :, ell].compressed().reshape(new_dimension, new_dimension)
        # This check is not necessary in principle, but it is useful to avoid singular
        # matrices
        if np.linalg.det(COV) == 0:
            idx = np.where(np.diag(COV) == 0)[0]
            COV = np.delete(COV, idx, axis=0)
            COV = np.delete(COV, idx, axis=1)

        res.append(np.linalg.inv(COV))
    return res[lmin:], masked_sigma.mask[:, :, lmin:]


def get_reduced_covariances(
    covariance: np.ndarray, lmin: int, lmax: int
) -> list[np.ndarray]:
    """Reduce the dimension of the covariance matrices given that they might be
    singular for some multipole ranges.

    Parameters:
        covariance (np.ndarray):
            The covariance matrix.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.

    Returns:
        List[np.ndarray]: List of reduced covariance matrices
    """
    reduced_covariance = []
    for ell in range(lmax + 1 - lmin):
        matrix = covariance[:, :, ell]
        # If the determinant is null, we need to reduce the covariance matrix
        idx = []
        if np.linalg.det(matrix) == 0:
            idx = np.where(np.diag(matrix) == 0)[0]
            matrix = np.delete(np.delete(matrix, idx, axis=0), idx, axis=1)
        reduced_covariance.append(matrix)
    return reduced_covariance


def get_reduced_data_vectors(
    N: int, covariance: np.ndarray, mask: np.ndarray, lmin: int, lmax: int
) -> list[int]:
    """Reduce the dimension of the data vectors given that some probe might not be
    defined in some multipole ranges or it might be excluded by the user.

    Parameters:
        N (int):
            Number of fields.
        covariance (np.ndarray):
            The covariance matrix.
        mask (np.ndarray):
            The mask array.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.

    Returns:
        List[int]: Dimensions of reduced data vectors for each multipole
    """
    reduced_dimension = []
    for ell in range(lmax + 1 - lmin):
        n_probes = int(N * (N + 1) / 2)
        dimension = n_probes - np.count_nonzero(np.diag(mask[:, :, ell]))
        reduced_dimension.append(dimension)
    return reduced_dimension


__all__ = [
    "sigma",
    "get_masked_sigma",
    "inv_sigma",
    "get_reduced_covariances",
    "get_reduced_data_vectors",
]
