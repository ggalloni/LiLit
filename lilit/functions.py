import numpy as np
from numpy.ma import MaskedArray
from camb import CAMBdata
from typing import List
from .binning import Bins

__all__ = [
    "get_chi_exact",
    "get_chi_gaussian",
    "get_chi_correlated_gaussian",
    "get_chi_HL",
    "get_chi_LoLLiPoP",
    "get_keys",
    "get_Gauss_keys",
    "get_reduced_covariances",
    "get_reduced_data_vectors",
    "cov_filling",
    "find_spectrum",
    "sigma",
    "inv_sigma",
    "CAMBres2dict",
    "txt2dict",
]


def CAMBres2dict(camb_results: CAMBdata, probes: list):
    """Takes the CAMB result product from get_cmb_power_spectra and convert it to a dictionary with the proper keys.

    Parameters:
        camb_results (CAMBdata):
            CAMB result product from the method get_cmb_power_spectra.
        probes (dict):
            Keys for the spectra dictionary we want to save.
    """
    ls = np.arange(camb_results["total"].shape[0], dtype=np.int64)
    # Mapping between the CAMB keys and the ones we want
    mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
    res = {"ell": ls}
    for probe, i in mapping.items():
        res[probe] = camb_results["total"][:, i]
    if "pp" in probes:
        cl_lens = camb_results.get("lens_potential")
        if cl_lens is not None:
            # Save it with the normalization to obtain phiphi
            array = cl_lens[:, 0].copy()
            array[2:] /= (res["ell"] * (res["ell"] + 1))[2:]
            res["pp"] = array
            if "pt" in probes and "pe" in probes:
                # Loop over the cross terms correcting the normalization to phiX
                for i, cross in enumerate(["pt", "pe"]):
                    array = cl_lens[:, i + 1].copy()
                    array[2:] /= np.sqrt(res["ell"] * (res["ell"] + 1))[2:]
                    res[cross] = array
                    res[cross[::-1]] = res[cross]
    return res


def txt2dict(
    txt: str, *, mapping_probe2colnum: dict = None, apply_ellfactor: bool = None
):
    """Takes a txt file and convert it to a dictionary. This requires a way to map the columns to the keys. Also, it is possible to apply an ell factor to the Cls.

    Parameters:
        txt (str):
            Path to txt file containing the spectra as columns.
        mapping (dict):
            Dictionary containing the mapping. Keywords will become the new keywords and values represent the index of the corresponding column.
    """
    assert (
        mapping_probe2colnum is not None
    ), "You must provide a way to map the columns of your txt to the keys of a dictionary"
    txt = np.loadtxt(txt)
    res = {}
    for probe, i in mapping_probe2colnum.items():
        ls = np.arange(len(txt[:, i]), dtype=np.int64)
        res["ell"] = ls
        if apply_ellfactor:
            res[probe] = txt[:, i] * ls * (ls + 1) / 2 / np.pi
        else:
            res[probe] = txt[:, i]
    return res


def get_keys(fields: list, *, debug: bool = False):
    """Extracts the keys that has to be used as a function of the requested fields. These will be the usual 2-points, e.g., tt, te, ee, etc.

    Parameters:
        fields (list):
            List of fields
        debug (bool, optional):
            If True, print out the requested keys
    """
    n = len(fields)
    res = [fields[i] + fields[j] for i in range(n) for j in range(i, n)]
    if debug:
        print(f"\nThe requested keys are {res}")
    return res


def get_Gauss_keys(n: int, keys: list, *, debug: bool = False):
    """Find the proper dictionary keys for the requested fields.

    Extracts the keys that has to be used as a function of the requested fields for the Gaussian likelihood. Indeed, the Gaussian likelihood is computed using 4-points, so the keys are different. E.g., there will be keys such as tttt, ttee, tete, etc.

    Parameters:
        n (int):
            Number of fields.
        keys (list):
            List of keys to use for the computation.
        debug (bool, optional):
            If set, print the keys that are used, by default False.
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


def cov_filling(
    fields: list,
    excluded_probes: list,
    absolute_lmin: int,
    absolute_lmax: int,
    cov_dict: dict,
    lmins: dict = {},
    lmaxs: dict = {},
):
    """Fill covariance matrix with appropriate spectra.

    Computes the covariance matrix once given a dictionary. Returns the covariance matrix of the considered fields, in a shape equal to (num_fields x num_fields x lmax). Note that if more than one lmax, or lmin, is specified, there will be null values in the matrices, making them singular. This will be handled in another method.

    Parameters:
        fields (list):
            The list of fields to consider.
        excluded_probes (list):
            The list of probes to exclude.
        absolute_lmin (int):
            The minimum multipole to consider.
        absolute_lmax (int):
            The maximum multipole to consider.
        cov_dict (dict):
            The input dictionary of spectra.
        lmins (dict):
            The dictionary of minimum multipole to consider for each field pair.
        lmaxs (dict):
            The dictionary of maximum multipole to consider for each field pair.
    """
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


def find_spectrum(lmin: int, lmax: int, input_dict: dict, key: str):
    """Find a spectrum in a given dictionary.

    Returns the corresponding power sepctrum for a given key. If the key is not found, it will try to find the reverse key. Otherwise it will fill the array with zeros.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        input_dict (dict):
            Dictionary where you want to search for keys.
        key (str):
            Key to search for.
    """
    res = np.zeros(lmax + 1)

    if key in input_dict:
        cov = input_dict[key]
    # if the key is not found, try the reverse key, otherwise fill with zeros
    else:
        cov = input_dict.get(key[::-1], np.zeros(lmax + 1))

    res[lmin : lmax + 1] = cov[lmin : lmax + 1]

    return res


def sigma(
    n: int,
    lmin: int,
    lmax: int,
    gauss_keys: dict,
    fiduDICT: dict,
    noiseDICT: dict,
    fsky: float = None,
    fskies: dict = {},
):
    """Define the covariance matrix for the Gaussian case.

    In case of Gaussian likelihood, this returns the covariance matrix needed for the computation of the chi2. Note that the inversion is done in a separate funciton.

    Parameters:
        n (int):
            Number of fields.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        keys (dict):
            Keys for the covariance elements.
        fiduDICT (dict):
            Dictionary with the fiducial spectra.
        noiseDICT (dict):
            Dictionary with the noise spectra.
        fsky (float, optional):
            The fraction of sky to consider. If not specified, it means that the fraction of sky is different for each field pair.
        fskies (dict, optional):
            The dictionary of fraction of sky to consider for each field pair.
    """
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
    gauss_keys: dict,
    sigma: List[np.ndarray],
    excluded_probes: list,
    lmins: dict = {},
    lmaxs: dict = {},
):
    """Mask the covariance matrix for the Gaussian case in certain ranges of multipoles.

    The covariance matrix is correctly built between lmin and lmax by the function "sigma". However, some observables might be missing in some multipole ranges, so we need to fill the matrix with zeros.

    Parameters:
        n (int):
            Number of fields.
        absolute_lmin (int):
            The minimum multipole to consider.
        absolute_lmax (int):
            The maximum multipole to consider.
        gauss_keys (dict):
            Keys for the covariance elements.
        sigma (ndarray):
            The covariance matrix.
        excluded_probes (list):
            List of probes to exclude.
        lmins (dict):
            The dictionary of minimum multipole to consider for each field pair.
        lmaxs (dict):
            The dictionary of maximum multipole to consider for each field pair.
    """
    n = int(n * (n + 1) / 2)
    mask = np.zeros(sigma.shape)

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

    return np.ma.masked_array(sigma, mask)


def inv_sigma(lmin: int, lmax: int, masked_sigma: List[MaskedArray]):
    """Invert the covariance matrix of the Gaussian case.

    Inverts the previously calculated sigma ndarray. Note that some elements may be null, thus the covariance may be singular. If so, this also reduces the dimension of the matrix by deleting the corresponding row and column.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        masked_sigma (np.ma.masked_array):
            Previously computed and masked covariance matrix (not inverted).
    """
    res = []
    for ell in range(lmax + 1):
        # Here we need to remove the masked elements to get the non null covariance matrix
        new_dimension = np.count_nonzero(np.diag(masked_sigma.mask[:, :, ell]) == False)
        COV = masked_sigma[:, :, ell].compressed().reshape(new_dimension, new_dimension)
        # This check is not necessary in principle, but it is useful to avoid singular matrices
        if np.linalg.det(COV) == 0:
            idx = np.where(np.diag(COV) == 0)[0]
            COV = np.delete(COV, idx, axis=0)
            COV = np.delete(COV, idx, axis=1)

        res.append(np.linalg.inv(COV))
    return res[lmin:], masked_sigma.mask[:, :, lmin:]


def get_reduced_covariances(covariance: np.ndarray, lmin: int, lmax: int):
    """Reduce the dimension of the covariance matrices given that they might be singular for some multipole ranges.

    Parameters:
        covariance (ndarray):
            The covariance matrix.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
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
):
    """Reduce the dimension of the data vectors given that some probe might not be defined in some multipole ranges or it might be excluded by the user.

    Parameters:
        N (int):
            Number of fields.
        covariance (ndarray):
            The covariance matrix.
        mask (ndarray):
            The mask matrix.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
    """
    reduced_data_vector = []
    for ell in range(lmax + 1 - lmin):
        vector = covariance[:, :, ell][np.triu_indices(N)]
        reduced_data_vector.append(
            np.ma.masked_array(vector, np.diag(mask[:, :, ell])).compressed()
        )
    return reduced_data_vector


def get_chi_exact(
    N: int,
    data: np.ndarray,
    coba: np.ndarray,
    lmin: int,
    lmax: int,
    fsky: float,
    bins: Bins,
):
    """Computes proper chi-square term for the exact likelihood case.

    Parameters:
        N (int):
            Number of fields.
        data (ndarray):
            The covariance matrix of the data.
        coba (ndarray):
            The covariance matrix on the MCMC step.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        fsky (float):
            The fraction of the sky if a unique number is provided. Otherwise, it is the geometrical mean of the fraction of the sky for each field pair. In other words an effective fraction of the sky.
    """
    ell = np.arange(lmin, lmax + 1, 1)
    if N != 1:
        reduced_data = get_reduced_covariances(data, lmin, lmax)
        reduced_coba = get_reduced_covariances(coba, lmin, lmax)

        M_ℓ = list(map(np.linalg.solve, reduced_coba, reduced_data))
        return (
            (2 * ell + 1)
            * fsky
            * [np.trace(M) - np.linalg.slogdet(M)[1] - M.shape[0] for M in M_ℓ]
        )
    else:
        M = data[0, 0, :] / coba[0, 0, :] - np.log(data[0, 0, :] / coba[0, 0, :]) - 1
        if bins:
            dl = bins.dl
            ell = np.concatenate((np.zeros(2), ell))
            ell = bins.bin_spectra(np.array([ell, ell, ell]))[1]
            M = np.concatenate((np.zeros(2), M))
            M = bins.bin_spectra(np.array([M, M, M]))[1]

        return (2 * ell + 1) * dl * fsky * M


def get_chi_gaussian(
    N: int,
    data: np.ndarray,
    coba: np.ndarray,
    mask: np.ndarray,
    inverse_covariance: List[np.ndarray],
    lmin: int,
    lmax: int,
    bins: Bins,
):
    """Computes proper chi-square term for the Gaussian likelihood case.

    Parameters:
        N (int):
            Number of fields.
        data (ndarray):
            The covariance matrix of the data.
        coba (ndarray):
            The covariance matrix on the MCMC step.
        mask (ndarray):
            Mask corresponding to both excluded probes and excluded multipole ranges.
        inverse_covariance (list(ndarray)):
            Inverse of the covaraince matrices for each multipole.
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
    """

    if N != 1:
        reduced_coba = get_reduced_data_vectors(
            N,
            coba,
            mask,
            lmin,
            lmax,
        )
        reduced_data = get_reduced_data_vectors(
            N,
            data,
            mask,
            lmin,
            lmax,
        )

        return [
            (reduced_coba[j] - reduced_data[j])
            @ inverse_covariance[j]
            @ (reduced_coba[j] - reduced_data[j])
            for j in range(lmax + 1 - lmin)
        ]
    else:
        M = coba[0, 0, :] - data[0, 0, :]
        inverse_covariance = np.array(inverse_covariance)[:, 0, 0]
        if bins:
            M = np.concatenate((np.zeros(2), M))
            M = bins.bin_spectra(np.array([M, M, M]))[1]
            inverse_covariance = np.concatenate((np.zeros(2), inverse_covariance))
            inverse_covariance = bins.bin_spectra(
                np.array([inverse_covariance, inverse_covariance, inverse_covariance])
            )[1]

        return M**2 * inverse_covariance


def get_chi_correlated_gaussian(
    data: np.ndarray,
    coba: np.ndarray,
    inverse_covariance: List[np.ndarray],
    bins: Bins,
):
    """Computes proper chi-square term for the Gaussian likelihood case.

    Parameters:
        data (ndarray):
            The covariance matrix of the data.
        coba (ndarray):
            The covariance matrix on the MCMC step.
        inverse_covariance (list(ndarray)):
            Inverse of the covaraince matrices for each multipole.
    """

    M = coba[0, 0, :] - data[0, 0, :]

    if bins:
        M = np.concatenate((np.zeros(2), M))
        M = bins.bin_spectra(np.array([M, M, M]))[1]

    return M @ inverse_covariance @ M


def get_chi_HL(
    data: np.ndarray,
    coba: np.ndarray,
    fidu: np.ndarray,
    offset: np.ndarray,
    inverse_covariance: List[np.ndarray],
    bins: Bins,
):
    """Computes proper chi-square term for the Hamimeche & Lewis likelihood case.

    Parameters:
        data (ndarray):
            The covariance matrix of the data.
        coba (ndarray):
            The covariance matrix on the MCMC step.
        inverse_covariance (list(ndarray)):
            Inverse of the covaraince matrices for each multipole.
    """

    M = np.array((data[0, 0, :] + offset[0, 0, :]) / (coba[0, 0, :] + offset[0, 0, :]))

    reference_spectrum = fidu[0, 0, :] + offset[0, 0, :]
    if bins:
        M = np.concatenate((np.zeros(2), M))
        M = bins.bin_spectra(np.array([M, M, M]))[1]
        reference_spectrum = np.concatenate((np.zeros(2), reference_spectrum))
        reference_spectrum = bins.bin_spectra(
            np.array([reference_spectrum, reference_spectrum, reference_spectrum])
        )[1]

    g = np.sign(M - 1) * np.sqrt(2 * (M - np.log(M) - 1))

    return (g * reference_spectrum) @ inverse_covariance @ (reference_spectrum * g)


def get_chi_LoLLiPoP(
    data: np.ndarray,
    coba: np.ndarray,
    fidu: np.ndarray,
    offset: np.ndarray,
    inverse_covariance: List[np.ndarray],
    bins: Bins,
):
    """Computes proper chi-square term for the Hamimeche & Lewis likelihood case.

    Parameters:
        data (ndarray):
            The covariance matrix of the data.
        coba (ndarray):
            The covariance matrix on the MCMC step.
        inverse_covariance (list(ndarray)):
            Inverse of the covaraince matrices for each multipole.
    """

    M = np.array((data[0, 0, :] + offset[0, 0, :]) / (coba[0, 0, :] + offset[0, 0, :]))

    reference_spectrum = fidu[0, 0, :] + offset[0, 0, :]
    if bins:
        M = np.concatenate((np.zeros(2), M))
        M = bins.bin_spectra(np.array([M, M, M]))[1]
        reference_spectrum = np.concatenate((np.zeros(2), reference_spectrum))
        reference_spectrum = bins.bin_spectra(
            np.array([reference_spectrum, reference_spectrum, reference_spectrum])
        )[1]

    g = (
        np.sign(M)
        * np.sign(np.abs(M) - 1)
        * np.sqrt(2 * (np.abs(M) - np.log(np.abs(M)) - 1))
    )

    return (g * reference_spectrum) @ inverse_covariance @ (reference_spectrum * g)


__docformat__ = "google"
__pdoc__ = {}
__pdoc__[
    "Likelihood"
] = "Likelihood class from Cobaya, refer to Cobaya documentation for more information."
