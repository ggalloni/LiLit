import numpy as np

__all__ = [
    "get_keys",
    "get_Gauss_keys",
    "cov_filling",
    "find_spectrum",
    "sigma",
    "inv_sigma",
    "CAMBres2dict",
    "txt2dict",
]


def get_keys(fields: list, *, debug: bool = False):
    """Extracts the keys that has to be used as a function of the requested fields. These will be the usual 2-points, e.g., tt, te, ee, etc.

    Parameters:
        fields (list):
            List of fields
        debug (bool, optional):
            If True, print out the requested keys
    """
    # List of all the possible combinations of the requested fields
    n = len(fields)
    res = [fields[i] + fields[j] for i in range(n) for j in range(i, n)]
    # Print the requested keys
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
    # Calculate the number of elements in the covariance matrix from the number of fields
    n = int(n * (n + 1) / 2)
    # Initialize a 3-d array to store the keys
    res = np.zeros((n, n, 4), dtype=str)
    # Loop over all the elements in the covariance matrix
    for i in range(n):
        for j in range(i, n):
            # Generate a key for the i-th and j-th element
            elem = keys[i] + keys[j]
            # Loop over all the characters in the key
            for k in range(4):
                # Add the k-th character to the i-th, j-th, and k-th
                # indices of the array
                res[i, j, k] = np.asarray(list(elem)[k])
                res[j, i, k] = res[i, j, k]
    # Print the keys if the debug flag is set
    # if debug:
    #     print(f"\nThe requested keys are {res}")
    # Return the keys
    return res


def cov_filling(
    fields: list,
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
    # Initialize output array
    n = len(fields)
    res = np.zeros((n, n, absolute_lmax + 1))

    # Loop over field1
    for i, field1 in enumerate(fields):
        # Loop over field2
        for j, field2 in enumerate(fields[i:]):
            # Get the index of field2
            j += i

            # Get the key of the covariance matrix
            key = field1 + field2

            # Get lmin and lmax for this field pair
            lmin = lmins.get(key, absolute_lmin)
            lmax = lmaxs.get(key, absolute_lmax)

            # Get the covariance for this field pair
            cov = cov_dict.get(key, np.zeros(lmax + 1))

            # Set the appropriate values in the covariance matrix
            res[i, j, lmin : lmax + 1] = cov[lmin : lmax + 1]
            # Fill the covariance matrix symmetrically
            res[j, i] = res[i, j]

    return res


def find_spectrum(lmin, lmax, input_dict, key):
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
    # create a zero array
    res = np.zeros(lmax + 1)

    # try to find the key in the dictionary
    if key in input_dict:
        cov = input_dict[key]
    # if the key is not found, try the reverse key
    else:
        cov = input_dict.get(key[::-1], np.zeros(lmax + 1))

    # fill the array with the requested spectrum
    res[lmin : lmax + 1] = cov[lmin : lmax + 1]

    return res


def sigma(n, lmin, lmax, keys, fiduDICT, noiseDICT, fsky=None, fskies=[]):
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
    """
    # The covariance matrix has to be symmetric.
    # The number of parameters in the likelihood is n.
    # The covariance matrix is a (n x n x lmax+1) ndarray.
    # We will store the covariance matrix in a (n x n x lmax+1) ndarray,
    # where n = int(n * (n + 1) / 2).
    n = int(n * (n + 1) / 2)
    res = np.zeros((n, n, lmax + 1))
    for i in range(n):  # Loop over all combinations of pairs of spectra
        for j in range(i, n):
            C_AC = find_spectrum(lmin, lmax, fiduDICT, keys[i, j, 0] + keys[i, j, 2])
            C_BD = find_spectrum(lmin, lmax, fiduDICT, keys[i, j, 1] + keys[i, j, 3])
            C_AD = find_spectrum(lmin, lmax, fiduDICT, keys[i, j, 0] + keys[i, j, 3])
            C_BC = find_spectrum(lmin, lmax, fiduDICT, keys[i, j, 1] + keys[i, j, 2])
            N_AC = find_spectrum(lmin, lmax, noiseDICT, keys[i, j, 0] + keys[i, j, 2])
            N_BD = find_spectrum(lmin, lmax, noiseDICT, keys[i, j, 1] + keys[i, j, 3])
            N_AD = find_spectrum(lmin, lmax, noiseDICT, keys[i, j, 0] + keys[i, j, 3])
            N_BC = find_spectrum(lmin, lmax, noiseDICT, keys[i, j, 1] + keys[i, j, 2])
            ell = np.arange(len(C_AC))
            if fsky is not None:
                res[i, j] = (
                    (C_AC + N_AC) * (C_BD + N_BD) + (C_AD + N_AD) * (C_BC + N_BC)
                ) / fsky
            else:
                AC = keys[i, j, 0] + keys[i, j, 2]
                BD = keys[i, j, 1] + keys[i, j, 3]
                AD = keys[i, j, 0] + keys[i, j, 3]
                BC = keys[i, j, 1] + keys[i, j, 2]
                AB = keys[i, j, 0] + keys[i, j, 1]
                CD = keys[i, j, 2] + keys[i, j, 3]
                res[i, j] = (
                    np.sqrt(fskies[AC] * fskies[BD]) * (C_AC + N_AC) * (C_BD + N_BD)
                    + np.sqrt(fskies[AD] * fskies[BC]) * (C_AD + N_AD) * (C_BC + N_BC)
                ) / (fskies[AB] * fskies[CD])
            res[i, j, 2:] /= 2 * ell[2:] + 1
            res[j, i] = res[i, j]
    return res


def inv_sigma(lmin, lmax, sigma):
    """Invert the covariance matrix of the Gaussian case.

    Inverts the previously calculated sigma ndarray. Note that some elements may be null, thus the covariance may be singular. If so, this also reduces the dimension of the matrix by deleting the corresponding row and column.

    Parameters:
        lmin (int):
            The minimum multipole to consider.
        lmax (int):
            The maximum multipole to consider.
        sigma (np.ndarray):
            (n x n x lmax+1) ndarray with the previously computed sigma (not inverted).
    """
    # Initialize array to store the inverted covariance matrices
    res = np.zeros(sigma.shape)

    # Loop over multipoles
    for ell in range(lmin, lmax + 1):
        # Check if matrix is singular
        COV = sigma[:, :, ell]
        if np.linalg.det(COV) == 0:
            # Get indices of null diagonal elements
            idx = np.where(np.diag(COV) == 0)[0]
            # Remove corresponding rows and columns
            COV = np.delete(COV, idx, axis=0)
            COV = np.delete(COV, idx, axis=1)
        # Invert matrix
        res[:, :, ell] = np.linalg.inv(COV)
    return res[:, :, lmin:]


def CAMBres2dict(camb_results, keys):
    """Takes the CAMB result product from get_cmb_power_spectra and convert it to a dictionary with the proper keys.

    Parameters:
        camb_results (CAMBdata):
            CAMB result product from the method get_cmb_power_spectra.
        keys (dict):
            Keys for the spectra we want to save.
    """
    # Get the number of multipoles
    ls = np.arange(camb_results["total"].shape[0], dtype=np.int64)
    # Mapping between the CAMB keys and the ones we want
    mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
    # Initialize the output dictionary
    res = {"ell": ls}
    # Loop over the keys we want
    for key, i in mapping.items():
        # Save the results
        res[key] = camb_results["total"][:, i]
    # Check if we want the lensing potential
    if "pp" in keys:
        # Get the lensing potential
        cl_lens = camb_results.get("lens_potential")
        # Check if it exists
        if cl_lens is not None:
            # Save it with the normalization to obtain phiphi
            array = cl_lens[:, 0].copy()
            array[2:] /= (res["ell"] * (res["ell"] + 1))[2:]
            res["pp"] = array
            # Check if we want the cross terms
            if "pt" in keys and "pe" in keys:
                # Loop over the cross terms
                for i, cross in enumerate(["pt", "pe"]):
                    # Save the result
                    array = cl_lens[:, i + 1].copy()
                    array[2:] /= np.sqrt(res["ell"] * (res["ell"] + 1))[2:]
                    res[cross] = array
                    # Save the symmetric term
                    res[cross[::-1]] = res[cross]
    return res


def txt2dict(txt, mapping=None, apply_ellfactor=None):
    """Takes a txt file and convert it to a dictionary. This requires a way to map the columns to the keys. Also, it is possible to apply an ell factor to the Cls.

    Parameters:
        txt (str):
            Path to txt file containing the spectra as columns.
        mapping (dict):
            Dictionary containing the mapping. Keywords will become the new keywords and values represent the index of the corresponding column.
    """
    # Define the ell values from the length of the txt file
    assert (
        mapping is not None
    ), "You must provide a way to map the columns of your txt to the keys of a dictionary"
    res = {}
    # Loop over the mapping and extract the corresponding column from the txt file
    # and store it in the dictionary under the corresponding keyword
    for key, i in mapping.items():
        ls = np.arange(len(txt[:, i]), dtype=np.int64)
        res["ell"] = ls
        if apply_ellfactor:
            res[key] = txt[:, i] * ls * (ls + 1) / 2 / np.pi
        else:
            res[key] = txt[:, i]
    return res
