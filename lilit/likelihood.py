import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from cobaya.likelihood import Likelihood


class LiLit(Likelihood):

    """Class defining the Likelihood for LiteBIRD (LiLit).

    Within LiLit, the most relevant study cases of LiteBIRD (T, E, B) are already tested and working. So, if you need to work with those, you should not need to look into the actual definition of the likelihood function, since you can proptly start running your MCMCs. Despite this, you should provide to the likelihood some file where to find the proper LiteBIRD noise power spectra, given that LiLit is implementing a simple inverse noise weighting just as a place-holder for something more realistic. As regards lensing, LiLit will need you to pass the reconstruction noise, since its computation is not coded, thus there is no place-holder for lensing.

    Parameters:
        name (str):
            The name for the likelihood, used in the output. It is necessary to pass it to LiLit. (default: None).
        fields (list):
            List of fields in the data file (default: None).
        lmax (int or list):
            Maximum multipole to consider (default: None).
        like (str, optional):
            Type of likelihood to use (default: "exact"). Currently supports "exact" and "gaussian".
        lmin (int or list):
            Minimum multipole to consider (default: 2).
        cl_file (str, optional):
            Path to Cl file (default: None).
        nl_file (str, optional):
            Path to noise file (default: None).
        experiment (str, optional):
            Name of experiment (default: None).
        nside (int, optional):
            Nside of the map (default: None).
        r (float, optional):
            Tensor-to-scalar ratio (default: None).
        nt (float, optional):
            Tensor spectral tilt (default: None).
        pivot_t (float, optional):
            Pivot scale of the tensor primordial power spectrum (default: 0.01).
        fsky (float or list):
            Sky fraction (default: 1).
        sep (str, optional):
            Separator used in the data file (default: "").
        debug (bool, optional):
            If True, produces more verbose output (default: None).


    Attributes:
        fields (list):
            List of fields in the data file.
        n_fields (int):
            Number of fields.
        keys (list):
            List of keywords for the dictionaries.
        gauss_keys (list):
            List of keywords for the Gaussian likelihood (4-points).
        sigma2 (np.ndarray):
            Array of covariances for the Gaussian likelihood case.
        lmax (int or list):
            List of lmax values.
        lmaxes (dict):
            Dictionary of lmax values.
        fsky (int or list):
            List of fsky values.
        fskies (dict):
            Dictionary of fsky values.
        lmin (int or list):
            Minimum multipole to consider.
        lmins (dict):
            Dictionary of lmin values.
        like (str):
            Type of likelihood to use.
        cl_file (str):
            Path to Cl file.
        fiduCLS (dict):
            Dictionary of fiducial Cls.
        noiseCLS (dict):
            Dictionary of noise Cls.
        fiduCOV (np.ndarray):
            Fiducial covariance matrix obtained from the corresponding dictionary.
        noiseCOV (np.ndarray):
            Noise covariance matrix obtained from the corresponding dictionary.
        data (np.ndarray):
            Data vector obtained by summing fiduCOV + noiseCOV.
        cobaCLS (dict):
            Dictionary of Cobaya Cls.
        cobaCOV (np.ndarray):
            Cobaya covariance matrix obtained from the corresponding dictionary.
        coba (np.ndarray):
            Cobaya vector obtained by summing cobaCOV + noiseCOV.
        nl_file (str):
            Path to noise file.
        experiment (str):
            Name of experiment.
        nside (int):
            Nside of the map.
        r (float):
            Tensor-to-scalar ratio.
        nt (float):
            Tensor spectral tilt.
        pivot_t (float):
            Pivot scale of the tensor primordial power spectrum.
        sep (str):
            Separator used in the data file.
        debug (bool):
            If True, produces more output.
    """

    def __init__(
        self,
        name=None,
        fields=None,
        lmax=None,
        like="exact",
        lmin=2,
        cl_file=None,
        nl_file=None,
        experiment=None,
        nside=None,
        r=None,
        nt=None,
        pivot_t=0.01,
        fsky=1,
        sep="",
        debug=None,
    ):
        # Check that the user has provided the name of the likelihood
        assert (
            name is not None
        ), "You must provide the name of the likelihood (e.g. 'BB' or 'TTTEEE')"
        # Check that the user has provided the fields
        assert (
            fields is not None
        ), "You must provide the fields (e.g. 'b' or ['t', 'e'])"
        # Check that the user has provided the maximum multipole
        assert lmax is not None, "You must provide the lmax (e.g. 300)"

        self.fields = fields
        self.n = len(fields)
        self.lmin = lmin
        self.like = like
        self.sep = sep
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.experiment = experiment
        if self.experiment is not None:
            # Check that the user has provided the nside if an experiment is used
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.debug = debug
        self.keys = self.get_keys()
        if "bb" in self.keys:
            # Check that the user has provided the tensor-to-scalar ratio if a BB likelihood is used
            assert (
                r is not None
            ), "You must provide the tensor-to-scalar ratio r for the fiducial production (defaul is at 0.01 Mpc^-1)"
            self.r = r
            self.nt = nt
            self.pivot_t = pivot_t

        self.set_lmin_lmax_fsky(lmin, lmax, fsky)

        Likelihood.__init__(self, name=name)

    def set_lmin_lmax_fsky(self, lmin, lmax, fsky):
        """Take lmin, lmax and fsky parameters and set the corresponding attributes.

        Sets the minimum multipole, the maximum multipole and the sky fraction. This handles automatically the case of a single value or a list of values. Note that the lmin, lmax and fsky for the cross-correlations are set to the geometrical mean of the lmin, lmax and fsky of the two fields. This approximation has been tested and found to be accurate, at least assuming that the two masks of the two considered multipoles are very overlapped.

        Parameters:
            lmin (int or list):
                Value or list of values of lmin.
            lmax (int or list):
                Value or list of values of lmax.
            fsky (float or list):
                Value or list of values of fsky.
        """

        self.lmins = {}
        self.lmaxs = {}
        self.fskies = {}

        # Set lmin
        if isinstance(lmin, list):
            assert (
                len(lmin) == self.n
            ), "If you provide multiple lmin, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    key = self.fields[i] + self.sep + self.fields[j]
                    self.lmins[key] = int(
                        np.ceil(np.sqrt(lmin[i] * lmin[j]))
                    )  # this approximaiton allows to gain some extra multipoles in the cross-correalation for which the SNR is still good.
                    self.lmins[key[::-1]] = int(np.ceil(np.sqrt(lmin[i] * lmin[j])))
            self.lmin = min(lmin)
        else:
            self.lmin = lmin

        # Set lmax
        if isinstance(lmax, list):
            assert (
                len(lmax) == self.n
            ), "If you provide multiple lmax, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    key = self.fields[i] + self.sep + self.fields[j]
                    self.lmaxs[key] = int(
                        np.floor(np.sqrt(lmax[i] * lmax[j]))
                    )  # this approximaiton allows to gain some extra multipoles in the cross-correalation for which the SNR is still good.
                    self.lmaxs[key[::-1]] = int(np.floor(np.sqrt(lmax[i] * lmax[j])))
            self.lmax = max(lmax)
        else:
            self.lmax = lmax

        # Set fsky
        if isinstance(fsky, list):
            assert (
                len(fsky) == self.n
            ), "If you provide multiple fsky, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    key = self.fields[i] + self.sep + self.fields[j]
                    self.fskies[key] = np.sqrt(
                        fsky[i] * fsky[j]
                    )  # this approximation for the cross-correlation is not correct in the case of two very different masks (verified with simulations)
                    self.fskies[key[::-1]] = np.sqrt(fsky[i] * fsky[j])
            self.fsky = None
        else:
            self.fsky = fsky
        return

    def cov_filling(self, cov_dict):
        """Fill covariance matrix with appropriate spectra.

        Computes the covariance matrix once given a dictionary. Returns the covariance matrix of the considered fields, in a shape equal to (num_fields x num_fields x lmax). Note that if more than one lmax, or lmin, is specified, there will be null values in the matrices, making them singular. This will be handled in another method.

        Parameters:
            cov_dict (dict):
                The input dictionary of spectra.
        """
        # Initialize output array
        res = np.zeros((self.n, self.n, self.lmax + 1))

        # Loop over field1
        for i, field1 in enumerate(self.fields):
            # Loop over field2
            for j, field2 in enumerate(self.fields[i:]):
                # Get the index of field2
                j += i

                # Get the key of the covariance matrix
                key = field1 + self.sep + field2

                # Get lmin and lmax for this field pair
                lmin = self.lmins.get(key, self.lmin)
                lmax = self.lmaxs.get(key, self.lmax)

                # Get the covariance for this field pair
                cov = cov_dict.get(key, np.zeros(lmax + 1))

                # Set the appropriate values in the covariance matrix
                res[i, j, lmin : lmax + 1] = cov[lmin : lmax + 1]
                # Fill the covariance matrix symmetrically
                res[j, i] = res[i, j]

        return res

    def get_keys(self):
        """Extracts the keys that has to be used as a function of the requested fields. These will be the usual 2-points, e.g., tt, te, ee, etc."""
        # List of all the possible combinations of the requested fields
        res = [
            self.fields[i] + self.sep + self.fields[j]
            for i in range(self.n)
            for j in range(i, self.n)
        ]
        # Print the requested keys
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def get_Gauss_keys(self):
        """Find the proper dictionary keys for the requested fields.

        Extracts the keys that has to be used as a function of the requested fields for the Gaussian likelihood. Indeed, the Gaussian likelihood is computed using 4-points, so the keys are different. E.g., there will be keys such as tttt, ttee, tete, etc.
        """
        # Calculate the number of elements in the covariance matrix
        n = int(self.n * (self.n + 1) / 2)
        # Initialize a 3-d array to store the keys
        res = np.zeros((n, n, 4), dtype=str)
        # Loop over all the elements in the covariance matrix
        for i in range(n):
            for j in range(i, n):
                # Generate a key for the i-th and j-th element
                elem = self.keys[i] + self.sep + self.keys[j]
                # Loop over all the characters in the key
                for k in range(4):
                    # Add the k-th character to the i-th, j-th, and k-th
                    # indices of the array
                    res[i, j, k] = np.asarray(list(elem)[k])
                    res[j, i, k] = res[i, j, k]
        # Print the keys if the debug flag is set
        if self.debug:
            print(f"\nThe requested keys are {res}")
        # Return the keys
        return res

    def find_spectrum(self, input_dict, key):
        """Find a spectrum in a given dictionary.

        Returns the corresponding power sepctrum for a given key. If the key is not found, it will try to find the reverse key. Otherwise it will fill the array with zeros.

        Parameters:
            input_dict (dict):
                Dictionary where you want to search for keys.

            key (str):
                Key to search for.
        """
        # create a zero array
        res = np.zeros(self.lmax + 1)

        # get lmin and lmax
        lmin = self.lmins.get(key, self.lmin)
        lmax = self.lmaxs.get(key, self.lmax)

        # try to find the key in the dictionary
        if key in input_dict:
            cov = input_dict[key]
        # if the key is not found, try the reverse key
        else:
            cov = input_dict.get(key[::-1], np.zeros(lmax + 1))

        # fill the array with the requested spectrum
        res[lmin : lmax + 1] = cov[lmin : lmax + 1]

        return res

    def sigma(self, keys, fiduDICT, noiseDICT):
        """Define the covariance matrix for the Gaussian case.

        In case of Gaussian likelihood, this returns the covariance matrix needed for the computation of the chi2. Note that the inversion is done in a separate funciton.

        Parameters:
            keys (dict):
                Keys for the covariance elements.

            fiduDICT (dict):
                Dictionary with the fiducial spectra.

            noiseDICT (dict):
                Dictionary with the noise spectra.
        """
        # The covariance matrix has to be symmetric.
        # The number of parameters in the likelihood is self.n.
        # The covariance matrix is a (self.n x self.n x self.lmax+1) ndarray.
        # We will store the covariance matrix in a (n x n x self.lmax+1) ndarray,
        # where n = int(self.n * (self.n + 1) / 2).
        n = int(self.n * (self.n + 1) / 2)
        res = np.zeros((n, n, self.lmax + 1))
        for i in range(n):  # Loop over all combinations of pairs of spectra
            for j in range(i, n):
                C_AC = self.find_spectrum(
                    fiduDICT, keys[i, j, 0] + keys[i, j, 2]
                )  # Find the fiducial spectra for each pair
                C_BD = self.find_spectrum(fiduDICT, keys[i, j, 1] + keys[i, j, 3])
                C_AD = self.find_spectrum(fiduDICT, keys[i, j, 0] + keys[i, j, 3])
                C_BC = self.find_spectrum(fiduDICT, keys[i, j, 1] + keys[i, j, 2])
                N_AC = self.find_spectrum(
                    noiseDICT, keys[i, j, 0] + keys[i, j, 2]
                )  # Find the noise spectra for each pair
                N_BD = self.find_spectrum(noiseDICT, keys[i, j, 1] + keys[i, j, 3])
                N_AD = self.find_spectrum(noiseDICT, keys[i, j, 0] + keys[i, j, 3])
                N_BC = self.find_spectrum(noiseDICT, keys[i, j, 1] + keys[i, j, 2])
                if self.fsky is not None:  # If self.fsky is defined, use the fsky value
                    res[i, j] = (
                        (C_AC + N_AC) * (C_BD + N_BD) + (C_AD + N_AD) * (C_BC + N_BC)
                    ) / self.fsky
                else:  # Otherwise, use the fsky values from the input spectra
                    AC = keys[i, j, 0] + keys[i, j, 2]
                    BD = keys[i, j, 1] + keys[i, j, 3]
                    AD = keys[i, j, 0] + keys[i, j, 3]
                    BC = keys[i, j, 1] + keys[i, j, 2]
                    AB = keys[i, j, 0] + keys[i, j, 1]
                    CD = keys[i, j, 2] + keys[i, j, 3]
                    res[i, j] = (
                        np.sqrt(self.fskies[AC] * self.fskies[BD])
                        * (C_AC + N_AC)
                        * (C_BD + N_BD)
                        + np.sqrt(self.fskies[AD] * self.fskies[BC])
                        * (C_AD + N_AD)
                        * (C_BC + N_BC)
                    ) / (self.fskies[AB] * self.fskies[CD])
                res[j, i] = res[i, j]
        return res

    def inv_sigma(self, sigma):
        """Invert the covariance matrix of the Gaussian case.

        Inverts the previously calculated sigma ndarray. Note that some elements may be null, thus the covariance may be singular. If so, this also reduces the dimension of the matrix by deleting the corresponding row and column.

        Parameters:
            ndarray (np.ndarray):
                (self.n x self.n x self.lmax+1) ndarray with the previously computed sigma (not inverted).
        """
        # Initialize array to store the inverted covariance matrices
        res = np.zeros(self.lmax + 1, dtype=object)

        # Loop over multipoles
        for i in range(self.lmax + 1):
            # Check if matrix is singular
            COV = sigma[:, :, i]
            if np.linalg.det(COV) == 0:
                # Get indices of null diagonal elements
                idx = np.where(np.diag(COV) == 0)[0]
                # Remove corresponding rows and columns
                COV = np.delete(COV, idx, axis=0)
                COV = np.delete(COV, idx, axis=1)
            # Invert matrix
            res[i] = np.linalg.inv(COV)
        return res[2:]

    def get_reduced_data(self, mat):
        """Find the reduced data eliminating the singularity of the matrix.

        Cuts the row and column corresponding to a zero diagonal value. Indeed, in case of different lmax, or lmin, for the fields, you will have singular marices.

        Parameters:
            ndarray (np.ndarray):
                A ndarray containing the covariance matrices, with some singular ones.
        """
        # Select the indices corresponding to the zero diagonal
        idx = np.where(np.diag(mat) == 0)[0]
        # Delete the rows and columns from the matrix
        return np.delete(np.delete(mat, idx, axis=0), idx, axis=1)

    def CAMBres2dict(self, camb_results):
        """Takes the CAMB result product from get_cmb_power_spectra and convert it to a dictionary with the proper keys.

        Parameters:
            camb_results (CAMBdata):
                CAMB result product from the method get_cmb_power_spectra.
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
        if "pp" in self.keys:
            # Get the lensing potential
            cl_lens = camb_results.get("lens_potential")
            # Check if it exists
            if cl_lens is not None:
                # Save it
                res["pp"] = cl_lens[:, 0].copy()
                # Check if we want the cross terms
                if "pt" in self.keys and "pe" in self.keys:
                    # Loop over the cross terms
                    for i, cross in enumerate(["pt", "pe"]):
                        # Save the result
                        res[cross] = cl_lens[:, i + 1].copy()
                        # Save the symmetric term
                        res[cross[::-1]] = res[cross]
        return res

    def txt2dict(self, txt, mapping=None, apply_ellfactor=None):
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
        ls = np.arange(txt.shape[0], dtype=np.int64)
        res = {"ell": ls}
        # Loop over the mapping and extract the corresponding column from the txt file
        # and store it in the dictionary under the corresponding keyword
        for key, i in mapping.items():
            if apply_ellfactor:
                res[key] = txt[:, i] * ls * (ls + 1) / 2 / np.pi
            else:
                res[key] = txt[:, i]
        return res

    def prod_fidu(self):
        """Produce fiducial spectra or read the input ones.

        If the user has not provided a Cl file, this function will produce the fiducial power spectra starting from the CAMB inifile for Planck2018. The extra keywords defined will maximize the accordance between the fiducial Cls and the ones obtained from Cobaya. If B-modes are requested, the tensor-to-scalar ratio and the spectral tilt will be set to the requested values. Note that if you do not provide a tilt, this will follow the standard single-field consistency relation. If instead you provide a custom file, stores that.
        """
        # If a custom file is provided, use that
        if self.cl_file is not None:
            # If the file is a pickle file, load it
            if self.cl_file.endswith(".pkl"):
                with open(self.cl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            # Otherwise, load it as text file
            else:
                txt = np.loadtxt(self.cl_file)
                mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
                res = self.txt2dict(txt, mapping)
            return res

        try:
            import camb
        except ImportError:
            print("CAMB seems to be not installed. Check the requirements.")

        # Read the ini file containing the parameters for CAMB
        path = os.path.dirname(os.path.abspath(__file__))
        planck_path = os.path.join(path, "planck_2018.ini")
        pars = camb.read_ini(planck_path)

        if "bb" in self.keys:  # If we want to include the tensor mode
            print(f"\nProducing fiducial spectra for r={self.r} and nt={self.nt}")
            pars.InitPower.set_params(
                As=2.100549e-9,
                ns=0.9660499,
                r=self.r,
                nt=self.nt,
                pivot_tensor=self.pivot_t,
                pivot_scalar=0.05,
                parameterization=2,
            )
            pars.WantTensors = True
            pars.Accuracy.AccurateBB = True
        pars.DoLensing = True
        # _pars.Accuracy.AccuracyBoost = 2 # This helps getting an extra squeeze on the accordance of Cobaya and Fiducial spectra

        if self.debug:
            print(pars)

        results = camb.get_results(pars)
        res = results.get_cmb_power_spectra(
            CMB_unit="muK",
            lmax=self.lmax,
            raw_cl=False,
        )
        return self.CAMBres2dict(res)

    def prod_noise(self):
        """Produce noise power spectra or read the input ones.

        If the user has not provided a noise file, this function will produce the noise power spectra for a given experiment with inverse noise weighting of white noise in each channel (TT, EE, BB). Note that you may want to have a look at the procedure since it is merely a place-holder. Indeed, you should provide a more realistic file from which to read the noise spectra, given that inverse noise weighting severely underestimates the amount of noise. If instead you provide the proper custom file, this method stores that.
        """
        # If the input noise file is a pickle file, load it.
        if self.nl_file is not None:
            if self.nl_file.endswith(".pkl"):
                with open(self.nl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            # If not, load the file as a text file
            else:
                _txt = np.loadtxt(self.nl_file)
                _mapping = {"bb": 0}
                # Convert the text file to a dictionary
                res = self.txt2dict(_txt, _mapping, apply_ellfactor=True)
            return res

        print(
            "***WARNING***: the inverse noise weighting performed here severely underestimates \
            the actual noise level of LiteBIRD. You should provide an input \
            noise power spectrum with a more realistic noise."
        )

        try:
            import yaml
            from yaml.loader import SafeLoader
            import healpy as hp
        except ImportError:
            print("YAML or Healpy seems to be not installed. Check the requirements.")

        assert (
            self.experiment is not None
        ), "You must specify the experiment you want to consider"
        print(f"\nComputing noise for {self.experiment}")

        path = os.path.dirname(os.path.abspath(__file__))
        experiments_path = os.path.join(path, "experiments.yaml")
        with open(experiments_path) as f:
            data = yaml.load(f, Loader=SafeLoader)

        # Get the instrument data from the saved data
        instrument = data[self.experiment]

        # Get the FWHM values from the instrument data
        fwhms = np.array(instrument["fwhm"])

        # Get the frequency values from the instrument data
        freqs = np.array(instrument["frequency"])

        # Get the depth values from the instrument data
        depth_p = np.array(instrument["depth_p"])
        depth_i = np.array(instrument["depth_i"])

        # Convert the depth to a pixel value
        depth_p /= hp.nside2resol(self.nside, arcmin=True)
        depth_i /= hp.nside2resol(self.nside, arcmin=True)
        depth_p = depth_p * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )
        depth_i = depth_i * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )

        # Get the number of frequencies
        n_freq = len(freqs)

        # Define the ell values as a numpy array
        ell = np.arange(0, self.lmax + 1, 1)

        # Define the keys for the dictionary that will be returned
        keys = ["tt", "ee", "bb"]

        sigma = np.radians(fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        sigma2 = sigma**2

        # Calculate the Gaussian beam function
        g = np.exp(ell * (ell + 1) * sigma2[:, np.newaxis])

        # Calculate the polarization factor
        pol_factor = np.array(
            [np.zeros(sigma2.shape), 2 * sigma2, 2 * sigma2, sigma2],
        )

        # Calculate the polarization factor as a function of ell
        pol_factor = np.exp(pol_factor)

        # Calculate the Gaussian beam function for each polarization
        G = []
        for i, arr in enumerate(pol_factor):
            G.append(g * arr[:, np.newaxis])
        g = np.array(G)

        # Initialize the dictionary that will be returned
        res = {key: np.zeros((n_freq, self.lmax + 1)) for key in keys}

        # Calculate the unnormalized power spectra
        res["tt"] = 1 / (g[0, :, :] * depth_i[:, np.newaxis] ** 2)
        res["ee"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)
        res["bb"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)

        # Calculate the normalized power spectra
        res["tt"] = ell * (ell + 1) / (np.sum(res["tt"], axis=0)) / 2 / np.pi
        res["ee"] = ell * (ell + 1) / (np.sum(res["ee"], axis=0)) / 2 / np.pi
        res["bb"] = ell * (ell + 1) / (np.sum(res["bb"], axis=0)) / 2 / np.pi

        res["tt"][:2] = [0, 0]
        res["ee"][:2] = [0, 0]
        res["bb"][:2] = [0, 0]

        return res

    def initialize(self):
        """Initializes the fiducial spectra and the noise power spectra."""
        # Compute the fiducial and noise power spectra
        self.fiduCLS = self.prod_fidu()
        self.noiseCLS = self.prod_noise()

        # Compute the covariance matrices
        self.fiduCOV = self.cov_filling(self.fiduCLS)
        self.noiseCOV = self.cov_filling(self.noiseCLS)

        # Print some information for debugging
        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            print("\nPrinting the first few values to check that it starts from 0...")
            field = self.fiduCLS.keys()[0]
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            field = self.noiseCLS.keys()[0]
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        # Compute the total covariance matrix
        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # Compute the inverse of the covariance matrix
        if self.like == "gaussian":
            self.gauss_keys = self.get_Gauss_keys()
            sigma2 = self.sigma(self.gauss_keys, self.fiduCLS, self.noiseCLS)
            self.sigma2 = self.inv_sigma(sigma2)

    def get_requirements(self):
        """Defines requirements of the likelihood, specifying quantities calculated by a theory code are needed. Note that you may want to change the overall keyword from 'Cl' to 'unlensed_Cl' if you want to work without considering lensing."""
        # The likelihood needs the lensed CMB angular power spectra. The keyword can be set to "unlensed_Cl" to get the unlensed ones
        requitements = {}
        requitements["Cl"] = {cl: self.lmax for cl in self.keys}
        # If debug is set to True, the likelihood will print the list of items required by the likelihood
        if self.debug:
            requitements["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {requitements}",
            )
        return requitements

    def data_vector(self, cov):
        """Get data vector from the covariance matrix.

        Extracts the data vector necessary for the Gaussian case. Note that this will cut the null value since some may be null when the fields have different values for lmax.

        Parameters:
            cov (np.ndarray):
                A ndarray containing the covariance matrices, with some null ones.
        """
        return cov[np.triu_indices(self.n)][cov[np.triu_indices(self.n)] != 0]

    def chi_exact(self, i=0):
        """Computes proper chi-square term for the exact likelihood case.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        # If the number of datasets is not equal to 1, then we have a
        # multi-dataset case, in which case we need to compute the
        # covariance matrix for each dataset.
        if self.n != 1:
            # We extract the covariance matrix and data for the ith
            # dataset.
            coba = self.coba[:, :, i]
            data = self.data[:, :, i]
            det = np.linalg.det(coba)
            # If the determinant is equal to 0, then we need to reduce
            # the dimensionality of the data and covariance matrix.
            if det == 0:
                data = self.get_reduced_data(data)
                coba = self.get_reduced_data(coba)
            # We compute the matrix M using the covariance matrix and
            # the data.
            M = np.linalg.solve(coba, data)
            # We compute the chi-square term using the trace of M, the
            # log determinant of M, and the number of fields.
            return np.trace(M) - np.linalg.slogdet(M)[1] - data.shape[0]
        # If the number of datasets is equal to 1, then we have a single
        # dataset case, in which case we do not need to loop over the
        # datasets.
        else:
            # We compute the matrix M using the covariance matrix and
            # the data.
            M = self.data / self.coba
            # We compute the chi-square term using M, the log of M, and
            # a constant value.
            return M - np.log(np.abs(M)) - 1

    def chi_gaussian(self, i=0):
        """Computes proper chi-square term for the Gaussian likelihood case.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        # If we have more than one data vector
        if self.n != 1:
            coba = self.data_vector(self.coba[:, :, i])
            data = self.data_vector(self.data[:, :, i])
            return (coba - data) @ self.sigma2[i] @ (coba - data)
        # If we have only one data vector
        else:
            coba = self.coba[0, 0, :]
            data = self.data[0, 0, :]
            res = (coba - data) * self.sigma2 * (coba - data)
            return res

    def compute_chi_part(self, i=0):
        """Chooses which chi-square term to compute.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        # check if the likelihood is "exact"
        if self.like == "exact":
            # if so, compute the chi-square term for the exact likelihood
            return self.chi_exact(i)
        # if not, check if it is "gaussian"
        elif self.like == "gaussian":
            # if so, compute the chi-square term for the gaussian likelihood
            return self.chi_gaussian(i)
        # if neither, print an error message
        else:
            print("You requested something different from 'exact or 'gaussian'!")
            return

    def log_likelihood(self):
        """Computes the log likelihood."""
        # Get the array of multipoles
        ell = np.arange(self.lmin, self.lmax + 1, 1)
        # Compute the log likelihood for each multipole
        if self.n != 1:
            logp_ℓ = np.zeros(ell.shape)
            for i in range(0, self.lmax + 1 - self.lmin):
                logp_ℓ[i] = -0.5 * (2 * ell[i] + 1) * self.compute_chi_part(i)
        else:
            logp_ℓ = -0.5 * (2 * ell + 1) * self.compute_chi_part()
        # Sum the log likelihood over multipoles
        return np.sum(logp_ℓ)

    def logp(self, **params_values):
        """Gets the log likelihood and pass it to Cobaya to carry on the MCMC process."""
        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        # Get the Cls from Cobaya
        self.cobaCLs = self.provider.get_Cl(ell_factor=True)

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            field = self.cobaCLs.keys()[0]
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {self.cobaCLs[field][0:5]}")

        # Fill the covariance matrix with the Cls from Cobaya
        self.cobaCOV = self.cov_filling(self, self.cobaCLs)

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            plt.loglog(ell, self.fiduCOV[0, 0, :], label="Fiducial CLs")
            plt.loglog(ell, self.cobaCOV[0, 0, :], label="Cobaya CLs", ls="--")
            plt.loglog(ell, self.noiseCOV[0, 0, :], label="Noise CLs")
            plt.xlim(2, None)
            plt.legend()
            plt.show()

        # Add the noise covariance to the covariance matrix filled with the Cls from Cobaya
        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # Compute the likelihood
        logp = self.log_likelihood()

        if self.debug:
            print(logp)
            exit()

        return logp


__all__ = ["LiLit"]

__docformat__ = "google"
__pdoc__ = {}
__pdoc__[
    "Likelihood"
] = "Likelihood class from Cobaya, refer to Cobaya documentation for more information."
