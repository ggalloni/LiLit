"""Define the likelihoods of this repository.

File providing the verbose definition of three likelihoods. The main one would be
the third, i.e. LiLit. The others are simple examples to get in touch with Cobaya.
"""
from cobaya.likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
import pickle


class exactXX(Likelihood):
    """Class to define the template for an exact likelihood encoding one single field, here named X."""

    def initialize(self):
        """Initialize the likelihood.

        Prepare any computation, importing any necessary code, files, etc. Note
        that this function is run only once at the very beginning of the routine. e.g.
        here you load a cl_file containing the fiducial values of the power spectra
        according to some fiducial cosmology you want to probe. Furthermore, you import
        the CLs of the noise you are considering and you check for basic consistencies.
        In this example, I precomputed the fiducial power spectra (coming as a dict),
        but you may want to produce them on the fly using a CAMB inifile, or whatever
        you want.

        Returns
        -------
            None
        """
        # Loading files
        _cl_file = "path/to/cls.pkl"
        _nl_file = "path/to/noise.pkl"
        with open(_cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(_nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        # Initialize useful quantities
        self.debug = True
        self.lmin = 2
        self.lmax = 300
        self.fsky = 0.5

        # Probably at this point you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            # For a field contained in the fiducial (here XX):
            _field = "xx"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {_field.upper()} ---> {self.fiduCLS[_field][0:5]}")
            print(f"Noise CLs for {_field.upper()} ---> {self.noiseCLS[_field][0:5]}")

        # Now we pass the fiducial spectra to the rest of the class
        self.data = (
            self.fiduCLS["xx"][self.lmin : self.lmax + 1]
            + self.noiseCLS["xx"][self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        """Define requirements of the likelihood, specifying quantities calculated by a theory code are needed.

        Returns
        -------
            dict: dict with requirements
        return dictionary
        """
        requirements = {}
        requirements["Cl"] = {"xx": self.lmax}
        if self.debug:
            requirements[
                "CAMBdata"
            ] = None  # This allows to get the complete product of the CAMB run
        return requirements

    def logp(self):
        """Compute the likelihood for each step of the chain. Note that this function has to return log-likelihood.

        Returns
        -------
            float: log-likelihood
        """
        # You may want to check whether every parameter has been set as desided, thus you can print
        # the parameters set in CAMB. This also allows you to do some extra computation with the CAMB
        # results if needed.
        if self.debug:
            _CAMBdata = self.provider.get_CAMBdata()
            _pars = _CAMBdata.Params
            print(_pars)

        # Retrieve CLs for the step
        _cobaCLs = self.provider.get_Cl(ell_factor=True)

        # Also here you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of Cobaya CLs ---> {_cobaCLs.keys()}")

            # For a field contained in the fiducial (here YY):
            _field = "xx"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {_field.upper()} ---> {_cobaCLs[_field][0:5]}")

        _coba = (
            _cobaCLs["xx"][self.lmin : self.lmax + 1]
            + self.noiseCLS["xx"][self.lmin : self.lmax + 1]
        )

        # At this point, you may want to check whether everything is consistent in terms of
        # normalizations, overall values, etc... Therefore here you can plot the considered field
        # and you can compare the fiducial and the cobaya spectra (+ noise eventually)
        if self.debug:
            _ell = np.arange(0, self.lmax + 1, 1)
            _field = "xx"
            plt.loglog(
                _ell[2 - self.lmin :],
                self.fiduCLS[_field][2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                _cobaCLs[_field][2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                self.noiseCLS[_field][2 - self.lmin :],
                label="Noise CLs",
            )

            plt.xlim(2, None)
            plt.legend()
            plt.show()

            # Since this function is called for every step, you want to kill it if you produce
            # this plot
            exit()

        # Now you want to compute the log-likelihood. This may be done in many different ways.
        # Here you can compute the log-likelihood using an exact likelihood. Note that this
        # example uses only one field. Thus for more complex likelihoods this may become more
        # involved
        _ell = np.arange(self.lmin, self.lmax + 1, 1)
        M = self.data / _coba
        logp_ℓ = -0.5 * (2 * _ell + 1) * self.fsky * (M - np.log(np.abs(M)) - 1)

        return np.sum(logp_ℓ)


class exactYYYKKK(Likelihood):
    """Class to define the template for an exact likelihood encoding two fields, here named here Y and K."""

    def initialize(self):
        """Initialize the likelihood.

        Prepare any computation, importing any necessary code, files, etc. Note
        that this function is run only once at the very beginning of the routine. e.g.
        here you load a cl_file containing the fiducial values of the power spectra
        according to some fiducial cosmology you want to probe. Furthermore, you import
        the CLs of the noise you are considering and you check for basic consistencies.
        In this example, I precomputed the fiducial power spectra (coming as a dict),
        but you may want to produce them on the fly using a CAMB inifile, or whatever
        you want.

        Returns
        -------
            None
        """
        # Loading files
        _cl_file = "path/to/cls.pkl"
        _nl_file = "path/to/noise.pkl"
        with open(_cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(_nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        # Initialize useful quantities
        self.debug = True
        self.lmin = 2
        self.lmaxYY = 500
        self.lmaxKK = 300
        self.lmax = np.max(self.lmaxYY, self.lmaxKK)
        self.fsky = 0.5

        # Probably at this point you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys}")

            # For a field contained in the fiducial (here YY):
            _field = "yy"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {_field.upper} ---> {self.fiduCLS[_field][0:5]}")
            print(f"Noise CLs for {_field.upper} ---> {self.noiseCLS[_field][0:5]}")

        # Now we pass the fiducial spectra to the rest of the class

        self.fiduCOV = np.array(
            [
                [self.fiduCLS["yy"], self.fiduCLS["yk"]],
                [self.fiduCLS["yk"], self.fiduCLS["kk"]],
            ],
        )
        self.noiseCOV = np.array(
            [
                [self.noiseCLS["yy"], self.noiseCLS["yk"]],
                [self.noiseCLS["yk"], self.noiseCLS["kk"]],
            ],
        )

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        """Define requirements of the likelihood, specifying quantities calculated by a theory code are needed.

        Returns
        -------
            dict: dict with requirements
        return dictionary
        """
        requirements = {}
        requirements["Cl"] = {"yy": self.lmax, "kk": self.lmax, "yk": self.lmax}
        if self.debug:
            requirements[
                "CAMBdata"
            ] = None  # This allows to get the complete product of the CAMB run
        return requirements

    def logp(self):
        """Compute the likelihood for each step of the chain. Note that this function has to return log-likelihood.

        Returns
        -------
            float: log-likelihood
        """
        # You may want to check whether every parameter has been set as desided, thus you can print
        # the parameters set in CAMB. This also allows you to do some extra computation with the CAMB
        # results if needed.
        if self.debug:
            _CAMBdata = self.provider.get_CAMBdata()
            _pars = _CAMBdata.Params
            print(_pars)

        # Retrieve CLs for the step
        self.cobaCLs = self.provider.get_Cl(ell_factor=True)

        # Also here you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            # For a field contained in the fiducial (here YY):
            _field = "yy"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {_field.upper()} ---> {self.cobaCLs[_field][0:5]}")

        self.cobaCOV = np.array(
            [
                [self.cobaCLS["yy"], self.cobaCLS["yk"]],
                [self.cobaCLS["yk"], self.cobaCLS["kk"]],
            ],
        )
        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # At this point, you may want to check whether everything is consistent in terms of
        # normalizations, overall values, etc... Therefore here you can plot the considered field
        # and you can compare the fiducial and the cobaya spectra (+ noise eventually)
        if self.debug:
            _ell = np.arange(0, self.lmax + 1, 1)
            _field = "yy"
            plt.loglog(
                _ell[2 - self.lmin :],
                self.fiduCLS[_field][2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                self.cobaCLs[_field][2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                self.noiseCLS[_field][2 - self.lmin :],
                label="Noise CLs",
            )
            plt.legend()
            plt.show()

            # Since this function is called for every step, you want to kill it if you produce
            # this plot
            exit()

        # Now you want to compute the log-likelihood. This may be done in many different ways.
        # Here you can compute the log-likelihood using an exact likelihood. Note that this
        # example uses only one field. Thus for more complex likelihoods this may become more
        # involved
        _ell = np.arange(self.lmin, self.lmax + 1, 1)
        logp_ℓ = np.zeros(_ell.shape)

        for i in range(0, self.lmax + 1 - self.lmin):
            if i <= self.lmaxKK:
                M = self.data[:, :, i] @ np.linalg.inv(self.coba[:, :, i])
                _norm = len(self.data[0, :, i])
                logp_ℓ[i] = (
                    -0.5
                    * (2 * _ell[i] + 1)
                    * self.fsky
                    * (np.trace(M) - np.linalg.slogdet(M)[1] - _norm)
                )
            else:
                M = self.data[0, 0, i] / self.coba[0, 0, i]
                _norm = 1
                logp_ℓ[i] = (
                    -0.5
                    * (2 * _ell[i] + 1)
                    * self.fsky
                    * (M - np.log(np.abs(M)) - _norm)
                )

        return np.sum(logp_ℓ)


class LiLit(Likelihood):
    """Class to define LiLit.

    This is a far more flexible likelihood than what presented above. LiLit encodes
    both the one field and the two fields cases. In fact, this implementation is
    independent from the number of fields (as soon as you are consistent with yourself
    in what you provide as an input). Also, one can specify lmax and fsky as lists
    corresponding to the values for different fields. I suggest to use this class, since
    the others are just quick examples of how Cobaya works.
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
        """Initialize LiLit.

        With the initializations of LiLit, I store useful quantities for the
        rest of the computation. These can be passed to the class at declaration.
        Note that the one required for a healthy run are initialized to None, so
        that you must provide them to avoid an error.
        """
        assert (
            name is not None
        ), "You must provide the name of the likelihood (e.g. 'BB' or 'TTTEEE')"
        assert (
            fields is not None
        ), "You must provide the fields (e.g. 'b' or ['t', 'e'])"
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
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.debug = debug
        self.keys = self.get_keys()
        if "bb" in self.keys:
            assert (
                r is not None
            ), "You must provide the tensor-to-scalar ratio r for the fiducial production (defaul is at 0.01 Mpc^-1)"
            self.r = r
            self.nt = nt
            self.pivot_t = pivot_t

        # This part is necesary to handle the case where the various fields have different lmax and fsky
        self.lmaxs = None
        if isinstance(lmax, list):
            assert (
                len(lmax) == self.n
            ), "If you provide multiple lmax, they must match the number of requested fields with the same order"
            self.lmaxs = {}
            for i in range(self.n):
                for j in range(i, self.n):
                    _key = self.fields[i] + self.sep + self.fields[j]
                    self.lmaxs[_key] = min(lmax[i], lmax[j])
                    self.lmaxs[_key[::-1]] = min(lmax[i], lmax[j])
            if self.debug:
                print(f"\nYou have requested the following lmax {self.lmaxs}")
            self.lmax = max(lmax)
        else:
            self.lmax = lmax

        self.fskies = None
        if isinstance(fsky, list):
            assert (
                len(fsky) == self.n
            ), "If you provide multiple fsky, they must match the number of requested fields with the same order"
            self.fskies = {}
            for i in range(self.n):
                for j in range(i, self.n):
                    _key = self.fields[i] + self.sep + self.fields[j]
                    self.fskies[_key] = min(fsky[i], fsky[j])
                    self.fskies[_key[::-1]] = min(fsky[i], fsky[j])
            if self.debug:
                print(f"\nYou have requested the following fsky {self.fskies}")
            self.fsky = None
        else:
            self.fsky = fsky
        Likelihood.__init__(self, name=name)

    def cov_filling(self, input_dictionary: dict) -> np.ndarray:
        """Fill covariance matrix with appropriate spectra.

        Compute the covariance matrix once given a dictionary. Returns the covariance
        matrix of the considered fields, in a shape equal to (num_fields x num_fields x lmax).
        Note that if more than one lmax is specified, there will be null values in the matrices,
        making them singular.

        Args:
        ----
            input_dictionary (dict[array]): input dictionary of spectra

        Returns:
        -------
            ndarray: covariance matrix of the considered fields of shape (num_fields x num_fields x lmax)
        """
        res = np.zeros((self.n, self.n, self.lmax + 1))
        for i in range(self.n):
            for j in range(i, self.n):
                _key = self.fields[i] + self.sep + self.fields[j]
                if self.lmaxs is not None:
                    res[i, j, : self.lmaxs[_key] + 1] = input_dictionary.get(
                        _key,
                        np.zeros(self.lmaxs[_key] + 1),
                    )[: self.lmaxs[_key] + 1]
                else:
                    res[i, j, : self.lmax + 1] = input_dictionary.get(
                        _key,
                        np.zeros(self.lmax + 1),
                    )[: self.lmax + 1]
                res[j, i] = res[i, j]
        return res

    def get_keys(self):
        """Extract the keys that has to be used as a function of the requested fields.

        Returns
        -------
            dict[str]: dict of the keys
        """
        res = []
        for i in range(self.n):
            for j in range(i, self.n):
                _key = self.fields[i] + self.sep + self.fields[j]
                res.append(_key)
        return res

    def get_Gauss_keys(self):
        """Find the proper dictionary keys for the requested fields.

        Get the proper combinations of fields in case of Gaussian likelihood.
        These will be used to build the covariance matrix in an automatic way.

        Returns
        -------
            dict[str]: dict of the keys
        """
        res = np.zeros(
            (int(self.n * (self.n + 1) / 2), int(self.n * (self.n + 1) / 2), 4),
            dtype=str,
        )
        for i in range(int(self.n * (self.n + 1) / 2)):
            for j in range(i, int(self.n * (self.n + 1) / 2)):
                _elem = self.keys[i] + self.sep + self.keys[j]
                for k in range(4):
                    res[i, j, k] = np.asarray(list(_elem)[k])
                    res[j, i, k] = res[i, j, k]
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def find_spectrum(self, dict, key):
        """Find a spectrum in a given dictionary.

        Return the corresponding power sepctrum for a given key. If the key is not found,
        it will try to find the reverse key. Otherwise it will fill the array with zeros.

        Args:
        ----
            dict (dict[array]): dictionary where you want to search for keys
            key (str): key to search for

        Returns:
        -------
            array: (self.lmax+1) array containing the requested spectrum
        """
        res = np.zeros(self.lmax + 1)
        if self.lmaxs is not None:
            if key in dict:
                res[: self.lmaxs[key] + 1] = dict[key][: self.lmaxs[key] + 1]
            else:
                res[: self.lmaxs[key] + 1] = dict.get(
                    key[::-1],
                    np.zeros(self.lmaxs[key] + 1),
                )[: self.lmaxs[key] + 1]
        else:
            if key in dict:
                res[: self.lmax + 1] = dict[key][: self.lmax + 1]
            else:
                res[: self.lmax + 1] = dict.get(key[::-1], np.zeros(self.lmax + 1))[
                    : self.lmax + 1
                ]
        return res

    def sigma(self, keys, fiduDICT, noiseDICT):
        """Define the covariance matrix for the Gaussian case.

        In case of Gaussian likelihood, this returns the covariance matrix needed
        for the computation of the chi2. Note that the inversion is done in a
        separate funciton.

        Args:
        ----
            keys (dict[str]): keys for the covariance elements
            fiduDICT (dict[array]): dictionary with the fiducial spectra
            noiseDICT (dict[array]): dictionary with the noise spectra

        Returns:
        -------
            ndarray: (self.n x self.n x self.lmax+1) ndarray
        """
        res = np.zeros(
            (
                int(self.n * (self.n + 1) / 2),
                int(self.n * (self.n + 1) / 2),
                self.lmax + 1,
            ),
        )
        for i in range(int(self.n * (self.n + 1) / 2)):
            for j in range(i, int(self.n * (self.n + 1) / 2)):
                _AB = keys[i, j, 0] + keys[i, j, 1]
                _AC = keys[i, j, 0] + keys[i, j, 2]
                _CD = keys[i, j, 2] + keys[i, j, 3]
                _BD = keys[i, j, 1] + keys[i, j, 3]
                _AD = keys[i, j, 0] + keys[i, j, 3]
                _BC = keys[i, j, 1] + keys[i, j, 2]
                _C_AC = self.find_spectrum(fiduDICT, _AC)
                _C_BD = self.find_spectrum(fiduDICT, _BD)
                _C_AD = self.find_spectrum(fiduDICT, _AD)
                _C_BC = self.find_spectrum(fiduDICT, _BC)
                _N_AC = self.find_spectrum(noiseDICT, _AC)
                _N_BD = self.find_spectrum(noiseDICT, _BD)
                _N_AD = self.find_spectrum(noiseDICT, _AD)
                _N_BC = self.find_spectrum(noiseDICT, _BC)
                if self.fsky is not None:
                    res[i, j] = (
                        (_C_AC + _N_AC) * (_C_BD + _N_BD)
                        + (_C_AD + _N_AD) * (_C_BC + _N_BC)
                    ) / self.fsky
                else:
                    res[i, j] = (
                        np.sqrt(self.fskies[_AC] * self.fskies[_BD])
                        * (_C_AC + _N_AC)
                        * (_C_BD + _N_BD)
                        + np.sqrt(self.fskies[_AD] * self.fskies[_BC])
                        * (_C_AD + _N_AD)
                        * (_C_BC + _N_BC)
                    ) / (self.fskies[_AB] * self.fskies[_CD])
                res[j, i] = res[i, j]
        return res

    def inv_sigma(self, sigma):
        """Invert the covariance matrix of the Gaussian case.

        Invert the previously calculated sigma ndarray. Note that some elements
        may be null, thus the covariance may be singular. In those cases, I
        reduce the dimension of the matrix by deleting the corresponding row and
        column.

        Args:
        ----
            ndarray: (self.n x self.n x self.lmax+1) ndarray with the previously computed sigma (not inverted)

        Returns:
        -------
            array : array of objects, each being a square matrix representing the covariance at that multipole
        """
        res = np.zeros(self.lmax + 1, dtype=object)

        for i in range(self.lmax + 1):
            # Check if matrix is singular
            COV = sigma[:, :, i]
            if np.linalg.det(COV) == 0:
                # Get indices of null diagonal elements
                _idx = np.where(np.diag(COV) == 0)[0]
                # Remove corresponding rows and columns
                COV = np.delete(COV, _idx, axis=0)
                COV = np.delete(COV, _idx, axis=1)
            # Invert matrix
            res[i] = np.linalg.inv(COV)
        return res[2:]

    def get_reduced_data(self, mat):
        """Find the reduced data eliminating the singularity of the matrix.

        Cut the row and column corresponding to a zero diagonal value.
        Indeed, in case of different lmax for the fields, you will
        have singular marices.

        Args:
        ----
            ndarray: A ndarray containing the covariance matrices, with some singular ones.

        Returns:
        -------
            ndarray : ndarray of the reduced input matrix
        """
        _idx = np.where(np.diag(mat) == 0)[0]
        mat = np.delete(mat, _idx, axis=0)
        return np.delete(mat, _idx, axis=1)

    def CAMBres2dict(self, camb_results):
        """Take the CAMB result product from get_cmb_power_spectra and convert it to a dictionary.

        Args:
        ----
            camb_results (CAMBdata instance): CAMB result product from the method get_cmb_power_spectra.

        Returns:
        -------
            dictionary (dict): dictionary containing the results under the proper keys.
        """
        _ls = np.arange(camb_results["total"].shape[0], dtype=np.int64)
        _mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
        res = {"ell": _ls}
        for key, i in _mapping.items():
            res[key] = camb_results["total"][:, i]
        if "pp" in self.keys:
            cl_lens = camb_results.get("lens_potential")
            if cl_lens is not None:
                res["pp"] = cl_lens[:, 0].copy()
                if "pt" in self.keys and "pe" in self.keys:
                    for i, cross in enumerate(["pt", "pe"]):
                        res[cross] = cl_lens[:, i + 1].copy()
                        res[cross[::-1]] = res[cross]
        return res

    def txt2dict(self, txt, mapping=None, apply_ellfactor=None):
        """Take a txt file and convert it to a dictionary. This requires a way to map the columns to the keys.

        Args:
        ----
            txt (txt file): txt file containing the spectra as columns
            mapping (dict): dictionary containing the mapping. Keywords will
            become the new keywords and values represent the index of the corresponding column

        Returns:
        -------
            dictionary (dict): dictionary containing the results under the proper keys.
        """
        assert (
            mapping is not None
        ), "You must provide a way to map the columns of your txt to the keys of a dictionary"
        _ls = np.arange(txt.shape[0], dtype=np.int64)
        res = {"ell": _ls}
        for key, i in mapping.items():
            if apply_ellfactor:
                res[key] = txt[:, i] * _ls * (_ls + 1) / 2 / np.pi
            else:
                res[key] = txt[:, i]
        return res

    def prod_fidu(self):
        """Produce fiducial spectra or read the input ones.

        Produce the fiducial power spectra starting from the CAMB inifile
        for Planck2018. If instead you provide a custom file, stores that.
        """
        if self.cl_file is not None:
            if self.cl_file.endswith(".pkl"):
                with open(self.cl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            else:
                _txt = np.loadtxt(self.cl_file)
                _mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
                res = self.txt2dict(_txt, _mapping)
            return res

        import os
        import camb

        _pars = camb.read_ini(os.path.join("./", "planck_2018.ini"))
        if "bb" in self.keys:
            print(f"\nProducing fiducial spectra for r={self.r} and nt={self.nt}")
            _pars.InitPower.set_params(
                As=2.100549e-9,
                ns=0.9660499,
                r=self.r,
                nt=self.nt,
                pivot_tensor=self.pivot_t,
                pivot_scalar=0.05,
                parameterization=2,
            )
            _pars.WantTensors = True
            _pars.Accuracy.AccurateBB = True
        _pars.DoLensing = True
        # _pars.Accuracy.AccuracyBoost = 2 # This helps getting an extra squeeze on the accordance of Cobaya and Fiducial spectra

        if self.debug:
            print(_pars)

        results = camb.get_results(_pars)
        res = results.get_cmb_power_spectra(
            CMB_unit="muK",
            lmax=self.lmax,
            raw_cl=False,
        )
        return self.CAMBres2dict(res)

    def prod_noise(self):
        """Produce noise power spectra or read the input ones.

        Produce the noise power spectra for a given experiment with
        inverse noise weighting of white noise in each channel (TT, EE, BB).
        Otherwise, if you specify a custom file, store that.
        """
        if self.nl_file is not None:
            if self.nl_file.endswith(".pkl"):
                with open(self.nl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            else:
                _txt = np.loadtxt(self.nl_file)
                _mapping = {"bb": 0}
                res = self.txt2dict(_txt, _mapping, apply_ellfactor=True)
            return res

        import os
        import yaml
        from yaml.loader import SafeLoader
        import healpy as hp

        assert (
            self.experiment is not None
        ), "You must specify the experiment you want to consider"
        print(f"\nComputing noise for {self.experiment}")

        with open(os.path.join("./", "experiments.yaml")) as f:
            _data = yaml.load(f, Loader=SafeLoader)

        _instrument = _data[self.experiment]
        _fwhms = np.array(_instrument["fwhm"])
        _freqs = np.array(_instrument["frequency"])
        _depth_p = np.array(_instrument["depth_p"])
        _depth_i = np.array(_instrument["depth_i"])
        _depth_p /= hp.nside2resol(self.nside, arcmin=True)
        _depth_i /= hp.nside2resol(self.nside, arcmin=True)
        _depth_p = _depth_p * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )
        _depth_i = _depth_i * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )
        _n_freq = len(_freqs)

        _ell = np.arange(0, self.lmax + 1, 1)

        _keys = ["tt", "ee", "bb"]

        _sigma = np.radians(_fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        _sigma2 = _sigma**2

        _g = np.exp(_ell * (_ell + 1) * _sigma2[:, np.newaxis])

        _pol_factor = np.array(
            [np.zeros(_sigma2.shape), 2 * _sigma2, 2 * _sigma2, _sigma2],
        )
        _pol_factor = np.exp(_pol_factor)

        _G = []
        for i, arr in enumerate(_pol_factor):
            _G.append(_g * arr[:, np.newaxis])
        _g = np.array(_G)

        res = {key: np.zeros((_n_freq, self.lmax + 1)) for key in _keys}

        res["tt"] = 1 / (_g[0, :, :] * _depth_i[:, np.newaxis] ** 2)
        res["ee"] = 1 / (_g[3, :, :] * _depth_p[:, np.newaxis] ** 2)
        res["bb"] = 1 / (_g[3, :, :] * _depth_p[:, np.newaxis] ** 2)

        res["tt"] = _ell * (_ell + 1) / (np.sum(res["tt"], axis=0)) / 2 / np.pi
        res["ee"] = _ell * (_ell + 1) / (np.sum(res["ee"], axis=0)) / 2 / np.pi
        res["bb"] = _ell * (_ell + 1) / (np.sum(res["bb"], axis=0)) / 2 / np.pi

        res["tt"][:2] = [0, 0]
        res["ee"][:2] = [0, 0]
        res["bb"][:2] = [0, 0]

        return res

    def initialize(self):
        """Initialize the fiducial spectra and the noise power spectra."""
        self.fiduCLS = self.prod_fidu()
        self.noiseCLS = self.prod_noise()

        self.fiduCOV = self.cov_filling(self.fiduCLS)
        self.noiseCOV = self.cov_filling(self.noiseCLS)

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            _field = "yy"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {_field.upper()} ---> {self.fiduCLS[_field][0:5]}")
            print(f"Noise CLs for {_field.upper()} ---> {self.noiseCLS[_field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        if self.like == "gaussian":
            self.gauss_keys = self.get_Gauss_keys()
            _sigma2 = self.sigma(self.gauss_keys, self.fiduCLS, self.noiseCLS)
            self.sigma2 = self.inv_sigma(_sigma2)

    def get_requirements(self):
        """Define requirements of the likelihood, specifying quantities calculated by a theory code are needed.

        Returns
        -------
            dict: dict with requirements
        return dictionary
        """
        requitements = {}
        requitements["Cl"] = {
            cl: self.lmax for cl in self.keys
        }  # Note that the keyword can be set to "Cl" to get the lensed ones
        if self.debug:
            requitements["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {requitements}",
            )
        return requitements

    def data_vector(self, cov):
        """Get data vector from the covariance matrix.

        Extract the data vector necessary for the Gaussian case.
        Note that this will cut the null value since some may be null
        when the fields have different values for lmax.

        Args:
        ----
            cov (ndarray): A ndarray containing the covariance matrices, with some null ones.

        Returns:
        -------
            array: array containing the data vector. Typically, this will be something like [YY, YK, KK].
        """
        return cov[np.triu_indices(self.n)][cov[np.triu_indices(self.n)] != 0]

    def chi_part(self, i=0):
        """Compute factor entering the chi-square expression in parenthesis.

        Returns
        -------
            array: array containing factor in parenthesis of the chi-square expression
        """
        if self.like == "exact":
            if self.n != 1:
                _coba = self.coba[:, :, i]
                _data = self.data[:, :, i]
                if np.linalg.det(_coba) == 0:
                    _data = self.get_reduced_data(_data)
                    _coba = self.get_reduced_data(_coba)
                M = _data @ np.linalg.inv(_coba)
                _norm = len(self.data[0, :, i][self.data[0, :, i] != 0])
                return np.trace(M) - np.linalg.slogdet(M)[1] - _norm
            else:
                M = self.data / self.coba
                return M - np.log(np.abs(M)) - 1
        elif self.like == "gaussian":
            if self.n != 1:
                _coba = self.data_vector(self.coba[:, :, i])
                _data = self.data_vector(self.data[:, :, i])
                res = (_coba - _data) @ self.sigma2[i] @ (_coba - _data)
            else:
                _coba = self.coba[0, 0, :]
                _data = self.data[0, 0, :]
                res = (_coba - _data) * self.sigma2 * (_coba - _data)
        return np.squeeze(res)

    def log_likelihood(self):
        """Compute the log likelihood.

        Returns
        -------
            float: value of the log likelihood already summed over multipoles
        """
        _ell = np.arange(self.lmin, self.lmax + 1, 1)
        if self.n != 1:
            logp_ℓ = np.zeros(_ell.shape)
            for i in range(0, self.lmax + 1 - self.lmin):
                logp_ℓ[i] = -0.5 * (2 * _ell[i] + 1) * self.chi_part(i)
        else:
            logp_ℓ = -0.5 * (2 * _ell + 1) * self.chi_part()
        return np.sum(logp_ℓ)

    def logp(self, **params_values):
        """Compute the log likelihood and pass it to Cobaya to carry on the MCMC process.

        Returns
        -------
            float: value of the log likelihood
            dict(Optional): dictionary of eventual deriver parameters computed by the likelihood function
        """
        if self.debug:
            _CAMBdata = self.provider.get_CAMBdata()
            _pars = _CAMBdata.Params
            print(_pars)

        self.cobaCLs = self.provider.get_Cl(ell_factor=True)

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            _field = "yy"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {_field.upper()} ---> {self.cobaCLs[_field][0:5]}")

        self.cobaCOV = self.cov_filling(self, self.cobaCLs)

        if self.debug:
            _ell = np.arange(0, self.lmax + 1, 1)
            plt.loglog(
                _ell[2 - self.lmin :],
                self.fiduCOV[0, 0, 2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                self.cobaCOV[0, 0, 2 - self.lmin :],
                label="Cobaya CLs",
                ls="--",
            )
            plt.loglog(
                _ell[2 - self.lmin :],
                self.noiseCOV[0, 0, 2 - self.lmin :],
                label="Noise CLs",
            )
            plt.xlim(2, None)
            plt.legend()
            plt.show()

        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        logp = self.log_likelihood()

        if self.debug:
            print(logp)
            exit()

        return logp
