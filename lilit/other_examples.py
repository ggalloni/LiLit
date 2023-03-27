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
