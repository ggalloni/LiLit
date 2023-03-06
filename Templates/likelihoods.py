from cobaya.likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
import pickle


class exactXX(Likelihood):
    """
    This class is a template for an exact likelihood encoding one single field, here named X.
    """

    def initialize(self):
        """
        Prepare any computation, importing any necessary code, files, etc. Note that this
        function is run only once at the very beginning of the routine.

        e.g. here you load a cl_file containing the fiducial values of the power spectra
        according to some fiducial cosmology you want to probe. Furthermore, you import
        the CLs of the noise you are considering and you check for basic consistencies.

        In this example, I precomputed the fiducial power spectra (coming as a dict),
        but you may want to produce them on the fly using a CAMB inifile, or whatever
        you want.
        """

        # Loading files
        cl_file = "path/to/cls.pkl"
        nl_file = "path/to/noise.pkl"
        with open(cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(nl_file, "rb") as pickle_file:
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
            field = "xx"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        # Now we pass the fiducial spectra to the rest of the class
        self.data = (
            self.fiduCLS["xx"][self.lmin : self.lmax + 1]
            + self.noiseCLS["xx"][self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        """
        return dictionary specifying quantities calculated by a theory code are needed

        e.g. here we need C_L^{xx} to lmax=300 assigned before
        """
        req = {}
        req["Cl"] = {"xx": self.lmax}
        if self.debug:
            req[
                "CAMBdata"
            ] = None  # This allows to get the complete product of the CAMB run
        return req

    def logp(self):
        """
        Here the likelihood is computed for each step of the chain.
        Note that this function has to return log-likelihood.

        e.g. here you calculate chi^2 using cls['xx'].
        """

        # You may want to check whether every parameter has been set as desided, thus you can print
        # the parameters set in CAMB. This also allows you to do some extra computation with the CAMB
        # results if needed.
        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        # Retrieve CLs for the step
        cobaCLs = self.provider.get_Cl(ell_factor=True)

        # Also here you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of Cobaya CLs ---> {cobaCLs.keys()}")

            # For a field contained in the fiducial (here YY):
            field = "xx"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {cobaCLs[field][0:5]}")

        coba = (
            cobaCLs["xx"][self.lmin : self.lmax + 1]
            + self.noiseCLS["xx"][self.lmin : self.lmax + 1]
        )

        # At this point, you may want to check whether everything is consistent in terms of
        # normalizations, overall values, etc... Therefore here you can plot the considered field
        # and you can compare the fiducial and the cobaya spectra (+ noise eventually)
        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            field = "xx"
            plt.loglog(
                ell[2 - self.lmin :],
                self.fiduCLS[field][2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                cobaCLs[field][2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                self.noiseCLS[field][2 - self.lmin :],
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
        ell = np.arange(self.lmin, self.lmax + 1, 1)
        M = self.data / coba
        logp_ℓ = -0.5 * (2 * ell + 1) * self.fsky * (M - np.log(np.abs(M)) - 1)

        return np.sum(logp_ℓ)


class exactYYYKKK(Likelihood):
    """
    This class is a template for an exact likelihood encoding two fields, here named here Y and K.
    """

    def initialize(self):
        """
        Prepare any computation, importing any necessary code, files, etc. Note that this
        function is run only once at the very beginning of the routine.

        e.g. here you load a cl_file containing the fiducial values of the power spectra
        according to some fiducial cosmology you want to probe. Furthermore, you import
        the CLs of the noise you are considering and you check for basic consistencies.

        In this example, I precomputed the fiducial power spectra (coming as a dict),
        but you may want to produce them on the fly using a CAMB inifile, or whatever
        you want.
        """

        # Loading files
        cl_file = "path/to/cls.pkl"
        nl_file = "path/to/noise.pkl"
        with open(cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(nl_file, "rb") as pickle_file:
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
            field = "yy"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper} ---> {self.noiseCLS[field][0:5]}")

        # Now we pass the fiducial spectra to the rest of the class

        self.fiduCOV = np.array(
            [
                [self.fiduCLS["yy"], self.fiduCLS["yk"]],
                [self.fiduCLS["yk"], self.fiduCLS["kk"]],
            ]
        )
        self.noiseCOV = np.array(
            [
                [self.noiseCLS["yy"], self.noiseCLS["yk"]],
                [self.noiseCLS["yk"], self.noiseCLS["kk"]],
            ]
        )

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        """
        return dictionary specifying quantities calculated by a theory code are needed

        e.g. here we need C_L^{xx} to lmax=300 assigned before
        """
        req = {}
        req["Cl"] = {"yy": self.lmax, "kk": self.lmax, "yk": self.lmax}
        if self.debug:
            req[
                "CAMBdata"
            ] = None  # This allows to get the complete product of the CAMB run
        return req

    def logp(self):
        """
        Here the likelihood is computed for each step of the chain.
        Note that this function has to return log-likelihood.

        e.g. here you calculate chi^2 using cls['yy'], cls['yk'] and cls['kk'].
        """

        # You may want to check whether every parameter has been set as desided, thus you can print
        # the parameters set in CAMB. This also allows you to do some extra computation with the CAMB
        # results if needed.
        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        # Retrieve CLs for the step
        self.cobaCLs = self.provider.get_Cl(ell_factor=True)

        # Also here you may want to check what fields are contained in the fiducial
        # dictionaries, the syntax of the keywords, or whether the lmin is 0, or 2. In case
        # something is not consistent you may want to add some extra lines to make them so.
        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            # For a field contained in the fiducial (here YY):
            field = "yy"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {self.cobaCLs[field][0:5]}")

        self.cobaCOV = np.array(
            [
                [self.cobaCLS["yy"], self.cobaCLS["yk"]],
                [self.cobaCLS["yk"], self.cobaCLS["kk"]],
            ]
        )
        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # At this point, you may want to check whether everything is consistent in terms of
        # normalizations, overall values, etc... Therefore here you can plot the considered field
        # and you can compare the fiducial and the cobaya spectra (+ noise eventually)
        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            field = "yy"
            plt.loglog(
                ell[2 - self.lmin :],
                self.fiduCLS[field][2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                self.cobaCLs[field][2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                self.noiseCLS[field][2 - self.lmin :],
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
        ell = np.arange(self.lmin, self.lmax + 1, 1)
        logp_ℓ = np.zeros(ell.shape)

        for i in range(0, self.lmax + 1 - self.lmin):
            if i <= self.lmaxKK:
                M = self.data[:, :, i] @ np.linalg.inv(self.coba[:, :, i])
                norm = len(self.data[0, :, i])
                logp_ℓ[i] = (
                    -0.5
                    * (2 * ell[i] + 1)
                    * self.fsky
                    * (np.trace(M) - np.linalg.slogdet(M)[1] - norm)
                )
            else:
                M = self.data[0, 0, i] / self.coba[0, 0, i]
                norm = 1
                logp_ℓ[i] = (
                    -0.5 * (2 * ell[i] + 1) * self.fsky * (M - np.log(np.abs(M)) - norm)
                )

        return np.sum(logp_ℓ)


class LiLit(Likelihood):
    """
    This is a far more flexible likelihood than what presented above. LiLit encodes both
    the one field and the two fields cases. In fact, this implementation is independent
    from the number of fields (as soon as you are consistent with yourself in what you
    provide as an input). I suggest to use this class, since the others are just quick
    examples of how Cobaya works.
    """

    def __init__(
        self,
        name=None,
        fields=None,
        lmax=None,
        lmin=2,
        cl_file="CLs.pkl",
        nl_file="noise.pkl",
        fsky=0.5,
        debug=False,
    ):
        """
        With the initializations of LiLit, I store useful quantities for the rest of the computation. These can
        be passed to the class at declaration. Note that the one required for a healthy
        run are initialized to None, so that you must provide them to avoid an error.
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
        self.lmax = lmax
        self.fsky = fsky
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.debug = debug
        Likelihood.__init__(self, name=name)

    def cov_filling(self, dict):
        """
        This function takes self and a dictionary as input and returns the covariance matrix
        of the considered fields, in a shape equal to (num_fields x num_fields x lmax).

        Args:
            dict: input dictionary of spectra

        Returns:
            ndarray: covariance matrix of the considered fields of shape (num_fields x num_fields x lmax)
        """
        res = np.zeros((self.n, self.n, self.lmax + 1))
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.fields[j]
                self.keys.append(key)
                res[i, j] = dict.get(key, np.zeros(self.lmax + 1))[: self.lmax + 1]
                res[j, i] = res[i, j]
        return res

    def get_keys(self):
        """
        This extract the keys that has to be used as a function of the requested fields
        """
        self.keys = []
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.fields[j]
                self.keys.append(key)

    def initialize(self):
        """
        This time, at this point all the useful quantities are already stored, thus I proceed to initialize the fiducial
        spectra.
        """

        with open(self.cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(self.nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        self.keys = self.get_keys(self)
        self.fiduCOV = self.cov_filling(self, self.fiduCLS)
        self.noiseCOV = self.cov_filling(self, self.noiseCLS)

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            field = "yy"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        req = {}
        req["unlensed_Cl"] = {
            cl: self.lmax for cl in self.keys
        }  # Note that the keyword can be set to "Cl" to get the lensed ones
        if self.debug:
            req["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {req}"
            )
        return req

    def log_likelihood(self, theo):
        ell = np.arange(self.lmin, self.lmax + 1, 1)
        if self.n != 1:
            logp_ℓ = np.zeros(ell.shape)
            for i in range(0, self.lmax + 1 - self.lmin):
                M = self.data[:, :, i] @ np.linalg.inv(theo[:, :, i])
                norm = len(self.data[0, :, i])
                logp_ℓ[i] = (
                    -0.5
                    * (2 * ell[i] + 1)
                    * self.fsky
                    * (np.trace(M) - np.linalg.slogdet(M)[1] - norm)
                )
        else:
            M = self.data / theo
            logp_ℓ = -0.5 * (2 * ell + 1) * self.fsky * (M - np.log(np.abs(M)) - 1)
        return np.sum(logp_ℓ)

    def logp(self, **params_values):

        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        cobaCLs = self.provider.get_unlensed_Cl(ell_factor=True)

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            field = "yy"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {self.cobaCLs[field][0:5]}")

        cobaCOV = self.cov_filling(self, cobaCLs)

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            plt.loglog(
                ell[2 - self.lmin :],
                self.fiduCOV[0, 0, 2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                self.cobaCOV[0, 0, 2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.legend()
            plt.show()
            exit()

        coba = (
            cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        logp = self.log_likelihood(self, coba)

        return logp
