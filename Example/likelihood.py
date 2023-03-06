from cobaya.likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
import pickle


class LiLit(Likelihood):
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

        res = np.zeros((self.n, self.n, self.lmax + 1))
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.fields[j]
                res[i, j] = dict.get(key, np.zeros(self.lmax + 1))[: self.lmax + 1]
                res[j, i] = res[i, j]
        return res

    def get_keys(self):
        self.keys = []
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.fields[j]
                self.keys.append(key)
        if self.debug:
            print(self.keys)

    def initialize(self):

        with open(self.cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(self.nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        self.get_keys()
        self.fiduCOV = self.cov_filling(self.fiduCLS)
        self.noiseCOV = self.cov_filling(self.noiseCLS)

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            field = "bb"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        req = {}
        req["unlensed_Cl"] = {cl: self.lmax for cl in self.keys}
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
            print(f"Keys of Cobaya CLs ---> {cobaCLs.keys()}")

            field = "bb"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {cobaCLs[field][0:5]}")

        cobaCOV = self.cov_filling(cobaCLs)

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            plt.loglog(
                ell[2 - self.lmin :],
                self.fiduCOV[0, 0, 2 - self.lmin :],
                label="Fiducial CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                cobaCOV[0, 0, 2 - self.lmin :],
                label="Cobaya CLs",
            )
            plt.xlim(2, None)
            plt.legend()
            plt.show()
            exit()

        coba = (
            cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        logp = self.log_likelihood(coba)

        return logp
