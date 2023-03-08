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
        like="exact",
        lmin=2,
        cl_file="CLs.pkl",
        nl_file="noise.pkl",
        fsky=1,
        sep="",
        debug=None,
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
        self.like = like
        self.fsky = fsky
        self.sep = sep
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.debug = debug
        self.keys = self.get_keys()

        self.lmaxs = None
        if isinstance(self.lmax, list):
            assert (
                len(self.lmax) == self.n
            ), "If you provide multiple lmax, they must match the number of requested fields with the same order"
            self.lmaxs = {}
            for i in range(self.n):
                for j in range(i, self.n):
                    key = self.fields[i] + self.sep + self.fields[j]
                    self.lmaxs[key] = min(self.lmax[i], self.lmax[j])
                    self.lmaxs[key[::-1]] = min(self.lmax[i], self.lmax[j])
            if self.debug:
                print(f"\nYou have requested the following lmax {self.lmaxs}")
            self.lmax = max(self.lmax)

        self.fskies = None
        if isinstance(self.fsky, list):
            assert (
                len(self.fsky) == self.n
            ), "If you provide multiple fsky, they must match the number of requested fields with the same order"
            self.fskies = {}
            for i in range(self.n):
                for j in range(i, self.n):
                    key = self.fields[i] + self.sep + self.fields[j]
                    self.fskies[key] = min(self.fsky[i], self.fsky[j])
                    self.fskies[key[::-1]] = min(self.fsky[i], self.fsky[j])
            if self.debug:
                print(f"\nYou have requested the following fsky {self.fskies}")
            self.fsky = None
        Likelihood.__init__(self, name=name)

    def cov_filling(self, dict):

        res = np.zeros((self.n, self.n, self.lmax + 1))
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.sep + self.fields[j]
                if self.lmaxs is not None:
                    res[i, j, : self.lmaxs[key] + 1] = dict.get(
                        key, np.zeros(self.lmaxs[key] + 1)
                    )[: self.lmaxs[key] + 1]
                else:
                    res[i, j, : self.lmax + 1] = dict.get(key, np.zeros(self.lmax + 1))[
                        : self.lmax + 1
                    ]
                res[j, i] = res[i, j]
        return res

    def get_keys(self):
        res = []
        for i in range(self.n):
            for j in range(i, self.n):
                key = self.fields[i] + self.sep + self.fields[j]
                res.append(key)
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def get_Gauss_keys(self):
        res = np.zeros(
            (int(self.n * (self.n + 1) / 2), int(self.n * (self.n + 1) / 2), 4),
            dtype=str,
        )
        for i in range(int(self.n * (self.n + 1) / 2)):
            for j in range(i, int(self.n * (self.n + 1) / 2)):
                elem = self.keys[i] + self.sep + self.keys[j]
                for k in range(4):
                    res[i, j, k] = np.asarray(list(elem)[k])
                    res[j, i, k] = res[i, j, k]
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def find_spectrum(self, dict, key):
        res = np.zeros(self.lmax + 1)
        if self.lmaxs is not None:
            if key in dict:
                res[: self.lmaxs[key] + 1] = dict[key][: self.lmaxs[key] + 1]
            else:
                res[: self.lmaxs[key] + 1] = dict.get(
                    key[::-1], np.zeros(self.lmaxs[key] + 1)
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
        res = np.zeros(
            (
                int(self.n * (self.n + 1) / 2),
                int(self.n * (self.n + 1) / 2),
                self.lmax + 1,
            )
        )
        for i in range(int(self.n * (self.n + 1) / 2)):
            for j in range(i, int(self.n * (self.n + 1) / 2)):
                AB = keys[i, j, 0] + keys[i, j, 1]
                AC = keys[i, j, 0] + keys[i, j, 2]
                CD = keys[i, j, 2] + keys[i, j, 3]
                BD = keys[i, j, 1] + keys[i, j, 3]
                AD = keys[i, j, 0] + keys[i, j, 3]
                BC = keys[i, j, 1] + keys[i, j, 2]
                C_AC = self.find_spectrum(fiduDICT, AC)
                C_BD = self.find_spectrum(fiduDICT, BD)
                C_AD = self.find_spectrum(fiduDICT, AD)
                C_BC = self.find_spectrum(fiduDICT, BC)
                N_AC = self.find_spectrum(noiseDICT, AC)
                N_BD = self.find_spectrum(noiseDICT, BD)
                N_AD = self.find_spectrum(noiseDICT, AD)
                N_BC = self.find_spectrum(noiseDICT, BC)
                if self.fsky is not None:
                    res[i, j] = (
                        (C_AC + N_AC) * (C_BD + N_BD) + (C_AD + N_AD) * (C_BC + N_BC)
                    ) / self.fsky
                else:
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
        res = np.zeros(self.lmax + 1, dtype=object)

        for i in range(self.lmax + 1):
            COV = sigma[:, :, i]
            if np.linalg.det(COV) == 0:
                idx = np.where(np.diag(COV) == 0)[0]
                COV = np.delete(COV, idx, axis=0)
                COV = np.delete(COV, idx, axis=1)
            res[i] = np.linalg.inv(COV)
        return res[2:]

    def get_reduced_data(self, mat):
        idx = np.where(np.diag(mat) == 0)[0]
        mat = np.delete(mat, idx, axis=0)
        return np.delete(mat, idx, axis=1)

    def initialize(self):

        with open(self.cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(self.nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        assert isinstance(
            self.fiduCLS, dict
        ), "fiduCLS must be a dictionary. You may want to add some line making it so."
        assert isinstance(
            self.noiseCLS, dict
        ), "noiseCLS must be a dictionary. You may want to add some line making it so."

        self.fiduCOV = self.cov_filling(self.fiduCLS)
        self.noiseCOV = self.cov_filling(self.noiseCLS)

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            field = "bb"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        if self.like == "gaussian":
            self.gauss_keys = self.get_Gauss_keys()
            sigma2 = self.sigma(self.gauss_keys, self.fiduCLS, self.noiseCLS)
            self.sigma2 = self.inv_sigma(sigma2)

    def get_requirements(self):
        req = {}
        req["unlensed_Cl"] = {cl: self.lmax for cl in self.keys}
        if self.debug:
            req["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {req}"
            )
        return req

    def data_vector(self, cov):
        return cov[np.triu_indices(self.n)][cov[np.triu_indices(self.n)] != 0]

    def chi_part(self, i=0):
        if self.like == "exact":
            if self.n != 1:
                coba = self.coba[:, :, i]
                data = self.data[:, :, i]
                if np.linalg.det(coba) == 0:
                    data = self.get_reduced_data(data)
                    coba = self.get_reduced_data(coba)
                M = data @ np.linalg.inv(coba)
                norm = len(self.data[0, :, i][self.data[0, :, i] != 0])
                print(norm)
                return np.trace(M) - np.linalg.slogdet(M)[1] - norm
            else:
                M = self.data / self.coba
                return M - np.log(np.abs(M)) - 1
        elif self.like == "gaussian":
            if self.n != 1:
                coba = self.data_vector(self.coba[:, :, i])
                data = self.data_vector(self.data[:, :, i])
                res = (coba - data) @ self.sigma2[i] @ (coba - data)
            else:
                coba = self.coba[0, 0, :]
                data = self.data[0, 0, :]
                res = (coba - data) * self.sigma2 * (coba - data)
        return np.squeeze(res)

    def log_likelihood(self):
        ell = np.arange(self.lmin, self.lmax + 1, 1)
        if self.n != 1:
            logp_ℓ = np.zeros(ell.shape)
            for i in range(0, self.lmax + 1 - self.lmin):
                logp_ℓ[i] = -0.5 * (2 * ell[i] + 1) * self.chi_part(i)
        else:
            logp_ℓ = -0.5 * (2 * ell + 1) * self.chi_part()
        return np.sum(logp_ℓ)

    def logp(self, **params_values):

        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        self.cobaCLs = self.provider.get_unlensed_Cl(ell_factor=True)

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            field = "bb"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {self.cobaCLs[field][0:5]}")

        self.cobaCOV = self.cov_filling(self.cobaCLs)

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
