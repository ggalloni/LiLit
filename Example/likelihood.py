from cobaya.likelihood import Likelihood
import numpy as np
import matplotlib.pyplot as plt
import pickle


class exactBB(Likelihood):
    def initialize(self):

        cl_file = "CLs.pkl"
        nl_file = "noise.pkl"
        with open(cl_file, "rb") as pickle_file:
            self.fiduCLS = pickle.load(pickle_file)
        with open(nl_file, "rb") as pickle_file:
            self.noiseCLS = pickle.load(pickle_file)

        self.debug = False
        self.lmin = 2
        self.lmax = 300
        self.fsky = 0.5

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            field = "bb"
            print(f"\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        self.data = (
            self.fiduCLS["bb"][self.lmin : self.lmax + 1]
            + self.noiseCLS["bb"][self.lmin : self.lmax + 1]
        )

    def get_requirements(self):
        req = {}
        req["unlensed_Cl"] = {"bb": self.lmax}
        if self.debug:
            req["CAMBdata"] = None
        return req

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

        coba = (
            cobaCLs["bb"][self.lmin : self.lmax + 1]
            + self.noiseCLS["bb"][self.lmin : self.lmax + 1]
        )

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            field = "bb"
            plt.loglog(
                ell[2 - self.lmin :],
                self.fiduCLS[field][2 - self.lmin : self.lmax + 1],
                label="Fiducial CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                self.noiseCLS[field][2 - self.lmin : self.lmax + 1],
                label="Noise CLs",
            )
            plt.loglog(
                ell[2 - self.lmin :],
                cobaCLs[field][2 - self.lmin : self.lmax + 1],
                label="Cobaya CLs",
            )
            plt.xlim(2, self.lmax)
            plt.legend()
            plt.show()
            exit()

        ell = np.arange(self.lmin, self.lmax + 1, 1)
        M = self.data / coba
        logp_ℓ = -0.5 * (2 * ell + 1) * self.fsky * (M - np.log(np.abs(M)) - 1)

        return np.sum(logp_ℓ)
