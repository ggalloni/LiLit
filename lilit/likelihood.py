import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from cobaya.likelihood import Likelihood

from .functions import (
    CAMBres2dict,
    cov_filling,
    get_Gauss_keys,
    get_keys,
    inv_sigma,
    sigma,
)


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
        cl_file (str, dict, optional):
            Path to Cl file or dictionary of fiducial spectra (default: None).
        nl_file (str, dict, optional):
            Path to noise file or dictionary of noise spectra (default: None).
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
        cl_file (str, dict):
            Path to Cl file or dictionary of fiducial spectra.
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
        nl_file (str, dict):
            Path to noise file or dictionary of the noise spectra.
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
        self.keys = get_keys(fields=self.fields, debug=self.debug)
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

    def get_reduced_data(self, mat):
        """Find the reduced data eliminating the singularity of the matrix.

        Cuts the row and column corresponding to a zero diagonal value. Indeed, in case of different lmax, or lmin, for the fields, you will have singular marices.

        Parameters:
            ndarray (np.ndarray):
                A ndarray containing the covariance matrices, with some singular ones.
        """
        idx = np.where(np.diag(mat) == 0)[0]
        return np.delete(np.delete(mat, idx, axis=0), idx, axis=1)

    def prod_fidu(self):
        """Produce fiducial spectra or read the input ones.

        If the user has not provided a Cl file, this function will produce the fiducial power spectra starting from the CAMB inifile for Planck2018. The extra keywords defined will maximize the accordance between the fiducial Cls and the ones obtained from Cobaya. If B-modes are requested, the tensor-to-scalar ratio and the spectral tilt will be set to the requested values. Note that if you do not provide a tilt, this will follow the standard single-field consistency relation. If instead you provide a custom file, stores that.
        """

        if self.cl_file is not None:
            if isinstance(self.cl_file, dict):
                return self.cl_file
            elif not self.cl_file.endswith(".pkl"):
                print(
                    "The file provided is not a pickle file. You should provide a pickle file containing a dictionary with keys such as 'tt', 'ee', 'te', 'bb' and 'tb'."
                )
                raise TypeError
            with open(self.cl_file, "rb") as pickle_file:
                return pickle.load(pickle_file)
        try:
            import camb
        except ImportError:
            print("CAMB seems to be not installed. Check the requirements.")

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

        if self.debug:
            print(pars)

        results = camb.get_results(pars)
        res = results.get_cmb_power_spectra(
            CMB_unit="muK",
            lmax=self.lmax,
            raw_cl=False,
        )
        return CAMBres2dict(res, self.keys)

    def prod_noise(self):
        """Produce noise power spectra or read the input ones.

        If the user has not provided a noise file, this function will produce the noise power spectra for a given experiment with inverse noise weighting of white noise in each channel (TT, EE, BB). Note that you may want to have a look at the procedure since it is merely a place-holder. Indeed, you should provide a more realistic file from which to read the noise spectra, given that inverse noise weighting severely underestimates the amount of noise. If instead you provide the proper custom file, this method stores that.
        """
        if self.nl_file is not None:
            if isinstance(self.nl_file, dict):
                return self.nl_file
            elif not self.nl_file.endswith(".pkl"):
                print(
                    "The file provided for the noise is not a pickle file. You should provide a pickle file containing a dictionary with keys such as 'tt', 'ee', 'te', 'bb' and 'tb'."
                )
                raise TypeError
            with open(self.nl_file, "rb") as pickle_file:
                return pickle.load(pickle_file)

        print(
            "***WARNING***: the inverse noise weighting performed here severely underestimates \
            the actual noise level of LiteBIRD. You should provide an input \
            noise power spectrum with a more realistic noise."
        )

        try:
            import healpy as hp
            import yaml
            from yaml.loader import SafeLoader
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

        instrument = data[self.experiment]

        fwhms = np.array(instrument["fwhm"])

        freqs = np.array(instrument["frequency"])

        depth_p = np.array(instrument["depth_p"])
        depth_i = np.array(instrument["depth_i"])

        # Convert to microK-rad
        depth_p /= hp.nside2resol(self.nside, arcmin=True)
        depth_i /= hp.nside2resol(self.nside, arcmin=True)
        depth_p *= np.sqrt(hp.nside2pixarea(self.nside, degrees=False))
        depth_i *= np.sqrt(hp.nside2pixarea(self.nside, degrees=False))

        n_freq = len(freqs)

        ell = np.arange(0, self.lmax + 1, 1)

        keys = ["tt", "ee", "bb"]

        sigma = np.radians(fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        sigma2 = sigma**2

        g = np.exp(ell * (ell + 1) * sigma2[:, np.newaxis])

        pol_factor = np.array(
            [np.zeros(sigma2.shape), 2 * sigma2, 2 * sigma2, sigma2],
        )

        pol_factor = np.exp(pol_factor)

        # Calculate the Gaussian beam function for each polarization
        G = []
        for i, arr in enumerate(pol_factor):
            G.append(g * arr[:, np.newaxis])
        g = np.array(G)

        res = {key: np.zeros((n_freq, self.lmax + 1)) for key in keys}

        res["tt"] = 1 / (g[0, :, :] * depth_i[:, np.newaxis] ** 2)
        res["ee"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)
        res["bb"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)

        res["tt"] = ell * (ell + 1) / (np.sum(res["tt"], axis=0)) / 2 / np.pi
        res["ee"] = ell * (ell + 1) / (np.sum(res["ee"], axis=0)) / 2 / np.pi
        res["bb"] = ell * (ell + 1) / (np.sum(res["bb"], axis=0)) / 2 / np.pi

        res["tt"][:2] = [0, 0]
        res["ee"][:2] = [0, 0]
        res["bb"][:2] = [0, 0]

        return res

    def initialize(self):
        """Initializes the fiducial spectra and the noise power spectra."""
        self.fiduCLS = self.prod_fidu()
        self.noiseCLS = self.prod_noise()

        self.fiduCOV = cov_filling(
            self.fields, self.lmin, self.lmax, self.fiduCLS, self.lmins, self.lmaxs
        )
        self.noiseCOV = cov_filling(
            self.fields, self.lmin, self.lmax, self.noiseCLS, self.lmins, self.lmaxs
        )

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            print("\nPrinting the first few values to check that it starts from 0...")
            field = list(self.fiduCLS.keys())[1]
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            field = list(self.noiseCLS.keys())[1]
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        if self.like == "gaussian":
            self.gauss_keys = get_Gauss_keys(n=self.n, keys=self.keys, debug=self.debug)
            sigma2 = sigma(
                self.n,
                self.lmin,
                self.lmax,
                self.gauss_keys,
                self.fiduCLS,
                self.noiseCLS,
                self.fsky,
                self.fskies,
            )
            self.sigma2 = inv_sigma(self.lmin, self.lmax, sigma2)

    def get_requirements(self):
        """Defines requirements of the likelihood, specifying quantities calculated by a theory code are needed. Note that you may want to change the overall keyword from 'Cl' to 'unlensed_Cl' if you want to work without considering lensing."""
        requitements = {}
        requitements["Cl"] = {cl: self.lmax for cl in self.keys}
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

        upper_triang = cov[np.triu_indices(self.n)]
        # Get indices of null diagonal elements
        idx = np.where(upper_triang == 0)[0]
        # Remove corresponding rows and columns
        upper_triang = np.delete(upper_triang, idx, axis=0)

        return upper_triang, idx

    def chi_exact(self, i=0):
        """Computes proper chi-square term for the exact likelihood case.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        if self.n != 1:
            ell = np.arange(0, self.lmax + 1, 1)
            coba = self.coba[:, :, i]
            data = self.data[:, :, i]
            if (
                np.linalg.det(coba) == 0
            ):  # If the determinant is null, we need to reduce the covariance matrix
                data = self.get_reduced_data(data)
                coba = self.get_reduced_data(coba)
            M = np.linalg.solve(coba, data)
            return (2 * ell[i] + 1) * (
                np.trace(M) - np.linalg.slogdet(M)[1] - data.shape[0]
            )
        else:
            ell = np.arange(2, self.lmax + 1, 1)
            M = self.data / self.coba
            return (2 * ell + 1) * (M - np.log(np.abs(M)) - 1)

    def chi_gaussian(self, i=0):
        """Computes proper chi-square term for the Gaussian likelihood case.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        if self.n != 1:
            coba, idx = self.data_vector(self.coba[:, :, i])
            data, _ = self.data_vector(self.data[:, :, i])
            COV = np.delete(self.sigma2[:, :, i], idx, axis=0)
            COV = np.delete(COV, idx, axis=1)
            return (coba - data) @ COV @ (coba - data)
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
        if self.like == "exact":
            return self.chi_exact(i)
        elif self.like == "gaussian":
            return self.chi_gaussian(i)
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
                logp_ℓ[i] = -0.5 * self.compute_chi_part(i)
        else:
            logp_ℓ = -0.5 * self.compute_chi_part()
        # Sum the log likelihood over multipoles
        return np.sum(logp_ℓ)

    def logp(self, **params_values):
        """Gets the log likelihood and pass it to Cobaya to carry on the MCMC process."""
        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

        # Get the Cls from Cobaya
        self.cobaCLS = self.provider.get_Cl(ell_factor=True)
        ell = np.arange(0, self.lmax + 1, 1)
        for key, value in self.cobaCLS.items():
            if key == "pp":
                value[2 : self.lmax + 1] = (
                    value[2 : self.lmax + 1] / (ell * (ell + 1))[2:]
                )
            elif "p" in key:
                value[2 : self.lmax + 1] = (
                    value[2 : self.lmax + 1] / np.sqrt(ell * (ell + 1))[2:]
                )
            self.cobaCLS[key] = value[: self.lmax + 1]

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLS.keys()}")

            field = list(self.cobaCLS.keys())[1]
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {field.upper()} ---> {self.cobaCLS[field][0:5]}")

        # Fill the covariance matrix with the Cls from Cobaya
        self.cobaCOV = cov_filling(
            self.fields, self.lmin, self.lmax, self.cobaCLS, self.lmins, self.lmaxs
        )

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            obs1 = 0
            obs2 = 0
            plt.plot(ell, self.fiduCOV[obs1, obs2, :], label="Fiducial CLs")
            plt.plot(ell, self.cobaCOV[obs1, obs2, :], label="Cobaya CLs", ls="--")
            plt.plot(ell, self.noiseCOV[obs1, obs2, :], label="Noise CLs")
            plt.loglog()
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
            print(f"Log-posterior -->  {logp}")
            exit()

        return logp


__all__ = ["LiLit"]

__docformat__ = "google"
__pdoc__ = {}
__pdoc__[
    "Likelihood"
] = "Likelihood class from Cobaya, refer to Cobaya documentation for more information."
