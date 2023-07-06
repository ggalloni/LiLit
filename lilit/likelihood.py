import os
import pickle
from typing import Union, Optional, List

import matplotlib.pyplot as plt
import numpy as np
from cobaya.likelihood import Likelihood

from .functions import (
    CAMBres2dict,
    cov_filling,
    get_chi_correlated_gaussian,
    get_chi_exact,
    get_chi_gaussian,
    get_chi_HL,
    get_Gauss_keys,
    get_keys,
    get_masked_sigma,
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
        lmin (int or list):
            Minimum multipole to consider (default: 2).
        lmax (int or list):
            Maximum multipole to consider (default: None).
        like_approx (str, optional):
            Type of likelihood to use (default: "exact"). Currently supports "exact" and "gaussian", soon "correlated_gaussian".
        cl_file (str, dict, optional):
            Path to Cl file or dictionary of fiducial spectra (default: None).
        nl_file (str, dict, optional):
            Path to noise file or dictionary of noise spectra (default: None).
        bias_file (str, dict, optional):
            Path to bias file or dictionary of bias spectra (default: None).
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
        excluded_probes (list, optional):
            List of probes to exclude (default: None).
        debug (bool, optional):
            If True, produces more verbose output (default: None).


    Attributes:
        fields (list):
            List of fields in the data file.
        N (int):
            Number of fields.
        keys (list):
            List of keywords for the dictionaries.
        gauss_keys (list):
            List of keywords for the Gaussian likelihood (4-points).
        inverse_covariance (list):
            List of covariances for the Gaussian likelihood case, one for each multipole.
        lmin (int or list):
            Minimum multipole to consider.
        lmins (dict):
            Dictionary of lmin values.
        lmax (int or list):
            List of lmax values.
        lmaxes (dict):
            Dictionary of lmax values.
        fsky (int or list):
            List of fsky values.
        fskies (dict):
            Dictionary of fsky values.
        like_approx (str):
            Type of likelihood to use.
        cl_file (str, dict):
            Path to Cl file or dictionary of fiducial spectra.
        nl_file (str, dict):
            Path to noise file or dictionary of the noise spectra.
        bias_file (str, dict):
            Path to bias file or dictionary of the bias spectra.
        fiduCLS (dict):
            Dictionary of fiducial Cls.
        noiseCLS (dict):
            Dictionary of noise Cls.
        biasCLS (dict):
            Dictionary of bias Cls.
        fiduCOV (np.ndarray):
            Fiducial covariance matrix obtained from the corresponding dictionary.
        noiseCOV (np.ndarray):
            Noise covariance matrix obtained from the corresponding dictionary.
        cobaCLS (dict):
            Dictionary of Cobaya Cls.
        cobaCOV (np.ndarray):
            Cobaya covariance matrix obtained from the corresponding dictionary.
        data (np.ndarray):
            Data vector obtained by summing fiduCOV + noiseCOV + biasCOV.
        coba (np.ndarray):
            Cobaya vector obtained by summing cobaCOV + noiseCOV.
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
        debug (bool):
            If True, produces more output.
    """

    def __init__(
        self,
        name: str = None,
        fields: List[str] = None,
        lmin: Optional[Union[int, List[int]]] = 2,
        lmax: Union[int, List[int]] = None,
        like: str = "exact",
        cl_file: Optional[Union[dict, str]] = None,
        nl_file: Optional[Union[dict, str]] = None,
        bias_file: Optional[Union[dict, str]] = None,
        external_covariance: Optional[np.ndarray] = None,
        experiment: Optional[str] = None,
        nside: Optional[int] = None,
        r: Optional[float] = None,
        nt: Optional[float] = None,
        pivot_t: Optional[float] = 0.01,
        fsky: Union[float, List[float]] = 1,
        excluded_probes: Optional[List[str]] = None,
        debug: Optional[bool] = None,
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
        self.N = len(fields)
        self.like_approx = like
        self.excluded_probes = excluded_probes
        if excluded_probes is not None:
            for probe in excluded_probes:
                self.excluded_probes.append(probe[::-1])
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.bias_file = bias_file
        self.external_covariance = external_covariance
        if self.like_approx == "correlated_gaussian":
            assert (
                self.external_covariance is not None
            ), "You must provide a covariance matrix for the correlated Gaussian likelihood"
        self.experiment = experiment
        if self.experiment is not None:
            # Check that the user has provided the nside if an experiment is used
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.debug = debug
        self.keys = get_keys(fields=self.fields, debug=self.debug)
        if "bb" in self.keys:
            # Check that the user has provided the tensor-to-scalar ratio if a BB likelihood is used
            if cl_file is None:
                assert (
                    r is not None
                ), "You must provide the tensor-to-scalar ratio r for the fiducial production (defaul is at 0.01 Mpc^-1)"
            self.r = r
            self.nt = nt
            self.pivot_t = pivot_t

        self.check_supported_approximations()
        self.set_lmin(lmin)
        self.set_lmax(lmax)
        self.set_fsky(fsky)

        Likelihood.__init__(self, name=name)

    def check_supported_approximations(self):
        """Check that the requested likelihood approximation is actually supported.

        Note: the correlated Gaussian is supported for a single field, not multiple ones.
        """
        self.supported = ["exact", "gaussian", "correlated_gaussian", "HL"]
        assert (
            self.like_approx in self.supported
        ), f"The likelihood approximation you specified, {self.like_approx}, is not supported!"

        return

    def set_lmin(self, lmin: Union[int, List[int]]):
        """Take lmin parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note that the lmin for the cross-correlations is set to the geometrical mean of the lmin of the two fields when the likelihood approximation is not exact. This approximation has been tested and found to be accurate, at least assuming that the two masks of the two considered multipoles are very overlapped. On the other hand, lmin is set to the maximum of the two other probes for the exact likelihood. Indeed, the geometrical mean causes some issues in this case.

        Parameters:
            lmin (int or list):
                Value or list of values of lmin.
        """
        self.lmins = {}
        if isinstance(lmin, list):
            assert (
                len(lmin) == self.N
            ), "If you provide multiple lmin, they must match the number of requested fields with the same order"
            for i in range(self.N):
                for j in range(i, self.N):
                    key = self.fields[i] + self.fields[j]
                    self.lmins[key] = int(max(lmin[i], lmin[j]))
                    if self.like_approx != "exact":
                        self.lmins[key] = int(np.ceil(np.sqrt(lmin[i] * lmin[j])))
                    self.lmins[key[::-1]] = self.lmins[key]
            self.lmin = min(lmin)
        else:
            self.lmin = lmin
        return

    def set_lmax(self, lmax: Union[int, List[int]]):
        """Take lmax parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note that the lmax for the cross-correlations is set to the geometrical mean of the lmax of the two fields when the likelihood approximation is not exact. This approximation has been tested and found to be accurate, at least assuming that the two masks of the two considered multipoles are very overlapped. On the other hand, lmax is set to the minimum of the two other probes for the exact likelihood. Indeed, the geometrical mean causes some issues in this case.

        Parameters:
            lmax (int or list):
                Value or list of values of lmax.
        """
        self.lmaxs = {}
        if isinstance(lmax, list):
            assert (
                len(lmax) == self.N
            ), "If you provide multiple lmax, they must match the number of requested fields with the same order"
            for i in range(self.N):
                for j in range(i, self.N):
                    key = self.fields[i] + self.fields[j]
                    self.lmaxs[key] = int(min(lmax[i], lmax[j]))
                    if self.like_approx != "exact":
                        self.lmaxs[key] = int(np.floor(np.sqrt(lmax[i] * lmax[j])))
                    self.lmaxs[key[::-1]] = self.lmaxs[key]
            self.lmax = max(lmax)
        else:
            self.lmax = lmax
        return

    def set_fsky(self, fsky: Union[float, List[float]]):
        """Take fsky parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note that the fsky for the cross-correlations is set to the geometrical mean of the fsky of the two fields. This approximation has been tested and found to be accurate, at least assuming that the two masks of the two considered multipoles are very overlapped.

        Parameters:
            fsky (float or list):
                Value or list of values of fsky.
        """
        self.fskies = {}
        if isinstance(fsky, list):
            assert (
                len(fsky) == self.N
            ), "If you provide multiple fsky, they must match the number of requested fields with the same order"
            for i in range(self.N):
                for j in range(i, self.N):
                    key = self.fields[i] + self.fields[j]
                    self.fskies[key] = np.sqrt(fsky[i] * fsky[j])
                    self.fskies[key[::-1]] = np.sqrt(fsky[i] * fsky[j])
            self.fsky = None
        else:
            self.fsky = fsky
        return

    def get_fiducial_spectra(self):
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

        if "bb" in self.keys:
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

    def get_noise_spectra(self):
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

    def get_bias_spectra(self):
        """Store the input spectra for the bias.

        The bias spectra stored here will be add to the fiducial power spectra, but not to the ones prodeced by Cobaya. In this way, one can study the case in which something is causing a bias in the spectra reconstruction (e.g. foregrounds, systematics and such).
        """

        if isinstance(self.bias, dict):
            return self.bias
        elif not self.bias.endswith(".pkl"):
            print(
                "The file provided is not a pickle file. You should provide a pickle file containing a dictionary with keys such as 'tt', 'ee', 'te', 'bb' and 'tb'."
            )
            raise TypeError
        with open(self.bias, "rb") as pickle_file:
            return pickle.load(pickle_file)

    def compute_covariance_Cl(self):
        "Compute the covariance matrix of the Cl."
        self.gauss_keys = get_Gauss_keys(n=self.N, keys=self.keys, debug=self.debug)

        sigma2 = sigma(
            n=self.N,
            lmin=self.lmin,
            lmax=self.lmax,
            gauss_keys=self.gauss_keys,
            fiduDICT=self.fiduCLS,
            noiseDICT=self.noiseCLS,
            fsky=self.fsky,
            fskies=self.fskies,
        )

        masked_sigma2 = get_masked_sigma(
            n=self.N,
            absolute_lmin=self.lmin,
            absolute_lmax=self.lmax,
            gauss_keys=self.gauss_keys,
            sigma=sigma2,
            excluded_probes=self.excluded_probes,
            lmins=self.lmins,
            lmaxs=self.lmaxs,
        )

        self.inverse_covariance, self.mask = inv_sigma(
            lmin=self.lmin, lmax=self.lmax, masked_sigma=masked_sigma2
        )
        return

    def initialize(self):
        """Initializes the fiducial spectra and the noise power spectra."""
        self.fiduCLS = self.get_fiducial_spectra()
        self.noiseCLS = self.get_noise_spectra()

        # If a bias is provided, this biases the fiducial spectra
        if self.bias_file is not None:
            self.biasCLS = self.get_bias_spectra()
            self.fiduCLS = {
                key: self.fiduCLS.get(key, 0) + self.biasCLS.get(key, 0)
                for key in set(self.fiduCLS)
            }

        self.fiduCOV = cov_filling(
            fields=self.fields,
            excluded_probes=self.excluded_probes,
            absolute_lmin=self.lmin,
            absolute_lmax=self.lmax,
            cov_dict=self.fiduCLS,
            lmins=self.lmins,
            lmaxs=self.lmaxs,
        )
        self.noiseCOV = cov_filling(
            fields=self.fields,
            excluded_probes=self.excluded_probes,
            absolute_lmin=self.lmin,
            absolute_lmax=self.lmax,
            cov_dict=self.noiseCLS,
            lmins=self.lmins,
            lmaxs=self.lmaxs,
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

        if self.like_approx == "exact" and self.fsky is None:
            effective_fsky = 1
            for k in self.fskies.keys():
                effective_fsky *= self.fskies[k]
            self.fsky = effective_fsky ** (1 / self.N**2)

        if self.like_approx == "gaussian":
            self.compute_covariance_Cl()

        if self.like_approx == "correlated_gaussian" or self.like_approx == "HL":
            # Note that the external covariance must be invertible. This means that the covariance should start from ell = 2.
            self.inverse_covariance = np.linalg.inv(self.external_covariance)

    def get_requirements(self):
        """Defines requirements of the likelihood, specifying quantities calculated by a theory code are needed. Note that you may want to change the overall keyword from 'Cl' to 'unlensed_Cl' if you want to work without considering lensing."""
        requirements = {}
        requirements["Cl"] = {cl: self.lmax for cl in self.keys}
        if self.debug:
            requirements["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {requirements}",
            )
        return requirements

    def log_likelihood(self):
        """Convert into log likelihood and sum over multipoles."""
        if self.like_approx == "exact":
            logp_ℓ = -0.5 * np.array(
                get_chi_exact(
                    N=self.N,
                    data=self.data,
                    coba=self.coba,
                    lmin=self.lmin,
                    lmax=self.lmax,
                    fsky=self.fsky,
                )
            )
        elif self.like_approx == "gaussian":
            logp_ℓ = -0.5 * np.array(
                get_chi_gaussian(
                    N=self.N,
                    data=self.data,
                    coba=self.coba,
                    mask=self.mask,
                    inverse_covariance=self.inverse_covariance,
                    lmin=self.lmin,
                    lmax=self.lmax,
                )
            )
        elif self.like_approx == "correlated_gaussian":
            logp_ℓ = -0.5 * np.array(
                get_chi_correlated_gaussian(
                    # N=self.N,
                    data=self.data,
                    coba=self.coba,
                    # mask=self.mask,
                    inverse_covariance=self.inverse_covariance,
                    # lmin=self.lmin,
                    # lmax=self.lmax,
                )
            )
        elif self.like_approx == "HL":
            logp_ℓ = -0.5 * np.array(
                get_chi_HL(
                    data=self.data,
                    coba=self.coba,
                    inverse_covariance=self.inverse_covariance,
                )
            )
        else:
            print(
                f"You requested some likelihood approximation (i.e. {self.like_approx}) which is not supported!"
            )
            raise KeyError

        return np.sum(logp_ℓ)

    def logp(self, **params_values):
        """Gets the log likelihood and pass it to Cobaya to carry on the MCMC process."""
        if self.debug:
            CAMBdata = self.provider.get_CAMBdata()
            pars = CAMBdata.Params
            print(pars)

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

        self.cobaCOV = cov_filling(
            self.fields,
            self.excluded_probes,
            self.lmin,
            self.lmax,
            self.cobaCLS,
            self.lmins,
            self.lmaxs,
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

        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

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
