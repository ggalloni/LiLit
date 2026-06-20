import matplotlib.pyplot as plt
import numpy as np
from cobaya.likelihood import Likelihood

from .binning import get_binning
from .core import ChiSquareCalculator, ChiSquareMethod
from .data import (
    BiasSpectraLoader,
    FiducialSpectraLoader,
    FiduGuessSpectraLoader,
    NoiseSpectraLoader,
    OffsetSpectraLoader,
)
from .math import (
    cov_filling,
    get_Gauss_keys,
    get_keys,
    get_masked_sigma,
    inv_sigma,
    sigma,
)


class LiLit(Likelihood):
    """Class defining the Likelihood for LiteBIRD (LiLit).

    Within LiLit, the most relevant study cases of LiteBIRD (T, E, B) are already tested
    and working. So, if you need to work with those, you should not need to look into
    the actual definition of the likelihood function, since you can promptly start
    running your MCMCs. Despite this, you should provide to the likelihood some file
    where to find the proper LiteBIRD noise power spectra, given that LiLit is
    implementing a simple inverse noise weighting just as a place-holder for something
    more realistic. As regards lensing, LiLit will need you to pass the reconstruction
    noise, since its computation is not coded, thus there is
    no place-holder for lensing.

    Parameters:
        name (str):
            The name for the likelihood, used in the output. It is necessary to pass it
            to LiLit. (default: None).
        fields (list):
            List of fields in the data file (default: None).
        lmin (int or list):
            Minimum multipole to consider (default: 2).
        lmax (int or list):
            Maximum multipole to consider (default: None).
        like_approx (str, optional):
            Type of likelihood to use (default: "exact"). Currently supports "exact"
            and "gaussian", soon "correlated_gaussian".
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
            List of covariances for the Gaussian likelihood case,
            one for each multipole.
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
        fields: list[str] = None,
        lmin: int | list[int] | None = 2,
        lmax: int | list[int] = None,
        like: str = "exact",
        cl_file: dict | str | None = None,
        nl_file: dict | str | None = None,
        bias_file: dict | str | None = None,
        fidu_guess_file: dict | str | None = None,
        offset_file: dict | str | None = None,
        external_covariance: np.ndarray | None = None,
        experiment: str | None = None,
        nside: int | None = None,
        r: float | None = None,
        nt: float | None = None,
        pivot_t: float | None = 0.01,
        fsky: float | list[float] = 1,
        excluded_probes: list[str] | None = None,
        want_binning: bool | None = False,
        delta_ell: int | None = 10,
        binning_transition: int | None = 35,
        binning_mode: str = "none",
        bin_lmins: list | np.ndarray | None = None,
        bin_lmaxs: list | np.ndarray | None = None,
        effective_dof: list | np.ndarray | None = None,
        debug: bool | None = None,
    ):
        # Check that the user has provided the name of the likelihood
        assert name is not None, (
            "You must provide the name of the likelihood (e.g. 'BB' or 'TTTEEE')"
        )
        # Check that the user has provided the fields
        assert fields is not None, "You must provide the fields "
        "(e.g. 'b' or ['t', 'e'])"
        # Check that the user has provided the maximum multipole
        assert lmax is not None, "You must provide the lmax (e.g. 300)"

        self.fields = fields
        self.N = len(fields)
        self.like_approx = like
        self.chi_method = ChiSquareMethod(like.lower())
        self.excluded_probes = excluded_probes
        if excluded_probes is not None:
            # Create a copy to avoid modifying the list we're iterating over
            self.excluded_probes = list(excluded_probes)
            for probe in excluded_probes:
                self.excluded_probes.append(probe[::-1])
            self.excluded_probes = list(set(self.excluded_probes))
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.bias_file = bias_file
        self.fidu_guess_file = fidu_guess_file
        self.offset_file = offset_file
        self.external_covariance = external_covariance
        if self.like_approx == "correlated_gaussian":
            assert self.external_covariance is not None, (
                "You must provide a covariance matrix for the correlated Gaussian "
                "likelihood"
            )
        if self.like_approx == "HL" or self.like_approx == "lollipop":
            assert self.fidu_guess_file is not None, (
                "You must provide a fiducial spectrum for the H&L likelihood"
            )
            assert self.external_covariance is not None, (
                "You must provide a covariance matrix for the H&L/LoLLiPoP likelihood"
            )
        self.experiment = experiment
        if self.experiment is not None:
            # Check that the user has provided the nside if an experiment is used
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.want_binning = want_binning
        self.delta_ell = delta_ell
        self.binning_transition = binning_transition
        # Externally-binned (bandpower) input path. Opt-in and backward-compatible:
        # when binned=True, cl_file/nl_file are interpreted as already-binned bandpowers
        # (one value per bin defined by bin_lmins/bin_lmaxs, inclusive edges) and the
        # likelihood is evaluated directly in bandpower space with a Knox covariance.
        # Binning mode (replaces the legacy, never-wired want_binning flag):
        #   "none"     -> standard per-multipole likelihood (unchanged default);
        #   "external" -> cl_file/nl_file are ALREADY bandpowers (length nbins);
        #   "internal" -> cl_file/nl_file are per-multipole and get binned here.
        # Both binned modes then share the exact same bandpower algebra downstream.
        self.binning_mode = (binning_mode or "none").lower()
        assert self.binning_mode in ("none", "internal", "external"), (
            f"binning_mode must be one of 'none', 'internal', 'external', "
            f"got '{self.binning_mode}'."
        )
        # The legacy want_binning is dead code (never read by any chi-square); keep it
        # accepted for backward compatibility but forbid mixing it with binning_mode.
        if self.binning_mode != "none" and want_binning:
            raise ValueError(
                "want_binning (legacy, no-op) cannot be combined with binning_mode; "
                "use binning_mode='internal' instead."
            )
        self.bin_lmins = None if bin_lmins is None else np.asarray(bin_lmins, dtype=int)
        self.bin_lmaxs = None if bin_lmaxs is None else np.asarray(bin_lmaxs, dtype=int)
        self.effective_dof = (
            None if effective_dof is None else np.asarray(effective_dof, dtype=float)
        )
        if self.binning_mode != "none":
            assert self.like_approx in ("gaussian", "exact", "correlated_gaussian"), (
                "binning_mode != 'none' currently supports like='gaussian', 'exact' "
                "or 'correlated_gaussian' (the latter needs external_covariance)."
            )
            assert self.N == 1, (
                "binning_mode != 'none' currently supports a single field (e.g. ['B'])."
            )
            # Bandpower edges are either given explicitly (bin_lmins/bin_lmaxs) or built
            # as uniform bands of width delta_ell from lmin to
            # lmax (see _build_bin_edges).
            if (self.bin_lmins is None) != (self.bin_lmaxs is None):
                raise ValueError(
                    "Provide both bin_lmins and bin_lmaxs, or neither "
                    "(neither -> uniform bins of width delta_ell from lmin to lmax)."
                )
            if self.bin_lmins is not None and len(self.bin_lmins) != len(self.bin_lmaxs):
                raise ValueError("bin_lmins and bin_lmaxs must have the same length.")
            if self.bin_lmins is None and not self.delta_ell:
                raise ValueError(
                    "binning_mode != 'none' needs either bin_lmins/bin_lmaxs "
                    "or delta_ell."
                )
        self.debug = debug
        self.keys = get_keys(fields=self.fields, debug=self.debug)

        # Always set these attributes, even if not using BB
        self.r = r
        self.nt = nt
        self.pivot_t = pivot_t

        if "bb" in self.keys:
            # Check that the user has provided the tensor-to-scalar ratio if a BB
            # likelihood is used
            if cl_file is None:
                assert r is not None, (
                    "You must provide the tensor-to-scalar ratio r for the fiducial "
                    "production (default is at 0.01 Mpc^-1)"
                )

        self.check_supported_approximations()
        self.set_lmin(lmin)
        self.set_lmax(lmax)
        self.set_fsky(fsky)

        Likelihood.__init__(self, name=name)

    def check_supported_approximations(self):
        """Check that the requested likelihood approximation is actually supported.

        Note: the correlated Gaussian is supported for a single field,
        not multiple ones.
        """
        self.supported = ["exact", "gaussian", "correlated_gaussian", "HL", "lollipop"]
        assert self.like_approx in self.supported, (
            f"The likelihood approximation you specified, {self.like_approx}, is not "
            f"supported! Available options are {self.supported}"
        )

        return

    def set_lmin(self, lmin: int | list[int]):
        """Take lmin parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note
        that the lmin for the cross-correlations is set to the geometrical mean of the
        lmin of the two fields when the likelihood approximation is not exact. This
        approximation has been tested and found to be accurate, at least assuming that
        the two masks of the two considered multipoles are very overlapped. On the
        other hand, lmin is set to the maximum of the two other probes for the exact
        likelihood. Indeed, the geometrical mean causes some issues in this case.

        Parameters:
            lmin (int or list):
                Value or list of values of lmin.
        """
        self.lmins = {}
        if isinstance(lmin, list):
            assert len(lmin) == self.N, (
                "If you provide multiple lmin, they must match the number of requested "
                "fields with the same order"
            )
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

    def set_lmax(self, lmax: int | list[int]):
        """Take lmax parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note
        that the lmax for the cross-correlations is set to the geometrical mean of the
        lmax of the two fields when the likelihood approximation is not exact. This
        approximation has been tested and found to be accurate, at least assuming that
        the two masks of the two considered multipoles are very overlapped. On the
        other hand, lmax is set to the minimum of the two other probes for the exact
        likelihood. Indeed, the geometrical mean causes some issues in this case.

        Parameters:
            lmax (int or list):
                Value or list of values of lmax.
        """
        self.lmaxs = {}
        if isinstance(lmax, list):
            assert len(lmax) == self.N, (
                "If you provide multiple lmax, they must match the number of requested "
                "fields with the same order"
            )
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

    def set_fsky(self, fsky: float | list[float]):
        """Take fsky parameter and set the corresponding attributes.

        This handles automatically the case of a single value or a list of values. Note
        that the fsky for the cross-correlations is set to the geometrical mean of the
        fsky of the two fields. This approximation has been tested and found to be
        accurate, at least assuming that the two masks of the two considered multipoles
        are very overlapped.

        Parameters:
            fsky (float or list):
                Value or list of values of fsky.
        """
        self.fskies = {}
        if isinstance(fsky, list):
            assert len(fsky) == self.N, (
                "If you provide multiple fsky, they must match the number of requested "
                "fields with the same order"
            )
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

        If the user has not provided a Cl file, this function will produce the fiducial
        power spectra starting from the CAMB inifile for Planck2018. The extra keywords
        defined will maximize the accordance between the fiducial Cls and the ones
        obtained from Cobaya. If B-modes are requested, the tensor-to-scalar ratio and
        the spectral tilt will be set to the requested values. Note that if you do not
        provide a tilt, this will follow the standard single-field consistency
        relation. If instead you provide a custom file, stores that.
        """
        loader = FiducialSpectraLoader(
            cl_file=self.cl_file,
            keys=self.keys,
            lmax=self.lmax,
            r=self.r,
            nt=self.nt,
            pivot_t=self.pivot_t,
            debug=self.debug,
        )
        return loader.load()

    def get_noise_spectra(self):
        """Produce noise power spectra or read the input ones.

        If the user has not provided a noise file, this function will produce the noise
        power spectra for a given experiment with inverse noise weighting of white
        noise in each channel (TT, EE, BB). Note that you may want to have a look at
        the procedure since it is merely a place-holder. Indeed, you should provide a
        more realistic file from which to read the noise spectra, given that inverse
        noise weighting severely underestimates the amount of noise. If instead you
        provide the proper custom file, this method stores that.
        """
        loader = NoiseSpectraLoader(
            nl_file=self.nl_file,
            experiment=self.experiment,
            lmax=self.lmax,
            nside=self.nside,
        )
        return loader.load()

    def get_bias_spectra(self):
        """Store the input spectra for the bias.

        The bias spectra stored here will be add to the fiducial power spectra, but not
        to the ones prodeced by Cobaya. In this way, one can study the case in which
        something is causing a bias in the spectra reconstruction (e.g. foregrounds,
        systematics and such).
        """
        loader = BiasSpectraLoader(bias_file=self.bias_file)
        return loader.load()

    def get_fidu_guess_spectra(self):
        """Store the input spectra for a fiducial guess on the spectrum of data.

        The bias spectra stored here will be add to the fiducial power spectra, but not
        to the ones produced by Cobaya. In this way, one can study the case in which
        something is causing a bias in the spectra reconstruction (e.g. foregrounds,
        systematics and such).
        """
        loader = FiduGuessSpectraLoader(fidu_guess_file=self.fidu_guess_file)
        return loader.load()

    def get_offset_spectra(self):
        """Store the input spectra for the offset (H&L approximation).

        The bias spectra stored here will be add to the fiducial power spectra, but not
        to the ones produced by Cobaya. In this way, one can study the case in which
        something is causing a bias in the spectra reconstruction (e.g. foregrounds,
        systematics and such).
        """
        loader = OffsetSpectraLoader(offset_file=self.offset_file)
        return loader.load()

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
            sigma_array=sigma2,
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
        if self.binning_mode != "none":
            self._initialize_binned()
            return
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

        self.bins = None
        if self.want_binning:
            self.bins = get_binning(
                lmin=self.lmin,
                lmax=self.lmax,
                delta_ell=self.delta_ell,
                transition=self.binning_transition,
            )

        if self.like_approx == "exact" and self.fsky is None:
            effective_fsky = 1
            for k in self.fskies.keys():
                effective_fsky *= self.fskies[k]
            self.fsky = effective_fsky ** (1 / self.N**2)

        if self.like_approx == "gaussian":
            self.compute_covariance_Cl()

        if (
            self.like_approx == "correlated_gaussian"
            or self.like_approx == "HL"
            or self.like_approx == "lollipop"
        ):
            # Note that the external covariance must be invertible. This means that
            # the covariance should start from ell = 2.
            try:
                self.inverse_covariance = np.linalg.inv(self.external_covariance)
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f"Cannot invert external covariance matrix for {self.like_approx} "
                    f"likelihood. The matrix must be square, non-singular, "
                    f"and positive definite. Original error: {e}"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Error processing external covariance matrix: {e}. "
                    f"Please check that the matrix has the correct shape and contains "
                    f"valid numerical values."
                ) from e

        if self.like_approx == "HL" or self.like_approx == "lollipop":
            self.fidu_guessCLS = self.get_fidu_guess_spectra()
            self.guessCOV = cov_filling(
                fields=self.fields,
                excluded_probes=self.excluded_probes,
                absolute_lmin=self.lmin,
                absolute_lmax=self.lmax,
                cov_dict=self.fidu_guessCLS,
                lmins=self.lmins,
                lmaxs=self.lmaxs,
            )
            self.guess = (
                self.guessCOV[:, :, self.lmin : self.lmax + 1]
                + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
            )
            if self.offset_file is not None:
                self.offsetCLS = self.get_offset_spectra()
                self.offsetCOV = cov_filling(
                    fields=self.fields,
                    excluded_probes=self.excluded_probes,
                    absolute_lmin=self.lmin,
                    absolute_lmax=self.lmax,
                    cov_dict=self.offsetCLS,
                    lmins=self.lmins,
                    lmaxs=self.lmaxs,
                )
                self.offset = self.offsetCOV[:, :, self.lmin : self.lmax + 1]
            else:
                self.offset = np.zeros_like(self.guess)

    def _get_binned_array(self, file, key):
        """Fetch a bandpower array from a dict, matching the field
        key case-insensitively."""
        if file is None:
            raise ValueError(
                "binned=True requires cl_file (and usually nl_file) as bandpower dicts."
            )
        for k in (key, key[::-1], key.lower(), key.upper()):
            if k in file:
                return np.asarray(file[k], dtype=float)
        raise KeyError(
            f"Could not find bandpower spectrum '{key}' in provided dict "
            f"(keys: {list(file.keys())})."
        )

    def _dell_to_cell(self, arr):
        """Convert D_ell = ell(ell+1)/(2pi) C_ell to C_ell (ell < 2 set to 0)."""
        arr = np.asarray(arr, dtype=float)
        ell = np.arange(arr.shape[0])
        factor = np.zeros_like(arr)
        factor[2:] = 2.0 * np.pi / (ell[2:] * (ell[2:] + 1.0))
        return arr * factor

    def _build_bin_edges(self):
        """Set self.bin_lmins/bin_lmaxs. Uses the explicit edges if provided, otherwise
        builds uniform bands of width delta_ell from lmin to lmax (the last band is
        clipped to lmax)."""
        if self.bin_lmins is not None and self.bin_lmaxs is not None:
            return
        lmins = np.arange(self.lmin, self.lmax + 1, self.delta_ell)
        lmaxs = lmins + self.delta_ell - 1
        # Keep only full-width bands inside [lmin, lmax]; drop a
        # partial trailing band so
        # that e.g. lmin=30, lmax=120, delta_ell=10 gives the 9 clean
        # bands [30,39]..[110,119]
        # (matching a NaMaster run cut at the same lmax), not a width-1 [120,120] band.
        keep = lmaxs <= self.lmax
        self.bin_lmins = lmins[keep].astype(int)
        self.bin_lmaxs = lmaxs[keep].astype(int)

    def _bin_average_operator(self):
        """Flat C_ell averaging operator P[b, ell] = 1/dl_b for ell in [lmin_b, lmax_b].

        This matches NaMaster flat-C_ell bandpowers (and the BBin flat averaging used to
        bin the data), so the theory is binned exactly the same way as the data."""
        lmax_bin = int(np.max(self.bin_lmaxs))
        P = np.zeros((len(self.bin_lmins), lmax_bin + 1))
        for b, (a, z) in enumerate(zip(self.bin_lmins, self.bin_lmaxs)):
            P[b, a : z + 1] = 1.0 / (z - a + 1)
        return P, lmax_bin

    def _as_bandpowers(self, arr, name):
        """Reduce a raw input spectrum to bandpowers, honouring binning_mode.

        - 'external': ``arr`` is already a bandpower vector (length nbins), used as-is.
        - 'internal': ``arr`` is a per-multipole spectrum, flat-binned with self._bin_P
          (same operator used for the theory), so internal and external paths
          coincide.
        """
        arr = np.asarray(arr, dtype=float)
        nbins = len(self.bin_lmins)
        if self.binning_mode == "external":
            if arr.shape[0] != nbins:
                raise ValueError(
                    f"binning_mode='external': {name} has length {arr.shape[0]} but "
                    f"{nbins} bandpowers are defined by bin_lmins/bin_lmaxs. "
                    f"(Did you mean binning_mode='internal' for per-multipole input?)"
                )
            return arr
        # internal: bin the per-multipole input
        if arr.shape[0] < self._lmax_bin + 1:
            arr = np.concatenate([arr, np.zeros(self._lmax_bin + 1 - arr.shape[0])])
        return self._bin_P @ arr[: self._lmax_bin + 1]

    def _initialize_binned(self):
        """Initialize the binned (bandpower) gaussian/exact likelihood.

        cl_file/nl_file are reduced to bandpowers (one value per bin defined by
        bin_lmins/bin_lmaxs) -- taken as-is for binning_mode='external', or binned here
        for 'internal'. The bandpower covariance is the Knox formula lifted to bandpower
        level, consistent with LiLit's per-multipole gaussian covariance:

            sigma_b^2 = 2 (C_b + N_b)^2 / (fsky * nu_b),
            nu_b = (lmax_b + 1)^2 - lmin_b^2   (exact number of modes in the band)

        which reduces to the per-ell Knox term 2 (C+N)^2 / ((2l+1) fsky) for
        unit bins.
        """
        self._build_bin_edges()
        if self.bin_lmaxs.max() > self.lmax or self.bin_lmins.min() < self.lmin:
            raise ValueError(
                f"Bandpower edges [{self.bin_lmins.min()}, "
                f"{self.bin_lmaxs.max()}] must lie "
                f"within [lmin, lmax] = [{self.lmin}, {self.lmax}]."
            )
        if self.fsky is None:
            raise ValueError("binning_mode != 'none' requires a single scalar fsky.")

        self._bin_P, self._lmax_bin = self._bin_average_operator()
        nbins = len(self.bin_lmins)

        # Obtain per-band signal/noise. In 'external' mode cl_file/nl_file are already
        # bandpowers (used as-is). In 'internal' mode we build the
        # per-multipole fiducial
        # and noise with the SAME loaders as the unbinned path -- so this works whether
        # the spectra come from a file or are generated from r +
        # experiment (cl_file=None)
        # -- and then flat-bin them. After this step both modes are identical
        # downstream.
        key = (self.fields[0] + self.fields[0]).upper()  # e.g. "BB"
        if self.binning_mode == "external":
            raw_S = self._get_binned_array(self.cl_file, key)
            raw_N = (
                self._get_binned_array(self.nl_file, key)
                if self.nl_file is not None
                else np.zeros(nbins)
            )
        else:  # internal
            # The loaders return D_ell = ell(ell+1)/(2pi) C_ell (fiducial: raw_cl=False;
            # noise: explicit ell(ell+1)/2pi factor). Convert to C_ell so the internal
            # bandpowers use the same flat-C_ell scheme as the external NaMaster
            # spectra.
            fiduCLS = self.get_fiducial_spectra()
            noiseCLS = self.get_noise_spectra()
            raw_S = self._dell_to_cell(self._get_binned_array(fiduCLS, key))
            raw_N = self._dell_to_cell(self._get_binned_array(noiseCLS, key))
        Sb = self._as_bandpowers(raw_S, "cl_file")
        Nb = self._as_bandpowers(raw_N, "nl_file")

        # Effective number of modes per band. Analytic default is the exact sum of
        # (2*ell + 1) over the band, scaled by fsky:
        #   nu_b = (lmax_b + 1)^2 - lmin_b^2 = (2*ell_center + 1) * delta_ell,
        # times fsky.
        # On a cut sky with mode-coupling the honest value is
        # nu_b = 2 (C_b / sigma_b)^2;
        # pass it via effective_dof to override.
        nu_b = ((self.bin_lmaxs + 1) ** 2 - self.bin_lmins**2).astype(float) * self.fsky
        if self.effective_dof is not None:
            if len(self.effective_dof) != nbins:
                raise ValueError(
                    f"effective_dof has length {len(self.effective_dof)} but there are "
                    f"{nbins} bins."
                )
            self.binned_modes = self.effective_dof
        else:
            self.binned_modes = nu_b

        # Knox bandpower variance, consistent with the per-multipole gaussian covariance
        # and with the Wishart dof: sigma_b^2 = 2 (C_b + N_b)^2 / nu_b  <=>
        # nu_b = 2 C^2/sigma^2.
        sigma2_b = 2.0 * (Sb + Nb) ** 2 / self.binned_modes

        # Assemble the data and inverse-covariance in EXACTLY the shapes the
        # per-multipole
        # machinery already expects: data is (N, N, naxis) and inverse_covariance is
        # (naxis, N, N). Here the last/leading axis simply indexes bandpowers instead of
        # multipoles, so the standard log_likelihood() -> ChiSquareCalculator path runs
        # unchanged -- no bespoke chi-square for the binned case. The Wishart 'exact'
        # kernel gets the same mode count via self.binned_modes (passed as 'modes').
        self.binned_noise = Nb
        self.data = (Sb + Nb).reshape(1, 1, -1)  # (N, N, nbins), N == 1
        if self.like_approx == "correlated_gaussian":
            # Full bandpower covariance supplied by the user (e.g. estimated from sims);
            # chi2 = diff^T C^-1 diff via ChiSquareCalculator.
            # _calculate_correlated_gaussian.
            # This replaces the Knox diagonal so off-diagonal band-band
            # correlations (mask
            # mode-coupling, foreground/noise residuals) are propagated into sigma(r).
            if self.external_covariance is None:
                raise ValueError(
                    "binned correlated_gaussian requires external_covariance, an "
                    "(nbins, nbins) bandpower covariance matrix."
                )
            cov = np.asarray(self.external_covariance, dtype=float)
            if cov.shape != (nbins, nbins):
                raise ValueError(
                    f"external_covariance has shape {cov.shape}, expected "
                    f"({nbins}, {nbins}) to match the {nbins} bandpowers."
                )
            self.inverse_covariance = np.linalg.inv(cov)
        else:
            self.inverse_covariance = (1.0 / sigma2_b).reshape(-1, 1, 1)
        self.mask = None  # unused by the N == 1 gaussian/exact branches

        if self.debug:
            print(
                f"[binned] nbins={nbins}, ell in [{self.bin_lmins.min()}, "
                f"{self.bin_lmaxs.max()}]"
            )
            print(f"[binned] modes/band = {self.binned_modes}")
            print(f"[binned] data bandpowers = {self.data[0, 0]}")

    def _bin_theory(self):
        """Reduce the current-step CAMB theory to C_ell bandpowers (flat C_ell average).

        Everything in the binned path works in flat-C_ell so that all cases share ONE
        binning scheme: 'external' bandpowers (NaMaster C_ell from the generate
        notebooks)
        and 'internal' (the loader D_ell converted to C_ell, see _initialize_binned) are
        binned identically, and the theory is binned the same way here. The three
        analyses
        (ideal / semi-realistic / realistic) then differ only in the input spectra."""
        cls = self.provider.get_Cl(ell_factor=False, units="muK2")
        key = (self.fields[0] + self.fields[0]).lower()  # e.g. "bb"
        cl_theory = np.asarray(cls[key], dtype=float)
        if cl_theory.shape[0] < self._lmax_bin + 1:
            cl_theory = np.concatenate([
                cl_theory,
                np.zeros(self._lmax_bin + 1 - cl_theory.shape[0]),
            ])
        return self._bin_P @ cl_theory[: self._lmax_bin + 1]

    def get_requirements(self):
        """Defines requirements of the likelihood, specifying quantities calculated by
        a theory code are needed. Note that you may want to change the overall keyword
        from 'Cl' to 'unlensed_Cl' if you want to work without considering lensing."""
        requirements = {}
        requirements["Cl"] = {cl: self.lmax for cl in self.keys}
        if self.debug:
            requirements["CAMBdata"] = None
            print(
                "\nYou requested that Cobaya provides to the likelihood the "
                f"following items: {requirements}",
            )
        return requirements

    def get_likelihood_kwargs(self):
        """Defines the keyword arguments to pass to the likelihood function."""
        return {
            "fields": self.fields,
            "lmin": self.lmin,
            "lmax": self.lmax,
            "fsky": self.fsky,
            "fskies": self.fskies,
            "like_approx": self.like_approx,
            "chi_method": self.chi_method,
            "excluded_probes": self.excluded_probes,
            "debug": self.debug,
            "N": self.N,
            "inverse_covariance": getattr(self, "inverse_covariance", None),
            "mask": getattr(self, "mask", None),
            "offset": getattr(self, "offset", None),
            "fidu": getattr(self, "guess", None),
            "bins": getattr(self, "bins", None),
            "modes": getattr(self, "binned_modes", None),
        }

    def log_likelihood(self):
        """Convert into log likelihood and sum over multipoles."""

        kwargs = self.get_likelihood_kwargs()
        chi_squared = ChiSquareCalculator.calculate(
            method=self.chi_method, data=self.data, coba=self.coba, **kwargs
        )
        return np.sum(-0.5 * chi_squared)

    def logp(self, **params_values):
        """Gets the log likelihood and pass it to Cobaya to carry on the MCMC
        process."""
        if self.binning_mode != "none":
            # Only the theory object differs (bandpowers, with binned noise added);
            # the chi-square itself is the standard one.
            self.coba = (self._bin_theory() + self.binned_noise).reshape(1, 1, -1)
            return self.log_likelihood()
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
__pdoc__["Likelihood"] = (
    "Likelihood class from Cobaya, refer to Cobaya documentation for more information."
)
