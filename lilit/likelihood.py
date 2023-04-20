import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from cobaya.likelihood import Likelihood

from functions import (
    get_keys,
    get_Gauss_keys,
    find_spectrum,
    cov_filling,
    sigma,
    inv_sigma,
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
        cl_file (str, optional):
            Path to Cl file (default: None).
        nl_file (str, optional):
            Path to noise file (default: None).
        mapping (dict, optional):
            Dictionary of mapping between the fields in the noise file and the fields in the likelihood, used only if nl_file has .txt extension (default: None).
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
        debug (bool, optional):
            If True, produces more verbose output (default: None).
        survey (str, optional):
            Name of the survey (default: "Euclid").


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
        mapping (dict):
            Dictionary of mapping between the fields in the noise file and the fields in the likelihood, used only if nl_file has .txt extension.
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
        survey (str):
            Name of the survey.
        zz (np.ndarray):
            Redshift array for the considered survey.
        dNdz (np.ndarray):
            dNdz array for the considered survey.
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
        mapping=None,
        experiment=None,
        nside=None,
        r=None,
        nt=None,
        pivot_t=0.01,
        fsky=1,
        debug=None,
        survey="Euclid",
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
        self.has_sources = False
        self.source_fields = []
        if "0" in self.fields:
            self.has_sources = True
            self.survey = survey
            self.set_source_fields()
        self.all_fields = self.fields + self.source_fields
        self.lmin = lmin
        self.like = like
        self.cl_file = cl_file
        self.nl_file = nl_file
        if self.nl_file.endswith(".txt"):
            self.mapping = mapping
        self.experiment = experiment
        if self.experiment is not None:
            # Check that the user has provided the nside if an experiment is used
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.debug = debug
        self.keys = get_keys(fields=self.all_fields, debug=self.debug)
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

    def set_source_fields(self):
        """Set the source fields.

        This method is used to set the source fields, i.e. the fields that are not considered in the likelihood but are used to compute the noise.

        Args:
            source_fields (list):
                List of source fields.
        """
        if self.has_sources:
            for field in self.fields.copy():
                try:
                    self.source_fields.append(str(int(field)))
                    self.fields.remove(field)
                except ValueError:
                    continue

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
                    key = self.all_fields[i] + self.all_fields[j]
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
                    key = self.all_fields[i] + self.all_fields[j]
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
                    key = self.all_fields[i] + self.all_fields[j]
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
                # Save it with the normalization to obtain phiphi
                array = cl_lens[:, 0].copy()
                array[2:] /= (res["ell"] * (res["ell"] + 1))[2:]
                res["pp"] = array
                # Check if we want the cross terms
                if "pt" in self.keys and "pe" in self.keys:
                    # Loop over the cross terms
                    for i, cross in enumerate(["pt", "pe"]):
                        # Save the result
                        array = cl_lens[:, i + 1].copy()
                        array[2:] /= np.sqrt(res["ell"] * (res["ell"] + 1))[2:]
                        res[cross] = array
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
        res = {}
        # Loop over the mapping and extract the corresponding column from the txt file
        # and store it in the dictionary under the corresponding keyword
        for key, i in mapping.items():
            ls = np.arange(len(txt[:, i]), dtype=np.int64)
            res["ell"] = ls
            if apply_ellfactor:
                res[key] = txt[:, i] * ls * (ls + 1) / 2 / np.pi  # TODO check this
            else:
                res[key] = txt[:, i]
        return res

    def add_sources_params(self, pars):
        from camb.model import NonLinear_both
        from camb.sources import SplinedSourceWindow

        pars.SourceTerms.counts_redshift = False
        pars.SourceTerms.counts_lensing = False
        pars.SourceTerms.limber_windows = True
        pars.SourceTerms.limber_phi_lmin = 100
        pars.SourceTerms.counts_velocity = False
        pars.SourceTerms.counts_radial = False
        pars.SourceTerms.counts_timedelay = False
        pars.SourceTerms.counts_ISW = True
        pars.SourceTerms.counts_potential = False
        pars.SourceTerms.counts_evolve = True
        pars.SourceTerms.line_phot_dipole = False
        pars.SourceTerms.line_phot_quadrupole = False
        pars.SourceTerms.line_basic = True
        pars.SourceTerms.line_distortions = False
        pars.SourceTerms.line_extra = False
        pars.SourceTerms.line_reionization = False
        pars.SourceTerms.use_21cm_mK = False
        pars.Want_CMB = True
        pars.NonLinear = NonLinear_both

        self.compute_dndz()

        if self.survey == "Euclid":
            pars.b1 = 1.0997727037892875
            pars.b2 = 1.220245876862528
            pars.b3 = 1.2723993083933989
            pars.b4 = 1.316624471897739
            pars.b5 = 1.35812370570578
            pars.b6 = 1.3998214171814918
            pars.b7 = 1.4446452851824907
            pars.b8 = 1.4964959071110084
            pars.b9 = 1.5652475842498528
            pars.b10 = 1.7429859437184225
            pars.SourceTerms.want_euclid = True
        elif self.survey == "LSST":
            pars.b1 = 1.08509147
            pars.b2 = 1.14382284
            pars.b3 = 1.2047005
            pars.b4 = 1.26740743
            pars.b5 = 1.3317134
            pars.b6 = 1.3974141
            pars.b7 = 1.46429394
            pars.b8 = 1.53212972
            pars.b9 = 1.60078307
            pars.b10 = 1.67017681
            pars.SourceTerms.want_lsst = True
        else:
            err = "Survey not supported"
            raise ValueError(err)

        sources_collection = []
        for i in range(np.array(self.dNdz).shape[0]):
            sources_collection.append(
                SplinedSourceWindow(
                    bias_z=np.ones(len(self.zz)), z=self.zz, W=self.dNdz[i]
                )
            )
        pars.SourceWindows = sources_collection
        return

    def store_sources_results(self, camb_results, results_dict):
        source_res = camb_results.get_source_cls_dict(raw_cl=False, lmax=self.lmax)
        ell = results_dict["ell"]
        for key, value in source_res.items():
            key = key.lower().replace("w", "").split("x")
            first_field = key[0]
            second_field = key[1]
            try:
                first_field = str(int(key[0]) - 1)
            except ValueError:
                pass
            try:
                second_field = str(int(key[1]) - 1)
            except ValueError:
                pass
            key = first_field + second_field
            if "p" in key:
                if "pp" in key:
                    value[2:] = value[2:] / (ell * (ell + 1))[2:]
                else:
                    value[2:] = value[2:] / (np.sqrt(ell * (ell + 1)))[2:]
            results_dict[key] = value

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

        print("\nProducing fiducial spectra from Planck 2018 best-fit values!")
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

        if "1" in self.fields:
            ## general. rel effects for source terms ##
            pars.SourceTerms.counts_redshift = False
            pars.SourceTerms.counts_lensing = False
            pars.SourceTerms.limber_windows = True
            pars.SourceTerms.limber_phi_lmin = 100
            pars.SourceTerms.counts_velocity = False
            pars.SourceTerms.counts_radial = False
            pars.SourceTerms.counts_timedelay = False
            pars.SourceTerms.counts_ISW = True
            pars.SourceTerms.counts_potential = False
            pars.SourceTerms.counts_evolve = True
            pars.SourceTerms.line_phot_dipole = False
            pars.SourceTerms.line_phot_quadrupole = False
            pars.SourceTerms.line_basic = True
            pars.SourceTerms.line_distortions = False
            pars.SourceTerms.line_extra = False
            pars.SourceTerms.line_reionization = False
            pars.SourceTerms.use_21cm_mK = False
            pars.set_for_lmax(self.lmax, lens_potential_accuracy=1)
            pars.Want_CMB = True
            pars.NonLinear = model.NonLinear_both

            from camb.sources import SplinedSourceWindow

        start = time.time()
        results = camb.get_results(pars)
        end = time.time()

        if self.debug:
            print(pars)
            print(f"\nCAMB took {end-start:.2f} seconds to run.")

        res = results.get_cmb_power_spectra(
            CMB_unit="muK",
            lmax=self.lmax,
            raw_cl=False,
        )
        res_dict = self.CAMBres2dict(res)

        if self.has_sources:
            self.store_sources_results(results, res_dict)

        return res_dict

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
                # Convert the text file to a dictionary
                res = self.txt2dict(_txt, self.mapping, apply_ellfactor=True)
            return res

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

        sigm = np.radians(fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        sigma2 = sigm**2

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
        self.fiduCOV = cov_filling(
            self.all_fields, self.lmin, self.lmax, self.fiduCLS, self.lmins, self.lmaxs
        )
        self.noiseCOV = cov_filling(
            self.all_fields, self.lmin, self.lmax, self.noiseCLS, self.lmins, self.lmaxs
        )

        # Print some information for debugging
        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            print("\nPrinting the first few values to check that it starts from 0...")
            field = list(self.fiduCLS.keys())[1]
            print(f"Fiducial CLs for {field.upper()} ---> {self.fiduCLS[field][0:5]}")
            field = list(self.noiseCLS.keys())[1]
            print(f"Noise CLs for {field.upper()} ---> {self.noiseCLS[field][0:5]}")

        # Compute the total covariance matrix
        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # Compute the inverse of the covariance matrix
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

    def dndz_Euclid(self):
        from scipy.special import erf

        self.zz = np.arange(0.001, 2.501, 0.001)
        z_med = 0.9
        z0 = z_med / (np.sqrt(2))
        c_b = 1.0
        z_b = 0.0
        sigma_b = 0.05
        c_0 = 1.0
        z_0 = 0.1
        sigma_0 = 0.05
        f_out = 0.1

        ## NZ ##
        def dndz(z, zm):
            return ((z / zm) ** 2) * np.exp(-((z / zm) ** (3.0 / 2.0)))

        z_bin_m = [
            0.001,
            0.418,
            0.560,
            0.678,
            0.789,
            0.900,
            1.019,
            1.155,
            1.324,
            1.576,
            2.500,
        ]
        self.z_med = [
            (z_bin_m[i + 1] + z_bin_m[i]) / 2 for i in range(len(z_bin_m) - 1)
        ]

        ## N(z) no tomography ##
        dndz_tot = dndz(self.zz, z0)

        ## N(z)_i ##
        def dndz_bin(z, z_bin_m, z_bin_p, c_b, z_b, sigma_b, c_0, z_0, sigma_0, f_out):
            return dndz_tot * ((1 - f_out) / (2 * c_b)) * (
                erf((z - c_b * z_bin_m - z_b) / (np.sqrt(2) * sigma_b * (1 + z)))
                - erf((z - c_b * z_bin_p - z_b) / (np.sqrt(2) * sigma_b * (1 + z)))
            ) + dndz_tot * (f_out / (2 * c_0)) * (
                erf((z - c_0 * z_bin_m - z_0) / (np.sqrt(2) * sigma_0 * (1 + z)))
                - erf((z - c_0 * z_bin_p - z_0) / (np.sqrt(2) * sigma_0 * (1 + z)))
            )

        ## Computing power spectrum ##
        self.dNdz = [
            dndz_bin(
                self.zz,
                z_bin_m[i],
                z_bin_m[i + 1],
                c_b,
                z_b,
                sigma_b,
                c_0,
                z_0,
                sigma_0,
                f_out,
            )
            for i in range(len(z_bin_m) - 1)
        ]

        z_med = [(z_bin_m[i + 1] + z_bin_m[i]) / 2 for i in range(len(z_bin_m) - 1)]

        return

    def dndz_LSST(self):
        from scipy.special import erf

        self.zz = np.arange(0.0, 1.601, 0.001)

        z0 = 0.24
        alpha = 0.90
        c_b = 1.0
        z_b = 0.0
        sigma_b = 0.03
        c_0 = 1.0
        z_0 = 0.1
        sigma_0 = 0.03
        f_out = 0

        ## NZ ##
        def dndz(z, zm, alpha):
            return (z**2) * np.exp(-((z / zm) ** (alpha)))

        z_bin_p = np.arange(0.2, 1.3, 0.1)

        dndz_tot = dndz(self.zz, z0, alpha)

        ## N(z)_i ##
        def dndz_bin(z, z_bin_m, z_bin_p, c_b, z_b, sigma_b, c_0, z_0, sigma_0, f_out):
            return dndz_tot * ((1 - f_out) / (2 * c_b)) * (
                erf((z - c_b * z_bin_m - z_b) / (np.sqrt(2) * sigma_b * (1 + z)))
                - erf((z - c_b * z_bin_p - z_b) / (np.sqrt(2) * sigma_b * (1 + z)))
            ) + dndz_tot * (f_out / (2 * c_0)) * (
                erf((z - c_0 * z_bin_m - z_0) / (np.sqrt(2) * sigma_0 * (1 + z)))
                - erf((z - c_0 * z_bin_p - z_0) / (np.sqrt(2) * sigma_0 * (1 + z)))
            )

        ### Computing power spectrum ##
        self.dNdz = [
            dndz_bin(
                self.zz,
                z_bin_p[i],
                z_bin_p[i + 1],
                c_b,
                z_b,
                sigma_b,
                c_0,
                z_0,
                sigma_0,
                f_out,
            )
            for i in range(len(z_bin_p) - 1)
        ]

        return

    def compute_dndz(self):
        if self.survey == "Euclid":
            self.dndz_Euclid()
        elif self.survey == "LSST":
            self.dndz_LSST()
        else:
            print("Survey not implemented yet (only Euclid and LSST)")
            return
        return

    def get_requirements(self):
        """Defines requirements of the likelihood, specifying quantities calculated by a theory code are needed. Note that you may want to change the overall keyword from 'Cl' to 'unlensed_Cl' if you want to work without considering lensing."""
        # The likelihood needs the lensed CMB angular power spectra. The keyword can be set to "unlensed_Cl" to get the unlensed ones
        requirements = {}
        N = len(self.fields)
        requirements["Cl"] = {
            self.fields[i] + self.fields[j]: self.lmax
            for i in range(N)
            for j in range(i, N)
        }
        # If debug is set to True, the likelihood will print the list of items required by the likelihood
        if self.has_sources:
            requirements["Cl"] = {"pp": self.lmax}
            self.compute_dndz()
            sources = {}
            for field in self.source_fields:
                try:
                    sources[field] = {
                        "function": "spline",
                        "z": self.zz,
                        "bias_z": np.ones(len(self.zz)),
                        "W": self.dNdz[int(field)],
                    }
                except ValueError:
                    pass
            requirements["source_Cl"] = {"non_linear": True, "limber": True}
            requirements["source_Cl"]["sources"] = sources

        if self.debug:
            requirements["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides the following items: {requirements}",
            )
        return requirements

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
        # If the number of datasets is not equal to 1, then we have a
        # multi-dataset case, in which case we need to compute the
        # covariance matrix for each dataset.
        ell = np.arange(0, self.lmax + 1, 1)
        if self.n != 1:
            # We extract the covariance matrix and data for the ith
            # dataset.
            coba = self.coba[:, :, i]
            data = self.data[:, :, i]
            # If the determinant is equal to 0, then we need to reduce
            # the dimensionality of the data and covariance matrix.
            if np.linalg.det(coba) == 0:
                data = self.get_reduced_data(data)
                coba = self.get_reduced_data(coba)
            # We compute the matrix M using the covariance matrix and
            # the data.
            M = np.linalg.solve(coba, data)
            # We compute the chi-square term using the trace of M, the
            # log determinant of M, and the number of fields.
            return (2 * ell[i] + 1) * (
                np.trace(M) - np.linalg.slogdet(M)[1] - data.shape[0]
            )
        # If the number of datasets is equal to 1, then we have a single
        # dataset case, in which case we do not need to loop over the
        # datasets.
        else:
            # We compute the matrix M using the covariance matrix and
            # the data.
            M = self.data / self.coba
            # We compute the chi-square term using M, the log of M, and
            # a constant value.
            return (2 * ell + 1) * (M - np.log(np.abs(M)) - 1)

    def chi_gaussian(self, i=0):
        """Computes proper chi-square term for the Gaussian likelihood case.

        Parameters:
            i (int, optional):
                ell index if needed. Defaults to 0.
        """
        # If we have more than one data vector
        if self.n != 1:
            coba, idx = self.data_vector(self.coba[:, :, i])
            data, _ = self.data_vector(self.data[:, :, i])
            COV = np.delete(self.sigma2[:, :, i], idx, axis=0)
            COV = np.delete(COV, idx, axis=1)
            return (coba - data) @ COV @ (coba - data)
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

        if self.has_sources:
            cobasourceCLs = self.provider.get_source_Cl()

            for key, value in cobasourceCLs.items():
                key = key[0] + key[1]
                key = key.lower().replace("w", "").replace("x", "")

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
            self.all_fields, self.lmin, self.lmax, self.cobaCLS, self.lmins, self.lmaxs
        )

        # Add the noise covariance to the covariance matrix filled with the Cls from Cobaya
        self.coba = (
            self.cobaCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        # Compute the likelihood
        logp = self.log_likelihood()

        if self.debug:
            ell = np.arange(0, self.lmax + 1, 1)
            print(logp)
            import matplotlib.ticker as tck

            fig, ax = plt.subplots()
            ax.tick_params(direction="in", which="both", labelsize=13, width=1.0)
            ax.yaxis.set_ticks_position("both")
            ax.xaxis.set_ticks_position("both")
            ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
            ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
            bin = 1
            plt.plot(ell, self.noiseCOV[0, 0, :], label="noisepp")
            plt.plot(ell, self.fiduCOV[0, 0, :], label="fidupp")
            plt.plot(ell, self.cobaCOV[0, 0, :], label="cobapp", ls="--")
            plt.plot(ell, self.noiseCOV[bin, bin, :], label="noisegg")
            plt.plot(ell, self.fiduCOV[bin, bin, :], label="fidugg")
            plt.plot(ell, self.cobaCOV[bin, bin, :], label="cobagg", ls="--")
            plt.plot(ell, self.fiduCOV[0, bin, :], label="fidupg")
            plt.plot(ell, self.cobaCOV[bin, 0, :], label="cobapg", ls="--")
            plt.legend()
            plt.loglog()
            plt.xlim(2, self.lmax)
            plt.show(block=True)

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
