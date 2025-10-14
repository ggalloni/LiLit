"""
Data loading utilities for various file formats.

This module provides functions to load and convert data from different formats
used in LiLit, including CAMB results, text files, and pickle files.
"""

import pickle

import numpy as np
from camb import CAMBdata


def load_pickle_spectra(file_path: str) -> dict:
    """Load spectra from pickle file.

    Parameters:
        file_path (str): Path to the pickle file.

    Returns:
        Dict: Loaded spectra dictionary.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def CAMBres2dict(camb_results: CAMBdata, probes: list[str]) -> dict:
    """Takes the CAMB result product from get_cmb_power_spectra and convert it to a
    dictionary with the proper keys.

    Parameters:
        camb_results (CAMBdata):
            CAMB result product from the method get_cmb_power_spectra.
        probes (List[str]):
            Keys for the spectra dictionary we want to save.

    Returns:
        Dict: Dictionary with spectra indexed by probe names.
    """
    ls = np.arange(camb_results["total"].shape[0], dtype=np.int64)
    # Mapping between the CAMB keys and the ones we want
    mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
    res = {"ell": ls}
    for probe, i in mapping.items():
        res[probe] = camb_results["total"][:, i]
    if "pp" in probes:
        cl_lens = camb_results.get("lens_potential")
        if cl_lens is not None:
            # Save it with the normalization to obtain phiphi
            array = cl_lens[:, 0].copy()
            array[2:] /= (res["ell"] * (res["ell"] + 1))[2:]
            res["pp"] = array
            if "pt" in probes and "pe" in probes:
                # Loop over the cross terms correcting the normalization to phiX
                for i, cross in enumerate(["pt", "pe"]):
                    array = cl_lens[:, i + 1].copy()
                    array[2:] /= np.sqrt(res["ell"] * (res["ell"] + 1))[2:]
                    res[cross] = array
                    res[cross[::-1]] = res[cross]
    return res


def txt2dict(
    txt: str, *, mapping_probe2colnum: dict[str, int] = None, apply_ellfactor: bool = None
) -> dict:
    """Takes a txt file and convert it to a dictionary. This requires a way to map the
    columns to the keys. Also, it is possible to apply an ell factor to the Cls.

    Parameters:
        txt (str):
            Path to txt file containing the spectra as columns.
        mapping_probe2colnum (Dict[str, int]):
            Dictionary containing the mapping. Keywords will become the new keywords and
            values represent the index of the corresponding column.
        apply_ellfactor (bool):
            If True, apply ell(ell+1)/(2π) factor to the spectra.

    Returns:
        Dict: Dictionary with loaded spectra.
    """
    assert mapping_probe2colnum is not None, (
        "You must provide a way to map the columns of your txt to the keys of a dict"
    )
    txt_data = np.loadtxt(txt)
    res = {}
    for probe, i in mapping_probe2colnum.items():
        ls = np.arange(len(txt_data[:, i]), dtype=np.int64)
        res["ell"] = ls
        if apply_ellfactor:
            res[probe] = txt_data[:, i] * ls * (ls + 1) / 2 / np.pi
        else:
            res[probe] = txt_data[:, i]
    return res


class SpectraLoader:
    """Unified interface for loading different file formats."""

    @staticmethod
    def load(file_input: str | dict) -> dict:
        """Load spectra from various formats.

        Parameters:
            file_input (Union[str, Dict]):
                Path to file or dictionary containing spectra.

        Returns:
            Dict: Loaded spectra dictionary.
        """
        if isinstance(file_input, dict):
            return file_input
        elif isinstance(file_input, str):
            if file_input.endswith(".pkl"):
                return load_pickle_spectra(file_input)
            elif file_input.endswith(".txt"):
                # For txt files, additional parameters would be needed
                # This is a basic implementation
                raise NotImplementedError(
                    "txt file loading requires mapping_probe2colnum parameter. "
                    "Use txt2dict() directly instead."
                )
            else:
                raise ValueError(f"Unsupported file format: {file_input}")
        else:
            raise TypeError(f"Unsupported input type: {type(file_input)}")


class SpectraLoaderBase:
    """Base class for all spectra loaders."""

    @staticmethod
    def _load_from_file_or_dict(file_input):
        """Load spectra from file path or return dict directly."""
        if isinstance(file_input, dict):
            return file_input
        elif isinstance(file_input, str):
            if file_input.endswith(".pkl"):
                return load_pickle_spectra(file_input)
            else:
                raise TypeError("File must be a pickle (.pkl) file or dictionary")
        else:
            raise TypeError("Input must be a file path string or dictionary")


class FiducialSpectraLoader(SpectraLoaderBase):
    """Loads or generates fiducial power spectra."""

    def __init__(
        self,
        cl_file=None,
        keys=None,
        lmax=100,
        r=None,
        nt=None,
        pivot_t=0.01,
        debug=False,
    ):
        """Initialize loader with parameters."""
        self.cl_file = cl_file
        self.keys = keys or ["tt", "ee", "te"]
        self.lmax = lmax
        self.r = r
        self.nt = nt
        self.pivot_t = pivot_t
        self.debug = debug

    def load(self):
        """Load fiducial spectra from file or generate with CAMB."""
        if self.cl_file is not None:
            return self._load_from_file_or_dict(self.cl_file)

        return self._generate_with_camb()

    def _generate_with_camb(self):
        """Generate fiducial spectra using CAMB."""
        try:
            import camb
        except ImportError:
            raise ImportError("CAMB not installed. Check requirements.")

        import os

        # Load Planck 2018 parameters
        path = os.path.dirname(os.path.abspath(__file__))
        planck_path = os.path.join(path, "..", "planck_2018.ini")
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
        res = results.get_cmb_power_spectra(CMB_unit="muK", lmax=self.lmax, raw_cl=False)
        return CAMBres2dict(res, self.keys)


class NoiseSpectraLoader(SpectraLoaderBase):
    """Loads or generates noise power spectra."""

    def __init__(self, nl_file=None, experiment=None, nside=None, lmax=None):
        """Initialize noise spectra loader."""
        self.nl_file = nl_file
        self.experiment = experiment
        self.nside = nside
        self.lmax = lmax

    def load(self):
        """Load noise spectra from file or generate for experiment."""
        if self.nl_file is not None:
            return self._load_from_file_or_dict(self.nl_file)

        if self.experiment is None:
            raise ValueError("Must provide either nl_file or experiment")

        return self._generate_for_experiment()

    def _generate_for_experiment(self):
        """Generate noise spectra for given experiment."""
        print(
            "***WARNING***: Inverse noise weighting severely underestimates actual noise."
        )

        try:
            import healpy as hp
        except ImportError:
            raise ImportError("Healpy not installed. Check requirements.")

        import os

        import yaml
        from yaml.loader import SafeLoader

        # Load experiment configuration
        path = os.path.dirname(os.path.abspath(__file__))
        experiments_path = os.path.join(path, "..", "experiments.yaml")

        with open(experiments_path) as f:
            data = yaml.load(f, Loader=SafeLoader)

        instrument = data[self.experiment]

        # Extract instrument parameters
        fwhms = np.array(instrument["fwhm"])
        freqs = np.array(instrument["frequency"])
        depth_p = np.array(instrument["depth_p"])
        depth_i = np.array(instrument["depth_i"])

        # Convert to proper units
        depth_p /= hp.nside2resol(self.nside, arcmin=True)
        depth_i /= hp.nside2resol(self.nside, arcmin=True)
        depth_p *= np.sqrt(hp.nside2pixarea(self.nside, degrees=False))
        depth_i *= np.sqrt(hp.nside2pixarea(self.nside, degrees=False))

        # Compute noise spectra
        ell = np.arange(0, self.lmax + 1, 1)
        n_freq = len(freqs)

        sigma = np.radians(fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        sigma2 = sigma**2

        g = np.exp(ell * (ell + 1) * sigma2[:, np.newaxis])

        pol_factor = np.array([np.zeros(sigma2.shape), 2 * sigma2, 2 * sigma2, sigma2])
        pol_factor = np.exp(pol_factor)

        G = [g * arr[:, np.newaxis] for arr in pol_factor]
        g = np.array(G)

        res = {key: np.zeros((n_freq, self.lmax + 1)) for key in ["tt", "ee", "bb"]}

        res["tt"] = 1 / (g[0, :, :] * depth_i[:, np.newaxis] ** 2)
        res["ee"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)
        res["bb"] = 1 / (g[3, :, :] * depth_p[:, np.newaxis] ** 2)

        # Sum over frequencies and apply ell factor
        for key in ["tt", "ee", "bb"]:
            res[key] = ell * (ell + 1) / np.sum(res[key], axis=0) / (2 * np.pi)
            res[key][:2] = [0, 0]  # Set ell=0,1 to zero

        return res


class BiasSpectraLoader(SpectraLoaderBase):
    """Loads bias spectra."""

    def __init__(self, bias_file):
        """Initialize bias spectra loader."""
        self.bias_file = bias_file

    def load(self):
        return self._load_from_file_or_dict(self.bias_file)


class FiduGuessSpectraLoader(SpectraLoaderBase):
    """Loads fiducial guess spectra."""

    def __init__(self, fidu_guess_file):
        """Initialize fiducial guess spectra loader."""
        self.fidu_guess_file = fidu_guess_file

    def load(self):
        return self._load_from_file_or_dict(self.fidu_guess_file)


class OffsetSpectraLoader(SpectraLoaderBase):
    """Loads offset spectra."""

    def __init__(self, offset_file):
        """Initialize offset spectra loader."""
        self.offset_file = offset_file

    def load(self):
        return self._load_from_file_or_dict(self.offset_file)


__all__ = [
    "load_pickle_spectra",
    "CAMBres2dict",
    "txt2dict",
    "SpectraLoader",
    "FiducialSpectraLoader",
    "NoiseSpectraLoader",
    "BiasSpectraLoader",
    "FiduGuessSpectraLoader",
    "OffsetSpectraLoader",
]
