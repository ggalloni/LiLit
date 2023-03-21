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
        self.like = like
        self.sep = sep
        self.cl_file = cl_file
        self.nl_file = nl_file
        self.experiment = experiment
        if self.experiment is not None:
            assert nside is not None, "You must provide an nside to compute the noise"
            self.nside = nside
        self.debug = debug
        self.keys = self.get_keys()
        if "bb" in self.keys:
            assert (
                r is not None
            ), "You must provide the tensor-to-scalar ratio r for the fiducial production (defaul is at 0.01 Mpc^-1)"
            self.r = r
            self.nt = nt
            self.pivot_t = pivot_t

        self.set_lmin_lmax_fsky(lmin, lmax, fsky)

        Likelihood.__init__(self, name=name)

    def set_lmin_lmax_fsky(self, lmin, lmax, fsky):
        self.lmins = {}
        self.lmaxs = {}
        self.fskies = {}

        if isinstance(lmin, list):
            assert (
                len(lmin) == self.n
            ), "If you provide multiple lmin, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    _key = self.fields[i] + self.sep + self.fields[j]
                    self.lmins[_key] = max(lmin[i], lmin[j])
                    self.lmins[_key[::-1]] = max(lmin[i], lmin[j])
            self.lmin = min(lmin)
        else:
            self.lmin = lmin

        if isinstance(lmax, list):
            assert (
                len(lmax) == self.n
            ), "If you provide multiple lmax, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    _key = self.fields[i] + self.sep + self.fields[j]
                    self.lmaxs[_key] = min(lmax[i], lmax[j])
                    self.lmaxs[_key[::-1]] = min(lmax[i], lmax[j])
            self.lmax = max(lmax)
        else:
            self.lmax = lmax

        if isinstance(fsky, list):
            assert (
                len(fsky) == self.n
            ), "If you provide multiple fsky, they must match the number of requested fields with the same order"
            for i in range(self.n):
                for j in range(i, self.n):
                    _key = self.fields[i] + self.sep + self.fields[j]
                    self.fskies[_key] = min(fsky[i], fsky[j])
                    self.fskies[_key[::-1]] = min(fsky[i], fsky[j])
            self.fsky = None
        else:
            self.fsky = fsky
        return

    def cov_filling(self, cov_dict):
        res = np.zeros((self.n, self.n, self.lmax + 1))

        for i, field1 in enumerate(self.fields):
            for j, field2 in enumerate(self.fields[i:]):
                j += i

                key = field1 + self.sep + field2
                lmin = self.lmins.get(key, self.lmin)
                lmax = self.lmaxs.get(key, self.lmax)
                cov = cov_dict.get(key, np.zeros(lmax + 1))

                res[i, j, lmin : lmax + 1] = cov[lmin : lmax + 1]
                res[j, i] = res[i, j]

        return res

    def get_keys(self):
        res = [
            self.fields[i] + self.sep + self.fields[j]
            for i in range(self.n)
            for j in range(i, self.n)
        ]
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def get_Gauss_keys(self):
        n = int(self.n * (self.n + 1) / 2)
        res = np.zeros((n, n, 4), dtype=str)
        for i in range(n):
            for j in range(i, n):
                elem = self.keys[i] + self.sep + self.keys[j]
                for k in range(4):
                    res[i, j, k] = np.asarray(list(elem)[k])
                    res[j, i, k] = res[i, j, k]
        if self.debug:
            print(f"\nThe requested keys are {res}")
        return res

    def find_spectrum(self, input_dict, key):

        res = np.zeros(self.lmax + 1)

        lmin = self.lmins.get(key, self.lmin)
        lmax = self.lmaxs.get(key, self.lmax)
        if key in input_dict:
            cov = input_dict[key]
        else:
            cov = input_dict.get(key[::-1], np.zeros(lmax + 1))
        res[lmin : lmax + 1] = cov[lmin : lmax + 1]

        return res

    def sigma(self, keys, fiduDICT, noiseDICT):
        n = int(self.n * (self.n + 1) / 2)
        res = np.zeros((n, n, self.lmax + 1))
        for i in range(n):
            for j in range(i, n):
                _C_AC = self.find_spectrum(fiduDICT, keys[i, j, 0] + keys[i, j, 2])
                _C_BD = self.find_spectrum(fiduDICT, keys[i, j, 1] + keys[i, j, 3])
                _C_AD = self.find_spectrum(fiduDICT, keys[i, j, 0] + keys[i, j, 3])
                _C_BC = self.find_spectrum(fiduDICT, keys[i, j, 1] + keys[i, j, 2])
                _N_AC = self.find_spectrum(noiseDICT, keys[i, j, 0] + keys[i, j, 2])
                _N_BD = self.find_spectrum(noiseDICT, keys[i, j, 1] + keys[i, j, 3])
                _N_AD = self.find_spectrum(noiseDICT, keys[i, j, 0] + keys[i, j, 3])
                _N_BC = self.find_spectrum(noiseDICT, keys[i, j, 1] + keys[i, j, 2])
                if self.fsky is not None:
                    res[i, j] = (
                        (_C_AC + _N_AC) * (_C_BD + _N_BD)
                        + (_C_AD + _N_AD) * (_C_BC + _N_BC)
                    ) / self.fsky
                else:
                    _AC = keys[i, j, 0] + keys[i, j, 2]
                    _BD = keys[i, j, 1] + keys[i, j, 3]
                    _AD = keys[i, j, 0] + keys[i, j, 3]
                    _BC = keys[i, j, 1] + keys[i, j, 2]
                    _AB = keys[i, j, 0] + keys[i, j, 1]
                    _CD = keys[i, j, 2] + keys[i, j, 3]
                    res[i, j] = (
                        np.sqrt(self.fskies[_AC] * self.fskies[_BD])
                        * (_C_AC + _N_AC)
                        * (_C_BD + _N_BD)
                        + np.sqrt(self.fskies[_AD] * self.fskies[_BC])
                        * (_C_AD + _N_AD)
                        * (_C_BC + _N_BC)
                    ) / (self.fskies[_AB] * self.fskies[_CD])
                res[j, i] = res[i, j]
        return res

    def inv_sigma(self, sigma):
        res = np.zeros(self.lmax + 1, dtype=object)

        for i in range(self.lmax + 1):
            COV = sigma[:, :, i]
            if np.linalg.det(COV) == 0:
                _idx = np.where(np.diag(COV) == 0)[0]
                COV = np.delete(COV, _idx, axis=0)
                COV = np.delete(COV, _idx, axis=1)
            res[i] = np.linalg.inv(COV)
        return res[2:]

    def get_reduced_data(self, mat):
        _idx = np.where(np.diag(mat) == 0)[0]
        return np.delete(np.delete(mat, _idx, axis=0), _idx, axis=1)

    def CAMBres2dict(self, camb_results):

        _ls = np.arange(camb_results["total"].shape[0], dtype=np.int64)
        _mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
        res = {"ell": _ls}
        for key, i in _mapping.items():
            res[key] = camb_results["total"][:, i]
        if "pp" in self.keys:
            cl_lens = camb_results.get("lens_potential")
            if cl_lens is not None:
                res["pp"] = cl_lens[:, 0].copy()
                if "pt" in self.keys and "pe" in self.keys:
                    for i, cross in enumerate(["pt", "pe"]):
                        res[cross] = cl_lens[:, i + 1].copy()
                        res[cross[::-1]] = res[cross]
        return res

    def txt2dict(self, txt, mapping=None, apply_ellfactor=None):

        assert (
            mapping is not None
        ), "You must provide a way to map the columns of your txt to the keys of a dictionary"
        _ls = np.arange(txt.shape[0], dtype=np.int64)
        res = {"ell": _ls}
        for key, i in mapping.items():
            if apply_ellfactor:
                res[key] = txt[:, i] * _ls * (_ls + 1) / 2 / np.pi
            else:
                res[key] = txt[:, i]
        return res

    def prod_fidu(self):

        if self.cl_file is not None:
            if self.cl_file.endswith(".pkl"):
                with open(self.cl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            else:
                _txt = np.loadtxt(self.cl_file)
                _mapping = {"tt": 0, "ee": 1, "bb": 2, "te": 3, "et": 3}
                res = self.txt2dict(_txt, _mapping)
            return res

        import os
        import camb

        _pars = camb.read_ini(os.path.join("./", "planck_2018.ini"))
        if "bb" in self.keys:
            print(f"\nProducing fiducial spectra for r={self.r} and nt={self.nt}")
            _pars.InitPower.set_params(
                As=2.100549e-9,
                ns=0.9660499,
                r=self.r,
                nt=self.nt,
                pivot_tensor=self.pivot_t,
                pivot_scalar=0.05,
                parameterization=2,
            )
            _pars.WantTensors = True
            _pars.Accuracy.AccurateBB = True
        _pars.DoLensing = True
        # _pars.Accuracy.AccuracyBoost = 2 # This helps getting an extra squeeze on the accordance of Cobaya and Fiducial spectra

        if self.debug:
            print(_pars)

        results = camb.get_results(_pars)
        res = results.get_cmb_power_spectra(
            CMB_unit="muK",
            lmax=self.lmax,
            raw_cl=False,
        )
        return self.CAMBres2dict(res)

    def prod_noise(self):

        if self.nl_file is not None:
            if self.nl_file.endswith(".pkl"):
                with open(self.nl_file, "rb") as pickle_file:
                    res = pickle.load(pickle_file)
            else:
                _txt = np.loadtxt(self.nl_file)
                _mapping = {"bb": 0}
                res = self.txt2dict(_txt, _mapping, apply_ellfactor=True)
            return res

        print(
            "***WARNING***: the inverse noise weighting performed here severely underestimates the actual noise level of LiteBIRD. You should provide an input noise power spectrum with a more realistic noise."
        )

        import os
        import yaml
        from yaml.loader import SafeLoader
        import healpy as hp

        assert (
            self.experiment is not None
        ), "You must specify the experiment you want to consider"
        print(f"\nComputing noise for {self.experiment}")

        with open(os.path.join("./", "experiments.yaml")) as f:
            _data = yaml.load(f, Loader=SafeLoader)

        _instrument = _data[self.experiment]
        _fwhms = np.array(_instrument["fwhm"])
        _freqs = np.array(_instrument["frequency"])
        _depth_p = np.array(_instrument["depth_p"])
        _depth_i = np.array(_instrument["depth_i"])
        _depth_p /= hp.nside2resol(self.nside, arcmin=True)
        _depth_i /= hp.nside2resol(self.nside, arcmin=True)
        _depth_p = _depth_p * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )
        _depth_i = _depth_i * np.sqrt(
            hp.pixelfunc.nside2pixarea(self.nside, degrees=False),
        )
        _n_freq = len(_freqs)

        _ell = np.arange(0, self.lmax + 1, 1)

        _keys = ["tt", "ee", "bb"]

        _sigma = np.radians(_fwhms / 60.0) / np.sqrt(8.0 * np.log(2.0))
        _sigma2 = _sigma**2

        _g = np.exp(_ell * (_ell + 1) * _sigma2[:, np.newaxis])

        _pol_factor = np.array(
            [np.zeros(_sigma2.shape), 2 * _sigma2, 2 * _sigma2, _sigma2],
        )
        _pol_factor = np.exp(_pol_factor)

        _G = []
        for i, arr in enumerate(_pol_factor):
            _G.append(_g * arr[:, np.newaxis])
        _g = np.array(_G)

        res = {key: np.zeros((_n_freq, self.lmax + 1)) for key in _keys}

        res["tt"] = 1 / (_g[0, :, :] * _depth_i[:, np.newaxis] ** 2)
        res["ee"] = 1 / (_g[3, :, :] * _depth_p[:, np.newaxis] ** 2)
        res["bb"] = 1 / (_g[3, :, :] * _depth_p[:, np.newaxis] ** 2)

        res["tt"] = _ell * (_ell + 1) / (np.sum(res["tt"], axis=0)) / 2 / np.pi
        res["ee"] = _ell * (_ell + 1) / (np.sum(res["ee"], axis=0)) / 2 / np.pi
        res["bb"] = _ell * (_ell + 1) / (np.sum(res["bb"], axis=0)) / 2 / np.pi

        res["tt"][:2] = [0, 0]
        res["ee"][:2] = [0, 0]
        res["bb"][:2] = [0, 0]

        return res

    def initialize(self):

        self.fiduCLS = self.prod_fidu()
        self.noiseCLS = self.prod_noise()

        self.fiduCOV = self.cov_filling(self.fiduCLS)
        self.noiseCOV = self.cov_filling(self.noiseCLS)

        if self.debug:
            print(f"Keys of fiducial CLs ---> {self.fiduCLS.keys()}")
            print(f"Keys of noise CLs ---> {self.noiseCLS.keys()}")

            _field = "bb"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Fiducial CLs for {_field.upper()} ---> {self.fiduCLS[_field][0:5]}")
            print(f"Noise CLs for {_field.upper()} ---> {self.noiseCLS[_field][0:5]}")

        self.data = (
            self.fiduCOV[:, :, self.lmin : self.lmax + 1]
            + self.noiseCOV[:, :, self.lmin : self.lmax + 1]
        )

        if self.like == "gaussian":
            self.gauss_keys = self.get_Gauss_keys()
            _sigma2 = self.sigma(self.gauss_keys, self.fiduCLS, self.noiseCLS)
            self.sigma2 = self.inv_sigma(_sigma2)

    def get_requirements(self):
        requirements = {}
        requirements["Cl"] = {cl: self.lmax for cl in self.keys}
        if self.debug:
            requirements["CAMBdata"] = None
            print(
                f"\nYou requested that Cobaya provides to the likelihood the following items: {requirements}"
            )
        return requirements

    def data_vector(self, cov):
        return cov[np.triu_indices(self.n)][cov[np.triu_indices(self.n)] != 0]

    def chi_exact(self, i=0):
        if self.n != 1:
            _coba = self.coba[:, :, i]
            _data = self.data[:, :, i]
            _det = np.linalg.det(_coba)
            if _det == 0:
                _data = self.get_reduced_data(_data)
                _coba = self.get_reduced_data(_coba)
            M = np.linalg.solve(_coba, _data)
            return np.trace(M) - np.linalg.slogdet(M)[1] - _data.shape[0]
        else:
            M = self.data / self.coba
            return M - np.log(np.abs(M)) - 1

    def chi_gaussian(self, i=0):
        if self.n != 1:
            _coba = self.data_vector(self.coba[:, :, i])
            _data = self.data_vector(self.data[:, :, i])
            return (_coba - _data) @ self.sigma2[i] @ (_coba - _data)
        else:
            _coba = self.coba[0, 0, :]
            _data = self.data[0, 0, :]
            res = (_coba - _data) * self.sigma2 * (_coba - _data)
            return res

    def compute_chi_part(self, i=0):
        if self.like == "exact":
            return self.chi_exact(i)
        elif self.like == "gaussian":
            return self.chi_gaussian(i)
        else:
            print("You requested something different from 'exact or 'gaussian'!")
            return

    def log_likelihood(self):
        _ell = np.arange(self.lmin, self.lmax + 1, 1)
        if self.n != 1:
            logp_ℓ = np.zeros(_ell.shape)
            for i in range(0, self.lmax + 1 - self.lmin):
                logp_ℓ[i] = -0.5 * (2 * _ell[i] + 1) * self.compute_chi_part(i)
        else:
            logp_ℓ = -0.5 * (2 * _ell + 1) * self.compute_chi_part()
        return np.sum(logp_ℓ)

    def logp(self, **params_values):

        if self.debug:
            _CAMBdata = self.provider.get_CAMBdata()
            _pars = _CAMBdata.Params
            print(_pars)

        self.cobaCLs = self.provider.get_Cl(ell_factor=True)

        if self.debug:
            print(f"Keys of Cobaya CLs ---> {self.cobaCLs.keys()}")

            _field = "bb"
            print("\nPrinting the first few values to check that it starts from 0...")
            print(f"Cobaya CLs for {_field.upper()} ---> {self.cobaCLs[_field][0:5]}")

        self.cobaCOV = self.cov_filling(self.cobaCLs)

        if self.debug:
            _ell = np.arange(0, self.lmax + 1, 1)
            plt.loglog(_ell, self.fiduCOV[0, 0, :], label="Fiducial CLs")
            plt.loglog(_ell, self.cobaCOV[0, 0, :], label="Cobaya CLs", ls="--")
            plt.loglog(_ell, self.noiseCOV[0, 0, :], label="Noise CLs")
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
