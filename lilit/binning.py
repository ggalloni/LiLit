import numpy as np


class Bins:
    """
    lmins : list of integers
        Lower bound of the bins
    lmaxs : list of integers
        Upper bound of the bins
    lmin_floor : int
        Minimum lmin value to keep (default 2, for spin-2 analyses).
    """

    def __init__(self, lmins, lmaxs, lmin_floor=2):
        if not (len(lmins) == len(lmaxs)):
            msg = "Incoherent inputs"
            raise ValueError(msg)

        lmins = np.asarray(lmins, dtype=int)
        lmaxs = np.asarray(lmaxs, dtype=int)
        if lmin_floor < 0:
            raise ValueError(f"lmin_floor must be >= 0, got {lmin_floor}")
        # Reject bins entirely below lmin_floor (default 2 reproduces the
        # legacy spin-2 floor; pass lmin_floor=1 for dipole or 0 for
        # monopole-aware analyses).
        keep = np.logical_and(lmaxs >= lmin_floor, lmins >= lmin_floor)
        self.lmins = lmins[keep]
        self.lmaxs = lmaxs[keep]
        self.lmin_floor = int(lmin_floor)

        self._derive_ext()

    @classmethod
    def fromdeltal(cls, lmin, lmax, delta_ell):
        """Create uniform bins with constant width.

        ``lmin`` doubles as the bin floor; values below 2 are honoured
        (e.g. ``Bins.fromdeltal(1, 4, 1)`` includes the dipole).
        """
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell - 1
        return cls(lmins, lmaxs, lmin_floor=lmin)

    def _derive_ext(self):
        if len(self.lmins) == 0:
            raise ValueError(
                f"No valid bins (all bins below lmin_floor={self.lmin_floor})"
            )

        for i, (l1, l2) in enumerate(zip(self.lmins, self.lmaxs)):
            if l1 > l2:
                raise ValueError(f"Bin {i}: lmin={l1} > lmax={l2}")

        # Check for overlaps (bins must be non-overlapping and sorted)
        order = np.argsort(self.lmins)
        self.lmins = self.lmins[order]
        self.lmaxs = self.lmaxs[order]
        for i in range(len(self.lmins) - 1):
            if self.lmins[i + 1] <= self.lmaxs[i]:
                raise ValueError(
                    f"Bins {i} and {i + 1} overlap: "
                    f"[{self.lmins[i]}, {self.lmaxs[i]}] and "
                    f"[{self.lmins[i + 1]}, {self.lmaxs[i + 1]}]"
                )

        self.lmin = int(np.min(self.lmins))
        self.lmax = int(np.max(self.lmaxs))
        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs) / 2.0
        self.dl = self.lmaxs - self.lmins + 1

    def bins(self):
        return (self.lmins, self.lmaxs)

    def cut_binning(self, lmin, lmax):
        sel = np.where((self.lmins >= lmin) & (self.lmaxs <= lmax))[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()

    def _bin_operators(self, Dl=False, cov=False):
        if Dl:
            ell2 = np.arange(self.lmax + 1)
            ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        else:
            ell2 = np.ones(self.lmax + 1)
        p = np.zeros((self.nbins, self.lmax + 1))
        q = np.zeros((self.lmax + 1, self.nbins))

        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            dl = z - a + 1
            p[b, a : z + 1] = ell2[a : z + 1] / dl
            if cov:
                q[a : z + 1, b] = 1 / ell2[a : z + 1] / dl
            else:
                q[a : z + 1, b] = 1 / ell2[a : z + 1]

        return p, q

    def bin_spectra(self, spectra, Dl=False):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        Return Cb
        """
        spectra = np.asarray(spectra)
        if self.lmin_floor > 0:
            pad = np.zeros((*spectra.shape[:-1], self.lmin_floor))
            spectra = np.concatenate([pad, spectra], axis=-1)
        minlmax = np.min([spectra.shape[-1] - 1, self.lmax])

        _p, _q = self._bin_operators(Dl=Dl)
        return np.dot(spectra[..., : minlmax + 1], _p.T[: minlmax + 1, ...])

    def bin_covariance(self, clcov):
        p, q = self._bin_operators(cov=True)
        return np.matmul(p, np.matmul(clcov, q))


def get_binning(lmin, lmax, delta_ell, transition=35):
    llmin = lmin
    llmax = transition
    hlmin = transition + 1
    lmins = list(range(llmin, llmax + 1)) + list(
        range(hlmin, lmax - delta_ell + 2, delta_ell)
    )
    lmaxs = list(range(llmin, llmax + 1)) + list(
        range(hlmin + delta_ell - 1, lmax + 1, delta_ell)
    )
    return Bins(lmins, lmaxs)
