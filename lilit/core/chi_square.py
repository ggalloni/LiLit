"""Chi-square calculations for LiLit likelihood computations.

This module provides a unified interface for different chi-square calculation methods
used in cosmological likelihood analysis, including exact, Gaussian, Hamimeche & Lewis,
and LoLLiPoP approximations.
"""

from enum import Enum

import numpy as np


class ChiSquareMethod(Enum):
    """Available chi-square calculation methods."""

    EXACT = "exact"
    GAUSSIAN = "gaussian"
    CORRELATED_GAUSSIAN = "correlated_gaussian"
    HAMIMECHE_LEWIS = "hl"
    LOLLIPOP = "lollipop"


class ChiSquareCalculator:
    """Unified interface for chi-square calculations in likelihood analysis.

    This class provides a factory pattern for different chi-square calculation methods,
    making it easy to switch between approximations and maintain consistent interfaces.
    """

    @staticmethod
    def calculate(
        method: ChiSquareMethod | str, data: np.ndarray, coba: np.ndarray, **kwargs
    ) -> np.ndarray | list:
        """Calculate chi-square using the specified method.

        Parameters:
            method (ChiSquareMethod or str):
                The chi-square calculation method to use.
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            **kwargs:
                Method-specific parameters.

        Returns:
            ndarray | list: Chi-square values, format depends on method.

        Raises:
            ValueError: If method is not supported or required parameters are missing.
        """
        if isinstance(method, str):
            try:
                method = ChiSquareMethod(method)
            except ValueError:
                raise ValueError(f"Unsupported chi-square method: {method}")

        method_map = {
            ChiSquareMethod.EXACT: ChiSquareCalculator._calculate_exact,
            ChiSquareMethod.GAUSSIAN: ChiSquareCalculator._calculate_gaussian,
            ChiSquareMethod.CORRELATED_GAUSSIAN: (
                ChiSquareCalculator._calculate_correlated_gaussian
            ),
            ChiSquareMethod.HAMIMECHE_LEWIS: ChiSquareCalculator._calculate_HL,
            ChiSquareMethod.LOLLIPOP: ChiSquareCalculator._calculate_LoLLiPoP,
        }

        return method_map[method](data, coba, **kwargs)

    @staticmethod
    def _calculate_exact(
        data: np.ndarray,
        coba: np.ndarray,
        N: int,
        lmin: int,
        lmax: int,
        fsky: float,
        **kwargs,
    ) -> np.ndarray:
        """Computes proper chi-square term for the exact likelihood case.

        Parameters:
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            N (int):
                Number of fields.
            lmin (int):
                The minimum multipole to consider.
            lmax (int):
                The maximum multipole to consider.
            fsky (float):
                The fraction of the sky if a unique number is provided. Otherwise, it is
                the geometrical mean of the fraction of the sky for each field pair. In
                other words an effective fraction of the sky.
        """
        # Import here to avoid circular imports
        from ..functions import get_reduced_covariances

        ell = np.arange(lmin, lmax + 1, 1)
        if N != 1:
            reduced_data = get_reduced_covariances(data, lmin, lmax)
            reduced_coba = get_reduced_covariances(coba, lmin, lmax)

            M_ℓ = list(map(np.linalg.solve, reduced_coba, reduced_data))
            return (
                (2 * ell + 1)
                * fsky
                * [np.trace(M) - np.linalg.slogdet(M)[1] - M.shape[0] for M in M_ℓ]
            )
        else:
            print(ell)
            ratio = data[0, 0, :] / coba[0, 0, :]
            return (2 * ell + 1) * fsky * (ratio - np.log(ratio) - 1)

    @staticmethod
    def _calculate_gaussian(
        data: np.ndarray,
        coba: np.ndarray,
        N: int,
        mask: np.ndarray,
        inverse_covariance: list[np.ndarray],
        lmin: int,
        lmax: int,
        **kwargs,
    ) -> list | np.ndarray:
        """Computes proper chi-square term for the Gaussian likelihood case.

        Parameters:
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            N (int):
                Number of fields.
            mask (ndarray):
                Mask corresponding to both excluded probes and excluded multipole ranges.
            inverse_covariance (list(ndarray)):
                Inverse of the covariance matrices for each multipole.
            lmin (int):
                The minimum multipole to consider.
            lmax (int):
                The maximum multipole to consider.
        """
        # Import here to avoid circular imports
        from ..functions import get_reduced_data_vectors

        if N != 1:
            reduced_coba = get_reduced_data_vectors(N, coba, mask, lmin, lmax)
            reduced_data = get_reduced_data_vectors(N, data, mask, lmin, lmax)

            return [
                (reduced_coba[j] - reduced_data[j])
                @ inverse_covariance[j]
                @ (reduced_coba[j] - reduced_data[j])
                for j in range(lmax + 1 - lmin)
            ]
        else:
            diff = coba[0, 0, :] - data[0, 0, :]
            return diff ** 2 * np.array(inverse_covariance)[:, 0, 0]

    @staticmethod
    def _calculate_correlated_gaussian(
        data: np.ndarray, coba: np.ndarray, inverse_covariance: list[np.ndarray], **kwargs
    ) -> float:
        """Computes proper chi-square term for the correlated Gaussian likelihood case.

        Parameters:
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            inverse_covariance (list(ndarray)):
                Inverse of the covariance matrices for each multipole.
        """
        diff = coba[0, 0, :] - data[0, 0, :]
        return diff @ inverse_covariance @ diff

    @staticmethod
    def _calculate_HL(
        data: np.ndarray,
        coba: np.ndarray,
        fidu: np.ndarray,
        offset: np.ndarray,
        inverse_covariance: list[np.ndarray],
        **kwargs,
    ) -> float:
        """Computes proper chi-square term for the Hamimeche & Lewis likelihood case.

        Parameters:
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            fidu (ndarray):
                The fiducial covariance matrix.
            offset (ndarray):
                The offset array for the calculation.
            inverse_covariance (list(ndarray)):
                Inverse of the covariance matrices for each multipole.
        """
        data_offset = data[0, 0, :] + offset[0, 0, :]
        coba_offset = coba[0, 0, :] + offset[0, 0, :]
        M = np.array(data_offset / coba_offset)
        g = np.sign(M - 1) * np.sqrt(2 * (M - np.log(M) - 1))

        weighted_g = g * (fidu[0, 0, :] + offset[0, 0, :])
        return weighted_g @ inverse_covariance @ weighted_g

    @staticmethod
    def _calculate_LoLLiPoP(
        data: np.ndarray,
        coba: np.ndarray,
        fidu: np.ndarray,
        offset: np.ndarray,
        inverse_covariance: list[np.ndarray],
        **kwargs,
    ) -> float:
        """Computes proper chi-square term for the LoLLiPoP likelihood case.

        Parameters:
            data (ndarray):
                The covariance matrix of the data.
            coba (ndarray):
                The covariance matrix on the MCMC step.
            fidu (ndarray):
                The fiducial covariance matrix.
            offset (ndarray):
                The offset array for the calculation.
            inverse_covariance (list(ndarray)):
                Inverse of the covariance matrices for each multipole.
        """
        data_offset = data[0, 0, :] + offset[0, 0, :]
        coba_offset = coba[0, 0, :] + offset[0, 0, :]
        M = np.array(data_offset / coba_offset)
        g = (
            np.sign(M)
            * np.sign(np.abs(M) - 1)
            * np.sqrt(2 * (np.abs(M) - np.log(np.abs(M)) - 1))
        )

        weighted_g = g * (fidu[0, 0, :] + offset[0, 0, :])
        return weighted_g @ inverse_covariance @ weighted_g


# Legacy function wrappers for backward compatibility
def get_chi_exact(
    N: int,
    data: np.ndarray,
    coba: np.ndarray,
    lmin: int,
    lmax: int,
    fsky: float,
) -> np.ndarray:
    """Legacy wrapper for exact chi-square calculation."""
    return ChiSquareCalculator.calculate(
        ChiSquareMethod.EXACT, data, coba, N=N, lmin=lmin, lmax=lmax, fsky=fsky
    )


def get_chi_gaussian(
    N: int,
    data: np.ndarray,
    coba: np.ndarray,
    mask: np.ndarray,
    inverse_covariance: list[np.ndarray],
    lmin: int,
    lmax: int,
) -> list | np.ndarray:
    """Legacy wrapper for Gaussian chi-square calculation."""
    return ChiSquareCalculator.calculate(
        ChiSquareMethod.GAUSSIAN,
        data,
        coba,
        N=N,
        mask=mask,
        inverse_covariance=inverse_covariance,
        lmin=lmin,
        lmax=lmax,
    )


def get_chi_correlated_gaussian(
    data: np.ndarray, coba: np.ndarray, inverse_covariance: list[np.ndarray]
) -> float:
    """Legacy wrapper for correlated Gaussian chi-square calculation."""
    return ChiSquareCalculator.calculate(
        ChiSquareMethod.CORRELATED_GAUSSIAN,
        data,
        coba,
        inverse_covariance=inverse_covariance,
    )


def get_chi_HL(
    data: np.ndarray,
    coba: np.ndarray,
    fidu: np.ndarray,
    offset: np.ndarray,
    inverse_covariance: list[np.ndarray],
) -> float:
    """Legacy wrapper for Hamimeche & Lewis chi-square calculation."""
    return ChiSquareCalculator.calculate(
        ChiSquareMethod.HAMIMECHE_LEWIS,
        data,
        coba,
        fidu=fidu,
        offset=offset,
        inverse_covariance=inverse_covariance,
    )


def get_chi_LoLLiPoP(
    data: np.ndarray,
    coba: np.ndarray,
    fidu: np.ndarray,
    offset: np.ndarray,
    inverse_covariance: list[np.ndarray],
) -> float:
    """Legacy wrapper for LoLLiPoP chi-square calculation."""
    return ChiSquareCalculator.calculate(
        ChiSquareMethod.LOLLIPOP,
        data,
        coba,
        fidu=fidu,
        offset=offset,
        inverse_covariance=inverse_covariance,
    )


__all__ = [
    "ChiSquareCalculator",
    "ChiSquareMethod",
    # Legacy functions for backward compatibility
    "get_chi_exact",
    "get_chi_gaussian",
    "get_chi_correlated_gaussian",
    "get_chi_HL",
    "get_chi_LoLLiPoP",
]
