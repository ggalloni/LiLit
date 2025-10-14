"""
Simple integration tests for chi-square calculations.

This module provides basic tests that don't rely on complex data reduction functions
and can verify that the core chi-square calculations work correctly.
"""

import numpy as np
import pytest

from lilit.core.chi_square import (
    ChiSquareCalculator,
    ChiSquareMethod,
    get_chi_exact,
)


class TestBasicChiSquare:
    """Basic tests for chi-square functionality."""

    def test_exact_single_field_simple(self):
        """Test exact calculation for single field with simple data."""
        N = 1
        lmin, lmax = 2, 5
        total_size = lmax + 1 - lmin

        # Create simple test data
        data = np.ones((N, N, total_size)) * 2.0
        coba = np.ones((N, N, total_size)) * 1.0
        fsky = 0.5

        result = ChiSquareCalculator.calculate(
            ChiSquareMethod.EXACT, data, coba, N=N, lmin=lmin, lmax=lmax, fsky=fsky
        )

        # Should return an array with proper length
        expected_length = lmax + 1 - lmin
        assert len(result) == expected_length
        assert np.all(np.isfinite(result))

    def test_correlated_gaussian_simple(self):
        """Test correlated Gaussian with simple data."""
        total_size = 6  # ell from 0 to 5

        # Create simple test data
        data = np.ones((1, 1, total_size)) * 2.0
        coba = np.ones((1, 1, total_size)) * 1.0

        # Simple identity covariance matrix
        inverse_cov = np.eye(total_size)

        result = ChiSquareCalculator.calculate(
            ChiSquareMethod.CORRELATED_GAUSSIAN,
            data,
            coba,
            inverse_covariance=inverse_cov,
        )

        # Should be a scalar
        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)
        # Should be 6.0 (sum of (2-1)^2 = 1 for each of 6 elements)
        expected = 6.0
        np.testing.assert_almost_equal(result, expected)

    def test_hamimeche_lewis_simple(self):
        """Test Hamimeche & Lewis calculation."""
        total_size = 4

        # Create simple test data (avoid zeros)
        data = np.ones((1, 1, total_size)) * 2.0
        coba = np.ones((1, 1, total_size)) * 1.0
        fidu = np.ones((1, 1, total_size)) * 1.5
        offset = np.ones((1, 1, total_size)) * 0.1

        inverse_cov = np.eye(total_size)

        result = ChiSquareCalculator.calculate(
            ChiSquareMethod.HAMIMECHE_LEWIS,
            data,
            coba,
            fidu=fidu,
            offset=offset,
            inverse_covariance=inverse_cov,
        )

        assert isinstance(result, (float, np.floating))
        assert np.isfinite(result)

    def test_legacy_compatibility(self):
        """Test that legacy functions still work."""
        N = 1
        lmin, lmax = 2, 4
        # Create full-size arrays like in real usage (0 to lmax)
        full_size = lmax + 1

        full_data = np.ones((N, N, full_size)) * 1.5
        full_coba = np.ones((N, N, full_size)) * 1.0
        fsky = 0.8

        # In real usage, data arrays are sliced to the multipole range before chi-square
        sliced_data = full_data[:, :, lmin : lmax + 1]
        sliced_coba = full_coba[:, :, lmin : lmax + 1]

        # Test legacy function with sliced data (as it would receive in real usage)
        legacy_result = get_chi_exact(N, sliced_data, sliced_coba, lmin, lmax, fsky)

        # Test modern function with sliced data
        modern_result = ChiSquareCalculator.calculate(
            ChiSquareMethod.EXACT,
            sliced_data,
            sliced_coba,
            N=N,
            lmin=lmin,
            lmax=lmax,
            fsky=fsky,
        )

        np.testing.assert_array_equal(legacy_result, modern_result)

    def test_string_vs_enum_methods(self):
        """Test that string and enum method calls give identical results."""
        total_size = 5
        data = np.ones((1, 1, total_size)) * 1.2
        coba = np.ones((1, 1, total_size)) * 1.0
        inverse_cov = np.eye(total_size)

        result_enum = ChiSquareCalculator.calculate(
            ChiSquareMethod.CORRELATED_GAUSSIAN,
            data,
            coba,
            inverse_covariance=inverse_cov,
        )

        result_string = ChiSquareCalculator.calculate(
            "correlated_gaussian", data, coba, inverse_covariance=inverse_cov
        )

        np.testing.assert_equal(result_enum, result_string)

    def test_error_handling(self):
        """Test proper error handling for invalid inputs."""
        data = np.ones((1, 1, 5))
        coba = np.ones((1, 1, 5))

        # Test invalid method
        with pytest.raises(ValueError, match="Unsupported chi-square method"):
            ChiSquareCalculator.calculate("invalid_method", data, coba)

        # Test missing parameters
        with pytest.raises(TypeError):
            ChiSquareCalculator.calculate(
                ChiSquareMethod.EXACT,
                data,
                coba,
                # Missing required parameters like N, lmin, lmax, fsky
            )
