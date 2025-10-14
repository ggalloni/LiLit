"""
Pytest configuration and shared fixtures for LiLit tests.

This module provides common test fixtures and configuration that can be
shared across all test modules.
"""

import os

import numpy as np
import pytest

mainpath = os.path.dirname(__file__)
data_path = os.path.join(mainpath, "data")


@pytest.fixture(scope="function")
def random_seed():
    """Set a global random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture(scope="session")
def small_test_data():
    """Generate small test data for quick unit tests."""
    N = 1
    lmin, lmax = 2, 5
    size = lmax + 1

    return {
        "N": N,
        "lmin": lmin,
        "lmax": lmax,
        "data": np.random.rand(N, N, size) + 0.1,
        "coba": np.random.rand(N, N, size) + 0.1,
        "fsky": 0.7,
    }


@pytest.fixture(scope="session")
def medium_test_data():
    """Generate medium-sized test data for more comprehensive tests."""
    N = 2
    lmin, lmax = 2, 20
    size = lmax + 1
    ell_range = lmax + 1 - lmin

    data = {
        "N": N,
        "lmin": lmin,
        "lmax": lmax,
        "ell_range": ell_range,
        "data": np.random.rand(N, N, size) + 0.1,
        "coba": np.random.rand(N, N, size) + 0.1,
        "fidu": np.random.rand(N, N, size) + 0.1,
        "offset": np.random.rand(N, N, size) * 0.01,
        "mask": np.zeros((N, N, size)),
        "fsky": 0.7,
    }

    # Add inverse covariance matrices
    data["inverse_covariance"] = [np.eye(N * N) for _ in range(ell_range)]
    data["inverse_covariance_single"] = np.eye(ell_range)

    return data


def _info_dict():
    info = {
        "params": {
            "As": 2.100549e-9,
            "ns": 0.9660499,
            "ombh2": 0.0223828,
            "omch2": 0.1201075,
            "omnuh2": 0.6451439e-03,
            "H0": 67.32117,
            "tau": 0.05430842,
            "nt": 0.0,
            "r": 0.02,
        },
        "force": True,
        "resume": False,
        "debug": True,
        "stop-at-error": True,
        "sampler": {
            "mcmc": {
                "Rminus1_cl_stop": 0.2,
                "Rminus1_stop": 0.01,
                "max_samples": 10,
                "max_tries": 10,
            },
        },
        "theory": {
            "camb": {
                "extra_args": {
                    "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                    "halofit_version": "mead",
                    "lens_potential_accuracy": 1,
                },
            },
        },
    }
    return info


@pytest.fixture(scope="session")
def info_dict():
    return _info_dict()


@pytest.fixture(scope="session")
def gauss_likelihood():
    from lilit import LiLit

    return LiLit(
        name="BB",
        fields="b",
        like="gaussian",
        r=0.02,
        nt=0.0,
        experiment="PTEPLiteBIRD",
        nside=128,
        debug=False,
        lmin=2,
        lmax=500,
        fsky=0.60,
    )


@pytest.fixture(scope="session")
def correlated_likelihood():
    """Session-scoped fixture to create correlated likelihood once per test session."""
    from lilit import LiLit

    filename = "cov.npy"
    cov = np.load(os.path.join(data_path, filename))

    return LiLit(
        name="BB",
        fields="b",
        like="correlated_gaussian",
        external_covariance=cov,
        r=0.02,
        nt=0.0,
        experiment="PTEPLiteBIRD",
        nside=128,
        debug=False,
        lmin=2,
        lmax=500,
        fsky=0.60,
    )


@pytest.fixture(scope="session")
def hl_likelihood():
    from lilit import LiLit

    filename = "cov.npy"
    cov = np.load(os.path.join(data_path, filename))

    return LiLit(
        name="BB",
        fields="b",
        like="HL",
        fidu_guess_file=os.path.join(data_path, "fiducial_cl.pkl"),
        external_covariance=cov,
        r=0.02,
        nt=0.0,
        experiment="PTEPLiteBIRD",
        nside=128,
        debug=False,
        lmin=2,
        lmax=500,
        fsky=0.60,
    )


@pytest.fixture(scope="session")
def lollipop_likelihood():
    from lilit import LiLit

    filename = "cov.npy"
    cov = np.load(os.path.join(data_path, filename))

    return LiLit(
        name="BB",
        fields="b",
        like="lollipop",
        r=0.02,
        nt=0.0,
        experiment="PTEPLiteBIRD",
        fidu_guess_file=os.path.join(data_path, "fiducial_cl.pkl"),
        external_covariance=cov,
        nside=128,
        debug=False,
        lmin=2,
        lmax=500,
        fsky=0.60,
    )


@pytest.fixture(scope="session")
def exact_likelihood():
    from lilit import LiLit

    return LiLit(
        name="BB",
        fields="b",
        like="exact",
        r=0.02,
        nt=0.0,
        experiment="PTEPLiteBIRD",
        nside=128,
        debug=False,
        lmin=2,
        lmax=500,
        fsky=0.60,
    )


@pytest.fixture(scope="session")
def ref_like_values():
    return {
        "exact": -96.97435583174286, # -96.97434706873389 -96.97435583174286
        "gauss": -90.11355227861259,
        "correlated": -90.11355227861257,
        "hl": -19.098927501435572,
        "lollipop": -19.098927501435572,
    }
