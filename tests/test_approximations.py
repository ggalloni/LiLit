"""Sample on B-modes."""

import numpy as np
from cobaya import get_model


def test_exact(info_dict, exact_likelihood, ref_like_values):
    """Test exact likelihood in isolation to check for initialization issues."""
    info_dict["likelihood"] = {"test": exact_likelihood}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values["exact"], decimal=5)

def test_gaussian(info_dict, gauss_likelihood, ref_like_values):
    """Test gaussian likelihood in isolation to check for initialization issues."""
    info_dict["likelihood"] = {"test": gauss_likelihood}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values["gauss"], decimal=5)

def test_correlated(info_dict, correlated_likelihood, ref_like_values):
    """Test correlated likelihood in isolation to check for initialization issues."""
    info_dict["likelihood"] = {"test": correlated_likelihood}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values["correlated"], decimal=5)

def test_hl(info_dict, hl_likelihood, ref_like_values):
    """Test Hamimeche-Lewis likelihood in isolation to check for initialization issues."""
    info_dict["likelihood"] = {"test": hl_likelihood}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values["hl"], decimal=5)

def test_lollipop(info_dict, lollipop_likelihood, ref_like_values):
    """Test lollipop likelihood in isolation to check for initialization issues."""
    info_dict["likelihood"] = {"test": lollipop_likelihood}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values["lollipop"], decimal=5)
