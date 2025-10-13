"""Sample on B-modes."""

import numpy as np
import pytest
from cobaya import get_model
from cobaya.run import run


@pytest.mark.parametrize("likelihood", ["exact", "gauss", "correlated", "hl", "lollipop"])
def test_model(
    likelihood,
    info_dict,
    exact_likelihood,
    gauss_likelihood,
    correlated_likelihood,
    hl_likelihood,
    lollipop_likelihood,
    ref_like_values,
):
    if likelihood == "exact":
        like = exact_likelihood
    elif likelihood == "gauss":
        like = gauss_likelihood
    elif likelihood == "correlated":
        like = correlated_likelihood
    elif likelihood == "hl":
        like = hl_likelihood
    elif likelihood == "lollipop":
        like = lollipop_likelihood
    else:
        raise ValueError(f"Unknown likelihood type: {likelihood}")
    info_dict["likelihood"] = {"test": like}
    model = get_model(info_dict)
    loglike = model.loglikes()[0][0]
    print(loglike)
    np.testing.assert_almost_equal(loglike, ref_like_values[likelihood], decimal=10)


def test_mcmc(info_dict):
    """Run a quick MCMC test."""

    info_dict["params"]["r"] = {
        "latex": "r_{0.01}",
        "prior": {"max": 3, "min": 0},
        "proposal": 0.0005,
        "ref": 0.02,
    }
    info_dict["params"]["nt"] = {
        "latex": "n_t",
        "prior": {"max": 5, "min": -5},
        "proposal": 0.1,
        "ref": 0.0,
    }

    upd_info, mcmc = run(info_dict)
