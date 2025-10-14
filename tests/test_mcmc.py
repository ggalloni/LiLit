"""Sample on B-modes."""

from cobaya.run import run


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
