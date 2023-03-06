import time
from likelihood import LiLit

debug = False
name = "TTTEEE"
lmax = 600

exactTTTEEE = LiLit(name=name, fields=["t", "e"], lmax=lmax, debug=debug)

info = {
    "likelihood": {name: exactTTTEEE},
    "params": {
        "As": {"latex": "A_\\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"},
        "H0": {"latex": "H_0", "max": 100, "min": 20},
        "cosmomc_theta": {
            "derived": False,
            "value": "lambda theta_MC_100: " "1.e-2*theta_MC_100",
        },
        "logA": {
            "drop": True,
            "latex": "\\log(10^{10} A_\\mathrm{s})",
            "prior": {"max": 3.91, "min": 1.61},
            "proposal": 0.001,
            "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
        },
        "mnu": 0.06,
        "ns": {
            "latex": "n_\\mathrm{s}",
            "prior": {"max": 1.2, "min": 0.8},
            "proposal": 0.002,
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
        },
        "ombh2": {
            "latex": "\\Omega_\\mathrm{b} h^2",
            "prior": {"max": 0.1, "min": 0.005},
            "proposal": 0.0001,
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
        },
        "omch2": {
            "latex": "\\Omega_\\mathrm{c} h^2",
            "prior": {"max": 0.99, "min": 0.001},
            "proposal": 0.0005,
            "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
        },
        "tau": {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"max": 0.8, "min": 0.01},
            "proposal": 0.003,
            "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
        },
        "theta_MC_100": {
            "drop": True,
            "latex": "100\\theta_\\mathrm{MC}",
            "prior": {"max": 10, "min": 0.5},
            "proposal": 0.0002,
            "ref": {"dist": "norm", "loc": 1.04109, "scale": 0.0004},
            "renames": "theta",
        },
    },
    "output": f"chains/exact{name}_lmax{lmax}",
    "force": True,
    "resume": False,
    "debug": debug,
    "stop-at-error": True,
    "sampler": {
        "mcmc": {
            "Rminus1_cl_stop": 0.2,
            "Rminus1_stop": 0.1,
        }
    },
    "theory": {
        "camb": {
            "extra_args": {
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                "halofit_version": "mead",
                "lens_potential_accuracy": 1,
                "nnu": 3.046,
                "num_massive_neutrinos": 1,
                "theta_H0_range": [20, 100],
            }
        }
    },
}

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from cobaya.run import run
from cobaya.log import LoggedError

start = time.time()

success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    pass

success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")

end = time.time()

print(f"ALL DONE IN {round(end-start, 2)} SECONDS!")
