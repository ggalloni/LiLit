# import the new likelihood from like_inflation.py file
from GitHub.LiLit.Templates.likelihoods import exactXX
import time

info = {
    "likelihood": {
        "XX": exactXX
    },  # Here "XX" will be the name Cobaya will use to refer to this likelihood, but it is completely arbitrary
    "params": {
        "A": {"derived": "lambda As: 1e9*As", "latex": "10^9 A_\\mathrm{s}"},
        "As": {"latex": "A_\\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"},
        "DHBBN": {
            "derived": "lambda DH: 10**5*DH",
            "latex": "10^5 \\mathrm{D}/\\mathrm{H}",
        },
        "H0": {"latex": "H_0", "max": 100, "min": 20},
        "YHe": {"latex": "Y_\\mathrm{P}"},
        "Y_p": {"latex": "Y_P^\\mathrm{BBN}"},
        "age": {"latex": "{\\rm{Age}}/\\mathrm{Gyr}"},
        "clamp": {
            "derived": "lambda As, tau: 1e9*As*np.exp(-2*tau)",
            "latex": "10^9 A_\\mathrm{s} e^{-2\\tau}",
        },
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
        "omega_de": {"latex": "\\Omega_\\Lambda"},
        "omegam": {"latex": "\\Omega_\\mathrm{m}"},
        "omegamh2": {
            "derived": "lambda omegam, H0: omegam*(H0/100)**2",
            "latex": "\\Omega_\\mathrm{m} h^2",
        },
        "r": {
            "latex": "r_{0.05}",
            "prior": {"max": 3, "min": 0},
            "proposal": 0.03,
            "ref": {"dist": "norm", "loc": 0, "scale": 0.03},
        },
        "rdrag": {"latex": "r_\\mathrm{drag}"},
        "s8h5": {
            "derived": "lambda sigma8, H0: sigma8*(H0*1e-2)**(-0.5)",
            "latex": "\\sigma_8/h^{0.5}",
        },
        "s8omegamp25": {
            "derived": "lambda sigma8, omegam: " "sigma8*omegam**0.25",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.25}",
        },
        "s8omegamp5": {
            "derived": "lambda sigma8, omegam: " "sigma8*omegam**0.5",
            "latex": "\\sigma_8 \\Omega_\\mathrm{m}^{0.5}",
        },
        "sigma8": {"latex": "\\sigma_8"},
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
        "zre": {"latex": "z_\\mathrm{re}"},
    },
    "output": "/path/to/exactXX_lmax300",  # Here "exactXX_lmax300" will be the name to of the output chain at "/path/to/" folder
    "force": True,  # This forces Cobaya to overwrite an eventual chain with the same name (note that this may be in contrast with the next keyword; typically you want to use force and resume in OR)
    "resume": True,  # This checks if there is a chain with the same name and if so it checks the compatibility of the input params of this dict with the one used to produce the other chain. If the are compatible, Cobaya will try to resume the sampling from where it stopped
    "debug": True,  # This will produce a very verbose output that is very helpful when debugging. Tho, I would not suggest to keep this True if nothing is wrong, since this slows down the MCMC
    "sampler": {
        "mcmc": {
            "Rminus1_cl_stop": 0.2,
            "Rminus1_stop": 0.01,
            "covmat": "auto",
            "drag": True,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
        }
    },
    "theory": {
        "camb": {
            "extra_args": {
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                "halofit_version": "mead",
                "lens_potential_accuracy": 1,
                "nnu": 3.046,
                "nt": None,
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
