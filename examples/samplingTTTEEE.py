"""Sample on CMB temperature and E-modes."""
import time
from mpi4py import MPI
from cobaya.run import run
from cobaya.log import LoggedError
from lilit import LiLit
from lilit import CAMBres2dict

debug = True
name = "TTTEEE"

# Note that the order of these list has to be the same of the fields keyword
lmin = [2, 20]
lmax = [1500, 1200]
fsky = [0.8, 0.5]

exactTTTEEE = LiLit(
    name=name,
    fields=["t", "e"],
    excluded_probes=None,
    like="exact",
    experiment="PTEPLiteBIRD",
    nside=256,
    lmin=lmin,
    lmax=lmax,
    fsky=fsky,
    debug=debug,
)

gaussTTTEEE = LiLit(
    name=name,
    fields=["t", "e"],
    excluded_probes=None,
    like="gaussian",
    experiment="PTEPLiteBIRD",
    nside=256,
    lmin=lmin,
    lmax=lmax,
    fsky=fsky,
    debug=debug,
)

info = {
    "likelihood": {name: exactTTTEEE},
    "params": {
        "As": {"latex": "A_\\mathrm{s}", "value": "lambda logA: 1e-10*np.exp(logA)"},
        "H0": {"latex": "H_0", "max": 100, "min": 20},
        # "H0": 67.32117,
        "cosmomc_theta": {
            "derived": False,
            "value": "lambda theta_MC_100: 1.e-2*theta_MC_100",
        },
        "logA": {
            "drop": True,
            "latex": "\\log(10^{10} A_\\mathrm{s})",
            "prior": {"max": 3.91, "min": 1.61},
            "proposal": 0.001,
            "ref": {"dist": "norm", "loc": 3.04478383213, "scale": 0.0001},
            # "ref": 3.04478383213,
        },
        "mnu": 0.06,
        "ns": {
            "latex": "n_\\mathrm{s}",
            "prior": {"max": 1.2, "min": 0.8},
            "proposal": 0.002,
            "ref": {"dist": "norm", "loc": 0.9660499, "scale": 0.0004},
            # "ref": 0.9660499,
        },
        "ombh2": {
            "latex": "\\Omega_\\mathrm{b} h^2",
            "prior": {"max": 0.1, "min": 0.005},
            "proposal": 0.0001,
            "ref": {"dist": "norm", "loc": 0.0223828, "scale": 0.00001},
            # "ref": 0.0223828,
        },
        "omch2": {
            "latex": "\\Omega_\\mathrm{c} h^2",
            "prior": {"max": 0.99, "min": 0.001},
            "proposal": 0.0005,
            "ref": {"dist": "norm", "loc": 0.1201075, "scale": 0.0001},
            # "ref": 0.1201075,
        },
        "tau": {
            "latex": "\\tau_\\mathrm{reio}",
            "prior": {"max": 0.8, "min": 0.01},
            "proposal": 0.003,
            "ref": {"dist": "norm", "loc": 0.05430842, "scale": 0.0006},
            # "ref": 0.05430842,
        },
        "theta_MC_100": {
            "drop": True,
            "latex": "100\\theta_\\mathrm{MC}",
            "prior": {"max": 10, "min": 0.5},
            "proposal": 0.0002,
            "ref": {"dist": "norm", "loc": 1.04109, "scale": 0.00004},
            # "ref": 1.041456798,
            "renames": "theta",
        },
    },
    "output": f"chains/exact{name}_lmax{lmax}",
    "force": True,
    "resume": False,
    # "debug": True,
    "stop-at-error": True,
    "sampler": {
        "mcmc": {
            "Rminus1_cl_stop": 0.2,
            "Rminus1_stop": 0.1,
        },
    },
    "theory": {
        "camb": {
            "extra_args": {
                "bbn_predictor": "PArthENoPE_880.2_standard.dat",
                "halofit_version": "mead",
                "lens_potential_accuracy": 1,
                "NonLinear": "NonLinear_both",  # This is necessary to be concordant with Planck2018 fiducial spectra
                "max_l": 2700,  # This is necessary to get accurate lensing B-modes
                "WantTransfer": True,  # This is necessary to be concordant with Planck2018 fiducial spectra
                "Transfer.high_precision": True,  # This is necessary to be concordant with Planck2018 fiducial spectra (this will impact negatively on the performance, so you might want to switch it off. However, remember to change the fiducial accordingly.)
                "parameterization": 2,
                "num_nu_massless": 2.046,
                "share_delta_neff": True,
                "YHe": 0.2454006,
                "pivot_tensor": 0.05,
                "num_massive_neutrinos": 1,
                "theta_H0_range": [20, 100],
            },
        },
    },
}


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


start = time.time()

success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError as err:
    print(err)

success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")

end = time.time()

print(f"******** ALL DONE IN {round(end-start, 2)} SECONDS! ********")
