import time
from mpi4py import MPI
from cobaya.run import run
from cobaya.log import LoggedError
from likelihood import LiLit

debug = False
name = "BB"
lmax = 500

r = 0.02
nt = 0.1

exactBB = LiLit(
    name=name,
    fields="b",
    like="exact",
    r=r,
    nt=nt,
    experiment="PTEPLiteBIRD",
    nside=128,
    debug=debug,
    lmax=lmax,
    fsky=0.49,
)
gaussBB = LiLit(
    name=name,
    fields="b",
    like="gaussian",
    r=r,
    nt=nt,
    experiment="PTEPLiteBIRD",
    nside=128,
    debug=debug,
    lmax=lmax,
    fsky=0.49,
)

info = {
    "likelihood": {name: exactBB},
    "params": {
        "As": 2.100549e-9,
        "ns": 0.9660499,
        "ombh2": 0.0223828,
        "omch2": 0.1201075,
        "omnuh2": 0.6451439e-03,
        "H0": 67.32117,
        "tau": 0.05430842,
        "nt": {
            "latex": "n_t",
            "prior": {"max": 5, "min": -5},
            "proposal": 0.1,
            "ref": {"dist": "norm", "loc": nt, "scale": 0.001},
        },
        "r": {
            "latex": "r_{0.01}",
            "prior": {"max": 3, "min": 1e-5},
            "proposal": 0.0002,
            "ref": {"dist": "norm", "loc": r, "scale": 0.0005},
        },
        "r005": {
            "derived": "lambda r, nt, ns: r * (0.05/0.01)**(nt - ns + 1)",
            "min": 0,
            "max": 3,
            "latex": "r_{0.05}",
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
            "Rminus1_stop": 0.01,
        }
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
                "Transfer.high_precision": True,  # This is necessary to be concordant with Planck2018 fiducial spectra (this will impact negatively on the performance, so you might want to switch it off. However, remember to chanfe the fiducial accordingly.)
                "parameterization": 2,
                "num_nu_massless": 2.046,
                "share_delta_neff": True,
                "YHe": 0.2454006,
                "pivot_tensor": 0.01,
                "num_massive_neutrinos": 1,
                "theta_H0_range": [20, 100],
                # "Accuracy.AccuracyBoost": 2, # This helps getting an extra squeeze on the accordance of Cobaya and Fiducial spectra
            }
        }
    },
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

start = time.time()

success = False
try:
    upd_info, mcmc = run(info)
    success = True
except LoggedError:
    pass

success = all(comm.allgather(success))

if not success and rank == 0:
    print("Sampling failed!")

end = time.time()

print(f"ALL DONE IN {round(end-start, 2)} SECONDS!")
