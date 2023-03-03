import os
import camb
import pickle

lmax = 600
As = 2.100549e-9
ns = 0.9660499
r = 0
nt = 0
pivot_s = 0.05
pivot_t = 0.05

pars = camb.read_ini(os.path.join("./", "planck_2018.ini"))
pars.InitPower.set_params(
    As=As,
    ns=ns,
    r=r,
    nt=nt,
    pivot_tensor=pivot_t,
    pivot_scalar=pivot_s,
    parameterization=2,
)
pars.Want_CMB = True
pars.WantTensors = True
pars.DoLensing = True
results = camb.get_results(pars)

# print(pars)

res = results.get_total_cls(CMB_unit="muK", lmax=lmax, raw_cl=False)

fiduCLs = {}
fiduCLs["tt"] = res[:, 0]
fiduCLs["ee"] = res[:, 1]
fiduCLs["te"] = res[:, 3]
fiduCLs["bb"] = res[:, 2]

print(f"Produced spectra for {fiduCLs.keys()} and lmax = {lmax}")

a_file = open("./noise.pkl", "wb")
pickle.dump(fiduCLs, a_file)
a_file.close()

print("\nSaved!")
