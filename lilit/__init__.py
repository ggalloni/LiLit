"""# Welcome to LiLit!

A Python package encoding the likelihood for LiteBIRD.

In this repository I provide some basic examples of forecasting likelihoods for LiteBIRD. These are implemented to be used a [Cobaya](https://github.com/CobayaSampler/cobaya) context ([J. Torrado and A. Lewis, 2020](https://arxiv.org/abs/2005.05290)), which encapsulates [CosmoMC](https://github.com/cmbant/CosmoMC) in a Python framework. The idea of this repository is to ease the creation of common framework among different  LiteBIRDers, trying to homogenize the post-PTEP works as we recently discussed among the collaboration. The main product of this repository is the Likelihood for LiteBIRD (LiLit), which you can find [here](lilit/likelihood.py). Within LiLit, the most relevant study cases of LiteBIRD (T, E, B) are already tested and working. So, if you need to work with those, you should not need to look into the actual definition of the likelihood function, since you can proptly start running your MCMCs. Despite this, you should provide to the likelihood some file where to find the proper LiteBIRD noise power spectra, given that LiLit is implementing a simple inverse noise weighting just as a place-holder for something more realistic. As regards lensing, LiLit will need you to pass the reconstruction noise, since its computation is not coded, thus there is no place-holder for lensing.

This repository should also give to new Cobaya-users a good starting point to build upon (see [this](examples/other_simple_likelihood.py)). 

The repository can be found at [https://github.com/ggalloni/LiLit](https://github.com/ggalloni/LiLit).

## Some further details on LiLit

If you want to start using LiLit, here are some further details on what you can do with it. Firstly, LiLit is independent of the number of fields considered and can be dynamically modified at declaration. Thus, it makes no difference if you want to use B-modes alone, or if you want to use CMB temperature, E-modes, lensing, etc. The only constraint is that the Boltzmann code you are using to provide the spectra should understand the fields you are asking to LiLit. Each of these fields may have their own $\ell_{\rm min}$, $\ell_{\rm max}$ and $f_{\rm sky}$. So, I implemented the possibility to pass all these quantities as lists to LiLit, which will then take case of the proper multipoles cuts and so on. The only requirement is that you should pass these lists following the order in which you passes the requested fields. For example:

```
fields = ["t", "e", "b"]

# lmax = [lmaxTT, lmaxEE, lmaxBB]
lmax = [1500, 1200, 900]

# lmin = [lminTT, lminEE, lminBB]
lmin = [20, 2, 2]

# fsky = [fskyTT, fskyEE, fskyBB]
fsky = [1.0, 0.8, 0.6]
```

If you do not want to pass custom fiducial power spectra for the fields you requested, you can exploit the fact that LiLit will internally compute every spectra according to _Planck_ 2018 best-fit values of the cosmological parameters. Note that if you requested B-modes, you must provide the value you want to assign to the tensor-to-scalar ratio $r$ (be careful on what pivot scale you are using here). If you do not pass the spectral index $n_t$, it will follow the standard consistency relation. 

As regards noise, as mentioned above you should pass realistic power spectra according to what you are working on. Just as a mere place-holder, LiLit will compute the inverse-weighted noise ([P. Campeti et al., 2020](https://arxiv.org/abs/2007.04241)) over each channel of LiteBIRD ([E. Allys et al., 2022](https://arxiv.org/abs/2202.02773)).

Once you have fixed all these quantities, you are ready to define your likelihood. The only remaining things to decide are what approximation you want to use to compute the $\chi^2$ and the actual name of the likelihood. LiLit implements the exact likelihood ([S. Hamimeche and A. Lewis, 2008](https://arxiv.org/abs/0801.0554)) and the Gaussian one. Thus, you can define your likelihood with:

```
name = exampleTEB

# using the exact likelihood approximation
exampleTEB = LiLit(name=name, fields=fields, like="exact", nl_file="/path/to/noise.pkl", lmax=lmax, fsky=fsky, debug=False)

# using the Gaussian likelihood approximation
exampleTEB = LiLit(name=name, fields=fields, like="gaussian", nl_file="/path/to/noise.pkl", lmax=lmax, fsky=fsky, debug=False)
```

Note that you may want to set debug to True in order to check that everything is OK.

I tried to be as modular as possible so that you can plug whatever existing function you have. Also, this should make parallelization easier if you need it.

## Other simple likelihood

[Here](examples/other_simple_likelihood.py), you can find two very simple examples of the skeleton of a single-field likelihood and a two-field one. If you are new to Cobaya, you may want to have a look at these before jumping to LiLit. 

As regards the single-field likelihood, what you can find there is a very basic example of how to run Cobaya on a generic field X. Some parts of this may require additional attention before running, however this should give you an idea of how to work with this package. For instance, here I am considering a very simple case in which I do not need to pass anything to the likelihood class but the spectra, and in which the likelihood does not provide any derived parameters.

## Examples

I provide also few working examples of the usage of LiLit. Some particular attention should be given to the dictionaries defined in the sampling scripts. If you are not familiar with their structure, have a look at Cobaya's [documentation](https://cobaya.readthedocs.io/en/latest/). Then, I customized these to maximize accordance with the fiducial _Planck_ 2018 spectra. These also assume that you will be using [CAMB](https://github.com/cmbant/CAMB) ([A. Lewis et al., 2000](https://arxiv.org/abs/astro-ph/9911177)); for the [CLASS](https://github.com/lesgourg/class_public) ([D. Blas et al., 2011](https://arxiv.org/abs/1104.2933)) equivalent of this, refer again to documentiation, since some of the parameters are renamed. Note that Cobaya will understand whatever newly defined parameter you added to the Boltzmann code.

In the parameters block of the sampling dictionary, parameters with a prior will be interpreted as open parameters, while all the others are essentially derived ones. Cobaya will figure out by itself whether it has to ask for some of them to the theory code (CAMB) or to other parameters. Also, it will figure out what parts of the routines need certain parameters. For example, you can pass $A_s$ to the likelihood function and do something with it, but at the same time it will pass it also to CAMB to compute the spectra.

Once you have finished preparing your sampling.py file, you can run it simply by using: 

`python sampling.py > log.txt`

Sending the inline output to a text file might help in recovering information on how the run is going. If you want to run it parallely, you may want to use something like:

`mpirun -np 4 --cpus-per-proc 4 python sampling.py > log.txt`

The number of processess translates on the number of chains that will be runned simultaneously.

## Dependencies
* cobaya
* numpy
* healpy
* pyyaml
* pickle
* matplotlib
* camb, or classy

## Documentation

The documentation can be found at [https://ggalloni.github.io/LiLit](https://ggalloni.github.io/LiLit).

## Developing the code

If you want to help developing the code, feel free to send a pull request. Also, feel free to write me so that we can discuss on eventual major developments.

## Support

For additional details on how Cobaya works, I suggest to see the [documentation](https://cobaya.readthedocs.io/en/latest/), which is very good to find whatever is not working in you case. If you cannot find the problem there, feel free to open an issue.
"""

from .likelihood import LiLit

__author__ = "Giacomo Galloni"
__version__ = "1.1.3"
__docformat__ = "numpy"
