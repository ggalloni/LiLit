# LiLit: Likelihood for LiteBIRD

Author: Giacomo Galloni

## Overview

In this repository I provide some basic examples of forecasting likelihoods for LiteBIRD. These are implemented to be used a [Cobaya](https://github.com/CobayaSampler/cobaya) context ([J. Torrado and A. Lewis, 2020](https://arxiv.org/abs/2005.05290)), which encapsulates [CosmoMC](https://github.com/cmbant/CosmoMC) in a Python framework. The idea of this repository is to ease the creation of common framework among different  LiteBIRDers, trying to homogenize the post-PTEP works as we recently discussed among the collaboration. The main product of this repository is the Likelihood for LiteBIRD (LiLit), which you can find [here](lilit/likelihood.py). Within LiLit, the most relevant study cases of LiteBIRD (T, E, B) are already tested and working. So, if you need to work with those, you should not need to look into the actual definition of the likelihood function, since you can proptly start running your MCMCs. Despite this, you should provide to the likelihood some file where to find the proper LiteBIRD noise power spectra, given that LiLit is implementing a simple inverse noise weighting just as a place-holder for something more realistic. As regards lensing, LiLit will need you to pass the reconstruction noise, since its computation is not coded, thus there is no place-holder for lensing.

This repository should also give to new Cobaya-users a good starting point to build upon (see [this](examples/other_simple_likelihood.py)). 

The repository can be found at [https://github.com/ggalloni/LiLit](https://github.com/ggalloni/LiLit).

## Installation

To install the LiLit package it is sufficient to do:

```
pip install lilit
```

Then, to access the LiLit class, it is sufficient to import it as:

```
from lilit import LiLit
```

Finally, to use the provided functions, you should import them as:

```
from lilit import CAMBres2dict
```

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

## Likelihood-specific details on LiLit

As mentioned above, LiLit implements two kinds of likelihood approximations: the exact one and the Gaussian one. Here, I will give some details on how they are implemented and I will clarify some underlying assumption.

### Exact likelihood
Firstly, let me explicitly report the formula used for this approximation, starting from the single field case:

```math
\log\mathcal{L} = -\frac{1}{2}\sum_{\ell}\left(2\ell+1\right)\left[\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}-\log\left(\frac{C_{\ell}^{\rm obs}}{C_{\ell}^{\rm th}}\right)-1\right]
```

Instead assuming to have $N$ fields, the corresponding multiple-field formula reads:

```math
\log\mathcal{L} = -\frac{1}{2}\sum_{\ell}\left(2\ell+1\right)\left[\text{Tr}\left(\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right) - \log\left|\mathcal{C}_{\rm obs}\mathcal{C}^{-1}_{\rm th}\right| - N\right]
```

where for example

```math
\mathcal{C}_{\rm obs} = \left(\begin{array}{ccc}
                            C_{\ell}^{XX} & C_{\ell}^{XY} & C_{\ell}^{XZ}\\
                            C_{\ell}^{YX} & C_{\ell}^{YY} & C_{\ell}^{YZ}\\
                            C_{\ell}^{ZX} & C_{\ell}^{ZY} & C_{\ell}^{ZZ}
                            \end{array}\right)
```

Here, each entry may look something like:

```math
C_{\ell}^{XX} = C_{\ell}^{CMB} + C_{\ell}^{FGs} + \dots + N_{\ell}^{X} + \dots 
```

The handling of the field-specific $\ell_{\rm min}$ and $\ell_{\rm max}$ is done at the level of the covarainces $\mathcal{C}$. In this case, the multipole range of the cross-correlation between two fields is given by the intersection of the two ranges (I encounter some issues with the geometrical mean of the two ranges, so for now I keep the intersection).

Then, for each field, the code will fill with zeros the entries corresponding to a given field if $\ell$ is outside the range $[\ell_{\rm min}, \ell_{\rm max}]$ for that field. Also, it is possible to arbitrarily exclude some probes (say $XZ$) from the likelihood through

`
excluded_probes = ["xz"]
`

These will be also sent to zero in the covariance matrix. This, together with the multipole ranges, will cause the covariance matrix to be singular. In order to avoid this, the code will check whether the determinant of the covariance is null. If so, it will identify the diagonal entries that are null and remove the correspondent row and column.

Note that the likelihood expressions above are exact in the full-sky case. In order to account for the sky cut, both the single-field and the multiple-field formulae are multiplied by a factor $f_{\rm sky}$. If the user provided a single value for the fraction of the sky, this results in reducing the number of modes in the sum by a factor $f_{\rm sky}$ for all the fields. If the user provided a list of values, the code first computes an effective $f_{\rm sky}$ as the geometrical mean of the ones provided. Then it will apply this factor to rescale the number of modes. As a consequence, this approach completely dsregards the coupling between multipoles introduced by the presence of a sky cut. 

### Gaussian likelihood
The Gaussian approximation is based on the following formula:

```math
\log\mathcal{L} = -\frac{1}{2}\sum_{\ell}\left(2\ell+1\right)\left[\frac{\left(C_{\ell}^{\rm obs} - C_{\ell}^{\rm th}\right)^2}{\sigma^{2}_{\ell}}\right]
```

where $\sigma^{2}_{\ell}$ is the variance of the observed power spectrum. In the single-field case, this is given by:

```math
\sigma^{2}_{\ell} = \frac{2}{\left(2\ell+1\right)f_{\rm sky}}\left(C_{\ell}^{\rm obs}\right)^2
```

Once again, the coupling between multipoles introduced by the sky cut is not taken into account. Instead, we just introduce a factor $f_{\rm sky}$ to rescale the variance of the observed power spectrum.

The multi-field case is slightly more involved. The data vector is obtained from the $\mathcal{C}$ matrices defined above as the upper trinaglular part of the matrix. For example, considering the one explicited above, the data vector would be:

```math
X_\ell = \left(C_{\ell}^{XX}, C_{\ell}^{XY}, C_{\ell}^{XZ}, C_{\ell}^{YY}, C_{\ell}^{YZ}, C_{\ell}^{ZZ}\right)
```

Thus, the covariance of this object will be a $6\times6$ matrix (with $N$ fields you get $N(N+1)/2$ different probes). Its expression is given by:

```math
\text{Cov}^{\rm ABCD}_{\ell} = \frac{1}{(2\ell+1)f_{\rm sky}^{AB}f_{\rm sky}^{CD}}\left( \sqrt{f_{\rm sky}^{AC}f_{\rm sky}^{BD}}C_\ell^{AC}C_\ell^{BD} + \sqrt{f_{\rm sky}^{AD}f_{\rm sky}^{BC}}C_\ell^{AD}C_\ell^{BC} \right)
```

where $A,B,C,D$ are the indices of the fields (e.g. $X, Y, Z$ following the example above).
Note that the factor $f_{\rm sky}$ for the cross-correlation of two fields is given by the geometrical average $f_{\rm sky}^{XY} = \sqrt{f_{\rm sky}^{XX}f_{\rm sky}^{YY}}$. This approximation holds as soon as the two masked regions do overlap consistently.

In the Gaussian case, the multipole ranges and excluded probes are handled differently w.r.t. the exact case. Firstly, the range of the cross-correlation of two fields is given by the geometrical mean of the two (e.g. $\ell_{\rm max}^{XY} = \sqrt{\ell_{\rm max}^{XX}\ell_{\rm max}^{YY}}$). Then, the code will fill compute the covariance matrix for all the probes in the largest multipole range. Before inverting, it will compute a mask that is swithced on for the multipoles that are outside the range for a given field (or are excluded a priori). This mask will be applied to the covariance matrix before inverting it and masked entries will be removed, ending up with a smaller matrix. The data vector $X_\ell$ will be masked in the same way, so that the matrix multiplication is performed correctly between equal-sized objects.

Finally, the likelihood reads:

```math
\log\mathcal{L} = -\frac{1}{2}\sum_{\ell}\left(2\ell+1\right)\left[X_\ell \times \text{Cov}^{-1}_{\ell}\times X_\ell^{\rm T}  \right]
```

Differently from the exact likelihood case, here each field retains its own $f_{\rm sky}$ factor. This is possible thank to the fact that the different fields are more easily separable in the Gaussian case.

### Correlated Gaussian likelihood

This likelihood is still in development. At the moment, only the single-field case is implemented. It represent the generalization to account for the correlation between different multipoles. In particular, the covariance matrix must be computed externally and provided to the LiLit class (let me call it $\text{Ext}$). Note that $\ell =0, 1$ must be excluded from the covariance matrix. The code will then proceed to invert the matrix and cumpute the likelihood very similarly to the Gaussian case. This time the likelihood reads:

```math
\log\mathcal{L} = -\frac{1}{2}\left[\left(\vec{C^{\rm obs}} - \vec{C^{\rm th}}\right) \times \text{Ext}^{-1} \times \left(\vec{C^{\rm obs}} - \vec{C^{\rm th}}\right)^{\rm T}\right]
```

where now $\vec{C^{\rm obs}}$ and $\vec{C^{\rm th}}$ are vector along the multipole range. Note that the covariance matrix is not rescaled by $f_{\rm sky}$, since this is already taken into account in the external covariance matrix.

The multi-field case is still to be implemented.

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
* matplotlib
* camb

## Documentation

The documentation can be found at [https://ggalloni.github.io/LiLit](https://ggalloni.github.io/LiLit).

## Developing the code

If you want to help developing the code, feel free to send a pull request. Also, feel free to write me so that we can discuss on eventual major developments.

## Support

For additional details on how Cobaya works, I suggest to see the [documentation](https://cobaya.readthedocs.io/en/latest/), which is very good to find whatever is not working in you case. If you cannot find the problem there, feel free to open an issue.
