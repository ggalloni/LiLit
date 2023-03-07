LiLit: Likelihood for LiteBIRD
==============================================

Authors: Giacomo Galloni

Contributors: Giulia Piccirilli

Overview
--------
In this repository I provide some basic examples of forecasting likelihoods for LiteBIRD. These
are implemented to be used a Cobaya context, which encapsulates CosmoMC in a Python framework. 
The idea of this repository is to ease the creation of common framework among different 
LiteBIRDers, trying to homogenize the post-PTEP works as we recently discussed among the 
collaboration. This should give to new Cobaya-users a good starting point to build upon. 

Templates
---------

In the "Templates" folder, you can find two very simple examples of the 
skeleton of a single-field likelihood and a two-field one. You can built upon those whatever 
specific case you are working on. 

Giving some more details about the single-field likelihood, what you can find there is a very 
basic example of how to run Cobaya on a generic field X. Some parts of this may 
require additional attention before running, however this should give you an idea of how 
to work with this package. For instance, here I am considering a very simple case in which
I do not need to pass anything to the likelihood class but the spectra, and in which the
likelihood does not provide any derived parameters.

Together with these two example, you will find the template of a far more flexible likelihood.
Indeed, once you have understood the basic ones, I suggest you to use what I called LiLit.
It is independent of the number of fields considered and can be dynamically modified at
declaration. As of now, the likelihood is able to compute the logP in two different ways:
the exact likelihood and the Gaussian one.

With LiLit, I tried to be as modular as possible so that you can plug whatever existing function you have.
Also, this should make parallelization easier if you need it.

The dictionary in sampling.py is taken from cobaya-cosmo-generator asking for 
the full analysis of Planck 2018 + tensors, thus it may not apply for the specific 
cases depicted as templates. 
However, this gives you a complete overview of what you can do with CAMB. Indeed, any 
parameters that CAMB undestands (even custom parameters you may have implemented) can be 
passed to the Cobaya framework in such a way. For the CLASS equivalent of this, refer again 
to cobaya-cosmo-generator, since some of the parameters are renamed.

In the parameters block of the same dict, params with a prior will be interpreted as open parameters, 
while all the others are essentially derived ones. Cobaya will figure out by itself 
whether it has to ask for some of them to the theory code (CAMB) or to other parameters.
Also, it will figure out what parts of the routines need certain parameters. For example,
you can pass A_s to the likelihood function and do something with it, but at the same time
it will pass it also to CAMB to compute the spectra.

Example
---------------

To show an actual example, I report two very simple MCMC analyses on BB and on TTTEEE in the "Example" folder. 
There, you can find both the likelihood (LiLit) and the simple Python scripts to run the
analyses. Also, in the subfolder "chains" I store the results, which can be analyzed with GetDist.
The two analyses are carryed out by running samplingBB.py or samplingTTTEEE.py, and the inline output is stored
in logBB.txt and logTTTEEE.txt. Usually I setup two extra scripts for the production of the fiducial power spectra 
and the noise power spectra (here I give two simple examples). Also you will find the Planck 2018 ini file provided
by CAMB.

Note that the BB run ihas converged in ~40 seconds, while the TTTEEE one has not converged yet (I interrupted it
since it is not the scope of this repo to report actual results).

LiteBIRD applications
---------------------

Finally, in the "LB applications" folder, you can find the actual likelihoods that are being used
in some of the post-PTEP papers.

Developing the code
-------------------

If you want to help developing the code, feel free to write me so that I can add you to the collaborators.
Also, if you want to share the likelihood you are using, contact me or send a pull request 
so that we can add it to the LiteBIRD applications already available.

Support
-------

For additional details on how Cobaya works, I suggest to see the documentation at 
https://cobaya.readthedocs.io/en/latest/, which is very good to find whatever 
is not working in you case. If you cannot find the problem there, feel free to open an issue.
