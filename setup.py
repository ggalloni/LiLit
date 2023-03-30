from setuptools import setup
from lilit import __version__


setup(
    name="lilit",
    version=__version__,
    description="A Python package encoding a likelihood for LiteBIRD.",
    license="GNU General Public License v3.0",
    license_files=["LICENSE"],
    author="Galloni, Giacomo",
    author_email="giacomo.galloni@roma2.infn.it",
    packages=["lilit"],
    package_dir={"lilit": "lilit"},
    url="https://github.com/ggalloni/LiLit",
    keywords="bayesian-statistics, markov-chain-mote-carlo, cosmic-microwave-background, cobaya",
    install_requires=[
        "cobaya",
        "numpy",
        "healpy",
        "pyyaml",
        "pickle-mixin",
        "matplotlib",
        "camb",
    ],
)
