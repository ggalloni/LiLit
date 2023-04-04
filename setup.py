from setuptools import setup, find_packages
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
    package_data={"lilit": ["experiments.yaml", "planck_2018.ini"]},
    url="https://github.com/ggalloni/LiLit",
    keywords="bayesian-statistics, markov-chain-mote-carlo, cosmic-microwave-background, cobaya",
    install_requires=[
        "cobaya",
        "numpy",
        "healpy",
        "pyyaml",
        "matplotlib",
        "camb",
    ],
    setup_requires=[
        "cobaya",
        "numpy",
        "healpy",
        "pyyaml",
        "matplotlib",
        "camb",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
