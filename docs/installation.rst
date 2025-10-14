Installation
============

LiLit can be installed via pip:

.. code-block:: bash

   pip install lilit

Or using uv (recommended for faster installation):

.. code-block:: bash

   uv add lilit

Development Installation
------------------------

For development, clone the repository and install in development mode:

Using pip:

.. code-block:: bash

   git clone https://github.com/ggalloni/LiLit.git
   cd LiLit
   pip install -e .[dev]

Using uv (recommended):

.. code-block:: bash

   git clone https://github.com/ggalloni/LiLit.git
   cd LiLit
   uv sync --extra dev

Requirements
------------

LiLit requires Python >3.9,<3.14 and the following packages:

* camb>=1.6.0
* cobaya>=3.5.7
* healpy>=1.17.3
* matplotlib>=3.9.4
* numpy>=2.0.2
* pyyaml>=6.0.3