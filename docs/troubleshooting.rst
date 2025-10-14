Troubleshooting
===============

This section addresses common issues you might encounter when using LiLit.

Installation Issues
-------------------

**Package not found after installation**

If you get ``ModuleNotFoundError: No module named 'lilit'`` after installation:

.. code-block:: bash

   # Check if lilit is properly installed
   pip list | grep lilit
   
   # If not found, try reinstalling
   pip install --upgrade lilit
   
   # Or install in development mode if working from source
   pip install -e .

**Dependency conflicts**

LiLit requires specific versions of dependencies. If you encounter conflicts:

.. code-block:: bash

   # Create a clean environment
   python -m venv lilit_env
   source lilit_env/bin/activate  # On Windows: lilit_env\Scripts\activate
   
   # Install lilit
   pip install lilit

Likelihood Setup Issues
-----------------------

**Singular covariance matrix errors**

This typically occurs when multipole ranges or field combinations create singular matrices:

.. code-block:: python

   # Enable debug mode to see what's happening
   likelihood = LiLit(
       fields=["t", "e", "b"],
       debug=True,  # This will show diagnostic information
       lmax=[2500, 2000, 500],
       fsky=[0.7, 0.7, 0.6]
   )

Common causes:
- Multipole ranges that don't overlap between fields
- Excluded probes that remove too many matrix elements
- Very small sky fractions leading to numerical issues

**Missing noise file**

If you get errors about missing noise files:

.. code-block:: python

   # LiLit will use placeholder noise if no file is provided
   likelihood = LiLit(
       fields=["t", "e"],
       # nl_file="/path/to/noise.pkl",  # Comment out if file doesn't exist
       lmax=[2500, 2000],
       fsky=[0.7, 0.7]
   )

**Field specification errors**

Make sure field names are recognized:

.. code-block:: python

   # Correct field names
   fields = ["t", "e", "b"]  # temperature, E-mode, B-mode
   
   # Not: ["T", "E", "B"] or ["temp", "pol"]

Cobaya Integration Issues
-------------------------

**LiLit not recognized by Cobaya**

Ensure LiLit is properly installed and importable:

.. code-block:: python

   # Test import
   try:
       from lilit import LiLit
       print("LiLit imported successfully")
   except ImportError as e:
       print(f"Import failed: {e}")

In Cobaya YAML files, use the full module path:

.. code-block:: yaml

   likelihood:
     lilit.LiLit:  # Note the module.class format
       fields: ["t", "e", "b"]
       lmax: [2500, 2000, 500]

**Parameter specification errors**

For tensor modes, ensure you include the required parameters:

.. code-block:: yaml

   params:
     # Standard parameters
     H0: {prior: {min: 60, max: 80}}
     omega_b: {prior: {min: 0.019, max: 0.025}}
     # ... other parameters
     
     # For B-mode analysis, add:
     r: {prior: {min: 0, max: 0.1}}
     # n_t can be derived from consistency relation

**Theory code compatibility**

LiLit works with both CAMB and CLASS through Cobaya:

.. code-block:: yaml

   # With CAMB
   theory:
     camb: null
   
   # With CLASS  
   theory:
     classy: null

Performance Issues
------------------

**Slow likelihood evaluation**

The exact likelihood can be computationally expensive. For faster evaluation:

.. code-block:: python

   # Use Gaussian approximation for speed
   likelihood = LiLit(
       fields=["t", "e", "b"],
       like="gaussian",  # Instead of "exact"
       lmax=[2500, 2000, 500]
   )
   
   # Or reduce lmax for testing
   likelihood = LiLit(
       fields=["t", "e", "b"], 
       like="exact",
       lmax=[1000, 800, 200]  # Lower lmax for testing
   )

**Memory issues with high lmax**

For very high multipoles:

.. code-block:: python

   # Process fields separately if memory is limited
   likelihood_tt = LiLit(fields=["t"], lmax=[5000])
   likelihood_pol = LiLit(fields=["e", "b"], lmax=[2000, 500])

Numerical Issues
----------------

**NaN or infinite values in likelihood**

This can occur with:
- Very small or zero power spectra
- Numerical precision issues with matrix operations
- Inappropriate multipole ranges

Debug with:

.. code-block:: python

   # Check your spectra
   import numpy as np
   
   # Ensure no zeros or NaNs in fiducial spectra
   assert np.all(np.isfinite(fiducial_spectra['tt']))
   assert np.all(fiducial_spectra['tt'] > 0)
   
   # Check multipole ranges are reasonable
   assert all(lmin < lmax for lmin, lmax in zip([2, 2, 2], [2500, 2000, 500]))

**Matrix inversion failures**

Enable debug mode and check for:

.. code-block:: python

   likelihood = LiLit(debug=True, ...)
   
   # Look for warnings about:
   # - Singular matrices being regularized
   # - Removed rows/columns due to zero entries
   # - Condition number warnings

Common Error Messages
---------------------

**"Fields must be a list of strings"**

.. code-block:: python

   # Wrong
   fields = "teb"
   
   # Correct
   fields = ["t", "e", "b"]

**"lmax must have same length as fields"**

.. code-block:: python

   # Wrong: 3 fields, 2 lmax values
   fields = ["t", "e", "b"]
   lmax = [2500, 2000]
   
   # Correct
   fields = ["t", "e", "b"]  
   lmax = [2500, 2000, 500]

**"Unknown approximation: exact"**

Check the spelling of the likelihood approximation:

.. code-block:: python

   # Wrong
   like = "Exact"
   
   # Correct
   like = "exact"  # or "gaussian"

Getting Help
------------

If you encounter issues not covered here:

1. **Check debug output**: Enable ``debug=True`` to see diagnostic information

2. **Verify installation**: Ensure all dependencies are properly installed

3. **Check the examples**: Look at working examples in the documentation

4. **GitHub Issues**: Report bugs or ask questions at https://github.com/ggalloni/LiLit/issues

5. **Community support**: Ask questions on relevant forums like CosmoCoffee

When reporting issues, please include:
- Your LiLit version: ``pip show lilit``  
- Python version: ``python --version``
- Complete error traceback
- Minimal code example that reproduces the issue
- Your system information (OS, etc.)