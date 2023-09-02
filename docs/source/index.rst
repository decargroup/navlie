.. navlie documentation master file, created by
   sphinx-quickstart on Wed Aug 24 15:11:06 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 1
   :hidden:

   Home <self>
   Tutorial <tutorial.rst>
   API <api>


Welcome to navlie's documentation!
----------------------------------

The core idea behind this project is to abstract-away the state definition such that a single estimator implementation can operate on a variety of state manifolds, such as the usual vector space, and any Lie group. At the moment, algorithms and features of this package include:

- Extended Kalman Filter
- Iterated Extended Kalman Filter
- Sigmapoint Kalman Filters (Unscented, Spherical Cubature, Gauss-Hermite)
- Interacting Multiple Model Filter
- Batch MAP Estimation
- A large collection of common process and measurement models
- Out-of-the-box on-manifold numerical jacobian using finite differencing
- Various utils for plotting, error, and consistency evaluation
- Monte Carlo experiment executor with result aggregation
- A preintegration module for linear, wheel odometry, and IMU process models

By implementing a few classes, the user can model almost any problem. Documentation can be found by https://decargroup.github.io/navlie

Setup
-----

Installation
^^^^^^^^^^^^

Clone this repo, change to its directory, and execute 

.. code-block:: bash

    pip install -e .

This command should automatically install all dependencies, including our package `pymlg` (found at https://github.com/decargroup/pymlg) for back-end Lie group mathematical operations.

Examples
^^^^^^^^
Some starting examples running EKFs can be found in the `examples/` folder. Simply run these as python3 scripts 