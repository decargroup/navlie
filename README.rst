navlie
======
.. image:: https://github.com/decargroup/navlie/actions/workflows/test_package.yml/badge.svg
    :target: https://github.com/decargroup/navlie/actions/workflows/test_package.yml
    :alt: Tests


.. image:: https://github.com/decargroup/navlie/actions/workflows/publish_docs.yml/badge.svg
    :target: https://github.com/decargroup/navlie/actions/workflows/publish_docs.yml
    :alt: Docs

.. image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
    :target: https://www.python.org/downloads/
    :alt: Python Version
    
.. image:: ./docs/source/fun_figs.png 
    :alt: Demo Figures
    :align: center
    :width: 100%


An on-manifold state estimation library for robotics.

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

By implementing a few classes, the user can model almost any problem. Documentation can be found at https://decargroup.github.io/navlie

Disclaimer
----------
While navlie is starting to get some maturity, its still definitely not perfect. The code is probably still simple enough that you can check out the  source directly to get a better understanding. If (when) you find a bug, please feel free to open an Github issue. Contributions/thoughts are welcome, and if anything regarding documentation is still grossly unclear, just let us know :). 

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

Tests
^^^^^
To run integration tests, 

.. code-block:: bash

    pytest tests

or, to run tests on a specific test file, 

.. code-block:: bash

    pytest -ra tests/integration/filename.py
    
The Core Concept - Defining a `State` Implementation
----------------------------------------------------

The algorithms and models in this repo are centered around the abstract `State` class. An instance of `State` is an object containing, at a minimum, the following attributes:

- `value`: a value of some sort;
- `dof`: the degrees of freedom associated with the state.

It will also contain the following mandatory methods that must be implemented by the user.

- `plus()`:  A generic "addition" operation given a `dx` vector with as many elements as the `dof` of this state.
- `minus()`:  A generic "subtraction" operation given another State object of the same type, which returns a numpy array of error values.
- `copy()`: A method that returns a new object of the same type, and with the same attibute values.

Optionally, it is often useful to assign a timestamp (`stamp`) and a label (`state_id`) to differentiate state instances from others. The snippet below shows how to define a simple vector-space state:  


.. code-block:: python

    from navlie.types import State 
    import numpy as np

    class VectorState(State):
        """
        A standard vector-based state, with value represented by a 1D numpy array.
        """

        def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
            super(VectorState, self).__init__(
                value=value,
                dof=value.size,
                stamp=stamp,
                state_id=state_id,
            )

        def plus(self, dx: np.ndarray):
            new = self.copy()
            new.value = new.value + dx
            return new

        def minus(self, x: "VectorState") -> np.ndarray:
            return self.value - x.value

        def copy(self) -> "VectorState":
            return VectorState(self.value.copy(), self.stamp, self.state_id)


As another more complicated example, a state object belonging to the SE(3) Lie group can be implemented as 

.. code-block:: python

    from navlie.types import State 
    from pymlg import SE3 
    import numpy as np 

    class SE3State(State):
        def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
            super(SE3State, self).__init__(
                value=value,
                dof=6,
                stamp=stamp,
                state_id=state_id,
            )
        
        def plus(self, dx: np.ndarray):
            new = self.copy()
            new.value = new.value @ SE3.Exp(dx)
            return new

        def minus(self, x: "SE3State") -> np.ndarray:
            return SE3.Log(SE3.inverse(x.value) @ self.value)

        def copy(self) -> "SE3State":
            return SE3State(self.value.copy(), self.stamp, self.state_id)


Process and Measurement Models
------------------------------
.. image:: ./docs/source/system_diagram.png
    :alt: System Diagram

There are a few more core types in this package. The main ones are the `ProcessModel` and `MeasurementModel` classes. Both of these are abstract classes requiring the user to implement

- an `evaluate()` method, 
- a `covariance()` method,
- and optionally a `jacobian()` method.

For example, a simple "single integrator" (velocity input) model can be implemented as follows:

.. code-block:: python

    class SingleIntegrator(ProcessModel):
        """
        The single-integrator process model is a process model of the form

            x_k = x_{k-1} + dt * u_{k-1}
        """

        def __init__(self, Q: np.ndarray):
            self._Q = Q

        def evaluate(self, x: VectorState, u: VectorInput, dt: float) -> np.ndarray:
            """
            Returns a state with an updated value according to a process model.
            """
            x.value = x.value + dt * u.value
            return x

        def jacobian(self, x: VectorState, u: VectorInput, dt: float) -> np.ndarray:
            """
            (optional) Jacobian of the process model with respect to the state.
            """
            return np.identity(x.dof)

        def covariance(self, x: VectorState, u: VectorInput, dt: float) -> np.ndarray:
            """
            Returns the covariance of the process model errors.
            """
            return dt**2 * self._Q


Similarly, a single distance-to-landmark measurement model can be implemented as 

.. code-block:: python 

    class RangePointToAnchor(MeasurementModel):
        """
        Range measurement from a point state to an anchor (which is also another
        point).
        """

        def __init__(self, anchor_position: List[float], R: float):
            self._r_cw_a = np.array(anchor_position)
            self._R = np.array(R)

        def evaluate(self, x: VectorState) -> np.ndarray:
            r_zw_a = x.value
            y = np.linalg.norm(self._r_cw_a - r_zw_a)
            return y

        def jacobian(self, x: VectorState) -> np.ndarray:
            r_zw_a = x.value
            r_zc_a = r_zw_a - self._r_cw_a
            y = np.linalg.norm(r_zc_a)
            return r_zc_a.reshape((1, -1)) / y

        def covariance(self, x: VectorState) -> np.ndarray:
            return self._R


In fact, for both `ProcessModel` and `MeasurementModel`, subclasses will inherit a finite-difference numerical differentiation method `jacobian_fd()`, which also serves as the default implementation if `jacobian` is not overriden. Nevertheless, it allows for an easy way to check your `jacobian()` implementation! (`evaluate()` method must be implemented for this to work, see some of the files in `tests/` for an example of this.)

Built-in Library
----------------
Many state, process, and measurement models are already written and part of the built-in library and, as an example, can be accessed with 

.. code-block:: python 

    from navlie.lib.states import VectorState, SE3State
    from navlie.lib.models import RangePoseToAnchor, Altitude

The following state types are currently part of the lib:

- `VectorState`
- `SO2State`
- `SO3State`
- `SE2State`
- `SE3State`
- `SE23State`
- `IMUState` (contains IMU biases as part of the state)
- `CompositeState` (for holding many sub-states as a single state)

The following process models are currently part of the lib:

- `SingleIntegrator`
- `BodyFrameVelocity`
- `RelativeBodyFrameVelocity`
- `CompositeProcessModel`
- `IMUKinematics`

The following measurement models are currently part of the lib:

- `RangePointToAnchor`
- `RangePoseToAnchor`
- `RangePoseToPose`
- `RangeRelativePose`
- `GlobalPosition`
- `Altitude` 
- `Gravitometer`
- and many more

Finally, this repo has the following state estimation algorithms implemented:

- `ExtendedKalmanFilter`
- `IteratedKalmanFilter`
- `UnscentedKalmanFilter`
- `InteractingModelFilter`
- and more


Contributing
------------
If you wish to make some changes, fork this repo, make your changes, and then make a pull request. Here are some conventions that should be followed:

- Code style should follow the PEP8 style guide. https://peps.python.org/pep-0008. We recommend using `black --line-length 80 .` to format the code.
- Everything should be type hinted as much as possible. Essentially, in the VS Code dark theme, you should not have any white text. 

The goal of this project is to write general algorithms that work for any implementation of the abstract `State`, `ProcessModel` and `MeasurementModel`. As such, please give thought to how this could be done to any algorithm you implement. As a rule of thumb, code outside of the `navlie/lib` folder should not depend on any of the classes in `navlie/lib`, although sometimes this rule is broken. 

If you want to discuss anything regarding this repo, feel free to email `charles.c.cossette@gmail.com`.


Contributing to the Documentation
---------------------------------
You must first install the dependencies for the documentation. This can be done by running

.. code-block:: bash

    pip install -r docs/requirements.txt

After this is done, change to the `./docs/` directory and run 

.. code-block:: bash

    make html

after which the documentation will be updated, and viewable by opening the ``docs/index.html`` file in your browser.  In terms of actually writing documentation, we use the numpy format, which can be seen in some of the existing docstrings in the code, and used as a template. 

Alternatively and prefereably, install the `autoDocstring extension for VSCode. <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>` and change the docstring format in the settings to `numpy`.



