# pynav
A very modular, reusable state estimation library for robotics.

The core idea behind this project is to abstract-away the state definition such that a single estimator implementation can operate on a variety of state manifolds, such as the usual vector space, and any Lie group. 

## Setup
### Dependencies
- python3.6+
- `pylie` Clone and install [this repo](https://github.com/decarsg/pylie) by following the README.
- All other dependencies should get installed automatically by `pip`

### Installation

Clone this repo, change to its directory, and execute 

    pip install -e .

### Examples
Some starting examples running EKFs can be found in the `examples/` folder. Simply run these as python3 scripts 

## The Core Concept - Defining a `State` Implementation

The algorithms and models in this repo are centered around the abstract `State` class. An instance of `State` is an object containing, at a minimum, the following attributes:

- `value`: a value of some sort;
- `dof`: the degrees of freedom associated with the state.

It will also contain the following mandatory methods that must be implemented by the user.
- `plus()`:  A generic "addition" operation given a `dx` vector with as many
        elements as the `dof` of this state.
- `minus()`:  A generic "subtraction" operation given another State object of the same type, which returns a numpy array of error values.
- `copy()`: A method that returns a new object of the same type, and with the same attibute values.

Optionally, it is often useful to assign a timestamp (`stamp`) and a label (`state_id`) to differentiate state instances from others. The snippet below shows how to define a simple vector-space state:  


```python 
from pynav.types import State 
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
        self.value = self.value + dx

    def minus(self, x: "VectorState") -> np.ndarray:
        return self.value - x.value

    def copy(self) -> "VectorState":
        return VectorState(self.value.copy(), self.stamp, self.state_id)

```

As another more complicated example, a state object belonging to the SE(3) Lie group can be implemented as 

```python
from pynav.types import State 
from pylie import SE3 
import numpy as np 

class SE3State(State):
    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        super(SE3State, self).__init__(
            value=value,
            dof=3,
            stamp=stamp,
            state_id=state_id,
        )
    
    def plus(self, dx: np.ndarray):
        self.value = self.value @ SE3.Exp(dx)

    def minus(self, x: "VectorState") -> np.ndarray:
        return SE3.Log(SE3.inverse(x.value) @ self.value)

    def copy(self) -> "VectorState":
        return SE3State(self.value.copy(), self.stamp, self.state_id)

```

## Process and Measurement Models
There are two more core types in this package, and they are the `ProcessModel` and `MeasurementModel` classes. Both of these are abstract classes requiring the user to implement

- an `evaluate()` method, 
- a `jacobian()` method,
- and a `covariance()` method.

For example, a simple "single integrator" (velocity input) model can be implemented as follows:

```python
class SingleIntegrator(ProcessModel):
    """
    The single-integrator process model is a process model of the form

        x_k = x_{k-1} + dt * u_{k-1}
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(self, x: VectorState, u: StampedValue, dt: float) -> np.ndarray:
        """
        Returns a state with an updated value according to a process model.
        """
        x.value = x.value + dt * u.value
        return x

    def jacobian(self, x: VectorState, u: StampedValue, dt: float) -> np.ndarray:
        """
        Jacobian of the process model with respect to the state.
        """
        return np.identity(x.dof)

    def covariance(self, x: VectorState, u: StampedValue, dt: float) -> np.ndarray:
        """
        Returns the covariance of the process model errors.
        """
        return dt**2 * self._Q
```

Similarly, a single distance-to-landmark measurement model can be implemented as 

```python 
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
```

In fact, for both `ProcessModel` and `MeasurementModel`, subclasses will inherit a finite-difference numerical differentiation method `jacobian_fd()`, that allows for a seamless way to check your `jacobian()` implementation! (`evaluate()` method must be implemented for this to work, see some of the files in `tests/` for an example of this.)
