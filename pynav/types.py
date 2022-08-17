import numpy as np
from typing import List, Any
from abc import ABC, abstractmethod


class StampedValue:
    """
    Generic data container with timestamped information.
    """

    __slots__ = ["value", "stamp"]

    def __init__(self, value: np.ndarray, stamp: float = 0.0):
        self.value = value
        self.stamp = stamp


class State(ABC):
    """
    An abstract state is an object containing the following attributes:
        - a value of some sort;
        - a certain number of degrees of freedom (dof);
        - an update rule that modified the state value given an update vector
          `dx` containing `dof` elements.

    Optionally, it is often useful to assign a timestamp (`stamp`) and a label
    (`state_id`) to differentiate state instances from others.
    """

    __slots__ = ["value", "dof", "stamp", "state_id"]

    def __init__(self, value: Any, dof: int, stamp: float = None, state_id=None):
        self.value = value
        self.stamp = stamp
        self.dof = dof
        self.state_id = state_id

    @abstractmethod
    def plus(self, dx: np.ndarray):
        """
        A generic "addition" operation given a `dx` vector with as many
        elements as the dof of this state.
        """
        pass

    @abstractmethod
    def minus(self, x: "State") -> np.ndarray:
        """
        A generic "subtraction" operation given another State object of the same
        type.
        """
        pass

    @abstractmethod
    def copy(self) -> "State":
        """
        Returns a copy of this State instance.
        """
        pass


class MeasurementModel(ABC):
    """
    An abstract measurement model base class.
    """

    @abstractmethod
    def evaluate(self, x: State) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, x: State) -> np.ndarray:
        pass

    @abstractmethod
    def covariance(self, x: State) -> np.ndarray:
        pass

    def jacobian_fd(self, x: State):
        """
        Calculates the model jacobian with finite difference.
        """
        N = x.dof
        y = self.evaluate(x)
        m = y.size
        jac_fd = np.zeros((m, N))
        h = 1e-6
        for i in range(N):
            dx = np.zeros((N, 1))
            dx[i, 0] = h
            x_temp = x.copy()
            x_temp.plus(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / h

        return jac_fd


class ProcessModel(ABC):
    @abstractmethod
    def evaluate(self, x: State, u: StampedValue, dt: float) -> State:
        pass

    @abstractmethod
    def jacobian(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        pass

    @abstractmethod
    def covariance(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        pass

    def jacobian_fd(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        Y_bar = self.evaluate(x.copy(), u, dt)
        jac_fd = np.zeros((x.dof, x.dof))
        h = 1e-6
        for i in range(x.dof):
            dx = np.zeros((x.dof, 1))
            dx[i, 0] = h
            x_pert = x.copy()  # Perturb current state
            x_pert.plus(dx)
            Y: State = self.evaluate(x_pert, u, dt)
            jac_fd[:, i] = Y.minus(Y_bar).flatten() / h

        return jac_fd


class Measurement:
    """
    A data container containing a generic measurement's value, timestamp,
    and corresponding model.
    """

    __slots__ = ["value", "stamp", "model"]

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        model: MeasurementModel = None,
    ):
        self.value = value
        self.stamp = stamp
        self.model = model


class StateWithCovariance:
    """
    A data container containing a State object and a covariance array.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state: State, covariance: np.ndarray):

        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")

        if covariance.shape[0] != state.dof:
            raise ValueError("Covariance matrix does not correspond with state DOF.")

        self.state = state
        self.covariance = covariance

    def symmetrize(self):
        """
        Enforces symmetry of the covariance matrix.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy()) 
