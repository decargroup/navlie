"""
This module contains the core primitive types used throughout navlie.
"""

import numpy as np
from typing import List, Any
from abc import ABC, abstractmethod


class Input(ABC):
    __slots__ = ["stamp", "dof", "covariance", "state_id"]
    """
    An abstract data container that holds a process model input value.
    """

    def __init__(
        self,
        dof: int,
        stamp: float = None,
        state_id: Any = None,
        covariance: np.ndarray = None,
    ):
        self.stamp = stamp  #:float: Timestamp
        self.dof = dof  #:int: Degrees of freedom of the object

        #:Any: Arbitrary optional identifier, possible to "assign" to a state.
        self.state_id = state_id

        #:np.ndarray: Covariance matrix of the object. Has shape (dof, dof)
        self.covariance = covariance

    @abstractmethod
    def plus(self, w: np.ndarray) -> "Input":
        """
        Generic addition operation to modify the internal value of the input,
        and return a new modified object.
        """
        pass

    @abstractmethod
    def copy(self) -> "Input":
        """ 
        Creates a deep copy of the object.
        """
        pass


class StampedValue(Input):
    """
    Generic data container for timestamped information.
    """

    __slots__ = ["value"]

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id: Any = None,
        covariance=None,
    ):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        #:numpy.ndarray:  Variable containing the data values
        self.value = value

        super().__init__(
            dof=value.size,
            stamp=stamp,
            state_id=state_id,
            covariance=covariance,
        )

    def plus(self, w: np.ndarray) -> "StampedValue":
        """
        Generic addition operation to modify the internal value.

        Parameters
        ----------
        w : np.ndarray
            to be added to the instance's .value
        """
        new = self.copy()
        og_shape = new.value.shape
        new.value = new.value.ravel() + w.ravel()
        new.value = new.value.reshape(og_shape)
        return new

    def copy(self) -> "StampedValue":
        """
        Returns a copy of the instance with fully seperate memory.
        """
        return StampedValue(
            self.value.copy(), self.stamp, self.state_id, self.covariance
        )

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, state_id={self.state_id})",
            f"{value_str}",
        ]
        return "\n".join(s)


class State(ABC):
    """
    An abstract state :math:`\mathbf{x}` is an object containing the following attributes:

        - a value of some sort;
        - a certain number of degrees of freedom (dof);
        - an update rule that modified the state value given an update vector
          `dx` containing `dof` elements.

    Optionally, it is often useful to assign a timestamp (`stamp`) and a label
    (`state_id`) to differentiate state instances from others.
    """

    __slots__ = ["value", "dof", "stamp", "state_id"]

    def __init__(
        self, value: Any, dof: int, stamp: float = None, state_id=None
    ):
        self.value = value  #:Any: State value
        self.dof = dof  #:int: Degree of freedom of the state
        self.stamp = stamp  #:float: Timestamp
        self.state_id = (
            state_id  #:Any: Some identifier associated with the state
        )

    @abstractmethod
    def plus(self, dx: np.ndarray) -> "State":
        """
        A generic "addition" operation given a `dx` numpy array with as many
        elements as the `dof` of this state.
        """
        pass

    @abstractmethod
    def minus(self, x: "State") -> np.ndarray:
        """
        A generic "subtraction" operation given another State object of the same
        type, always returning a numpy array.
        """
        pass

    @abstractmethod
    def copy(self) -> "State":
        """
        Returns a copy of this State instance.
        """
        pass

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        """
        Jacobian of the `plus` operator. For Lie groups, this is known as the
        *group Jacobian*.
        """
        return self.plus_jacobian_fd(dx)

    def plus_jacobian_fd(self, dx, step_size=1e-8) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        dx_bar = dx
        jac_fd = np.zeros((self.dof, self.dof))
        Y_bar = self.plus(dx_bar)
        for i in range(self.dof):
            dx = np.zeros((self.dof,))
            dx[i] = step_size
            Y: State = self.plus(dx_bar.ravel() + dx)
            jac_fd[:, i] = Y.minus(Y_bar).flatten() / step_size

        return jac_fd

    def minus_jacobian(self, x: "State") -> np.ndarray:
        """
        Jacobian of the `minus` operator with respect to self. That is, if

            y = x1.minus(x2)

        then this is the Jacobian of `y` with respect to `x1`.
        For Lie groups, this is the inverse of the *group Jacobian* evaluated at
        `dx = x1.minus(x2)`.
        """
        return self.minus_jacobian_fd(x)

    def minus_jacobian_fd(self, x: "State", step_size=1e-8) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        x_bar = x
        jac_fd = np.zeros((self.dof, self.dof))

        y_bar = self.minus(x_bar)
        for i in range(self.dof):
            dx = np.zeros((self.dof,))
            dx[i] = step_size
            y: State = self.plus(dx).minus(x_bar)
            jac_fd[:, i] = (y - y_bar).flatten() / step_size

        return jac_fd

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp}, state_id={self.state_id})",
            f"{value_str}",
        ]
        return "\n".join(s)


class MeasurementModel(ABC):
    """
    Abstract measurement model base class, used to implement measurement models
    of the form

    .. math::
        \mathbf{y} = \mathbf{g}(\mathbf{x}) + \mathbf{v}

    where :math:`\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})`.

    """

    @abstractmethod
    def evaluate(self, x: State) -> np.ndarray:
        """
        Evaluates the measurement model :math:`\mathbf{g}(\mathbf{x})`.
        """
        pass

    @abstractmethod
    def covariance(self, x: State) -> np.ndarray:
        """
        Returns the covariance :math:`\mathbf{R}` associated with additive Gaussian noise.
        """
        pass

    def jacobian(self, x: State) -> np.ndarray:
        """
        Evaluates the measurement model Jacobian
        :math:`\mathbf{G} = \partial \mathbf{g}(\mathbf{x})/ \partial \mathbf{x}`.
        """
        return self.jacobian_fd(x)

    def jacobian_fd(self, x: State, step_size=1e-6):
        """
        Calculates the model jacobian with finite difference.
        """
        N = x.dof
        y = self.evaluate(x)
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N, 1))
            dx[i, 0] = step_size
            x_temp = x.plus(dx)
            jac_fd[:, i] = (self.evaluate(x_temp) - y).flatten() / step_size

        return jac_fd

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def sqrt_information(self, x: State):
        R = np.atleast_2d(self.covariance(x))
        return np.linalg.cholesky(np.linalg.inv(R))


class ProcessModel(ABC):
    """
    Abstract process model base class for process models of the form

    .. math::
        \mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t) + \mathbf{w}_{k}

    where :math:`\mathbf{u}` is the input, :math:`\Delta t` is the time
    period between the two states, and :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`
    is additive Gaussian noise.
    """

    @abstractmethod
    def evaluate(self, x: State, u: Input, dt: float) -> State:
        """
        Implementation of :math:`\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t)`.

        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : Input
            The input value :math:`\mathbf{u}` provided as a Input object.
            The actual numerical value is accessed via `u.value`.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        State
            State at time :math:`k`.
        """
        pass

    @abstractmethod
    def covariance(self, x: State, u: Input, dt: float) -> np.ndarray:
        """
        Covariance matrix math:`\mathbf{Q}_k` of the additive Gaussian
        noise :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`.

        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : Input
            The input value :math:`\mathbf{u}` provided as a Input object.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        np.ndarray
            Covariance matrix :math:`\mathbf{Q}_k`.
        """
        pass

    
    def jacobian(self, x: State, u: Input, dt: float) -> np.ndarray:
        """
        Implementation of the process model Jacobian with respect to the state.

        .. math::
            \mathbf{F} = \partial \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t)
            / \partial \mathbf{x}_{k-1}


        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : Input
            The input value :math:`\mathbf{u}` provided as a Input object.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        np.ndarray
            Process model Jacobian with respect to the state :math:`\mathbf{F}`.
        """
        return self.jacobian_fd(x, u, dt)

    def jacobian_fd(
        self, x: State, u: Input, dt: float, step_size=1e-6, *args, **kwargs
    ) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        Y_bar = self.evaluate(x.copy(), u, dt, *args, **kwargs)
        jac_fd = np.zeros((x.dof, x.dof))
        for i in range(x.dof):
            dx = np.zeros((x.dof, 1))
            dx[i, 0] = step_size
            x_pert = x.plus(dx)
            Y: State = self.evaluate(x_pert, u, dt, *args, **kwargs)
            jac_fd[:, i] = Y.minus(Y_bar).flatten() / step_size

        return jac_fd

    def input_jacobian_fd(
        self, x: State, u: Input, dt: float, step_size=1e-6, *args, **kwargs
    ) -> np.ndarray:
        """
        Calculates the input jacobian with finite difference.
        """
        Y_bar = self.evaluate(x.copy(), u.copy(), dt, *args, **kwargs)
        jac_fd = np.zeros((x.dof, u.dof))
        for i in range(u.dof):
            du = np.zeros((u.dof,))
            du[i] = step_size
            Y: State = self.evaluate(x.copy(), u.plus(du), dt, *args, **kwargs)
            jac_fd[:, i] = Y.minus(Y_bar).flatten() / step_size

        return jac_fd

    def __repr__(self):
        return f"{self.__class__.__name__} at {hex(id(self))}"

    def sqrt_information(self, x: State, u: Input, dt: float) -> np.ndarray:
        Q = np.atleast_2d(self.covariance(x, u, dt))
        return np.linalg.cholesky(np.linalg.inv(Q))


class Measurement:
    """
    A data container containing a generic measurement's value, timestamp,
    and corresponding model.
    """

    __slots__ = ["value", "stamp", "model", "state_id"]

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        model: MeasurementModel = None,
        state_id: Any = None,
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            the value of the measurement reading
        stamp : float, optional
            timestamp, by default None
        model : MeasurementModel, optional
            model for this measurement, by default None
        state_id : Any, optional
            optional state ID, by default None
        """
        #:numpy.ndarray: Container for the measurement value
        self.value = np.array(value) if np.isscalar(value) else value
        #:float: Timestamp
        self.stamp = stamp
        #:navlie.types.MeasurementModel: measurement model associated with this measurement.
        self.model = model
        #:Any: Optional, ID of the state this measurement is associated.
        self.state_id = state_id

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])

        s = [
            f"Measurement(stamp={self.stamp}, state_id={self.state_id})"
            + f" of {self.model}",
            value_str,
        ]
        return "\n".join(s)

    def minus(self, y_check: np.ndarray) -> np.ndarray:
        """Evaluates the difference between the current measurement
        and a predicted measurement.

        By default, assumes that the measurement is a column vector,
        and thus, the `minus` operator is simply vector subtraction.
        """

        return self.value.reshape((-1, 1)) - y_check.reshape((-1, 1))


class StateWithCovariance:
    """
    A data container containing a State object and a covariance array.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state: State, covariance: np.ndarray):
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")

        if covariance.shape[0] != state.dof:
            raise ValueError(
                "Covariance matrix does not correspond with state DOF."
            )

        #:navlie.types.State: state object
        self.state = state

        #:numpy.ndarray: covariance associated with state
        self.covariance = covariance

    @property
    def stamp(self):
        return self.state.stamp

    @stamp.setter
    def stamp(self, stamp):
        self.state.stamp = stamp

    def symmetrize(self):
        """
        Enforces symmetry of the covariance matrix.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy())

    def __repr__(self):
        return f"StateWithCovariance(stamp={self.stamp})"


class Dataset(ABC):
    """A container to store a dataset.

    Contains abstract methods to get the groundtruth data,
    the input data, and measurement data.
    """

    @abstractmethod
    def get_ground_truth(self) -> List[State]:
        """Returns a list of groundtruth states."""
        pass

    @abstractmethod
    def get_input_data(self) -> List[Input]:
        """Retruns a list of inputs."""
        pass

    @abstractmethod
    def get_meas_data(self) -> List[Measurement]:
        """Returns a list of measurements."""
        pass
