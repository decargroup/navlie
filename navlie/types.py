"""
This module contains the core primitive types used throughout navlie.
"""

import numpy as np
from typing import List, Any, Tuple
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


class State(ABC):
    """
    An abstract state :math:`\mathcal{X}` is an object containing the following attributes:

    - a value of some sort;
    - a certain number of degrees of freedom (dof);
    - ``plus`` and ``minus`` methods that generalize addition and subtracting to
      to this object.

    Optionally, it is often useful to assign a timestamp (``stamp``) and a label
    (``state_id``) to differentiate state instances from others.

    When implementing a new state type, you should inherit from this class as
    shown in the tutorial.

    .. note::
        The ``plus`` and ``minus`` must correspond to each other, in the sense
        that the following must hold:

        .. math::

            \delta \mathbf{x} = (\mathcal{X} \oplus \delta \mathbf{x}) \ominus \mathcal{X}

        for any state :math:`\mathcal{X}` and any perturbation :math:`\delta \mathbf{x}`.
        In practice this can be tested with something along the lines of:

        .. code-block:: python

            x = State(...) # some state object
            dx = np.random.randn(x.dof)
            dx_test = x.plus(dx).minus(x)
            assert np.allclose(dx, dx_test)

    """

    __slots__ = ["value", "dof", "stamp", "state_id"]

    def __init__(self, value: Any, dof: int, stamp: float = None, state_id=None):
        self.value = value  #:Any: State value
        self.dof = dof  #:int: Degree of freedom of the state
        self.stamp = stamp  #:float: Timestamp
        self.state_id = state_id  #:Any: Some identifier associated with the state

    @abstractmethod
    def plus(self, dx: np.ndarray) -> "State":
        """
        A generic "addition" operation given a ``dx`` numpy array with as many
        elements as the ``dof`` of this state.
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
        Jacobian of the ``plus`` operator. That is, using Lie derivative notation,

        .. math::

            \mathbf{J} = \\frac{D (\mathcal{X} \oplus \delta \mathbf{x})}{D \delta \mathbf{x}}


        For Lie groups, this is known as the *group Jacobian*.
        """
        return self.plus_jacobian_fd(dx)

    def plus_jacobian_fd(self, dx, step_size=1e-8) -> np.ndarray:
        """
        Calculates the plus jacobian with finite difference.
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
        Jacobian of the ``minus`` operator with respect to self.

        .. math::

            \mathbf{J} = \\frac{D (\mathcal{Y} \ominus \mathcal{X})}{D \mathcal{Y}}

        That is, if ``dx = y.minus(x)`` then this is the Jacobian of ``dx`` with respect to ``y``.
        For Lie groups, this is the inverse of the *group Jacobian* evaluated at
        ``dx = x1.minus(x2)``.
        """
        return self.minus_jacobian_fd(x)

    def minus_jacobian_fd(self, x: "State", step_size=1e-8) -> np.ndarray:
        """
        Calculates the minus jacobian with finite difference.
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
            f"{self.__class__.__name__}(stamp={self.stamp}, dof={self.dof}, state_id={self.state_id})",
            f"{value_str}",
        ]
        return "\n".join(s)


class MeasurementModel(ABC):
    """
    Abstract measurement model base class, used to implement measurement models
    of the form

    .. math::

        \mathbf{y} = \mathbf{g}(\mathcal{X}) + \mathbf{v}

    where :math:`\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})`. To
    implement a measurement model, you must inherit from this class and
    implement the ``evaluate`` method, which *must* return a numpy array. You
    must also specify covariance matrix :math:`\mathbf{R}` by implementing the
    ``covariance`` method.

    """

    @abstractmethod
    def evaluate(self, x: State) -> np.ndarray:
        """
        Evaluates the measurement model :math:`\mathbf{g}(\mathcal{X})`.
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
        Evaluates the measurement model Jacobian with respect to the state.

        .. math::

            \mathbf{G} = \\frac{D \mathbf{g}(\mathcal{X})}{D \mathcal{X}}
        """
        return self.jacobian_fd(x)

    def evaluate_with_jacobian(self, x: State) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the measurement model and simultaneously returns the Jacobian
        as its second output argument. This is useful to override for
        performance reasons when the model evaluation and Jacobian have a lot of
        common calculations, and it is more efficient to calculate them in the
        same function call.
        """
        return self.evaluate(x), self.jacobian(x)

    def jacobian_fd(self, x: State, step_size=1e-6):
        """
        Calculates the model jacobian with finite difference.
        """
        N = x.dof
        y = self.evaluate(x)
        m = y.size
        jac_fd = np.zeros((m, N))
        for i in range(N):
            dx = np.zeros((N,))
            dx[i] = step_size
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
        \mathcal{X}_k = f(\mathcal{X}_{k-1}, \mathbf{u}_{k-1}, \Delta t) \oplus \mathbf{w}_{k}

    where :math:`\mathbf{u}_{k-1}` is the input, :math:`\Delta t` is the time period
    between the two states, and :math:`\mathbf{w}_{k} \sim
    \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)` is additive Gaussian noise.

    To define a process model, you must inherit from this class and implement
    the ``evaluate`` method. You must also specify covariance information in one
    of either two ways.

    **1. Specifying the covariance matrix directly:**

    The first way is to specify the :math:`\mathbf{Q}_k` covariance matrix
    directly by overriding the ``covariance`` method. This covariance matrix
    represents the distribution of process model errors directly.

    **2. Specifing the covariance of additive noise on the input:**
    The second way is to specify the covariance of noise that is additive to
    the input. That is, if the process model is of the form

    .. math::

            \mathcal{X}_k = f(\mathcal{X}_{k-1}, \mathbf{u}_{k-1} +
            \mathbf{w}^u_{k-1}, \Delta t)

    where :math:`\mathbf{w}^u_{k-1} \sim \mathcal{N}(\mathbf{0},
    \mathbf{Q}^u_{k-1})`. In this case, you should override the
    ``input_covariance`` method, at which point the covariance of the process
    model error is approximated using a linearization procedure,

    .. math::

        \mathbf{Q}_k = \mathbf{L} \mathbf{Q}^u_{k-1} \mathbf{L}^T

    where :math:`\mathbf{L} = D \mathbf{f}(\mathcal{X}_{k-1}, \mathbf{u}_{k-1}, dt) /
    D \mathbf{u}_{k-1}` is the *input jacobian*. This is calculated using finite
    difference by default, but can be overridden by implementing the
    ``input_jacobian`` method.
    """

    @abstractmethod
    def evaluate(self, x: State, u: Input, dt: float) -> State:
        """
        Implementation of :math:`{f}(\mathcal{X}_{k-1}, \mathbf{u}, \Delta t)`.

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

    def covariance(self, x: State, u: Input, dt: float) -> np.ndarray:
        """
        Covariance matrix :math:`\mathbf{Q}_k` of the additive Gaussian
        noise :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`.
        If this method is not overridden, the covariance of the process model
        error is approximated from the input covariance using a linearization
        procedure, with the input Jacobian evaluated using finite difference.

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
        L = np.atleast_2d(self.input_jacobian_fd(x, u, dt))
        Q = np.atleast_2d(self.input_covariance(x, u, dt))
        return L @ Q @ L.T

    def jacobian(self, x: State, u: Input, dt: float) -> np.ndarray:
        """
        Implementation of the process model Jacobian with respect to the state.

        .. math::
            \mathbf{F} = \\frac{D {f}(\mathcal{X}_{k-1}, \mathbf{u}, \Delta t)}{D \mathcal{X}_{k-1}}


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

    def evaluate_with_jacobian(
        self, x: State, u: Input, dt: float
    ) -> (State, np.ndarray):
        """
        Evaluates the process model and simultaneously returns the Jacobian as
        its second output argument. This is useful to override for
        performance reasons when the model evaluation and Jacobian have a lot of
        common calculations, and it is more efficient to calculate them in the
        same function call.
        """
        return self.evaluate(x, u, dt), self.jacobian(x, u, dt)

    def jacobian_fd(
        self, x: State, u: Input, dt: float, step_size=1e-6, *args, **kwargs
    ) -> np.ndarray:
        """
        Calculates the model jacobian with finite difference.
        """
        Y_bar = self.evaluate(x.copy(), u, dt, *args, **kwargs)
        jac_fd = np.zeros((x.dof, x.dof))
        for i in range(x.dof):
            dx = np.zeros((x.dof))
            dx[i] = step_size
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

    def input_covariance(self, x: State, u: Input, dt: float) -> np.ndarray:
        """
        Covariance matrix of additive noise *on the input*.

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
            Covariance matrix :math:`\mathbf{R}_k`.
        """
        raise NotImplementedError(
            "input_covariance must be implemented "
            + "if the covariance method is not overridden."
        )


class Measurement:
    """
    A simple data container containing a generic measurement's value, timestamp,
    and corresponding model stored as a ``MeasurementModel`` object. This
    container can be used as-is without inheritance.
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
        and thus, the ``minus`` operator is simply vector subtraction.
        """
        return self.value.reshape((-1, 1)) - y_check.reshape((-1, 1))


class StateWithCovariance:
    """
    A data container containing a ``State`` object and a covariance array.
    This class can be used as-is without inheritance.
    """

    __slots__ = ["state", "covariance"]

    def __init__(self, state: State, covariance: np.ndarray):
        """

        Parameters
        ----------
        state : State
            A state object, usually representing the mean of a distribution.
        covariance : np.ndarray
            A square, symmetric covariance matrix associated with the state.

        Raises
        ------
        ValueError
            If the covariance matrix is not square.
        ValueError
            If the covariance matrix does not correspond with the state degrees
            of freedom.
        """
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be an n x n array.")

        if covariance.shape[0] != state.dof:
            raise ValueError("Covariance matrix does not correspond with state DOF.")

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
    def get_measurement_data(self) -> List[Measurement]:
        """Returns a list of measurements."""
        pass
