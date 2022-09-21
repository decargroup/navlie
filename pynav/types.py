"""
This module contains the core primitive types used throughout pynav.

"""

import numpy as np
from typing import List, Any
from abc import ABC, abstractmethod


class StampedValue:
    """
    Generic data container for timestamped information.
    """

    __slots__ = ["value", "stamp"]
    # TODO: we should probably let variable by Any, such as a Tuple 
    # this would require also adding an optional `plus()` method 
    def __init__(self, value: np.ndarray, stamp: float = 0.0):
        self.value = value #:numpy.ndarray:  Variable containing the data values
        self.stamp = stamp #:float: Timestamp

    def plus(self, w: np.ndarray):
        """
        Generic addition operation to modify the internal value.
        
        Parameters
        ----------
        w : np.ndarray
            to be added to the instance's .value
        """
        og_shape = self.value.shape 
        self.value = self.value.ravel() + w.ravel()
        self.value.reshape(og_shape)

    def copy(self) -> "StampedValue":
        """ 
        Returns a copy of the instance with fully seperate memory.
        """
        return StampedValue(self.value.copy(), self.stamp)
        

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

    def __init__(self, value: Any, dof: int, stamp: float = None, state_id=None):
        self.value = value #:Any: State value
        self.dof = dof #:int: Degree of freedom of the state
        self.stamp = stamp #:float: Timestamp 
        self.state_id = state_id #:Any: Some identifier associated with the state 

    @abstractmethod
    def plus(self, dx: np.ndarray):
        """
        A generic "addition" operation given a `dx` numpy array with as many
        elements as the `dof` of this state.
        """
        # TODO: investigate the cost of returning a copy instead. This would 
        # be much closer to the __add__ method in python and how numpy does things.
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


    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of the `plus` operator. For Lie groups, this is known as the
        *group Jacobian*.
        """
        return np.identity(self.dof)

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
    def jacobian(self, x: State) -> np.ndarray:
        """ 
        Evaluates the measurement model Jacobian 
        :math:`\mathbf{G} = \partial \mathbf{g}(\mathbf{x})/ \partial \mathbf{x}`.
        """
        pass

    @abstractmethod
    def covariance(self, x: State) -> np.ndarray:
        """
        Returns the covariance :math:`\mathbf{R}` associated with additive Gaussian noise.
        """
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
    """
    Abstract process model base class for process models of the form 

    .. math::
        \mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t) + \mathbf{w}_{k}

    where :math:`\mathbf{u}` is the input, :math:`\Delta t` is the time 
    period between the two states, and :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`
    is additive Gaussian noise.
    """
    @abstractmethod
    def evaluate(self, x: State, u: StampedValue, dt: float) -> State:
        """
        Implementation of :math:`\mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t)`.

        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : StampedValue
            The input value :math:`\mathbf{u}` provided as a StampedValue object.
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
    def jacobian(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        """
        Implementation of the process model Jacobian with respect to the state.

        .. math::
            \mathbf{F} = \partial \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}, \Delta t)
            / \partial \mathbf{x}_{k-1}


        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : StampedValue
            The input value :math:`\mathbf{u}` provided as a StampedValue object.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        np.ndarray
            Process model Jacobian with respect to the state :math:`\mathbf{F}`.
        """
        pass

    @abstractmethod
    def covariance(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        """
        Covariance matrix math:`\mathbf{Q}_k` of the additive Gaussian 
        noise :math:`\mathbf{w}_{k} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k)`.

        Parameters
        ----------
        x : State
            State at time :math:`k-1`.
        u : StampedValue
            The input value :math:`\mathbf{u}` provided as a StampedValue object.
        dt : float
            The time interval :math:`\Delta t` between the two states.

        Returns
        -------
        np.ndarray
            Covariance matrix :math:`\mathbf{Q}_k`.
        """
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
        #:numpy.ndarray: Container for the measurement value
        self.value = np.array(value) if np.isscalar(value) else value
        #:float: Timestamp
        self.stamp = stamp
        #:pynav.types.MeasurementModel: measurement model associated with this measurement.
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

        #:pynav.types.State: state object
        self.state = state

        #:numpy.ndarray: covariance associated with state
        self.covariance = covariance

    def symmetrize(self):
        """
        Enforces symmetry of the covariance matrix.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def copy(self) -> "StateWithCovariance":
        return StateWithCovariance(self.state.copy(), self.covariance.copy()) 
