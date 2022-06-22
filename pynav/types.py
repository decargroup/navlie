import numpy as np
from typing import List, Any
from copy import deepcopy
from abc import ABC, abstractmethod


class StampedValue:
    """
    Generic data container with timestamped information.
    """

    __slots__ = ["value", "stamp"]

    def __init__(self, value: np.ndarray, stamp: float):
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


class VectorState(State):
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """

    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        super(VectorState, self).__init__(
            value=value.flatten(),
            dof=value.size,
            stamp=stamp,
            state_id=state_id,
        )

    def plus(self, dx: np.ndarray):
        self.value: np.ndarray = self.value.flatten() + dx.flatten()

    def minus(self, x: "VectorState") -> np.ndarray:
        return self.value - x.value

    def copy(self) -> "VectorState":
        return VectorState(self.value.copy(), self.stamp, self.state_id)


class CompositeState(State):
    """
    A "composite" state object intended to hold a list of State objects as a
    single conceptual "state". The intended use is to hold a list of poses
    as a single state at a specific time.

    PARAMETERS:
    -----------
    state_list: list of states that forms this composite state

    Each state in the provided list has an index (the index in the list), as
    well as an id, which is found as an attribute in the corresponding state
    object.

    It is possible to access sub-states in the composite states both by index
    and by ID.
    """

    __slots__ = ["substates", "_slices"]

    def __init__(self, state_list: List[State], stamp: float = None, state_id=None):

        self.value = state_list
        self.dof = sum([x.dof for x in state_list])
        self.stamp = stamp
        self.state_id = state_id

        # Alternate way to access, may be a more appropriate name
        self.substates = self.value

        # Compute the slices for each individual state.
        self._slices = []
        counter = 0
        for state in state_list:
            self._slices.append(slice(counter, counter + state.dof))
            counter += state.dof

    def get_index_by_id(self, state_id):
        """
        Get index of a particular state_id in the list of states.
        """
        return [x.state_id for x in self.value].index(state_id)

    def get_slice_by_id(self, state_id):
        """
        Get slice of a particular state_id in the list of states.
        """
        idx = self.get_index_by_id(state_id)
        return self._slices[idx]  # TODO: recalculate every time?

    def get_value_by_id(self, state_id):
        """
        Get state value by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].value

    def get_dof_by_id(self, state_id):
        """
        Get degrees of freedom of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.dof[idx]

    def get_stamp_by_id(self, state_id):
        """
        Get timestamp of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.stamp[idx]

    def set_stamp_by_id(self, stamp: float, state_id):
        """
        Set the timestamp of a sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].stamp = stamp

    def set_stamp_for_all(self, stamp: float):
        """
        Set the timestamp of all substates.
        """
        for state in self.value:
            state.stamp = stamp

    def to_list(self):
        """
        Converts the CompositeState object back into a list of states.
        """
        return self.value

    def copy(self) -> "CompositeState":
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
        return CompositeState([state.copy() for state in self.value])

    def plus(self, dx, new_stamp: float = None):
        """
        Updates the value of each sub-state given a dx. Interally parses
        the dx vector.
        """
        for i, s in enumerate(self._slices):
            sub_dx = dx[s]
            self.value[i].plus(sub_dx)

        if new_stamp is not None:
            self.set_stamp_for_all(new_stamp)

    def update_by_id(self, dx, state_id: int, new_stamp: float = None):
        """
        Updates a specific sub-state.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].plus(dx)
        if new_stamp is not None:
            self.set_stamp_by_id(new_stamp, state_id)


class MeasurementModel:
    """
    A generic residual, AKA innovation, AKA measurement error.
    """

    def evaluate(self, x: State) -> np.ndarray:
        raise NotImplementedError

    def jacobian(self, x: State) -> np.ndarray:
        raise NotImplementedError

    def covariance(self, x: State) -> np.ndarray:
        raise NotImplementedError

    def check_jacobian(self, x: State):
        """
        Calculates the model jacobian with finite difference, and returns the
        difference between the user-implemented jacobian().
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

        jac = self.jacobian(x)

        return jac - jac_fd


class ProcessModel:
    def evaluate(self, x: State, u: StampedValue, dt: float) -> State:
        raise NotImplementedError

    def jacobian(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        raise NotImplementedError

    def covariance(self, x: State, u: StampedValue, dt: float) -> np.ndarray:
        raise NotImplementedError


class Measurement:
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
