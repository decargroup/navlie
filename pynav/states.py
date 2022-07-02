from pylie import SO2, SO3, SE2, SE3, SE23
from pylie.numpy.base import MatrixLieGroup
import numpy as np
from .types import State
from typing import List 

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



class MatrixLieGroupState(State):
    """
    The MatrixLieGroupState class is a "meta" class (although not actually a
    real python metaclass). Using this group-general meta-class, one can create
    a group-specific `State` class by passing a `pylie.numpy.base.MatrixLieGroup`
    class as a parameter when inheriting.
    """

    __slots__ = ["direction"]

    def __init__(
        self,
        value: np.ndarray,
        group: MatrixLieGroup,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        self.direction = direction
        self.group = group
        super(MatrixLieGroupState, self).__init__(
            value, self.group.dof, stamp, state_id
        )

    def plus(self, dx: np.ndarray):
        if self.direction == "right":
            self.value: np.ndarray = self.value @ self.group.Exp(dx)
        elif self.direction == "left":
            self.value: np.ndarray = self.group.Exp(dx) @ self.value
        else:
            raise ValueError("direction must either be 'left' or 'right'.")

    def minus(self, x: "MatrixLieGroupState") -> np.ndarray:
        if self.direction == "right":
            return self.group.Log(self.group.inverse(x.value) @ self.value)
        elif self.direction == "left":
            return self.group.Log(self.value @ self.group.inverse(x.value)) 
        else:
            raise ValueError("direction must either be 'left' or 'right'.")

    def copy(self) -> "MatrixLieGroupState":
        return self.__class__(
            self.value.copy(),
            self.stamp,
            self.state_id,
            self.direction,
        )

    @property
    def attitude(self) -> np.ndarray:
        raise NotImplementedError("{0} does not have attitude property".format(self.__class__.__name__))

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError("{0} does not have position property".format(self.__class__.__name__))

    @property
    def velocity(self) -> np.ndarray:
        raise NotImplementedError("{0} does not have velocity property".format(self.__class__.__name__))


class SO2State(MatrixLieGroupState):
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, SO2, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value

    @attitude.setter
    def attitude(self, C):
        self.value = C


class SO3State(MatrixLieGroupState):
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, SO3, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value

    @attitude.setter
    def attitude(self, C):
        self.value = C


class SE2State(MatrixLieGroupState):
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, SE2, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value[0:2, 0:2]

    @attitude.setter
    def attitude(self, C):
        self.value[0:2, 0:2] = C

    @property
    def position(self):
        return self.value[0:2, 2]

    @position.setter
    def position(self, r):
        self.value[0:2, 2] = r

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None, position: np.ndarray = None
    ):


        for jac in [attitude, position]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 1))
        if position is None:
            position = np.zeros((dim, 2))

        return np.block([attitude, position])


class SE3State(MatrixLieGroupState):
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, SE3, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self):
        return self.value[0:3, 3]

    @position.setter
    def position(self, r):
        self.value[0:3, 3] = r

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None, position: np.ndarray = None
    ):

        for jac in [attitude, position]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))

        return np.block([attitude, position])


class SE23State(MatrixLieGroupState):
    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, SE23, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self):
        return self.value[0:3, 4]

    @position.setter
    def position(self, r):
        self.value[0:3, 4] = r

    @property
    def velocity(self):
        return self.value[0:3, 3]

    @velocity.setter
    def velocity(self, r):
        self.value[0:3, 3] = r

    @staticmethod
    def jacobian_from_blocks(
        attitude: np.ndarray = None,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
    ):
        for jac in [attitude, position, velocity]:
            if jac is not None:
                dim = jac.shape[0]

        if attitude is None:
            attitude = np.zeros((dim, 3))
        if position is None:
            position = np.zeros((dim, 3))
        if velocity is None:
            velocity = np.zeros((dim, 3))

        return np.block([attitude, velocity, position])
