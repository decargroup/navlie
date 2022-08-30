from pylie import SO2, SO3, SE2, SE3, SE23
from pylie.numpy.base import MatrixLieGroup
import numpy as np
from ..types import State
from typing import List


class VectorState(State):
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """

    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        value = np.array(value).ravel()
        super(VectorState, self).__init__(
            value=value,
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


class MatrixLieGroupState(State):
    """
    The MatrixLieGroupState class.
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
        raise NotImplementedError(
            "{0} does not have attitude property".format(self.__class__.__name__)
        )

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have position property".format(self.__class__.__name__)
        )

    @property
    def velocity(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have velocity property".format(self.__class__.__name__)
        )

    def jacobian_from_blocks(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()


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

    @staticmethod
    def jacobian_from_blocks(attitude: np.ndarray):

        return attitude


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
    def jacobian_from_blocks(attitude: np.ndarray = None, position: np.ndarray = None):

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
    def jacobian_from_blocks(attitude: np.ndarray = None, position: np.ndarray = None):

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
        r = r.flatten()
        self.value[0:3, 4] = r

    @property
    def velocity(self):
        return self.value[0:3, 3]

    @velocity.setter
    def velocity(self, v):
        v = v.flatten()
        self.value[0:3, 3] = v

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


class IMUState(State):
    def __init__(
        self,
        nav_state: np.ndarray,
        bg: np.ndarray,
        ba: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        """Instantiate and IMUState object.

        Parameters
        ----------
        nav_state : np.ndarray
            The navigation state stored as an element of SE_2(3).
            Contains orientation, velocity, and position.
        bg : np.ndarray
            Gyroscope bias, [3 x 1] matrix
        ba : np.ndarray
            Accelerometer bias, [3 x 1] matrix
        stamp : float, optional
            Timestamp, by default None
        state_id : _type_, optional
            Unique identifier, by default None
        direction : str, optional
            Direction of the perturbation for the nav state, by default "right"
        """
        nav_state = SE23State(nav_state, stamp, state_id, direction)
        bg = VectorState(bg, stamp, state_id)
        ba = VectorState(ba, stamp, state_id)

        # Gravity resolved in the inertial frame
        self.gravity: np.ndarray = np.array([0, 0, -9.81]).reshape((-1, 1))

        self.value: List[State] = [nav_state, bg, ba]
        self.state_id = state_id
        self.stamp = stamp
        self.dof = 15
        self.direction = direction

        self._slices = []
        counter = 0
        for state in self.value:
            self._slices.append(slice(counter, counter + state.dof))
            counter += state.dof

    @property
    def attitude(self):
        return self.value[0].attitude

    @attitude.setter
    def attitude(self, C):
        self.value[0].attitude = C

    @property
    def velocity(self):
        return self.value[0].velocity

    @velocity.setter
    def velocity(self, v):
        self.value[0].velocity = v

    @property
    def position(self):
        return self.value[0].position

    @position.setter
    def position(self, r):
        self.value[0].position = r

    @property
    def bg(self) -> np.ndarray:
        return self.value[1].value

    @bg.setter
    def bg(self, gyro_bias):
        self.value[1].value = gyro_bias

    @property
    def ba(self) -> np.ndarray:
        return self.value[2].value

    @ba.setter
    def ba(self, accel_bias):
        self.value[2].value = accel_bias

    @property
    def nav_state(self) -> np.ndarray:
        return self.value[0].value

    def plus(self, dx: np.ndarray, new_stamp: float = None):
        """
        Updates the value of each of the IMU state, given a perturbation dx.
        """
        if dx.shape[0] != 15:
            raise ValueError("Perturbation must be dimension 15!")

        for i, s in enumerate(self._slices):
            sub_dx = dx[s]
            self.value[i].plus(sub_dx)

        if new_stamp is not None:
            self.set_stamp_for_all(new_stamp)

    def minus(self, x: "IMUState") -> np.ndarray:
        dx = []
        for i, v in enumerate(x.value):
            dx.append(self.value[i].minus(x.value[i]))

        return np.vstack(dx)

    def copy(self):
        """
        Returns a new composite state object where the state values have also
        been copied.
        """
        return IMUState(
            self.nav_state.copy(),
            self.bg.copy(),
            self.ba.copy(),
            self.stamp,
            self.state_id,
        )


class CompositeState(State):
    """
    A "composite" state object intended to hold a list of State objects as a
    single conceptual "state". The intended use is to hold a list of poses
    as a single state at a specific time.

    Parameters
    ----------
    state_list: List[State]
        List of State that forms this composite state


    Each state in the provided list has an index (the index in the list), as
    well as a state_id, which is found as an attribute in the corresponding State
    object.

    It is possible to access sub-states in the composite states both by index
    and by ID.
    """

    __slots__ = ["substates", "_slices"]

    def __init__(self, state_list: List[State], stamp: float = None, state_id=None):

        #:List[State]: The substates are the CompositeState 's value.
        self.value = state_list

        self.stamp = stamp
        self.state_id = state_id

        # Compute the slices for each individual state.
        self._slices = []
        counter = 0
        for state in state_list:
            self._slices.append(slice(counter, counter + state.dof))
            counter += state.dof

    @property
    def dof(self):
        return sum([x.dof for x in self.value])

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
        return self._slices[idx]

    def get_value_by_id(self, state_id):
        """
        Get state value by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].value

    def get_state_by_id(self, state_id):
        """
        Get state object by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx]

    def get_dof_by_id(self, state_id):
        """
        Get degrees of freedom of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].dof[idx]

    def get_stamp_by_id(self, state_id):
        """
        Get timestamp of sub-state by id.
        """
        idx = self.get_index_by_id(state_id)
        return self.value[idx].stamp[idx]

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
        return CompositeState(
            [state.copy() for state in self.value], self.stamp, self.state_id
        )

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

    def minus(self, x: "CompositeState") -> np.ndarray:
        dx = []
        for i, v in enumerate(x.value):
            dx.append(self.value[i].minus(x.value[i]))

        return np.vstack(dx)

    def plus_by_id(self, dx, state_id: int, new_stamp: float = None):
        """
        Updates a specific sub-state.
        """
        idx = self.get_index_by_id(state_id)
        self.value[idx].plus(dx)
        if new_stamp is not None:
            self.set_stamp_by_id(new_stamp, state_id)

    def jacobian_from_blocks(self, block_dict: dict):
        """
        Returns the jacobian of the entire composite state given jacobians
        associated with some of the substates. These are provided as a dictionary
        with the the keys being the substate IDs.
        """
        block: np.ndarray = block_dict.values()[0]
        m = block.shape[0]  # Dimension of "y" value
        jac = np.zeros((m, self.dof))
        for state_id, block in block_dict.items():
            slc = self.get_slice_by_id(state_id)
            jac[:, slc] = block

        return jac
