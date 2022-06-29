from pylie import SO2, SO3, SE2, SE3, SE23
from pylie.numpy.base import MatrixLieGroup
import numpy as np
from .types import State


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
            return self.group.Log(x.value @ self.group.inverse(self.value))
        elif self.direction == "left":
            return self.group.Log(self.group.inverse(self.value) @ x.value) 
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
