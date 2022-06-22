from pylie import SO2, SO3, SE2, SE3, SE23
from pylie.numpy.base import MatrixLieGroup
import numpy as np
from .types import State


class MatrixLieGroupState(State):

    __slots__ = ["direction"]

    def __init_subclass__(cls, group: MatrixLieGroup) -> None:
        """
        This function is called whenever a class inherits from this class.
        """
        super().__init_subclass__()
        cls._group = group

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        """
        This function is called whenever a subclass is instantiated.
        """
        self.direction = direction
        super(MatrixLieGroupState, self).__init__(
            value, self._group.dof, stamp, state_id
        )

    def plus(self, dx: np.ndarray):
        if self.direction == "right":
            self.value = self.value @ self._group.Exp(dx)
        elif self.direction == "left":
            self.value = self._group.Exp(dx) @ self.value
        else:
            raise ValueError("direction must either be 'left' or 'right'.")

    def copy(self) -> "MatrixLieGroupState":
        return self.__class__(
            self.value.copy(), self.dof, self.stamp, self.state_id, self.direction
        )


class SO2State(MatrixLieGroupState, group=SO2):
    pass


class SO3State(MatrixLieGroupState, group=SO3):
    pass


class SE2State(MatrixLieGroupState, group=SE2):
    pass


class SE3State(MatrixLieGroupState, group=SE3):
    pass


class SE23State(MatrixLieGroupState, group=SE23):
    pass
