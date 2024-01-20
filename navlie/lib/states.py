from pymlg import SO2, SO3, SE2, SE3, SE23, SL3
from pymlg.numpy.base import MatrixLieGroup
import numpy as np
from navlie.types import State, Input
from typing import Any

try:
    # We do not want to make ROS a hard dependency, so we import it only if
    # available.
    from geometry_msgs.msg import PoseStamped, QuaternionStamped
    import rospy
except ImportError:
    pass  # ROS is not installed
except:
    raise


from navlie.composite import CompositeState  # For backwards compatibility


class VectorState(State):
    """
    A standard vector-based state, with value represented by a 1D numpy array.
    """

    def __init__(self, value: np.ndarray, stamp: float = None, state_id=None):
        value = np.array(value, dtype=np.float64).ravel()
        super(VectorState, self).__init__(
            value=value,
            dof=value.size,
            stamp=stamp,
            state_id=state_id,
        )
        self.value: np.ndarray = self.value  # just for type hinting

    def plus(self, dx: np.ndarray) -> "VectorState":
        new = self.copy()
        if dx.size == self.dof:
            new.value: np.ndarray = new.value.ravel() + dx.ravel()
            return new
        else:
            raise ValueError("Array of mismatched size added to VectorState.")

    def minus(self, x: "VectorState") -> np.ndarray:
        og_shape = self.value.shape
        return (self.value.ravel() - x.value.ravel()).reshape(og_shape)

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        return np.identity(self.dof)

    def minus_jacobian(self, x: State) -> np.ndarray:
        return np.identity(self.dof)

    def copy(self) -> "VectorState":
        return VectorState(self.value.copy(), self.stamp, self.state_id)


class MatrixLieGroupState(State):
    """
    The MatrixLieGroupState class. Although this class can technically be used
    directly, it is recommended to use one of the subclasses instead, such as
    ``SE2State`` or ``SO3State``.
    """

    # TODO. Add identity() and random() functions to this
    __slots__ = ["group", "direction"]

    def __init__(
        self,
        value: np.ndarray,
        group: MatrixLieGroup,
        stamp: float = None,
        state_id: Any = None,
        direction="right",
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            Value of of the state. If the value has as many elements as the
            DOF of the group, then it is assumed to be a vector of exponential
            coordinates. Otherwise, the value must be a 2D numpy array representing
            a direct element of the group in matrix form.
        group : MatrixLieGroup
            A `pymlg.MatrixLieGroup` class, such as `pymlg.SE2` or `pymlg.SO3`.
        stamp : float, optional
            timestamp, by default None
        state_id : Any, optional
            optional state ID, by default None
        direction : str, optional
            either "left" or "right", by default "right". Defines the perturbation
            :math:`\\delta \mathbf{x}` as either

            .. math::
                \mathbf{X} = \mathbf{X} \exp(\delta \mathbf{x}^\wedge) \text{ (right)}

                \mathbf{X} = \exp(\delta \mathbf{x}^\wedge) \mathbf{X} \text{ (left)}
        """
        if isinstance(value, list):
            value = np.array(value)

        if value.size == group.dof:
            value = group.Exp(value)
        elif value.shape[0] != value.shape[1]:
            raise ValueError(
                f"value must either be a {group.dof}-length vector of exponential"
                "coordinates or a matrix direct element of the group."
            )

        self.direction = direction
        self.group = group
        super(MatrixLieGroupState, self).__init__(
            value, self.group.dof, stamp, state_id
        )
        self.value: np.ndarray = self.value  # just for type hinting

    def plus(self, dx: np.ndarray) -> "MatrixLieGroupState":
        new = self.copy()
        if self.direction == "right":
            new.value = self.value @ self.group.Exp(dx)
        elif self.direction == "left":
            new.value = self.group.Exp(dx) @ self.value
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return new

    def minus(self, x: "MatrixLieGroupState") -> np.ndarray:
        if self.direction == "right":
            diff = self.group.Log(self.group.inverse(x.value) @ self.value)
        elif self.direction == "left":
            diff = self.group.Log(self.value @ self.group.inverse(x.value))
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return diff.ravel()

    def copy(self) -> "MatrixLieGroupState":
        ## Check if instance of this class as opposed to a child class
        if type(self) == MatrixLieGroupState:
            return MatrixLieGroupState(
                self.value.copy(),
                self.group,
                self.stamp,
                self.state_id,
                self.direction,
            )
        else:
            return self.__class__(
                self.value.copy(),
                self.stamp,
                self.state_id,
                self.direction,
            )

    def plus_jacobian(self, dx: np.ndarray) -> np.ndarray:
        if self.direction == "right":
            jac = self.group.right_jacobian(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    def minus_jacobian(self, x: "MatrixLieGroupState") -> np.ndarray:
        dx = self.minus(x)
        if self.direction == "right":
            jac = self.group.right_jacobian_inv(dx)
        elif self.direction == "left":
            jac = self.group.left_jacobian_inv(dx)
        else:
            raise ValueError("direction must either be 'left' or 'right'.")
        return jac

    def __repr__(self):
        value_str = str(self.value).split("\n")
        value_str = "\n".join(["    " + s for s in value_str])
        s = [
            f"{self.__class__.__name__}(stamp={self.stamp},"
            + f" state_id={self.state_id}, direction={self.direction})",
            f"{value_str}",
        ]
        return "\n".join(s)

    def dot(self, other: "MatrixLieGroupState") -> "MatrixLieGroupState":
        new = self.copy()
        new.value = self.value @ other.value
        return new

    @property
    def attitude(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have attitude property".format(
                self.__class__.__name__
            )
        )

    @property
    def position(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have position property".format(
                self.__class__.__name__
            )
        )

    @property
    def velocity(self) -> np.ndarray:
        raise NotImplementedError(
            "{0} does not have velocity property".format(
                self.__class__.__name__
            )
        )

    def jacobian_from_blocks(self, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class SO2State(MatrixLieGroupState):
    """
    A state object for rotations in 2D. The value of this state is stored as a
    2x2 numpy array representing a direct element of the SO2 group. I.e.,

    .. math::
        \mathbf{C} \in \mathbb{R}^{2 \times 2}, \quad
        \mathbf{C}^T \mathbf{C} = \mathbf{I}, \quad \det(\mathbf{C}) = \mathbf{1}

    """

    group = SO2

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        # check if value is a single number
        if isinstance(value, (int, float)):
            value = np.array(value).reshape((1, 1))
        super().__init__(value, self.group, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value

    @attitude.setter
    def attitude(self, C):
        self.value = C

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SO2State(
            SO2.random(), stamp=stamp, state_id=state_id, direction=direction
        )


class SO3State(MatrixLieGroupState):
    """
    A state object for rotations in 3D. The value of this state is stored as a
    3x3 numpy array representing a direct element of the SO3 group. I.e.,

    .. math::

        \mathbf{C} \in \mathbb{R}^{3 \times 3}, \quad
        \mathbf{C}^T \mathbf{C} = \mathbf{I}, \quad \det(\mathbf{C}) = \mathbf{1}

    """

    group = SO3

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)

    @property
    def attitude(self):
        return self.value

    @attitude.setter
    def attitude(self, C):
        self.value = C

    @staticmethod
    def jacobian_from_blocks(attitude: np.ndarray):
        return attitude

    @staticmethod
    def from_ros(
        msg: "QuaternionStamped", state_id=None, direction="right"
    ) -> "SO3State":
        """
        Create a SO3State from a ROS QuaternionStamped message.

        Parameters
        ----------
        msg : QuaternionStamped
            ROS quaternion
        state_id : Any, optional
            If not provided, the frame_id of the message will be used
        direction : str, optional
            perturbation direction, by default "right"

        Returns
        -------
        SO3State
            a new instance of SO3State
        """
        if state_id is None:
            state_id = msg.header.frame_id

        return SO3State(
            SO3.from_ros(msg.quaternion),
            msg.header.stamp.to_sec(),
            state_id,
            direction,
        )

    def to_ros(self) -> "QuaternionStamped":
        """
        Convert to ROS QuaternionStamped message.

        Returns
        -------
        QuaternionStamped
            ROS quaternion
        """
        msg = QuaternionStamped()
        msg.header.stamp = rospy.Time.from_sec(self.stamp)
        msg.header.frame_id = self.state_id
        msg.quaternion = SO3.to_ros(self.attitude)
        return msg

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SO3State(
            SO3.random(), stamp=stamp, state_id=state_id, direction=direction
        )


class SE2State(MatrixLieGroupState):
    """ 
    A state object for 2D rigid body transformations. The value of this state
    is stored as a 3x3 numpy array representing a direct element of the SE2
    group. I.e.,
    
    .. math::

        \mathbf{T} = \begin{bmatrix}
            \mathbf{C} & \mathbf{r} \\
            \mathbf{0} & 1
        \end{bmatrix}, \quad \mathbf{C} \in SO(2), \quad \mathbf{r} \in \mathbb{R}^2.


    """

    group = SE2

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:2, 0:2]

    @attitude.setter
    def attitude(self, C):
        self.value[0:2, 0:2] = C

    @property
    def position(self) -> np.ndarray:
        return self.value[0:2, 2]

    @position.setter
    def position(self, r):
        self.value[0:2, 2] = r

    @property
    def pose(self) -> np.ndarray:
        return self.value

    @pose.setter
    def pose(self, T):
        self.value = T

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

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SE2State(
            SE2.random(), stamp=stamp, state_id=state_id, direction=direction
        )


class SE3State(MatrixLieGroupState):
    """
    A state object for 3D rigid body transformations. The value of this state
    is stored as a 4x4 numpy array representing a direct element of the SE3
    group. I.e.,

    .. math::
    
            \mathbf{T} = \begin{bmatrix}
                \mathbf{C} & \mathbf{r} \\
                \mathbf{0} & 1
            \end{bmatrix}, \quad \mathbf{C} \in SO(3), \quad \mathbf{r} \in \mathbb{R}^3.
    
    
    """

    group = SE3

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self) -> np.ndarray:
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

    @staticmethod
    def from_ros(
        msg: "PoseStamped", state_id: Any = None, direction="right"
    ) -> "SE3State":
        """
        Convert a ROS PoseStamped message to a SE3State.

        Parameters
        ----------
        msg : PoseStamped
            ROS PoseStamped message
        state_id : Any, optional
            If not provided, the frame_id of the message will be used
        direction : str, optional
            perturbation direction, by default "right"

        Returns
        -------
        SE3State
            a new instance of SE3State
        """
        C = SO3.from_ros(msg.pose.orientation)
        r = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ]
        )
        if state_id is None:
            state_id = msg.header.frame_id

        return SE3State(
            SE3.from_components(C, r),
            msg.header.stamp.to_sec(),
            state_id=state_id,
            direction=direction,
        )

    def to_ros(self, frame_id: str = None) -> "PoseStamped":
        """
        Convert a SE3State to a ROS PoseStamped message.

        Parameters
        ----------
        frame_id : str, optional
            If not provided, the state_id will be used.

        Returns
        -------
        PoseStamped
            ROS PoseStamped message
        """
        if frame_id is None:
            frame_id = str(self.state_id)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.from_sec(self.stamp)
        if frame_id is not None:
            msg.header.frame_id = frame_id

        msg.pose = SE3.to_ros(self.value)

        return msg

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SE2State(
            SE2.random(), stamp=stamp, state_id=state_id, direction=direction
        )


class SE23State(MatrixLieGroupState):
    """
    A state object for 2D rigid body transformations with velocity. The value
    of this state is stored as a 5x5 numpy array representing a direct element
    of the SE23 group. I.e.,

    .. math::

        \mathbf{T} = \begin{bmatrix}
            \mathbf{C} & \mathbf{v} & \mathbf{r} \\
            \mathbf{0} & 1 & 0 \\
            \mathbf{0} & 0 & 1
        \end{bmatrix}, \quad \mathbf{C} \in SO(3), \quad 
        \mathbf{v}, \mathbf{r} \in \mathbb{R}^3.
    """

    group = SE23

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)

    @property
    def pose(self) -> np.ndarray:
        return self.value[0:5, 0:5]

    @pose.setter
    def pose(self, T):
        self.value[0:5, 0:5] = T

    @property
    def attitude(self) -> np.ndarray:
        return self.value[0:3, 0:3]

    @attitude.setter
    def attitude(self, C):
        self.value[0:3, 0:3] = C

    @property
    def position(self) -> np.ndarray:
        return self.value[0:3, 4]

    @position.setter
    def position(self, r):
        self.value[0:3, 4] = r.ravel()

    @property
    def velocity(self) -> np.ndarray:
        return self.value[0:3, 3]

    @velocity.setter
    def velocity(self, v) -> np.ndarray:
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

    @staticmethod
    def random(stamp: float = None, state_id=None, direction="right"):
        return SE23State(
            SE23.random(), stamp=stamp, state_id=state_id, direction=direction
        )


class SL3State(MatrixLieGroupState):
    """
    A state object representing the special linear group in 3D. The value of
    this state is stored as a 3x3 numpy array representing a direct element of
    the SL3 group. I.e.,

    .. math::

            \mathbf{C} \in SL(3), \quad \mathbf{C} \in \mathbb{R}^{3 \times 3}, \quad
            \det(\mathbf{C}) = \mathbf{1}.
    """

    group = SL3

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id=None,
        direction="right",
    ):
        super().__init__(value, self.group, stamp, state_id, direction)


class VectorInput(Input):
    """
    A standard vector-based input, with value represented by a 1D numpy array.
    """

    __slots__ = ["value"]

    def __init__(
        self,
        value: np.ndarray,
        stamp: float = None,
        state_id: Any = None,
        covariance=None,
    ):
        """
        Parameters
        ----------
        value : np.ndarray
            Value of of the input.
        stamp : float, optional
            timestamp, by default None
        state_id : Any, optional
            optional container for identifying information, can be useful to
            associate an input with a state. This has no functionality unless
            the user uses it in a process model, or something else. By default
            None.
        covariance : np.ndarray, optional
            covariance matrix describing additive noise on the input, by default
            None
        """
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

    def plus(self, w: np.ndarray) -> "VectorInput":
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

    def copy(self) -> "VectorInput":
        """
        Returns a copy of the instance with fully seperate memory.
        """
        return VectorInput(
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


class StampedValue(VectorInput):
    pass  # For backwards compatibility
