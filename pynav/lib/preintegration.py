from abc import ABC, abstractmethod
from typing import List, Any, Callable
from pynav.types import StampedValue, ProcessModel, Input
from pynav.lib.imu import (
    IMU,
    IMUState,
    U_matrix,
    adjoint_IE3,
    inverse_IE3,
    L_matrix,
    G_matrix,
)
from pynav.lib.states import MatrixLieGroupState, VectorState
import numpy as np
from pylie import SO3, SE3, SE2, SE23, MatrixLieGroup


class RelativeMotionIncrement(Input):
    __slots__ = ["value", "stamps", "covariance", "state_id"]

    def __init__(self, dof: int):
        self.dof = dof

        #:Any: the value of the RMI
        self.value = None

        #:List[float, float]: the two timestamps i, j associated with the RMI
        self.stamps = [None, None]

        #:np.ndarray: the covariance matrix of the RMI
        self.covariance = None

        #:Any: an ID associated with the RMI
        self.state_id = None

    @property
    def stamp(self):
        return self.stamps[1]

    @abstractmethod
    def increment(self, u, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        pass

    @abstractmethod
    def new(self) -> "RelativeMotionIncrement":
        pass

    def increment_many(
        self, u_list: List[Input], end_stamp: float = None
    ) -> "RelativeMotionIncrement":
        """
        Creates an RMI object directly from a list of input measurements.
        The two timestamps i, j are inferred from the first and last measurements
        in the list, unless the optional `end_stamp` argument is provided.
        In this case, the ending timestamp is set to `end_stamp` and the
        last input in the list is used to increment the RMI from `u_list[-1].stamp`
        to `end_stamp`.

        Parameters
        ----------
        u_list : List[StampedValue]
            List of input measurements to construct RMI from
        end_stamp : float, optional
            final timestamp of RMI, by default None

        Returns
        -------
        RelativeMotionIncrement
            The RMI
        """

        for k in range(len(u_list) - 1):
            dt = u_list[k + 1].stamp - u_list[k].stamp
            self.increment(u_list[k], dt)

        if end_stamp is not None:
            dt = end_stamp - u_list[-1].stamp
            self.increment(u_list[-1], dt)

        return self

    def symmetrize(self):
        """
        Symmetrize the covariance matrix of the RMI.
        """
        self.covariance = 0.5 * (self.covariance + self.covariance.T)


class IMUIncrement(RelativeMotionIncrement):
    __slots__ = [
        "value",
        "original_value",
        "original_bias",
        "stamps",
        "covariance",
        "input_covariance",
        "bias_jacobian",
        "gravity",
        "state_id",
        "_estimating_bias",
    ]

    def __init__(
        self,
        input_covariance: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        state_id: Any = None,
        gravity=None,
    ):
        """
        Initializes an "identity" IMU RMI.

        Parameters
        ----------
        input_covariance : np.ndarray with shape (12, 12) or (6, 6)
            If a 6 x 6 array is provided, this is the covariance of gyro, accel
            measurements. If a 12 x 12 array is provided, this is the covariance
            of gyro, accel, gyro bias random walk, and accel bias random walk.
        """
        # TODO: add tests for when (6,6) input covariance is given
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])

        if input_covariance.shape[0] == 12:
            super().__init__(dof=15)
            self._estimating_bias = True
            self.covariance = np.zeros((15, 15))
        elif input_covariance.shape[0] == 6:
            super().__init__(dof=9)
            self._estimating_bias = False
            self.covariance = np.zeros((9, 9))
        else:
            raise ValueError("Input covariance must be 12x12 or 6x6")

        self.original_value = np.identity(5)
        self.original_bias = np.hstack(
            [np.array(gyro_bias).ravel(), np.array(accel_bias).ravel()]
        )
        self.value = self.original_value
        self.stamps = [None, None]
        self.input_covariance = input_covariance
        self.bias_jacobian = np.zeros((9, 6))
        self.gravity = np.array(gravity).ravel()
        self.state_id = state_id

    @property
    def gyro_bias(self):
        return self.original_bias[:3]

    @property
    def accel_bias(self):
        return self.original_bias[3:]

    def increment(self, u: IMU, dt: float):
        u = u.copy()
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_gyro = u.gyro - self.gyro_bias
        unbiased_accel = u.accel - self.accel_bias

        U = U_matrix(unbiased_gyro, unbiased_accel, dt)
        self.original_value = self.original_value @ U

        U_inv = inverse_IE3(U)
        A = adjoint_IE3(U_inv)
        L = L_matrix(unbiased_gyro, unbiased_accel, dt)
        Q = self.input_covariance

        A_full = np.zeros((15, 15))
        A_full[0:9, 0:9] = A
        A_full[0:9, 9:15] = -L
        A_full[9:15, 9:15] = np.identity(6)

        L_full = np.zeros((15, 12))
        L_full[0:9, 0:6] = L
        L_full[9:15, 6:12] = dt * np.identity(6)

        if not self._estimating_bias:
            A_full = A_full[0:9, 9:9]
            L_full = L_full[0:9, 0:6]

        self.covariance = (
            A_full @ self.covariance @ A_full.T + L_full @ Q @ L_full.T
        )
        self.bias_jacobian = A @ self.bias_jacobian - L
        self.value = self.original_value
        self.symmetrize()

    def bias_update(
        self, new_gyro_bias: np.ndarray, new_accel_bias: np.ndarray
    ):
        """
        Updates the RMI given new bias values

        Parameters
        ----------
        new_gyro_bias: np.ndarray with size 3
            new gyro bias value
        new_accel_bias: np.ndarray with size 3
            new accel bias value
        """
        # TODO. this needs some tests.
        # TODO. bias jacobian also needs some tests.
        # bias change
        new_bias = np.vstack(
            [
                new_gyro_bias.reshape((-1, 1)),
                new_accel_bias.reshape((-1, 1)),
            ]
        )

        db = new_bias - self.original_bias.reshape((-1, 1))
        self.value = self.original_value @ SE23.Exp(self.bias_jacobian @ db)

    def plus(self, w: np.ndarray) -> "IMUIncrement":
        """
        Adds noise to the RMI

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        RelativeMotionIncrement
            The updated RMI
        """
        new = self.copy()
        new.value = new.value @ SE23.Exp(w[0:9])
        return new

    def copy(self) -> "IMUIncrement":
        """
        Returns
        -------
        IMUIncrement
            A copy of the RMI
        """
        new = IMUIncrement(
            self.input_covariance,
            self.gyro_bias,
            self.accel_bias,
            self.state_id,
            self.gravity,
        )
        new.value = self.value.copy()
        new.original_value = self.original_value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

    def new(self, gyro_bias=None, accel_bias=None) -> "IMUIncrement":
        """
        Parameters
        ----------
        gyro_bias : np.ndarray
            The new gyro bias to use in this RMI
        accel_bias : np.ndarray
            The new accel bias to use in this RMI

        Returns
        -------
        IMUIncrement
            A copy of the RMI with reinitialized values
        """
        if gyro_bias is None:
            gyro_bias = self.gyro_bias

        if accel_bias is None:
            accel_bias = self.accel_bias

        new = IMUIncrement(
            self.input_covariance,
            gyro_bias,
            accel_bias,
            self.state_id,
            self.gravity,
        )
        return new


class PreintegratedIMUKinematics(ProcessModel):
    def __init__(self, gravity=None):
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])
        self.gravity = gravity

    def evaluate(self, x: IMUState, rmi: IMUIncrement, dt=None) -> IMUState:
        x = x.copy()
        rmi = rmi.copy()
        rmi.bias_update(x.bias_gyro, x.bias_accel)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        x.pose = DG @ x.pose @ DU
        return x

    def jacobian(self, x: IMUState, rmi: IMUIncrement, dt=None) -> np.ndarray:
        x = x.copy()
        rmi = rmi.copy()
        rmi.bias_update(x.bias_gyro, x.bias_accel)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        A = np.identity(15)
        if x.direction == "right":
            A[0:9, 0:9] = adjoint_IE3(inverse_IE3(DU))
            A[0:9, 9:15] = rmi.bias_jacobian
        elif x.direction == "left":
            Ad = SE23.adjoint(DG @ x.pose @ DU)
            A[0:9, 0:9] = adjoint_IE3(DG)
            A[0:9, 9:15] = Ad @ rmi.bias_jacobian
        return A

    def covariance(self, x: IMUState, rmi: IMUIncrement, dt=None) -> np.ndarray:
        x = x.copy()
        rmi = rmi.copy()
        rmi.bias_update(x.bias_gyro, x.bias_accel)
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value

        if x.direction == "right":
            L = np.identity(15)
        elif x.direction == "left":
            L = np.identity(15)
            Ad = SE23.adjoint(DG @ x.pose @ DU)
            L[0:9, 0:9] = Ad
        return L @ rmi.covariance @ L.T


class BodyVelocityIncrement(RelativeMotionIncrement):
    """
    This is a general preintegration class for any process model of the form

    .. math::
        \mathbf{T}_{k} = \mathbf{T}_{k-1} \exp(\Delta t \mathbf{u}_{k-1}^\wedge).

    Preintegration is trivially done with

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{U}_{ij},

    where :math:`\Delta \mathbf{U}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{U}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{u}_{k}^\wedge).
    """

    def __init__(
        self,
        group: MatrixLieGroup,
        Q: np.ndarray,
        bias: np.ndarray = None,
        state_id=None,
    ):
        super().__init__(group.dof)
        self.bias = bias
        self.group = group
        self.covariance = np.zeros((group.dof, group.dof))
        self.input_covariance = Q
        self.value: np.ndarray = group.identity()
        self.bias_jacobian = np.zeros((group.dof, group.dof))
        self.stamps = [None, None]
        self.state_id = state_id
        if bias is None:
            self.bias = np.zeros((group.dof))

    def increment(self, u: StampedValue, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_velocity = u.value - self.bias

        # Increment the value
        U = self.group.Exp(unbiased_velocity * dt)
        self.value = self.value @ U

        # Increment the covariance
        A = self.group.adjoint(self.group.inverse(U))
        L = dt * self.group.left_jacobian(-unbiased_velocity * dt)
        self.covariance = (
            A @ self.covariance @ A.T + L @ self.input_covariance @ L.T
        )

        # Increment the bias jacobian
        self.bias_jacobian = A @ self.bias_jacobian + L
        self.symmetrize()

    def bias_update(self, new_bias: np.ndarray):
        """
        Internally updates the RMI given a new bias.

        Parameters
        ----------
        new_bias : np.ndarray
            New bias values
        """
        db = new_bias - self.bias

        self.value = self.value @ self.group.Exp(self.bias_jacobian @ db)
        self.bias = new_bias

    def plus(self, w: np.ndarray):
        """
        Adds noise to the RMI

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        RelativeMotionIncrement
            The updated RMI
        """
        new = self.copy()
        new.value = new.value @ new.group.Exp(w)
        if w.size > new.group.dof:
            new_bias = new.bias + w[new.group.dof :]
            new.bias_update(new_bias)

        return new

    def copy(self):
        """
        Returns
        -------
        RelativeMotionIncrement
            A copy of the RMI
        """
        new = self.new()
        new.value = self.value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

    def new(self):
        """
        Returns
        -------
        RelativeMotionIncrement
            A copy of the RMI
        """
        new = self.__class__(self.group, self.input_covariance, self.bias)
        return new


class PreintegratedBodyVelocity(ProcessModel):
    """
    Process model that performs prediction of the state given an RMI
    :math:`\Delta \mathbf{U}_{ij}` according to the equation

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{U}_{ij}.

    The covariance is also propagated accordingly
    """

    def __init__(self):
        pass

    def evaluate(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt=None
    ) -> MatrixLieGroupState:
        x = x.copy()
        x.value = x.value @ rmi.value
        return x

    def jacobian(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt=None
    ) -> np.ndarray:
        dof = x.dof
        if rmi.input_covariance.shape[0] == dof:
            estimating_bias = False
            total_dof = x.dof
        elif rmi.input_covariance.shape[0] == 2 * dof:
            estimating_bias = True
            total_dof = 2 * x.dof
        else:
            raise ValueError("Input covariance has incorrect shape")

        A = np.identity(total_dof)

        if x.direction == "right":
            A[0:dof, 0:dof] = x.group.adjoint(x.group.inverse(rmi.value))
            if estimating_bias:
                A[0:dof, dof:total_dof] = rmi.bias_jacobian

        elif x.direction == "left":
            Ad = x.group.adjoint(rmi.value)
            if estimating_bias:
                A[0:dof, dof:total_dof] = Ad @ rmi.bias_jacobian

        return A

    def covariance(
        self, x: MatrixLieGroupState, rmi: BodyVelocityIncrement, dt: float
    ) -> np.ndarray:
        dof = x.dof
        if rmi.input_covariance.shape[0] == dof:
            total_dof = x.dof
        elif rmi.input_covariance.shape[0] == 2 * dof:
            total_dof = 2 * x.dof
        else:
            raise ValueError("Input covariance has incorrect shape")

        if x.direction == "right":
            L = np.identity(total_dof)
        elif x.direction == "left":
            L = np.identity(total_dof)
            Ad = x.group.adjoint(x.value @ rmi.value)
            L[0:dof, 0:dof] = Ad

        return L @ rmi.covariance @ L.T


class AngularVelocityIncrement(BodyVelocityIncrement):
    """
    This is a preintegration class for angular velocity measurements, on only
    attitude. Give a rotation matrix :math:`\mathbf{C}_{k}`, the preintegrated
    process model is of the form

    .. math::
        \mathbf{C}_{j} = \mathbf{C}_{i} \Delta \mathbf{\Omega}_{ij},

    where :math:`\Delta \mathbf{\Omega}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{\Omega}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{\omega}_{k}^\wedge)

    and :math:`\mathbf{\omega}_{k}` is the angular velocity measurement at time :math:`k`.

    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id=None):
        super().__init__(SO3, Q, bias, state_id=state_id)


class WheelOdometryIncrement(BodyVelocityIncrement):
    """
    This is a preintegration class for wheel odometry measurements on SE(n).
    The preintegrated process model is of the form

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{V}_{ij},

    where :math:`\Delta \mathbf{V}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{V}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{\\varpi}_{k}^\wedge)

    and :math:`\mathbf{\\varpi}_{k} = [\mathbf{\omega}_k^T \;
    \mathbf{v}_k^T]^T` is the angular and translational velocity of the
    robot at time :math:`k`.
    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id=None):
        if Q.shape == (6, 6):
            group = SE3
        elif Q.shape == (3, 3):
            group = SE2
        else:
            raise ValueError("Input covariance has incorrect shape")

        super().__init__(group, Q, bias, state_id=state_id)


# Alternate names for classes
class PreintegratedAngularVelocity(PreintegratedBodyVelocity):
    pass


class PreintegratedWheelOdometry(PreintegratedBodyVelocity):
    pass


class LinearIncrement(RelativeMotionIncrement):
    """
    For any process model of the form

    .. math::
        \mathbf{x}_{k} = \mathbf{A}_{k-1} \mathbf{x}_{k-1}
        + \mathbf{B}_{k-1} \mathbf{u}_{k-1}

    This class will compute the preintegrated quantities
    :math:`\mathbf{A}_{ij}, \Delta \\bar{\mathbf{u}}_{ij}, \mathbf{Q}_{ij}` where

    .. math::
        \mathbf{x}_k = \mathbf{A}_{ij} \mathbf{x}_i + \Delta \mathbf{u}_{ij}

    and :math:`\mathbf{Q}_{ij}` where :math:`\Delta \mathbf{u}_{ij} \sim \mathcal{N}(\Delta \\bar{\mathbf{u}}_{ij}, \mathbf{Q}_{ij})`.

    """

    def __init__(
        self,
        input_covariance: np.ndarray,
        state_matrix: Callable[[StampedValue, float], np.ndarray],
        input_matrix: Callable[[StampedValue, float], np.ndarray],
        dof: int,
        bias: np.ndarray = None,
        state_id: Any=None,
    ):
        """

        Parameters
        ----------
        input_covariance : np.ndarray
            Covariance associated with the input. If a bias is also supplied,
            then this should also contain the covariance of the bias random walk.
        state_matrix : Callable[[StampedValue, float], np.ndarray]
            The state transition matrix, supplied as a function of the input
            and time interval `dt`.
        input_matrix : Callable[[StampedValue, float], np.ndarray]
            The input matrix, supplied as a function of the input and time
            interval `dt`.
        dof : int
            the dof of the state
        bias : np.ndarray, optional
            If provided, this bias will be subtracted from the input values. 
            Furthermore, the covariance associated with this RMI will also 
            be augmented to include a bias state.
        state_id : Any, optional
            Optional container for other identifying information, by default None.
        """
        super().__init__(dof=dof) 
        self.input_covariance = input_covariance
        self.state_matrix = state_matrix
        self.input_matrix = input_matrix
        self.state_id = state_id

        #:List[np.ndarray, np.ndarray]: List with elements :math:`(\mathbf{A}_{ij}, \Delta \mathbf{u}_{ij})`
        self.value = [np.identity(dof), np.zeros((dof, 1))]  # [A_ij, Du_ij]

        if bias is not None:
            bias = np.array(bias).ravel()
            self.covariance = np.zeros((dof + bias.size, dof + bias.size))
            self.bias_jacobian = np.zeros((dof, bias.size))
        else:
            self.covariance = np.zeros((dof, dof))
            self.bias_jacobian = None

        self.bias = bias

    def increment(self, u: StampedValue, dt: float):
        u = u.copy()
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        # Remove bias is supplied
        if self.bias is not None:
            u.value = u.value.ravel() - self.bias.ravel()

        # Get the state and input matrices, increment the RMI value
        A = self.state_matrix(u, dt)
        B = self.input_matrix(u, dt)
        self.value[0] = A @ self.value[0]
        self.value[1] = A @ self.value[1].reshape(
            (-1, 1)
        ) + B @ u.value.reshape((-1, 1))
        
        if self.bias is not None:
            A_full = np.zeros((self.dof + self.bias.size, self.dof + self.bias.size))
            A_full[:self.dof, :self.dof] = A
            A_full[self.dof:, self.dof:] = np.identity(self.bias.size)
            A_full[:self.dof, self.dof:] = -B 

            B_full = np.zeros((self.dof + self.bias.size, 2*self.bias.size))
            B_full[:self.dof, :self.bias.size] = B
            B_full[self.dof:, self.bias.size:] = dt * np.identity(self.bias.size)
        else: 
            A_full = A
            B_full = B

        self.covariance = (
            A_full @ self.covariance @ A_full.T + B_full @ self.input_covariance @ B_full.T
        )

        if self.bias_jacobian is None:
            self.bias_jacobian = np.zeros((self.dof, u.dof))

        self.bias_jacobian = A @ self.bias_jacobian - B

    def plus(self, w: np.ndarray) -> "LinearIncrement":
        """
        Increment the RMI itself

        Parameters
        ----------
        w : np.ndarray
            The noise to add

        Returns
        -------
        LinearIncrement
            The updated RMI
        """
        new = self.copy()
        new.value[1] = new.value[1] + w
        return new

    def copy(self) -> "LinearIncrement":
        new = self.new()
        new.value = [self.value[0].copy(), self.value[1].copy()]
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

    def new(self) -> "LinearIncrement":
        """
        Returns
        -------
        LinearIncrement
            A copy of the RMI with reinitialized values
        """
        new = self.__class__(
            self.input_covariance,
            self.state_matrix,
            self.input_matrix,
            self.dof,
            self.bias,
            self.state_id,
        )
        return new


class PreintegratedLinearModel(ProcessModel):
    """
    Process model that applies a preintegrated `LinearIncrement` to predict
    a state forward in time using the equation 

    .. math::
        \mathbf{x}_k = \mathbf{A}_{ij} \mathbf{x}_i + \Delta \mathbf{u}_{ij}
    """
    def __init__(self):
        pass

    def evaluate(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> VectorState:

        x = x.copy()
        A_ij = rmi.value[0]
        Du_ij = rmi.value[1]

        if rmi.bias is not None:
            x_i = x.value[0:-rmi.bias.size].reshape((-1,1))
        else:
            x_i = x.value.reshape((-1,1))

        x_j = A_ij @ x_i + Du_ij

        if rmi.bias is not None:
            x_j = np.vstack((x_j, x.value[-rmi.bias.size:].reshape((-1,1))))

        x.value = x_j.ravel()
        return x

    def jacobian(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> np.ndarray:

        if rmi.bias is not None:
            A_ij = rmi.value[0]
            B_ij = rmi.bias_jacobian
            state_dof = x.dof - rmi.bias.size
            A = np.zeros((x.dof, x.dof))
            A[:state_dof, :state_dof] = A_ij
            A[state_dof:, state_dof:] = np.identity(rmi.bias.size)
            A[:state_dof, state_dof:] = B_ij
        else:
            A = rmi.value[0]

        return A 

    def covariance(
        self, x: VectorState, rmi: LinearIncrement, dt=None
    ) -> np.ndarray:
        return rmi.covariance
