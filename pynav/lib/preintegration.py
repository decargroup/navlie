from abc import ABC, abstractmethod
from typing import List, Any
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
from pynav.lib.states import MatrixLieGroupState
import numpy as np
from pylie import SO3, SE3, SE2, SE23, MatrixLieGroup


class RelativeMotionIncrement(Input):
    __slots__ = ["value", "stamps", "covariance"]

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

    @abstractmethod
    def increment(self, u, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        pass


class IMUIncrement(RelativeMotionIncrement):
    def __init__(
        self,
        input_covariance: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        gravity=None,
        state_id: Any = None
    ):
        """
        Initializes an "identity" IMU RMI.

        Parameters
        ----------
        input_covariance : np.ndarray with shape (12, 12)
            covariance of gyro, accel measurements
        """
        # TODO: allow the case where biases are not being estimated. 
        # in this case the covariance would 9 x 9 with 9 dof
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])

        super().__init__(dof=15)

        self.value = np.identity(5)
        self.stamps = [None, None]
        self.covariance = np.zeros((15, 15))
        self.input_covariance = input_covariance
        self.bias_jacobian = np.zeros((9, 6))
        self.accel_bias = np.array(accel_bias).ravel()
        self.gyro_bias = np.array(gyro_bias).ravel()
        self.gravity = np.array(gravity).ravel()
        self.state_id = state_id

    def increment(self, u: IMU, dt: float):
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_gyro = u.gyro - self.gyro_bias
        unbiased_accel = u.accel - self.accel_bias

        U = U_matrix(unbiased_gyro, unbiased_accel, dt)
        self.value = self.value @ U

        U_inv = inverse_IE3(U)
        A = adjoint_IE3(U_inv)
        L = L_matrix(unbiased_gyro, unbiased_accel, dt)
        Q = self.covariance

        A_full = np.zeros((15, 15))
        A_full[0:9, 0:9] = A
        A_full[0:9, 9:15] = L
        A_full[9:15, 9:15] = np.identity(6)

        L_full = np.zeros((15, 12))
        L_full[0:9, 0:6] = L
        L_full[9:15, 6:12] = dt * np.identity(6)

        self.covariance = (
            A_full @ Q @ A_full.T + L_full @ self.input_covariance @ L_full.T
        )
        self.bias_jacobian = A @ self.bias_jacobian + L

    def bias_update(
        self, new_gyro_bias: np.ndarray, new_accel_bias: np.ndarray
    ):
        """
        Updates the RMI given a small bias change

        Parameters
        ----------
        db : np.ndarray with size 6
            change to bias
        """
        db = np.vstack(
            [
                (new_gyro_bias - self.gyro_bias).reshape((-1, 1)),
                (new_accel_bias - self.accel_bias).reshape((-1, 1)),
            ]
        )

        self.value = self.value @ SE23.Exp(self.bias_jacobian @ db)
        self.gyro_bias = new_gyro_bias
        self.accel_bias = new_accel_bias

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
        new.value = new.value @ SE23.Exp(w[0:9])
        if w.size > 9:
            new.bias_update(w[9:12], w[12:15])
        return new

    def copy(self):
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
            self.gravity,
        )
        new.value = self.value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

class PreintegratedIMUKinematics(ProcessModel):
    def __init__(self, gravity=None):
        if gravity is None:
            gravity = np.array([0, 0, -9.80665])
        self.gravity = gravity

    def evaluate(self, x: IMUState, rmi: IMUIncrement, dt=None) -> IMUState:
        x = x.copy()
        dt = rmi.stamps[1] - rmi.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = rmi.value
        x.pose = DG @ x.pose @ DU
        return x

    def jacobian(self, x: IMUState, rmi: IMUIncrement, dt=None) -> np.ndarray:
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
        \mathbf{T}_{k} = \mathbf{T}_{k-1} \exp(\Delta t \mbf{u}_{k-1}^\wedge).

    Preintegration is trivially done with

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{U}_{ij},

    where :math:`\Delta \mathbf{U}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{U}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mbf{u}_{k}^\wedge).
    """

    def __init__(
        self, group: MatrixLieGroup, Q: np.ndarray, bias: np.ndarray = None, state_id = None
    ):
        super().__init__(group.dof)
        self.bias = bias
        self.group = group
        self.covariance = np.zeros((group.dof, group.dof))
        self.input_covariance = Q
        self.value :np.ndarray = group.identity()
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
            new.bias_update(w[new.group.dof:])

        return new

    def copy(self):
        """
        Returns
        -------
        RelativeMotionIncrement
            A copy of the RMI
        """
        new = BodyVelocityIncrement(self.group, self.input_covariance, self.bias)
        new.value = self.value.copy()
        new.covariance = self.covariance.copy()
        new.bias_jacobian = self.bias_jacobian.copy()
        new.stamps = self.stamps.copy()
        return new

class PreintegratedBodyVelocity(ProcessModel):
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
        \Delta \mathbf{\Omega}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mbf{\omega}_{k}^\wedge)

    and :math:`\mbf{\omega}_{k}` is the angular velocity measurement at time :math:`k`.

    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id = None):
        super().__init__(SO3, Q, bias, state_id = state_id)


class WheelOdometryIncrement(BodyVelocityIncrement):
    """
    This is a preintegration class for wheel odometry measurements on SE(n).
    The preintegrated process model is of the form

    .. math::
        \mathbf{T}_{j} = \mathbf{T}_{i} \Delta \mathbf{V}_{ij},

    where :math:`\Delta \mathbf{V}_{ij}` is the preintegrated increment given by

    .. math::
        \Delta \mathbf{V}_{ij} = \prod_{k=i}^{j-1} \exp(\Delta t \mathbf{\\varpi}_{k}^\wedge)

    and :math:`\mathbf{\\varpi}_{k} = [\mathbf{\omega}_k^\\trans \;
    \mathbf{v}_k^\trans]^\trans` is the angular and translational velocity of the
    robot at time :math:`k`.
    """

    def __init__(self, Q: np.ndarray, bias: np.ndarray = None, state_id = None):
        if Q.shape == (6, 6):
            group = SE3
        elif Q.shape == (3, 3):
            group = SE2
        else:
            raise ValueError("Input covariance has incorrect shape")

        super().__init__(group, Q, bias, state_id = state_id)

# Alternate names for classes
class PreintegratedAngularVelocity(PreintegratedBodyVelocity):
    pass

class PreintegratedWheelOdometry(PreintegratedBodyVelocity):
    pass
