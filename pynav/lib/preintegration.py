from abc import ABC, abstractmethod
from typing import List
from pynav.types import StampedValue, State, StateWithCovariance
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
from pylie import SE23, MatrixLieGroup


class RelativeMotionIncrement(ABC):
    __slots__ = ["value", "stamps", "covariance"]

    def __init__(self):
        #:Any: the value of the RMI
        self.value = None

        #:List[float, float]: the two timestamps i, j associated with the RMI
        self.stamps = [None, None]

        #:np.ndarray: the covariance matrix of the RMI
        self.covariance = None

    def increment_many(
        self, u_list: List[StampedValue], end_stamp: float = None
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
            self.increment(u_list[k].value, dt)

        if end_stamp is not None:
            dt = end_stamp - u_list[-1].stamp
            self.increment(u_list[-1].value, dt)

        return self

    @abstractmethod
    def increment(self, u, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        pass

    @abstractmethod
    def predict(self, x: StateWithCovariance) -> StateWithCovariance:
        """
        Parameters
        ----------
        x : State
            The state to apply the RMI to

        Returns
        -------
        State
            The updated state
        """
        pass


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
        self, group: MatrixLieGroup, Q: np.ndarray, bias: np.ndarray = None
    ):
        self.bias = bias
        self.group = group
        self.covariance = np.zeros((group.dof, group.dof))
        self.input_covariance = Q
        self.value = group.identity()
        self.bias_jacobian = np.zeros((group.dof, group.dof))
        if bias is None:
            self.bias = np.zeros((group.dof))

    def increment(self, u: StampedValue, dt):
        """
        In-place updating the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
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

    def predict(self, x: StateWithCovariance) -> StateWithCovariance:
        """
        Parameters
        ----------
        x : State
            The state to apply the RMI to

        Returns
        -------
        State
            The updated state
        """
        x = x.copy()
        x_state: MatrixLieGroupState = x.state
        x_state.value = x_state.value @ self.value

        dof = self.group.dof
        if self.input_covariance.shape[0] ==  dof:
            estimating_bias = False
            total_dof =  self.group.dof
        elif self.input_covariance.shape[0] == 2 * dof:
            estimating_bias = True
            total_dof = 2 *  self.group.dof
        else:
            raise ValueError("Input covariance has incorrect shape")    

        A = np.identity(total_dof)
        L = np.identity(total_dof)

        if x_state.direction == "right":
            A[0:dof, 0:dof] = self.group.adjoint(self.group.inverse(self.value))
            if estimating_bias:
                A[0:dof, dof : total_dof] = self.bias_jacobian

        elif x_state.direction == "left":
            Ad = self.group.adjoint(x_state.value)
            if estimating_bias:
                A[0:dof, dof:total_dof] = Ad @ self.bias_jacobian

            L[0:dof, 0:dof] = Ad

        x.state.value = x_state.value
        x.covariance = A @ x.covariance @ A.T + L @ self.covariance @ L.T
        return x


class IMUIncrement(RelativeMotionIncrement):
    def __init__(
        self,
        input_covariance: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        gravity=[0, 0, -9.80665],
    ):
        """
        Initializes an "identity" IMU RMI.

        Parameters
        ----------
        input_covariance : np.ndarray with shape (6, 6)
            covariance of gyro, accel measurements
        """
        self.value = np.identity(5)
        self.stamps = [None, None]
        self.covariance = np.zeros((15, 15))
        self.input_covariance = input_covariance
        self.bias_jacobian = np.zeros((9, 6))
        self.accel_bias = np.array(accel_bias).ravel()
        self.gyro_bias = np.array(gyro_bias).ravel()
        self.gravity = np.array(gravity).ravel()

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

    def predict(self, x: StateWithCovariance) -> StateWithCovariance:
        """
        Parameters
        ----------
        x : State
            The state to apply the RMI to

        Returns
        -------
        State
            The updated state
        """
        x = x.copy()
        dt = self.stamps[1] - self.stamps[0]
        DG = G_matrix(self.gravity, dt)
        DU = self.value
        x_state: IMUState = x.state
        x_state.pose = DG @ x_state.pose @ DU

        A = np.identity(15)
        L = np.identity(15)
        if x_state.direction == "right":
            A[0:9, 0:9] = adjoint_IE3(inverse_IE3(DU))
            A[0:9, 9:15] = self.bias_jacobian
        elif x_state.direction == "left":
            Ad = SE23.adjoint(x_state.pose)
            A[0:9, 0:9] = adjoint_IE3(DG)
            A[0:9, 9:15] = Ad @ self.bias_jacobian
            L[0:9, 0:9] = Ad

        x.covariance = A @ x.covariance @ A.T + L @ self.covariance @ L.T

        return x
