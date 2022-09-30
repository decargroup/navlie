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
    G_matrix
)
import numpy as np
from pylie import SE23


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


class IMUIncrement(RelativeMotionIncrement):
    def __init__(
        self,
        input_covariance: np.ndarray,
        gyro_bias: np.ndarray,
        accel_bias: np.ndarray,
        gravity = [0, 0, -9.80665],
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
        L_full[9:15, 6:12] = dt*np.identity(6)

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