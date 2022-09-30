from abc import ABC, abstractmethod
from pynav.types import State
from pynav.lib.imu import (
    IMU,
    IMUState,
    U_matrix,
    adjoint_IE3,
    get_unbiased_imu,
    inverse_IE3,
    L_matrix,
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

    @staticmethod
    def from_measurements(u_list, x_list):
        """
        Parameters
        ----------
        u_list : List[StampedValue]
            List of input measurements to construct RMI from
        x_list : List[State]
            List of state measurements to construct RMI from

        Returns
        -------
        RelativeMotionIncrement
            The RMI
        """
        pass

    @abstractmethod
    def increment(self, u, dt):
        """
        Updates the RMI given an input measurement `u` and a duration `dt`
        over which to preintegrate.
        """
        pass

    @abstractmethod
    def apply_to_state(self, x: State):
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
        self.covariance = np.zeros((5, 5))
        self.input_covariance = input_covariance
        self.bias_jacobian = np.zeros((9, 6))
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias

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
        self.covariance = A @ Q @ A.T + L @ self.input_covariance @ L.T
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
