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
    def increment(self, x, u, dt):
        # Updates the value, stamps, covariance
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
    def __init__(self, input_covariance: np.ndarray):
        self.value = np.identity(5)
        self.stamps = [None, None]
        self.covariance = np.zeros((5, 5))
        self._input_covariance = input_covariance

    def increment(self, x: IMUState, u: IMU, dt):
        if self.stamps[0] is None:
            self.stamps[0] = u.stamp
            self.stamps[1] = u.stamp + dt
        else:
            self.stamps[1] += dt

        unbiased_gyro, unbiased_accel = get_unbiased_imu(x, u)
        U = U_matrix(unbiased_gyro, unbiased_accel, dt)
        self.value = self.value @ U

        U_inv = inverse_IE3(U)
        A = adjoint_IE3(U_inv)
        L = L_matrix(x, u, dt)
        Q = self.covariance
        self.covariance = A @ Q @ A.T + L @ self._input_covariance @ L.T
        