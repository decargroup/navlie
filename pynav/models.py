from pynav.types import ProcessModel, MeasurementModel, VectorState, StampedValue
import numpy as np
from typing import List

class SingleIntegrator(ProcessModel):
    """
    The single-integrator process model is a process model of the form

        x_dot = u .
    """

    def __init__(self, Q: np.ndarray):

        if not isinstance(Q, np.ndarray):
            Q = np.array(Q).reshape((1, 1))

        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a scalar or n x n matrix.")

        self.Q = Q
        self.dim = Q.shape[0]

    def evaluate(self, x: VectorState, u: StampedValue, dt: float) -> np.ndarray:
        x.value = x.value + dt * u.value
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        return np.identity(self.dim)

    def covariance(self, x, u, dt) -> np.ndarray:
        return dt**2 * self.Q


class AnchorRangeModel(MeasurementModel):
    def __init__(self, anchor_position: List[float], R: float):
        self.r_cw_a = np.array(anchor_position).flatten()
        self.R = np.array(R)

    def evaluate(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        y = np.linalg.norm(self.r_cw_a - r_zw_a)
        return y

    def jacobian(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        r_cz_a: np.ndarray = self.r_cw_a - r_zw_a
        y = np.linalg.norm(r_cz_a)
        return r_cz_a.reshape((1, -1)) / y

    def covariance(self, x: VectorState) -> np.ndarray:
        return self.R