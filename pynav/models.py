from pynav.types import (
    ProcessModel,
    MeasurementModel,
    StampedValue,
)
from pynav.states import MatrixLieGroupState, SE3State, VectorState
from pylie import SO2, SO3
import numpy as np
from typing import List


class SingleIntegrator(ProcessModel):
    """
    The single-integrator process model is a process model of the form

        x_dot = u .
    """

    def __init__(self, Q: np.ndarray):

        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be an n x n matrix.")

        self._Q = Q
        self.dim = Q.shape[0]

    def evaluate(
        self, x: VectorState, u: StampedValue, dt: float
    ) -> np.ndarray:
        x.value = x.value + dt * u.value
        return x

    def jacobian(self, x, u, dt) -> np.ndarray:
        return np.identity(self.dim)

    def covariance(self, x, u, dt) -> np.ndarray:
        return dt**2 * self._Q

class BodyFrameVelocity(ProcessModel):
    """
    The body-frame velocity process model assumes that the input contains 
    both translational and angular velocity measurements, both relative to 
    a local reference frame, but resolved in the robot body frame. 

    This is commonly the process model associated with SE(n). 
    """

    def __init__(self, Q: np.ndarray):
        self._Q = Q

    def evaluate(self, x: SE3State, u: StampedValue, dt: float) -> SE3State:
        x.value = x.value @ x.group.Exp(u.value * dt)
        return x

    def jacobian(self, x: SE3State, u: StampedValue, dt: float) -> np.ndarray:
        if x.direction == "right":
            return x.group.adjoint(x.group.Exp(-u.value * dt))
        else: 
            raise NotImplementedError("TODO: left jacobian not yet implemented.")

    def covariance(self, x: SE3State, u: StampedValue, dt: float) -> np.ndarray:
        if x.direction == "right":
            L = dt * x.group.left_jacobian(-u.value*dt)
            return L @ self._Q @ L.T
        else: 
            raise NotImplementedError("TODO: left covariance not yet implemented.")
        
class RangePointToAnchor(MeasurementModel):
    def __init__(self, anchor_position: List[float], R: float):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = np.array(R)

    def evaluate(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        y = np.linalg.norm(self._r_cw_a - r_zw_a)
        return y

    def jacobian(self, x: VectorState) -> np.ndarray:
        r_zw_a = x.value.flatten()
        r_zc_a: np.ndarray = r_zw_a - self._r_cw_a
        y = np.linalg.norm(r_zc_a)
        return r_zc_a.reshape((1, -1)) / y

    def covariance(self, x: VectorState) -> np.ndarray:
        return self._R


class RangePoseToAnchor(MeasurementModel):
    
    def __init__(
        self,
        anchor_position: List[float],
        tag_body_position: List[float],
        R: float,
    ):
        self._r_cw_a = np.array(anchor_position).flatten()
        self._R = R 
        self._r_tz_b = np.array(tag_body_position).flatten()

    def evaluate(self, x: MatrixLieGroupState)-> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1,1)) + r_zw_a.reshape((-1,1))
        r_tc_a :np.ndarray = r_tw_a - self._r_cw_a.reshape((-1,1))
        return np.linalg.norm(r_tc_a) 

    def jacobian(self, x: MatrixLieGroupState) -> np.ndarray:
        r_zw_a = x.position
        C_ab = x.attitude
        if C_ab.shape == (2,2):
            att_group = SO2
        elif C_ab.shape == (3,3):
            att_group = SO3

        r_tw_a = C_ab @ self._r_tz_b.reshape((-1,1)) + r_zw_a.reshape((-1,1))
        r_tc_a :np.ndarray = r_tw_a - self._r_cw_a.reshape((-1,1))
        rho = r_tc_a/np.linalg.norm(r_tc_a)
        jac_attitude = rho.T @ C_ab @ att_group.odot(self._r_tz_b) 
        jac_position = rho.T @ C_ab
        jac = x.jacobian_from_blocks(attitude=jac_attitude, position=jac_position)
        return jac
        
    def covariance(self, x: MatrixLieGroupState) -> np.ndarray:
        return self._R

class GlobalPosition(MeasurementModel):
    pass