from navlie.lib import (
    VectorState,
    SO2State,
    SO3State,
    SE2State,
    SE3State,
    SE23State,
    SL3State,
    IMUState,
    MatrixLieGroupState
)
from navlie.types import State
from pymlg import SO3, SE3
import numpy as np
import pytest
import sys
from typing import Dict

np.random.normal(0)
sample_states: Dict[str, State] = {
    "vector": VectorState([1, 2, 3]),
    "so2": SO2State(0.1),
    "so3": SO3State([0.1, 0.2, 0.3]),
    "se2": SE2State([0.1, 0.2, 0.3]),
    "se3": SE3State([0.1, 0.2, 0.3, 4, 5, 6]),
    "se23": SE23State([0.1, 0.2, 0.3, 4, 5, 6, 7, 8, 9]),
    "sl3": SL3State([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    "imu": IMUState([0.1, 0.2, 0.3, 4, 5, 6, 7, 8, 9], [1, 2, 3], [4, 5, 6]),
    "mlg": MatrixLieGroupState(SE3.random(), SE3)
}

@pytest.mark.parametrize(
    "s", ["vector", "so2", "so3", "se2", "se3", "se23", "sl3", "imu", "mlg"]
)
def test_plus_minus(s: str):
    x = sample_states[s]
    dx = np.random.randn(x.dof)
    x2 = x.plus(dx)
    dx_test = x2.minus(x).ravel()
    assert np.allclose(dx, dx_test)
    

@pytest.mark.parametrize(
    "s", ["vector", "so2", "so3", "se2", "se3", "se23", "sl3", "imu", "mlg"]
)
def test_plus_jacobian(s: str):
    x = sample_states[s]
    dx = np.random.randn(x.dof)
    jac = x.plus_jacobian(dx)
    jac_test = x.plus_jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-5)

@pytest.mark.parametrize(
    "s", ["vector", "so2", "so3", "se2", "se3", "se23", "sl3", "imu","mlg"]
)
def test_minus_jacobian(s: str):
    x = sample_states[s]
    dx = np.random.randn(x.dof)
    x2 = x.plus(dx)
    jac = x.minus_jacobian(x2)
    jac_test = x.minus_jacobian_fd(x2)
    assert np.allclose(jac, jac_test, atol=1e-5)


@pytest.mark.parametrize(
    "s", ["so2", "so3", "se2", "se3", "se23", "sl3", "mlg"]
)
def test_mlg_dot(s: str):
    x = sample_states[s]
    dx = np.random.randn(x.dof)
    x2 = x.plus(dx)
    xdot = x.dot(x2)
    assert np.allclose(xdot.value, x.value @ x2.value, atol=1e-5)

@pytest.mark.skipif('geometry_msgs' not in sys.modules,
                    reason="requires ROS1 to be installed")
def test_se3_ros():
    T = SE3.random()
    x = SE3State(T, stamp=1, state_id="test")
    x_ros = x.to_ros()
    x2 = SE3State.from_ros(x_ros)
    assert np.allclose(x.value, x2.value)
    assert x.stamp == x2.stamp
    assert x.state_id == x2.state_id
    assert x.state_id == x_ros.header.frame_id

@pytest.mark.skipif('geometry_msgs' not in sys.modules,
                    reason="requires ROS1 to be installed")
def test_so3_ros():
    C = SO3.random()
    x = SO3State(C, stamp=1, state_id="test")
    x_ros = x.to_ros()
    x2 = SO3State.from_ros(x_ros)
    assert np.allclose(x.value, x2.value)
    assert x.stamp == x2.stamp
    assert x.state_id == x2.state_id
    assert x.state_id == x_ros.header.frame_id
