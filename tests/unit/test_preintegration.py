from pynav.lib.imu import IMU, IMUKinematics, IMUState
from pynav.lib.preintegration import (
    BodyVelocityIncrement,
    IMUIncrement,
    PreintegratedBodyVelocity,
    PreintegratedIMUKinematics,
    LinearIncrement,
    PreintegratedLinearModel,
)
from pynav.lib.models import BodyFrameVelocity, DoubleIntegrator, DoubleIntegratorWithBias
from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import SE3State, VectorState
import numpy as np
from pylie import SE23, SE2, SE3, SO3
from pynav.types import StampedValue, StateWithCovariance
import pytest

np.set_printoptions(precision=8, linewidth=200)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_imu_preintegration(direction):
    """
    Tests to make sure IMU preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(12)
    accel_bias = [1, 2, 3]
    gyro_bias = [0.1, 0.2, 0.3]

    model = IMUKinematics(Q)
    dt = 0.01
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        gyro_bias,
        accel_bias,
        0,
        direction=direction,
    )
    P0 = np.identity(15)
    ekf = ExtendedKalmanFilter(model)
    x0 = StateWithCovariance(x, P0)
    x_dr = x0.copy()
    rmi = IMUIncrement(Q, gyro_bias=gyro_bias, accel_bias=accel_bias)
    preint_model = PreintegratedIMUKinematics()

    # Do both dead reckoning and preintegration
    for i in range(100):
        x_dr = ekf.predict(x_dr, u, dt)
        rmi.increment(u, dt)

    # Apply the rmi to the state
    ekf.process_model = preint_model
    x_pre = ekf.predict(x0.copy(), rmi, dt)

    # Compare the results
    print(x_dr.covariance - x_pre.covariance)
    assert np.allclose(x_dr.state.pose, x_pre.state.pose)
    assert np.allclose(x_dr.state.bias, x_pre.state.bias)
    assert np.allclose(x_dr.covariance, x_pre.covariance)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_odometry_preintegration_se3(direction):
    """
    Tests to make sure preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(6)
    bias = [0, 0, 0, 0, 0, 0]

    model = BodyFrameVelocity(Q)
    dt = 0.01
    u = StampedValue([1, 2, 3, 4, 5, 6], 0)
    x = SE3State(SE3.random(), stamp=0.0, direction=direction)
    P0 = np.identity(6)
    ekf = ExtendedKalmanFilter(model)
    x0 = StateWithCovariance(x, P0)
    x_dr = x0.copy()
    rmi = BodyVelocityIncrement(x.group, Q, bias=bias)
    preint_model = PreintegratedBodyVelocity()

    # Do both dead reckoning and preintegration
    for i in range(1):
        x_dr = ekf.predict(x_dr, u, dt)
        rmi.increment(u, dt)

    # Apply the rmi to the state
    ekf.process_model = preint_model
    x_pre = ekf.predict(x0.copy(), rmi, dt)

    # Compare the results
    print(x_dr.covariance - x_pre.covariance)
    assert np.allclose(x_dr.state.value, x_pre.state.value)
    assert np.allclose(x_dr.covariance, x_pre.covariance)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_preintegrated_process_jacobian_body_velocity(direction):
    """
    Numerically checks the jacobian of the RMI-based process model.
    """
    Q = np.identity(6)
    bias = [0, 0, 0, 0, 0, 0]
    dt = 0.01
    u = StampedValue([1, 2, 3, 4, 5, 6], 0)
    x = SE3State(SE3.random(), stamp=0.0, direction=direction)
    rmi = BodyVelocityIncrement(x.group, Q, bias=bias)
    preint_model = PreintegratedBodyVelocity()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    jac = preint_model.jacobian(x, rmi, dt)
    jac_fd = preint_model.jacobian_fd(x, rmi, dt)

    print(jac - jac_fd)
    assert np.allclose(jac, jac_fd, atol=1e-4)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_preintegrated_process_jacobian_imu(direction):
    Q = np.identity(12)
    accel_bias = [1, 2, 3]
    gyro_bias = [0.1, 0.2, 0.3]

    dt = 0.01
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        gyro_bias,
        accel_bias,
        0,
        direction=direction,
    )
    rmi = IMUIncrement(Q, gyro_bias=gyro_bias, accel_bias=accel_bias)
    preint_model = PreintegratedIMUKinematics()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    jac = preint_model.jacobian(x, rmi, dt)
    jac_fd = preint_model.jacobian_fd(x, rmi, dt)

    print(jac - jac_fd)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_double_integrator_preintegration():
    """
    Tests to make sure preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(2)

    dt = 0.01
    u = StampedValue([1, 2], 0)
    x = VectorState([0, 0, 0, 0], stamp=0.0)
    P0 = np.identity(4)

    model = DoubleIntegrator(Q)
    rmi = LinearIncrement(
        input_covariance = Q,
        state_matrix = lambda u, dt: model._state_jacobian(dt),
        input_matrix = lambda u, dt: model._input_jacobian(dt),
        dof = 4,
    )
    preint_model = PreintegratedLinearModel()

    ekf = ExtendedKalmanFilter(model)
    x0 = StateWithCovariance(x, P0)
    x_dr = x0.copy()
 
    # Do both dead reckoning and preintegration
    for i in range(100):
        x_dr = ekf.predict(x_dr, u, dt)
        rmi.increment(u, dt)

    # Apply the rmi to the state
    ekf.process_model = preint_model
    x_pre = ekf.predict(x0.copy(), rmi, dt)

    # Compare the results
    print(x_dr.covariance - x_pre.covariance)
    assert np.allclose(x_dr.state.value, x_pre.state.value)
    assert np.allclose(x_dr.covariance, x_pre.covariance)

def test_double_integrator_preintegration_with_bias():
    """
    Tests to make sure preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(4)
    bias = [0, 2]

    dt = 0.01
    u = StampedValue([1, 2], 0)
    x = VectorState([0, 0, 0, 0] + bias, stamp=0.0)
    P0 = np.identity(6)

    model = DoubleIntegratorWithBias(Q)
    rmi = LinearIncrement(
        input_covariance = Q,
        state_matrix = lambda u, dt: model._state_jacobian(dt),
        input_matrix = lambda u, dt: model._input_jacobian(dt),
        dof = 4,
        bias=bias,
    )
    preint_model = PreintegratedLinearModel()

    ekf = ExtendedKalmanFilter(model)
    x0 = StateWithCovariance(x, P0)
    x_dr = x0.copy()
 
    # Do both dead reckoning and preintegration
    for i in range(100):
        x_dr = ekf.predict(x_dr, u, dt)
        rmi.increment(u, dt)

    # Apply the rmi to the state
    ekf.process_model = preint_model
    x_pre = ekf.predict(x0.copy(), rmi, dt)

    # Compare the results
    print(x_dr.covariance - x_pre.covariance)
    assert np.allclose(x_dr.state.value, x_pre.state.value)
    assert np.allclose(x_dr.covariance, x_pre.covariance)


if __name__ == "__main__":
    test_double_integrator_preintegration_with_bias()
