from navlie.lib.imu import IMU, IMUKinematics, IMUState
from navlie.lib.preintegration import (
    BodyVelocityIncrement,
    IMUIncrement,
    PreintegratedBodyVelocity,
    PreintegratedIMUKinematics,
    LinearIncrement,
    PreintegratedLinearModel,
)
from navlie.lib.models import BodyFrameVelocity, DoubleIntegrator, DoubleIntegratorWithBias
from navlie.filters import ExtendedKalmanFilter
from navlie.lib.states import SE3State, VectorState
import numpy as np
from pymlg import SE23, SE2, SE3, SO3
from navlie.types import StampedValue, StateWithCovariance
import pytest

np.set_printoptions(precision=5, suppress=True, linewidth=200)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_imu_preintegration_equivalence(direction):
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

def test_imu_update_bias():
    Q = np.identity(12)
    accel_bias = [1, 2, 3]
    gyro_bias = [0.1, 0.2, 0.3]

    new_accel_bias = [1.1, 2.1, 3.1]
    new_gyro_bias = [0.11, 0.21, 0.31]

    dt = 0.01
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    rmi1 = IMUIncrement(Q, gyro_bias=gyro_bias, accel_bias=accel_bias)
    rmi2 = IMUIncrement(Q, gyro_bias=new_gyro_bias, accel_bias=new_accel_bias)

    # Do both dead reckoning and preintegration
    for i in range(100):
        rmi1.increment(u, dt)
        rmi2.increment(u, dt)

    # Correct the first rmi with the new bias value
    rmi1.update_bias(np.array(new_gyro_bias + new_accel_bias))

    assert np.allclose(rmi1.value, rmi2.value, atol=1e-3)





@pytest.mark.parametrize("direction", ["left", "right"])
def test_odometry_preintegration_se3_equivalence(direction):
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
    rmi = IMUIncrement(Q, gyro_bias=[2,3,4], accel_bias=[0.1,0.2,0.3])
    preint_model = PreintegratedIMUKinematics()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    jac = preint_model.jacobian(x, rmi, dt)
    jac_fd = preint_model.jacobian_fd(x, rmi, dt)

    print(jac - jac_fd)
    assert np.allclose(jac, jac_fd, atol=1e-4)

@pytest.mark.parametrize("direction", ["left", "right"])
def test_preintegrated_process_covariance_imu(direction):
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
    rmi = IMUIncrement(Q, gyro_bias=[2,3,4], accel_bias=[0.1,0.2,0.3])
    preint_model = PreintegratedIMUKinematics()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    Qd = preint_model.covariance(x, rmi, dt)
    L = preint_model.input_jacobian_fd(x, rmi, dt)
    Qd_fd = L @ rmi.covariance @ L.T

    # We are unable to test the covariance associated with the bias RMI
    # since it is always zero and not actually included in the IMUIncrement object.
    assert np.allclose(Qd[:9,:9], Qd_fd[:9,:9], atol=1e-4)


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
        state_matrix = lambda u, dt: model.jacobian(None, None, dt),
        input_matrix = lambda u, dt: model.input_jacobian(dt),
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
    model_di = DoubleIntegrator(Q[:2,:2])
    rmi = LinearIncrement(
        input_covariance = Q,
        state_matrix = lambda u, dt: model_di.jacobian(None, None, dt),
        input_matrix = lambda u, dt: model_di.input_jacobian(dt),
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

def test_double_integrator_bias_equivalence():
    """
    Tests to make sure preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(4)
    bias = [0, 2]

    dt = 0.01
    u = StampedValue([1, 2], 0)
    u2 = StampedValue([1, 2, 0, 0], 0)
    x = VectorState([0, 0, 0, 0] + bias, stamp=0.0)
    P0 = np.identity(6)

    model = DoubleIntegratorWithBias(Q)
    model_di = DoubleIntegrator(Q[:2,:2])
    rmi1 = LinearIncrement(
        input_covariance = Q,
        state_matrix = lambda u, dt: model_di.jacobian(None, None, dt),
        input_matrix = lambda u, dt: model_di.input_jacobian(dt),
        dof = 4,
        bias=bias,
    )
    rmi2 = LinearIncrement(
        input_covariance = Q,
        state_matrix = lambda u, dt: model.jacobian(None, None, dt),
        input_matrix = lambda u, dt: model.input_jacobian(dt),
        dof = 6,
        bias=None,
    )

    # Do both dead reckoning and preintegration
    for i in range(100):
        rmi1.increment(u, dt)
        rmi2.increment(u2, dt)

    ekf = ExtendedKalmanFilter(PreintegratedLinearModel())
    x0 = StateWithCovariance(x, P0)
    x1 = ekf.predict(x0.copy(), rmi1, dt)
    x2 = ekf.predict(x0.copy(), rmi2, dt)

    # Compare the results
    assert np.allclose(x1.state.value, x2.state.value)
    assert np.allclose(x1.covariance, x2.covariance)
    assert np.allclose(rmi1.covariance, rmi2.covariance)


if __name__ == "__main__":
    test_odometry_preintegration_se3_equivalence("left")
