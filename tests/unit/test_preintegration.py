
from pynav.lib.imu import IMU, IMUKinematics, IMUState
from pynav.lib.preintegration import BodyVelocityIncrement, IMUIncrement, PreintegratedBodyVelocity, PreintegratedIMUKinematics
from pynav.lib.models import BodyFrameVelocity
from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import SE3State
import numpy as np 
from pylie import SE23, SE2, SE3, SO3
from pynav.types import StampedValue, StateWithCovariance

np.set_printoptions(precision=4, suppress=True, linewidth=200)

def _test_imu_preintegration(direction):
    """
    Tests to make sure IMU preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(12)
    accel_bias = [1,2,3]
    gyro_bias = [0.1,0.2,0.3]

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
    rmi = IMUIncrement(Q,gyro_bias=gyro_bias,accel_bias=accel_bias)
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
    assert np.allclose(x_dr.covariance, x_pre.covariance)

def _test_odometry_preintegration_se3(direction):
    """
    Tests to make sure preintegration and regular dead reckoning
    are equivalent.
    """
    Q = np.identity(6)
    bias = [0,0,0,0,0,0]

    model = BodyFrameVelocity(Q)
    dt = 0.01
    u = StampedValue([1, 2, 3, 4, 5, 6], 0)
    x = SE3State(SE3.random(), stamp = 0.0, direction=direction)
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

def _test_preintegrated_process_imu(direction):
    Q = np.identity(12)
    accel_bias = [1,2,3]
    gyro_bias = [0.1,0.2,0.3]

    dt = 0.01
    u = IMU([1, 2, 3], [2, 3, 1], 0)
    x = IMUState(
        SE23.Exp([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        gyro_bias,
        accel_bias,
        0,
        direction=direction,
    )
    rmi = IMUIncrement(Q,gyro_bias=gyro_bias,accel_bias=accel_bias)
    preint_model = PreintegratedIMUKinematics()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    jac = preint_model.jacobian(x, rmi, dt)
    jac_fd = preint_model.jacobian_fd(x, rmi, dt)

    print(jac - jac_fd)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def _test_preintegrated_process_body_velocity(direction):
    Q = np.identity(6)
    bias = [0,0,0,0,0,0]
    dt = 0.01
    u = StampedValue([1, 2, 3, 4, 5, 6], 0)
    x = SE3State(SE3.random(), stamp = 0.0, direction=direction)
    rmi = BodyVelocityIncrement(x.group, Q, bias=bias)
    preint_model = PreintegratedBodyVelocity()

    # Do preintegration
    for i in range(10):
        rmi.increment(u, dt)

    jac = preint_model.jacobian(x, rmi, dt)
    jac_fd = preint_model.jacobian_fd(x, rmi, dt)

    print(jac - jac_fd)
    assert np.allclose(jac, jac_fd, atol=1e-4)

def test_imu_preintegration_right():
    _test_imu_preintegration("right")

def test_imu_preintegration_left():
    _test_imu_preintegration("left")

def test_preintegrated_process_imu_right():
    _test_preintegrated_process_imu("right")

def test_preintegrated_process_imu_left():
    _test_preintegrated_process_imu("left")

def test_odometry_preintegration_se3_right():
    _test_odometry_preintegration_se3("right")

def test_odometry_preintegration_se3_left():
    _test_odometry_preintegration_se3("left")

def test_preintegrated_process_body_velocity_right():
    _test_preintegrated_process_body_velocity("right")

def test_preintegrated_process_body_velocity_left():
    _test_preintegrated_process_body_velocity("left")

if __name__ == "__main__":
    test_preintegrated_process_imu_right()