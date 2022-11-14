from pynav.lib.states import VectorState, SO2State, SO3State, SE2State, SE3State, SE23State, SL3, SL3State
from pylie import SO2, SO3, SE2, SE3, SE23, SL3
import numpy as np 

try:
    # We do not want to make ROS a hard dependency, so we import it only if
    # available.
    from geometry_msgs.msg import PoseStamped, QuaternionStamped
    import rospy
except ImportError:
    pass  # ROS is not installed
except:
    raise

def test_plus_minus_vector():
    x1 = VectorState([1,2,3])
    dx = np.array([4,5,6])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_se3():
    x1 = SE3State(SE3.Exp([0.1,0.2,0.3,4,5,6]))
    dx = np.array([0.1, 0.2, 0.4, 0.2, 0.2, 0.2])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_se2():
    x1 = SE2State(SE2.Exp([0.1,5,6]))
    dx = np.array([0.1, 0.2, 0.4])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_so2():
    x1 = SO2State(SO2.Exp([0.1]))
    dx = np.array([0.3])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_so3():
    x1 = SO3State(SO3.Exp([0.1, 0.2, 0.3]))
    dx = np.array([0.3, 0.4, 0.5])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_plus_minus_se23():
    x1 = SE23State(SE23.Exp(0.1*np.array([1,2,3,4,5,6,7,8,9])))
    dx = 0.1*np.array([3,4,5,6,7,8,9,1,2])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx, dx_test)

def test_se3_ros():
    T = SE3.random()
    x = SE3State(T, stamp=1, state_id="test")
    x_ros = x.to_ros()
    x2 = SE3State.from_ros(x_ros)
    assert np.allclose(x.value, x2.value)
    assert x.stamp == x2.stamp
    assert x.state_id == x2.state_id
    assert x.state_id == x_ros.header.frame_id

def test_so3_ros():
    C = SO3.random()
    x = SO3State(C, stamp=1, state_id="test")
    x_ros = x.to_ros()
    x2 = SO3State.from_ros(x_ros)
    assert np.allclose(x.value, x2.value)
    assert x.stamp == x2.stamp
    assert x.state_id == x2.state_id
    assert x.state_id == x_ros.header.frame_id

def test_jacobian_vector():
    x1 = VectorState([1,2,3])
    dx = np.array([4,5,6])
    jac = x1.jacobian(dx)
    jac_test = x1.jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-8)

def test_jacobian_se3():
    x1 = SE3State(SE3.Exp([0.1,0.2,0.3,4,5,6]))
    dx = np.array([0.1, 0.2, 0.4, 0.2, 0.2, 0.2])
    jac = x1.jacobian(dx)
    jac_test = x1.jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-8)

def test_jacobian_se2():
    x1 = SE2State(SE2.Exp([0.1,5,6]))
    dx = np.array([0.1, 0.2, 0.4])
    jac = x1.jacobian(dx)
    jac_test = x1.jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-8)

def test_jacobian_so2():
    x1 = SO2State(SO2.Exp([0.1]))
    dx = np.array([0.3])
    jac = x1.jacobian(dx)
    jac_test = x1.jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-8)

def test_jacobian_so3():
    x1 = SO3State(SO3.Exp([0.1, 0.2, 0.3]))
    dx = np.array([0.3, 0.4, 0.5])
    jac = x1.jacobian(dx)
    jac_test = x1.jacobian_fd(dx)
    assert np.allclose(jac, jac_test, atol=1e-8)

def test_plus_minus_sl3():
    x1 = SL3State(SL3.Exp([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    dx = 0.1*np.array([3,4,5,6,7,8,9,1])
    x2 = x1.plus(dx)
    dx_test = x2.minus(x1)
    assert np.allclose(dx,dx_test, atol=1e-8)

if __name__ == "__main__":
    test_jacobian_so2()