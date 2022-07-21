from pynav.states import SE2State, SE3State, SE23State
from pynav.models import BodyFrameVelocity, RangePoseToAnchor
from pylie import SO2, SO3, SE3, SE2, SE3, SE23
import numpy as np

from pynav.types import StampedValue


def test_body_velocity_se3():
    x = SE3State(
        SE3.random(),
        direction="right",
    )
    u = StampedValue(np.array([1, 2, 3, 4, 5, 6]))
    dt = 0.1
    Q = np.identity(6)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)


def test_body_velocity_se2():
    x = SE2State(
        SE2.random(),
        direction="right",
    )
    u = StampedValue(np.array([1, 2, 3]))
    dt = 0.1
    Q = np.identity(3)
    process_model = BodyFrameVelocity(Q)
    jac = process_model.jacobian(x, u, dt)
    jac_fd = process_model.jacobian_fd(x, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-4)
