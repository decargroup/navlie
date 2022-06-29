from pynav.liegroups import SE2State, SE3State, SE23State
from pynav.models import RangePoseToAnchor
from pylie import SO2, SO3, SE3, SE2, SE3, SE23
import numpy as np

np.random.seed(0)

def test_range_pose_anchor_se2():
    x = SE2State(
        SE2.random(),
        direction="right",
    )
    model = RangePoseToAnchor([1, 2], [0.3, 0.1], 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_pose_anchor_se3():
    x = SE3State(
        SE3.random(),
        direction="right",
    )
    model = RangePoseToAnchor([1, 2, 0], [0.3, 0.1, 0], 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_pose_anchor_se23():
    x = SE23State(
        SE23.random(),
        direction="right",
    )
    model = RangePoseToAnchor([1, 2, 0], [0.3, 0.1, 0], 1)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)



if __name__ == "__main__":
    test_range_pose_anchor_se23()
