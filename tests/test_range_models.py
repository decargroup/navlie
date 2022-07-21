from pynav.states import SE2State, SE3State, SE23State, CompositeState
from pynav.models import RangePoseToAnchor, RangePoseToPose, RangeRelativePose
from pylie import SO2, SO3, SE3, SE2, SE3, SE23
import numpy as np

from pynav.types import StampedValue

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


def test_range_pose_to_pose_se2():
    T12 = SE2State(SE2.Exp([0, 0, 0]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([0, 1, 0]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17], [-0.17, 0.17]]

    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    y = model.evaluate(x)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_pose_to_pose_se3():
    T12 = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, state_id=2)
    T13 = SE3State(SE3.Exp([0, 1, 0, 1, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17, 0], [-0.17, 0.17, 0]]

    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    y = model.evaluate(x)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_pose_to_pose_se23():
    T12 = SE23State(SE23.random(), stamp=0.0, state_id=2)
    T13 = SE23State(SE23.random(), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    tags = [[0.17, 0.17, 0], [-0.17, 0.17, 0]]

    model = RangePoseToPose(tags[0], tags[1], T12.state_id, T13.state_id, 1)
    y = model.evaluate(x)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


if __name__ == "__main__":
    test_range_pose_to_pose_se3()
