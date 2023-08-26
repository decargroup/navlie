from navlie.utils import state_interp
from navlie.lib.states import SE3State, SE3, SO3
import numpy as np
import pytest


@pytest.mark.parametrize("direction", ["left", "right"])
def test_state_interp_se3_position(direction):
    x0 = SE3State(SE3.Exp([0.0, 0.0, 0.0, 1, 2, 3]), 0, direction=direction)
    x1 = SE3State(SE3.Exp([0.0, 0.0, 0.0, 2, 3, 4]), 1, direction=direction)
    x = state_interp(0.5, [x0, x1])
    assert np.allclose(x.stamp, 0.5)
    assert x.value[0, 3] == 1.5
    assert x.value[1, 3] == 2.5
    assert x.value[2, 3] == 3.5


@pytest.mark.parametrize("direction", ["left", "right"])
def test_state_interp_se3_attitude(direction):
    x0 = SE3State(SE3.Exp([0.1, 0.2, 0.3, 0, 0, 0]), 0, direction=direction)
    x1 = SE3State(SE3.Exp([0.0, 0.0, 0.0, 0, 0, 0]), 1, direction=direction)
    x = state_interp(0.5, [x0, x1])
    assert np.allclose(x.stamp, 0.5)
    assert np.allclose(x.value[0:3, 0:3], SO3.Exp([0.05, 0.1, 0.15]))


def test_state_interp_out_of_bounds():
    x0 = SE3State(SE3.Exp([0.0, 0.0, 0.0, 1, 2, 3]), 0)
    x1 = SE3State(SE3.Exp([0.0, 0.0, 0.0, 2, 3, 4]), 1)
    x = state_interp(1.5, [x0, x1])
    assert np.allclose(x.stamp, 1.5)
    assert np.allclose(x.value, x1.value)

    x = state_interp(-0.5, [x0, x1])
    assert np.allclose(x.stamp, -0.5)
    assert np.allclose(x.value, x0.value)


def test_state_interp_multiple():
    x_data = [SE3State.random(stamp=i) for i in range(10)]
    x_query = [x_data[2], x_data[5], x_data[8]]
    x_interp = state_interp(x_query, x_data)
    assert len(x_interp) == len(x_query)
    for i in range(len(x_interp)):
        assert np.allclose(x_interp[i].stamp, x_query[i].stamp)
        assert np.allclose(x_interp[i].value, x_query[i].value)


if __name__ == "__main__":
    test_state_interp_out_of_bounds()
