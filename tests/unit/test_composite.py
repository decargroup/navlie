from pynav.lib.states import SE2State, CompositeState
from pynav.types import StampedValue
from pynav.lib.models import BodyFrameVelocity, CompositeProcessModel, RangeRelativePose
from pylie import SE2
import numpy as np
import pickle
import os

def test_composite_process_model():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    Q = np.diag([0.1**2, 0.1**2, 0.001**2])
    x0 = CompositeState([T12, T13])
    process_model = CompositeProcessModel([BodyFrameVelocity(Q), BodyFrameVelocity(Q)])
    u = StampedValue(np.array([np.array([0.3, 1, 0]), np.array([-0.3, 2, 0])]), 1)
    dt = 1
    jac = process_model.jacobian(x0, u, dt)
    jac_fd = process_model.jacobian_fd(x0, u, dt)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_range_relative_pose():

    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    my_tag = [0.17, 0.17]
    tag = [-0.16, 0.2]
    model = RangeRelativePose(my_tag, tag, T12.state_id, 0.1**2)
    jac = model.jacobian(x)
    jac_fd = model.jacobian_fd(x)
    assert np.allclose(jac, jac_fd, atol=1e-6)


def test_composite_pickling():
    T12 = SE2State(SE2.Exp([0.5, 1, -1]), stamp=0.0, state_id=2)
    T13 = SE2State(SE2.Exp([-0.5, 1, 1]), stamp=0.0, state_id=3)
    x = CompositeState([T12, T13])
    with open("test.pkl", "wb") as f:
        pickle.dump(x, f)
    
    with open("test.pkl", "rb") as f:
        y = pickle.load(f)

    os.remove("test.pkl")


if __name__ == "__main__":
    test_composite_pickling()
