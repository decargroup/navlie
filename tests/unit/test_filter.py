from pynav.lib.models import (
    SingleIntegrator,
    RangePointToAnchor,
)
from pynav.lib.states import VectorState
from pynav import DataGenerator, StateWithCovariance
from pynav.filters import IteratedKalmanFilter, run_filter
from pynav.utils import randvec
import numpy as np


def test_iteratedEKF_no_line_search():
    # Check if we can run the filter without line search
    # ##############################################################################
    # Problem Setup

    x0 = VectorState(np.array([1, 0]), stamp=0)
    P0 = np.diag([1, 1])
    R = 0.1**2
    Q = 0.1 * np.identity(2)
    range_models = [
        RangePointToAnchor([0, 4], R),
    ]
    range_freqs = [2]
    process_model = SingleIntegrator(Q)
    input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])
    input_covariance = Q
    input_freq = 10
    noise_active = True
    # ##############################################################################
    # Data Generation

    dg = DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        range_models,
        range_freqs,
    )

    gt_data, input_data, meas_data = dg.generate(x0, 0, 1, noise=noise_active)

    # ##############################################################################
    # Run Filter
    if noise_active:
        x0 = x0.plus(randvec(P0))

    x = StateWithCovariance(x0, P0)
    try:
        filter = IteratedKalmanFilter(process_model, line_search=False)
        run_filter(filter, x0, P0, input_data, meas_data)
    except Exception as e:
        print(e)
        assert False
    else:
        assert True
