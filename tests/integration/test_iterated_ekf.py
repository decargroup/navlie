from navlie.lib.states import SE3State
from navlie.lib.models import BodyFrameVelocity, GlobalPosition
from navlie.datagen import DataGenerator
from navlie.filters import ExtendedKalmanFilter, IteratedKalmanFilter
from navlie.utils import GaussianResult, GaussianResultList, randvec, monte_carlo
from navlie.types import StateWithCovariance, State
from pymlg import SE3
import numpy as np
from typing import List
import pytest

np.random.seed(0)


# ##############################################################################
# Test parameter definitions

Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
test1_kwargs = {
    "duration": 5,
    "x0": SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, direction="right"),
    "P0": np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1]),
    "process_model": BodyFrameVelocity(Q),
    "input_profile": lambda t, x: np.array(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
    ),
    "input_freq": 50,
    "input_covariance": Q,
    "measurement_models": [GlobalPosition(np.diag([0.1**2, 0.1**2, 0.1**2]))],
    "measurement_freq": 10,
    "filter": IteratedKalmanFilter,
}

test2_kwargs = {
    "duration": 5,
    "x0": SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, direction="left"),
    "P0": np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1]),
    "process_model": BodyFrameVelocity(Q),
    "input_profile": lambda t, x: np.array(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
    ),
    "input_freq": 50,
    "input_covariance": Q,
    "measurement_models": [GlobalPosition(np.diag([0.1**2, 0.1**2, 0.1**2]))],
    "measurement_freq": 10,
    "filter": IteratedKalmanFilter,
}

test3_kwargs = {
    "duration": 5,
    "x0": SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0, direction="right"),
    "P0": np.diag([0.1**2, 0.1**2, 0.1**2, 1, 1, 1]),
    "process_model": BodyFrameVelocity(Q),
    "input_profile": lambda t, x: np.array(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
    ),
    "input_freq": 50,
    "input_covariance": Q,
    "measurement_models": [GlobalPosition(np.diag([0.1**2, 0.1**2, 0.1**2]))],
    "measurement_freq": 10,
    "filter": ExtendedKalmanFilter,
}

################################################################################
################################################################################

def _filter_trial(
    trial_number: int,
    filter: ExtendedKalmanFilter,
    x0: State,
    P0: np.ndarray,
    process_model,
    measurement_models,
    input_profile,
    input_freq,
    input_covariance,
    duration,
    measurement_freq,
):
    np.random.seed(trial_number)
    filter = filter(process_model=process_model)
    dg = DataGenerator(
        process_model,
        input_profile,
        input_covariance,
        input_freq,
        measurement_models,
        measurement_freq,
    )
    state_true, input_data, meas_data = dg.generate(x0, 0, duration, noise=True)
    x0 = x0.plus(randvec(P0))
    x = StateWithCovariance(x0, P0)

    meas_idx = 0
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):
        results_list.append(GaussianResult(x, state_true[k]))

        u = input_data[k]

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = filter.correct(x, y, u)
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        dt = input_data[k + 1].stamp - x.stamp
        x = filter.predict(x, u, dt)

    return GaussianResultList(results_list)

@pytest.mark.parametrize("kwargs", [test1_kwargs, test2_kwargs, test3_kwargs])
def test_iterated_ekf(kwargs):

    results = monte_carlo(lambda n: _filter_trial(n, **kwargs), 5)
    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 2 * results.nees_upper_bound(0.99)
    )
    nt = results.average_nees.shape[0]
    # Proportion of time NEES remains below 2*upper_bound bigger than 90%
    assert nees_in_correct_region / nt > 0.90

    # Make sure we essentially never get a completely absurd NEES.
    nees_in_correct_region = np.count_nonzero(
        results.average_nees < 50 * results.nees_upper_bound(0.99)
    )
    assert nees_in_correct_region / nt > 0.9999


