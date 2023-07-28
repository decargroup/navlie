import pytest
from navlie import (
    GaussianResult,
    GaussianResultList,
    DataGenerator,
    ExtendedKalmanFilter,
    monte_carlo,
    run_filter
)

from navlie.lib import (
    VectorState,
    SE3State,
    SingleIntegrator,
    RangePointToAnchor,
    BodyFrameVelocity,
    Magnetometer,
    Gravitometer,
    RangePoseToAnchor,
)
from pymlg import SE3

import numpy as np
from typing import List

# TODO. simplify this. 

def make_range_models_ekf_vector(R):
    range_models = [
        RangePointToAnchor([0, 4], R),
        RangePointToAnchor([-2, 0], R),
        RangePointToAnchor([2, 0], R),
    ]
    return range_models


def make_invariant_so3_models(R):
    mag_model = Magnetometer(R)
    grav_model = Gravitometer(R)
    return [mag_model, grav_model]


def make_range_models_iterated_ekf(R):
    range_models = [
        RangePoseToAnchor([1, 0, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([1, 0, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([-1, 0, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([-1, 0, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 0], [0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 0], [-0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 2], [0.17, 0.17, 0], R),
        RangePoseToAnchor([0, 2, 2], [-0.17, 0.17, 0], R),
    ]
    return range_models


def make_filter_trial_prediction_noiseless(dg, x0_true, P0, t_max, ekf):
    def ekf_trial(trial_number: int) -> List[GaussianResult]:

        np.random.seed(trial_number)
        state_true, input_data, meas_data = dg.generate(
            x0_true, 0, t_max, noise=False
        )
        x0_check = x0_true.copy()
        estimate_list = run_filter(ekf, x0_check, P0, input_data, meas_data)

        results = GaussianResultList.from_estimates(estimate_list, state_true)
        return results

    return ekf_trial


@pytest.mark.parametrize(
    "x0, P0, input_profile, process_model, measurement_model, Q, R, input_freq, measurement_freq",
    [
        (
            VectorState(np.array([1, 0]), stamp=0.0),
            np.diag([1, 1]),
            lambda t, x: np.array([np.sin(t), np.cos(t)]),
            SingleIntegrator,
            make_range_models_ekf_vector,
            0.1 * np.identity(2),
            0.1**2,
            10,
            [1, 1, 1],
        ),
        (
            VectorState(np.array([1, 0]), stamp=0.0),
            np.diag([1, 1]),
            lambda t, x: np.array([np.sin(t), np.cos(t)]),
            SingleIntegrator,
            make_range_models_ekf_vector,
            0.1 * np.identity(2),
            1e-8,
            10,
            [1, 1, 1],
        ),
        (
            SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0),
            1 * np.identity(6),
            lambda t, x: np.array(
                [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
            ),
            BodyFrameVelocity,
            make_range_models_iterated_ekf,
            np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1]),
            0.1**2,
            200,
            10,
        ),
    ],
)
def test_noiseless(
    x0,
    P0,
    input_profile,
    process_model,
    measurement_model,
    Q,
    R,
    input_freq,
    measurement_freq,
):
    dt = 1 / input_freq
    t_max = dt * 1000
    N = 1
    kf = ExtendedKalmanFilter(process_model(Q))
    dg = DataGenerator(
        process_model(Q),
        input_profile,
        Q,
        input_freq,
        measurement_model(R),
        measurement_freq,
    )
    ekf_trial = make_filter_trial_prediction_noiseless(dg, x0, P0, t_max, kf)
    results = monte_carlo(ekf_trial, N)

    for lv1, result in enumerate(results.trial_results):
        assert np.allclose(result.error, np.zeros(result.error.shape))
