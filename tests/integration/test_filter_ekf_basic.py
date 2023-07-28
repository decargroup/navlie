import pytest
from navlie import (
    GaussianResult,
    GaussianResultList,
    randvec,
    run_filter,
    DataGenerator,
    ExtendedKalmanFilter,
    monte_carlo
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


def make_filter_trial(dg, x0_true, P0, t_max, ekf):
    def ekf_trial(trial_number: int) -> List[GaussianResult]:
        """
        A single trial in a monte carlo experiment. This function accepts the trial
        number and must return a list of GaussianResult objects.
        """

        # By using the trial number as the seed for the random generator, we can
        # make sure our experiments are perfectly repeatable, yet still have
        # independent noise samples from trial-to-trial.
        np.random.seed(trial_number)
        state_true, input_list, meas_list = dg.generate(x0_true, 0, t_max, True)

        x0_check = x0_true.plus(randvec(P0))
        estimate_list = run_filter(ekf, x0_check, P0, input_list, meas_list)

        results = GaussianResultList.from_estimates(estimate_list, state_true)
        return results

    return ekf_trial


@pytest.mark.parametrize(
    "x0, P0, input_profile, process_model, measurement_model, Q, R, input_freq, measurement_freq",
    [
        (
            VectorState(np.array([1, 0]), 0.0),
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
            VectorState(np.array([1, 0]), 0.0),
            np.diag([1e-8, 1e-8]),
            lambda t, x: np.array([np.sin(t), np.cos(t)]),
            SingleIntegrator,
            make_range_models_ekf_vector,
            0.1 * np.identity(2),
            1e-5,
            10,
            [1, 1, 1],
        ),
        (
            SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0),
            0.1**2 * np.identity(6),
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
def test_reasonable_nees(
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
    N = 2
    kf = ExtendedKalmanFilter(process_model(Q))
    dg = DataGenerator(
        process_model(Q),
        input_profile,
        Q,
        input_freq,
        measurement_model(R),
        measurement_freq,
    )
    ekf_trial = make_filter_trial(dg, x0, P0, t_max, kf)
    results = monte_carlo(ekf_trial, N)

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
