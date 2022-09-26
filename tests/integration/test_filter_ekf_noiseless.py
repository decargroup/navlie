import pytest
from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import VectorState, SO3State, SE3State
from pynav.datagen import DataGenerator
from pynav.types import StampedValue, StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, MonteCarloResult
from pynav.utils import randvec

from pynav.utils import monte_carlo, plot_error
from pynav.lib.models import (
    DoubleIntegrator,
    OneDimensionalPositionVelocityRange,
)
from pynav.lib.models import SingleIntegrator, RangePointToAnchor
from pynav.lib.models import (
    BodyFrameVelocity,
    InvariantMeasurement,
    Magnetometer,
    Gravitometer,
)
from pynav.lib.models import RangePoseToAnchor
from pylie import SO3, SE3

import numpy as np
from typing import List
import time
from matplotlib import pyplot as plt


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


def make_filter_trial_prediction_noiseless(dg, x0_true, P0, t_max, kf):
    def ekf_trial(trial_number: int) -> List[GaussianResult]:

        np.random.seed(trial_number)
        state_true, input_data, meas_data = dg.generate(
            x0_true, 0, t_max, noise=False
        )
        x0_check = x0_true.copy()
        x = StateWithCovariance(x0_check, P0)

        meas_idx = 0
        y = meas_data[meas_idx]
        results_list = []
        for k in range(len(input_data) - 1):
            results_list.append(GaussianResult(x, state_true[k]))
            u = input_data[k]

            # Fuse any measurements that have occurred.
            while y.stamp < input_data[k + 1].stamp and meas_idx < len(
                meas_data
            ):

                x = kf.correct(x, y, u)
                meas_idx += 1
                if meas_idx < len(meas_data):
                    y = meas_data[meas_idx]

            dt = input_data[k + 1].stamp - x.stamp
            x = kf.predict(x, u, dt)
        return GaussianResultList(results_list)

    return ekf_trial


@pytest.mark.parametrize(
    "x0, P0, input_profile, process_model, measurement_model, Q, R, input_freq, measurement_freq",
    [
        (
            VectorState(np.array([1, 0])),
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
            VectorState(np.array([1, 0])),
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
