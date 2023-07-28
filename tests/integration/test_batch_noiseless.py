from concurrent.futures import process
import pytest


from navlie.lib.states import SO3State, VectorState, SE3State
from navlie.lib.models import (
    BodyFrameVelocity,
    RangePointToAnchor,
    RangePoseToAnchor,
    Magnetometer,
    Gravitometer,
    SingleIntegrator,
)
from navlie.datagen import DataGenerator
from navlie.filters import ExtendedKalmanFilter, run_filter
from navlie.utils import GaussianResult, GaussianResultList, plot_error, randvec
from navlie.batch.estimator import BatchEstimator
from pymlg import SO3, SE3
import numpy as np
import matplotlib.pyplot as plt

# Flag to plot the errors to verify that they are 
# zero in the absense of noise.
plot_flag = False


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
def test_noiseless_batch(
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

    noise_active = False
    process_model = process_model(Q)

    # Generate data with no noise
    dg = DataGenerator(
        process_model,
        input_profile,
        Q,
        input_freq,
        measurement_model(R),
        measurement_freq,
    )

    state_true, input_list, meas_list = dg.generate(x0, 0, 5, noise_active)

    # Run batch
    estimator = BatchEstimator(max_iters=20)
    estimate_list = estimator.solve(x0, P0, input_list, meas_list, process_model)
    results = GaussianResultList(
        [
            GaussianResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]
    )

    if plot_flag:
        fig, ax = plot_error(results)
        plt.show()
