import pytest
from navlie.filters import ExtendedKalmanFilter
from navlie.lib.states import VectorState, SE3State
from navlie.datagen import DataGenerator
from navlie.utils import GaussianResult
from navlie.utils import randvec

from navlie.utils import monte_carlo, plot_error
from navlie.lib.models import DoubleIntegrator, OneDimensionalPositionVelocityRange
from navlie.lib.models import SingleIntegrator, RangePointToAnchor
from navlie.lib.models import (
    BodyFrameVelocity,
    Magnetometer,
    Gravitometer,
)
from navlie.lib.models import RangePoseToAnchor
from pymlg import SE3

import numpy as np
from typing import List
from navlie.imm import InteractingModelFilter, run_imm_filter
from navlie.imm import IMMResultList
from navlie.imm import  IMMResult
# TODO this test is very complicated. we need to simplify this.

PLOT_FLAG = False


def make_onedimensional_range(R):
    range_models = [OneDimensionalPositionVelocityRange(R)]
    return range_models


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


def make_filter_trial(dg, x0_true, P0, t_max, imm, process_model, Q_profile):
    def imm_trial(trial_number: int) -> List[GaussianResult]:
        """
        A single Interacting Multiple Model Filter trial
        """
        np.random.seed(trial_number)
        state_true, input_list, meas_list = dg.generate(x0_true, 0, t_max, True)

        x0_check = x0_true.copy()
        x0_check = x0_check.plus(randvec(P0))

        estimate_list = run_imm_filter(
            imm, x0_check, P0, input_list, meas_list
        )

        results = [
            IMMResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]

        return IMMResultList(results)

    return imm_trial


def make_c_profile_varying(t_max):
    def c_profile(t):
        if t <= t_max / 4:
            c = 1
        if t > t_max / 4 and t <= t_max * 3 / 4:
            c = 9
        if t > t_max * 3 / 4:
            c = 1
        return c

    return c_profile


def make_c_profile_const(t_max):
    def c_profile(t):
        return 1.0

    return c_profile


@pytest.mark.parametrize(
    "x0, P0, input_profile, process_model_true, measurement_model, \
                        make_c_profile, R, input_freq, measurement_freq, imm_process_model_list, Q_ref",
    [
        (
            VectorState(np.array([1, 0]), 0.0),
            np.diag([1, 1]),
            lambda t, x: np.array([np.sin(t), np.cos(t)]),
            SingleIntegrator,
            make_range_models_ekf_vector,
            make_c_profile_varying,
            0.1**2,
            10,
            [1, 1, 1],
            [SingleIntegrator(1 * np.eye(2)), SingleIntegrator(9 * np.eye(2))],
            np.eye(2),
        ),
        (
            VectorState(np.array([1, 0]), 0.0),
            np.diag([1, 1]),
            lambda t, x: np.array([np.sin(t)]),
            DoubleIntegrator,
            make_onedimensional_range,
            make_c_profile_varying,
            0.1**4,
            10,
            5,
            [DoubleIntegrator(np.eye(1)), DoubleIntegrator(9 * np.eye(1))],
            np.eye(1),
        ),
        (
            SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0),
            0.1**2 * np.identity(6),
            lambda t, x: np.array(
                [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
            ),
            BodyFrameVelocity,
            make_range_models_iterated_ekf,
            make_c_profile_varying,
            0.1**2,
            10,
            5,
            [
                BodyFrameVelocity(
                    np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
                ),
                BodyFrameVelocity(
                    9 * np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
                ),
            ],
            np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1]),
        ),
    ],
)
def test_reasonable_nees_imm(
    x0,
    P0,
    input_profile,
    process_model_true,
    measurement_model,
    make_c_profile,
    R,
    input_freq,
    measurement_freq,
    imm_process_model_list,
    Q_ref,
):
    dt = 1 / input_freq
    t_max = dt * 100
    N = 10
    Q_dg = np.eye(x0.value.shape[0])
    n_models = len(imm_process_model_list)
    kf_list = [ExtendedKalmanFilter(pm) for pm in imm_process_model_list]
    off_diag_p = 0.02
    Pi = np.ones((n_models, n_models)) * off_diag_p
    Pi = Pi + (1 - off_diag_p * (n_models)) * np.diag(np.ones(n_models))
    c_profile = make_c_profile(t_max)
    Q_profile = lambda t: c_profile(t) * Q_ref

    class VaryingNoiseProcessModel(process_model_true):
        def __init__(self, Q_profile):
            self.Q_profile = Q_profile
            super().__init__(Q_profile(0))

        def covariance(self, x, u, dt) -> np.ndarray:
            self._Q = self.Q_profile(x.stamp)
            return super().covariance(x, u, dt)

    dg = DataGenerator(
        VaryingNoiseProcessModel(Q_profile),
        input_profile,
        Q_profile,
        input_freq,
        measurement_model(R),
        measurement_freq,
    )

    imm_trial = make_filter_trial(
        dg,
        x0,
        P0,
        t_max,
        InteractingModelFilter(kf_list, Pi),
        process_model_true,
        Q_profile,
    )
    results = monte_carlo(imm_trial, N)

    if PLOT_FLAG:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme()

        fig, ax = plt.subplots(1, 1)
        ax.plot(results.stamp, results.average_nees)
        ax.plot(results.stamp, results.expected_nees, color="r", label="Expected NEES")
        ax.plot(
            results.stamp,
            results.nees_lower_bound(0.99),
            color="k",
            linestyle="--",
            label="99 percent c.i.",
        )
        ax.plot(
            results.stamp,
            results.nees_upper_bound(0.99),
            color="k",
            linestyle="--",
        )
        ax.set_title("{0}-trial average NEES".format(results.num_trials))
        ax.set_ylim(0, None)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("NEES")
        ax.legend()

        if N < 15:

            fig, axs = plt.subplots(2, 1)
            axs: List[plt.Axes] = axs
            for result in results.trial_results:
                plot_error(result, axs=axs)

            axs[0].set_title("Estimation error")
            axs[1].set_xlabel("Time (s)")

        average_model_probabilities = np.average(
            np.array([t.model_probabilities for t in results.trial_results]), axis=0
        )
        fig, ax = plt.subplots(1, 1)
        for lv1 in range(n_models):
            ax.plot(results.stamp, average_model_probabilities[lv1, :])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Model Probabilities")

        plt.show()

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
