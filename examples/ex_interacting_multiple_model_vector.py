import pytest
from pynav.filters import ExtendedKalmanFilter, run_filter
from pynav.lib.states import VectorState
from pynav.datagen import DataGenerator
from pynav.utils import GaussianResult, GaussianResultList
from pynav.utils import randvec

from pynav.utils import monte_carlo, plot_error
from pynav.lib.models import DoubleIntegrator, OneDimensionalPositionVelocityRange

import numpy as np
from typing import List
from matplotlib import pyplot as plt
from pynav.imm import InteractingModelFilter, run_interacting_multiple_model_filter
from pynav.imm import IMMResultList, IMMResult


"""This example runs an Interacting Multiple Model filter to estimate the process model noise matrix
for a state that is on a vector space. The performance is compared to an EKF that knows the ground
truth process model noise. 
"""
# TODO. Remove monte carlo. this makes the example more complicated than it needs to be
# and is not necessary to demonstrate the IMM filter.

# TODO. The IMM seems to have an issue when the user accidently modifies the 
# provided state in the process model.

# Measurement model
R = 0.1**4
range_models = [OneDimensionalPositionVelocityRange(R)]

# Process model noise setup
c_list = [1, 9]
Q_ref = np.eye(1)
input_freq = 10
dt = 1 / input_freq
t_max = dt * 1000


def Q_profile(t):
    if t <= t_max / 4:
        c = c_list[0]
    if t > t_max / 4 and t <= t_max * 3 / 4:
        c = c_list[1]
    if t > t_max * 3 / 4:
        c = c_list[0]
    Q = c * Q_ref
    return Q


# Setup
x0 = VectorState(np.array([1, 0]), 0.0)
P0 = np.diag([1, 1])
input_profile = lambda t, x: np.array([np.sin(t)])
process_model_true = DoubleIntegrator
measurement_freq = 5

# The two models correspond to the DoubleIntegrator which uses two different Q matrices.
# The two different Q matrices are a Q_ref matrix which is scaled by a scalar, c.
imm_process_model_list = [
    DoubleIntegrator(c_list[0] * Q_ref),
    DoubleIntegrator(c_list[1] * Q_ref),
]


class VaryingNoiseProcessModel(process_model_true):
    def __init__(self, Q_profile):
        self.Q_profile = Q_profile
        super().__init__(Q_profile(0))

    def covariance(self, x, u, dt) -> np.ndarray:
        self._Q = self.Q_profile(x.stamp)
        return super().covariance(x, u, dt)


N = 5
Q_dg = np.eye(x0.value.shape[0])
n_inputs = input_profile(0, np.zeros(x0.value.shape[0])).shape[0]
n_models = len(imm_process_model_list)

# Kalman Filter bank
kf_list = [ExtendedKalmanFilter(pm) for pm in imm_process_model_list]

# Set up probability transition matrix
off_diag_p = 0.02
Pi = np.ones((n_models, n_models)) * off_diag_p
Pi = Pi + (1 - off_diag_p * (n_models)) * np.diag(np.ones(n_models))
imm = InteractingModelFilter(kf_list, Pi)

dg = DataGenerator(
    VaryingNoiseProcessModel(Q_profile),
    input_profile,
    Q_profile,
    input_freq,
    range_models,
    measurement_freq,
)


def imm_trial(trial_number: int) -> List[GaussianResult]:
    """
    A single Interacting Multiple Model Filter trial
    """
    np.random.seed(trial_number)
    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, True)

    x0_check = x0.plus(randvec(P0))

    estimate_list = run_interacting_multiple_model_filter(
        imm, x0_check, P0, input_list, meas_list
    )

    results = [
        IMMResult(estimate_list[i], state_true[i]) for i in range(len(estimate_list))
    ]

    return IMMResultList(results)


def ekf_trial(trial_number: int) -> List[GaussianResult]:
    """
    A single trial in a monte carlo experiment. This function accepts the trial
    number and must return a list of GaussianResult objects.
    """

    # By using the trial number as the seed for the random generator, we can
    # make sure our experiments are perfectly repeatable, yet still have
    # independent noise samples from trial-to-trial.
    np.random.seed(trial_number)

    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, noise=True)
    x0_check = x0.plus(randvec(P0))
    ekf = ExtendedKalmanFilter(VaryingNoiseProcessModel(Q_profile))

    estimate_list = run_filter(ekf, x0_check, P0, input_list, meas_list)
    results = [
        GaussianResult(estimate_list[i], state_true[i])
        for i in range(len(estimate_list))
    ]

    return GaussianResultList(results)


results = monte_carlo(imm_trial, N)
results_ekf = monte_carlo(ekf_trial, N)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(1, 1)
ax.plot(results.stamp, results.average_nees, label="IMM NEES")
ax.plot(results.stamp, results_ekf.average_nees, label="EKF using GT Q NEES")
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

    axs[0].set_title("Estimation error IMM")
    axs[1].set_xlabel("Time (s)")

    fig, axs = plt.subplots(2, 1)
    axs: List[plt.Axes] = axs
    for result in results_ekf.trial_results:
        plot_error(result, axs=axs)

    axs[0].set_title("Estimation error EKF GT")
    axs[1].set_xlabel("Time (s)")

    average_model_probabilities = np.average(
        np.array([t.model_probabilities for t in results.trial_results]), axis=0
    )
    fig, ax = plt.subplots(1, 1)
    for lv1 in range(n_models):
        ax.plot(results.stamp, average_model_probabilities[lv1, :])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model Probabilities")

fig, ax = plt.subplots(1, 1)
Q_ = np.zeros(results.stamp.shape)
for lv1 in range(n_models):
    Q_ = Q_ + average_model_probabilities[lv1, :] * c_list[lv1] * Q_ref[0, 0]

ax.plot(results.stamp, Q_, label=r"$Q_{00}$, Estimated")
ax.plot(
    results.stamp,
    np.array([Q_profile(t)[0, 0] for t in results.stamp]),
    label=r"$Q_{00}$, GT",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$Q_{00}$")

plt.show()
