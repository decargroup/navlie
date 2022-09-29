import pytest
from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import VectorState, SO3State, SE3State
from pynav.datagen import DataGenerator
from pynav.types import StampedValue, StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, MonteCarloResult
from pynav.utils import randvec

from pynav.utils import monte_carlo, plot_error
from pynav.lib.models import DoubleIntegrator, OneDimensionalPositionVelocityRange
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
from pynav.imm import InteractingModelFilter, run_interacting_multiple_model_filter, run_time_varying_Q_filter
from pynav.imm import ImmResultList


def make_onedimensional_range(R):
    range_models=[
        OneDimensionalPositionVelocityRange(R)
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
        estimate_list, model_probabilities = run_interacting_multiple_model_filter(imm, x0_check, P0, input_list, meas_list)

        results = \
        [
            GaussianResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]
        
        return ImmResultList(results, model_probabilities)
    
    return imm_trial


def make_ekf_trial(dg, x0_true, P0, t_max, ekf, process_model, Q_profile):
        
    def ekf_trial(trial_number: int) -> List[GaussianResult]:
        """
        A single EKF trial for an EKF which uses a time-varying process noise matrix.
        """
        np.random.seed(trial_number)
        state_true, input_list, meas_list = dg.generate(x0_true, 0, t_max, True)

        x0_check = x0_true.copy()
        x0_check = x0_check.plus(randvec(P0))
        estimate_list = run_time_varying_Q_filter(ekf, process_model, Q_profile, x0_check, P0, input_list, meas_list)

        results = GaussianResultList(
        [
            GaussianResult(estimate_list[i], state_true[i])
            for i in range(len(estimate_list))
        ]
        )
        return results
    
    return ekf_trial

# The two models correspond to the BodyVelocityModel which uses two different Q matrices. 
# The two different Q matrices are a Q_ref matrix which is scaled by a scalar, c. 
c_list = [1, 9]
Q_ref = np.eye(1)
def make_nd_Q_profile_varying(t_max, n_inputs):
    def Q_profile(t):
        if t <= t_max/4:
            c = c_list[0]
        if t > t_max/4 and t <= t_max*3/4:
            c = c_list[1]
        if t > t_max*3/4:
            c = c_list[0]
        Q = c*Q_ref
        return Q
    return Q_profile

x0, P0, input_profile, process_model_true, measurement_model, \
    make_Q_profile, R, input_freq, measurement_freq, imm_process_model_list = \
                    (VectorState(np.array([1, 0]), 0.0), np.diag([1, 1]), 
                    lambda t, x: np.array([np.sin(t)]),
                    DoubleIntegrator, make_onedimensional_range, 
                    make_nd_Q_profile_varying, 0.1**4, 
                    10, 5, [DoubleIntegrator(c_list[0]*Q_ref), DoubleIntegrator(c_list[1]*Q_ref)])
                      
dt = 1/input_freq
t_max = dt*1000
N = 10
Q_dg = np.eye(x0.value.shape[0])
n_inputs = input_profile(0, np.zeros(x0.value.shape[0])).shape[0]
N_MODELS = len(imm_process_model_list)

# Kalman Filter bank
kf_list = [ExtendedKalmanFilter(pm) for pm in imm_process_model_list] 

# Set up probability transition matrix
off_diag_p = 0.02
Pi = np.ones((N_MODELS,N_MODELS))*off_diag_p
Pi = Pi+(1-off_diag_p*(N_MODELS))*np.diag(np.ones(N_MODELS))

Q_profile = make_Q_profile(t_max, n_inputs)
dg = DataGenerator(
    process_model_true(Q_dg),
    input_profile,
    Q_profile,
    input_freq,
    measurement_model(R),
    measurement_freq,
    )

imm_trial = make_filter_trial(dg, x0, P0, t_max, InteractingModelFilter(kf_list, Pi), process_model_true, Q_profile)
results = monte_carlo(imm_trial, N)

ekf_trial = make_ekf_trial(dg, x0, P0, t_max, ExtendedKalmanFilter, process_model_true, Q_profile)
results_ekf = monte_carlo(ekf_trial, N)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(1, 1)
ax.plot(results.stamp, results.average_nees, label = "IMM NEES")
ax.plot(results.stamp, results_ekf.average_nees, label = "EKF using GT Q NEES")
ax.plot(results.stamp, results.expected_nees, color="r", label="Expected NEES")
ax.plot( results.stamp, results.nees_lower_bound(0.99), color="k", linestyle="--", label="99 percent c.i.",)
ax.plot(results.stamp, results.nees_upper_bound(0.99), color="k", linestyle="--",)
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
    for lv1 in range(N_MODELS):
        ax.plot(results.stamp, average_model_probabilities[lv1, :])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model Probabilities")

fig, ax = plt.subplots(1, 1)
Q_ = np.zeros(results.stamp.shape)
for lv1 in range(N_MODELS):
    Q_ = Q_ + average_model_probabilities[lv1, :]*c_list[lv1]*Q_ref[0,0]

ax.plot(results.stamp, Q_, label =r'$Q_{00}$, Estimated')
ax.plot(results.stamp, np.array([Q_profile(t)[0, 0] for t in results.stamp]), label =r'$Q_{00}$, GT')
ax.set_xlabel("Time (s)")
ax.set_ylabel(r'$Q_{00}$')

plt.show()

