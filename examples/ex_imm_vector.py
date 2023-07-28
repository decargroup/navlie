"""
This example runs an Interacting Multiple Model filter to estimate the process
model noise matrix for a state that is on a vector space. The performance is
compared to an EKF that knows the ground truth process model noise. 
"""

import navlie as nav
from navlie.lib.models import DoubleIntegrator, RangePointToAnchor, VectorState
from navlie.imm import InteractingModelFilter, run_imm_filter, IMMResultList
import numpy as np
from typing import List
from matplotlib import pyplot as plt

# Measurement model
R = 0.1**4

range_model = RangePointToAnchor(anchor_position=0.0, R=R)

# Process model noise setup
c_list = [1, 9]
Q_ref = np.eye(1)
input_freq = 10
dt = 1 / input_freq
t_max = dt * 1000

# For data generation, assume that the process model noise Q follows this profile.
def Q_profile(t):
    if t <= t_max / 4:
        c = c_list[0]
    if t > t_max / 4 and t <= t_max * 3 / 4:
        c = c_list[1]
    if t > t_max * 3 / 4:
        c = c_list[0]
    Q = c * Q_ref
    return Q

def main():

    # Setup
    x0 = VectorState(np.array([1, 0]), 0.0)
    P0 = np.diag([1, 1])
    input_profile = lambda t, x: np.array([np.sin(t)])
    measurement_freq = 5

    # The two models correspond to the DoubleIntegrator which uses two different Q matrices.
    # The two different Q matrices are a Q_ref matrix which is scaled by a scalar, c.
    imm_process_model_list = [
        DoubleIntegrator(c_list[0] * Q_ref),
        DoubleIntegrator(c_list[1] * Q_ref),
    ]


    n_models = len(imm_process_model_list)

    # Kalman Filter bank
    kf_list = [nav.ExtendedKalmanFilter(pm) for pm in imm_process_model_list]

    # Set up probability transition matrix
    off_diag_p = 0.02
    prob_trans_matrix = np.ones((n_models, n_models)) * off_diag_p
    prob_trans_matrix = prob_trans_matrix + (1 - off_diag_p * (n_models)) * np.diag(np.ones(n_models))
    imm = InteractingModelFilter(kf_list, prob_trans_matrix)

    # Generate some data with varying Q matrix
    dg = nav.DataGenerator(
        DoubleIntegrator(Q_ref),
        input_profile,
        Q_profile,
        input_freq,
        range_model,
        measurement_freq,
    )
    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, True)


    # Add some noise to the initial state
    x0_check = x0.plus(nav.randvec(P0))

    estimate_list = run_imm_filter(
        imm, x0_check, P0, input_list, meas_list
    )

    results = IMMResultList.from_estimates(estimate_list, state_true)
    return results


if __name__ == "__main__":
    results = main()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    # NEES Plot
    fig, ax = nav.plot_nees(results)
    ax.set_ylim(0, None)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("NEES")
    ax.legend()

    # Error plot
    fig, axs = nav.plot_error(results)
    axs[0].set_title("Estimation error IMM")
    axs[0].set_ylabel("Position Error (m)")
    axs[1].set_ylabel("Velocity Error (m/s)")
    axs[1].set_xlabel("Time (s)")

    # Model probabilities
    fig, ax = plt.subplots(1, 1)
    ax.plot(results.stamp, results.model_probabilities[:, 0], label="Model 1")
    ax.plot(results.stamp, results.model_probabilities[:, 1], label="Model 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Model Probabilities")
    ax.legend()

    # Process noise as estimated by the IMM
    fig, ax = plt.subplots(1, 1)
    estimated_Q = np.sum(c_list*results.model_probabilities, axis=1)
    true_Q = np.array([Q_profile(t)[0, 0] for t in results.stamp])
    ax.plot(results.stamp, estimated_Q, label="Estimated")
    ax.plot(results.stamp, true_Q, label="True")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimated vs True Process Noise")
    ax.legend()
    plt.show()
