from navlie.lib import SE3State, BodyFrameVelocity, RangePoseToAnchor
import navlie as nav

import numpy as np
from typing import List
"""
This example runs an Interacting Multiple Model filter to estimate the process model noise matrix
for a state that is on a Lie group. The performance is compared to an EKF that knows the ground
truth process model noise. 
"""

# Create the process model noise profile
c_list = [1, 9]
Q_ref = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])


def Q_profile(t):
    if t <= t_max / 4:
        c = c_list[0]
    if t > t_max / 4 and t <= t_max * 3 / 4:
        c = c_list[1]
    if t > t_max * 3 / 4:
        c = c_list[0]
    Q = c * Q_ref
    return Q


# Measurement model
R = 0.1**2
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

# Setup
x0 = SE3State([0, 0, 0, 0, 0, 0], stamp=0.0)
P0 = 0.1**2 * np.identity(6)
input_profile = lambda t, x: np.array(
    [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
)
process_model_true = BodyFrameVelocity
input_freq = 10
dt = 1 / input_freq
t_max = dt * 100
measurement_freq = 5

# The two models correspond to the BodyVelocityModel which uses two different Q matrices.
# The two different Q matrices are a Q_ref matrix which is scaled by a scalar, c.
imm_process_model_list = [
    BodyFrameVelocity(c_list[0] * Q_ref),
    BodyFrameVelocity(c_list[1] * Q_ref),
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
n_models = len(imm_process_model_list)

# Kalman Filter bank
kf_list = [nav.ExtendedKalmanFilter(pm) for pm in imm_process_model_list]

# Set up probability transition matrix
off_diag_p = 0.02
Pi = np.ones((n_models, n_models)) * off_diag_p
Pi = Pi + (1 - off_diag_p * (n_models)) * np.diag(np.ones(n_models))
imm = nav.imm.InteractingModelFilter(kf_list, Pi)


dg = nav.DataGenerator(
    VaryingNoiseProcessModel(Q_profile),
    input_profile,
    Q_profile,
    input_freq,
    range_models,
    measurement_freq,
)


def imm_trial(trial_number: int) -> List[nav.GaussianResult]:
    """
    A single Interacting Multiple Model Filter trial
    """
    np.random.seed(trial_number)
    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, True)

    x0_check = x0.plus(nav.randvec(P0))

    estimate_list = nav.imm.run_imm_filter(
        imm, x0_check, P0, input_list, meas_list
    )

    results = [
        nav.imm.IMMResult(estimate_list[i], state_true[i]) for i in range(len(estimate_list))
    ]

    return nav.imm.IMMResultList(results)


def ekf_trial(trial_number: int) -> List[nav.GaussianResult]:
    """
    A single trial in a monte carlo experiment. This function accepts the trial
    number and must return a list of GaussianResult objects.
    """

    # By using the trial number as the seed for the random generator, we can
    # make sure our experiments are perfectly repeatable, yet still have
    # independent noise samples from trial-to-trial.
    np.random.seed(trial_number)

    state_true, input_list, meas_list = dg.generate(x0, 0, t_max, noise=True)
    x0_check = x0.plus(nav.randvec(P0))
    ekf = nav.ExtendedKalmanFilter(VaryingNoiseProcessModel(Q_profile))

    estimate_list = nav.run_filter(ekf, x0_check, P0, input_list, meas_list)
    return nav.GaussianResultList.from_estimates(estimate_list, state_true)


results = nav.monte_carlo(imm_trial, N)
results_ekf = nav.monte_carlo(ekf_trial, N)

if __name__ == "__main__":
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

        fig, axs = plt.subplots(3, 2)
        axs: List[plt.Axes] = axs
        for result in results.trial_results:
            nav.plot_error(result, axs=axs)

        fig.suptitle("Estimation error IMM")
        axs[1,0].set_xlabel("Time (s)")

        fig, axs = plt.subplots(3, 2)
        axs: List[plt.Axes] = axs
        for result in results_ekf.trial_results:
            nav.plot_error(result, axs=axs)

        fig.suptitle("Estimation error EKF GT")
        axs[1,0].set_xlabel("Time (s)")

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
    ax.set_title("Estimated vs. GT Process Noise")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$Q_{00}$")

    plt.show()
