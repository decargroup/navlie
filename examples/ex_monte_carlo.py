# %%
"""
This script shows how to use the monte carlo utils. The monte_carlo() 
function will automatically execute, and aggregate the results from a user-
provided callable trial function. Average NEES, its probability bounds, and 
expected NEES are all automatically calculated for you.
"""

from pynav.lib.states import SE3State
from pynav.lib.models import BodyFrameVelocity, RangePoseToAnchor
from pynav.datagen import DataGenerator
from pynav.filters import ExtendedKalmanFilter
from pynav.utils import GaussianResult, GaussianResultList, monte_carlo, plot_error, randvec
from pynav.types import StateWithCovariance
import time
from pylie import SE3
import numpy as np
from typing import List

x0_true = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0)
P0 = np.diag([0.1**2, 0.1**2, 0.1**2, 0.3**3, 0.3**2, 0.3**2])
Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
process_model = BodyFrameVelocity(Q)
def input_profile(t):
    return np.array([np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0])
range_models = [
    RangePoseToAnchor([1, 0, 0], [0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([1, 0, 0], [-0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([-1, 0, 0], [0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([-1, 0, 0], [-0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([0, 2, 0], [0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([0, 2, 0], [-0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([0, 2, 2], [0.17, 0.17, 0], 0.1**2),
    RangePoseToAnchor([0, 2, 2], [-0.17, 0.17, 0], 0.1**2),
]
dg = DataGenerator(process_model, input_profile, Q, 100, range_models, 10)


def ekf_trial(trial_number:int) -> List[GaussianResult]:
    """
    A single trial in a monte carlo experiment. This function accepts the trial
    number and must return a list of GaussianResult objects.
    """

    # By using the trial number as the seed for the random generator, we can
    # make sure our experiments are perfectly repeatable, yet still have 
    # independent noise samples from trial-to-trial.
    np.random.seed(trial_number)


    state_true, input_data, meas_data = dg.generate(x0_true, 0, 10, noise=True)
    x0_check = x0_true.copy()
    x0_check.plus(randvec(P0))
    x = StateWithCovariance(x0_check, P0)
    ekf = ExtendedKalmanFilter(process_model)

    meas_idx = 0
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):
        u = input_data[k]
        x = ekf.predict(x, u)
        
        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = ekf.correct(x, y, u)
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        
        results_list.append(GaussianResult(x, state_true[k]))

    return GaussianResultList(results_list)

# %% Run the monte carlo experiment

N = 5

results = monte_carlo(ekf_trial, N)

# %% Plot
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

fig, ax = plt.subplots(1,1)
ax.plot(results.stamp, results.average_nees)
ax.plot(results.stamp, results.expected_nees, color = 'r', label = "Expected NEES")
ax.plot(results.stamp, results.nees_lower_bound(0.99), color='k', linestyle="--", label="99 percent c.i.")
ax.plot(results.stamp, results.nees_upper_bound(0.99), color='k', linestyle="--",)
ax.set_title("{0}-trial average NEES".format(results.num_trials))
ax.set_ylim(0,None)
ax.set_xlabel("Time (s)")
ax.set_ylabel("NEES")
ax.legend()


if N < 15:

    fig, axs = plt.subplots(2, 1)
    axs: List[plt.Axes] = axs
    for result in results.trial_results:
        plot_error(result, axs = axs)

    axs[0].set_title("Estimation error")
    axs[1].set_xlabel("Time (s)")

plt.show()