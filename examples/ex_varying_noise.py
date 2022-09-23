# %%

from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import VectorState
from pynav.datagen import DataGenerator
from pynav.types import StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, monte_carlo, plot_error, randvec

from pynav.lib.models import DoubleIntegrator, OneDimensionalPositionVelocityRange
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

"""
This is an example script showing how to define time-varying noise matrices
and to then run an EKF using these same noise matrices. 
"""


x0_true = VectorState(np.array([1, 0]))
P0 = np.diag([0.5, 0.5])
R = 0.01**2
Q = 0.1 * np.identity(1)
N = 50 # Number MC trials

range_models = [
    OneDimensionalPositionVelocityRange(R),
]


range_freqs = [10]
input_freq = 10
dt = 1/input_freq

def Q_profile(t):
    if t <= t_max/4:
        Q = 1
    if t > t_max/4 and t <= t_max*3/4:
        Q = 9
    if t > t_max*3/4:
        Q = 1
    Q = np.array(Q).reshape(1,1)
    return Q


# For data generation, the Q for the process model does not matter as
# only the evaluate method is used. 
process_model_dg = DoubleIntegrator(Q)
input_profile = lambda t, x: np.sin(t)
t_max = 10

dg = DataGenerator(
    process_model_dg,
    input_profile,
    Q_profile,
    input_freq,
    range_models,
    range_freqs
)





def ekf_trial(trial_number:int) -> List[GaussianResult]:
    """
    A single trial in a monte carlo experiment. This function accepts the trial
    number and must return a list of GaussianResult objects.
    """

    np.random.seed(trial_number)
    state_true, input_data, meas_data = dg.generate(x0_true, 0, t_max, noise=True)

    x0_check = x0_true.copy()
    x0_check.plus(randvec(P0))
    x = StateWithCovariance(x0_check, P0)

    
    
    meas_idx = 0
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):
        u = input_data[k]
        ekf = ExtendedKalmanFilter(DoubleIntegrator(Q_profile(u.stamp)))

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):
            x = ekf.correct(x, y, u)
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        if x.state.stamp is not None:
            dt = u.stamp-x.state.stamp
        else: 
            dt = 0
        x = ekf.predict(x, u, dt=dt)
        results_list.append(GaussianResult(x, state_true[k]))

    return GaussianResultList(results_list)

# %% Run the monte carlo experiment
results = monte_carlo(ekf_trial, N)

# %% Plot

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