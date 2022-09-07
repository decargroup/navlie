from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import VectorState
from pynav.datagen import DataGenerator
from pynav.types import StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, monte_carlo, randvec
from pynav.lib.models import DoubleIntegrator, RangePointToAnchor
import numpy as np
from typing import List
import time

# ##############################################################################
# Problem Setup

x0 = VectorState(np.array([1, 0, 0, 0]))
P0 = 0.0001**2*np.identity(x0.dof)
R = 0.1**2
Q = 0.1 * np.identity(2)
range_models = [
    RangePointToAnchor([0, 4], R),
    RangePointToAnchor([-2, 0], R),
    RangePointToAnchor([2, 0], R),
]
range_freqs = [1, 1, 1]
process_model = DoubleIntegrator(Q)
input_profile = lambda t: randvec(Q).ravel() # random walk.
input_covariance = 0
input_freq = 100

dg = DataGenerator(
    process_model,
    input_profile,
    input_covariance,
    input_freq,
    range_models,
    range_freqs,
)

# ##############################################################################
# Trial function

def trial(i):
    gt_data, input_data, meas_data = dg.generate(x0, 0, 50, noise=True)

    x = x0.copy()
    x.plus(randvec(P0))
    x = StateWithCovariance(x, P0)

    ekf = ExtendedKalmanFilter(process_model)

    meas_idx = 0
    y = meas_data[meas_idx]
    results_list = []
    for k in range(len(input_data) - 1):
        u = input_data[k]
        u.value = np.zeros(u.value.shape) # Zero-out the input

        # Fuse any measurements that have occurred.
        while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

            x = ekf.correct(x, y, u)

            # Load the next measurement
            meas_idx += 1
            if meas_idx < len(meas_data):
                y = meas_data[meas_idx]

        x = ekf.predict(x, u)
        results_list.append(GaussianResult(x, gt_data[k]))

    return GaussianResultList(results_list)

results = monte_carlo(trial, 20)


import matplotlib.pyplot as plt

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


plt.show()
