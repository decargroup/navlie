from pynav.filters import ExtendedKalmanFilter, IteratedKalmanFilter
from pynav.lib.states import VectorState
from pynav.datagen import DataGenerator
from pynav.types import StateWithCovariance
from pynav.utils import GaussianResult, GaussianResultList, randvec
from pynav.lib.models import SingleIntegrator, RangePointToAnchor
import numpy as np
from typing import List
import time

"""
This is an example script showing how to define a custom process model and
measurement model, generate data using those models, and then run an EKF 
on that data.
"""

# ##############################################################################
# Problem Setup

x0 = VectorState(np.array([1, 0]), stamp=0)
P0 = np.diag([1, 1])
R = 0.1**2
Q = 0.1 * np.identity(2)
range_models = [
    RangePointToAnchor([0, 4], R),
    RangePointToAnchor([-2, 0], R),
    RangePointToAnchor([2, 0], R),
]
range_freqs = [50, 50, 50]
process_model = SingleIntegrator(Q)
input_profile = lambda t, x: np.array([np.sin(t), np.cos(t)])
input_covariance = Q
input_freq = 180
noise_active = False
# ##############################################################################
# Data Generation

dg = DataGenerator(
    process_model,
    input_profile,
    input_covariance,
    input_freq,
    range_models,
    range_freqs,
)

gt_data, input_data, meas_data = dg.generate(x0, 0, 10, noise=noise_active)

# ##############################################################################
# Run Filter
x0.plus(randvec(P0))
x = StateWithCovariance(x0, P0)

ekf = ExtendedKalmanFilter(process_model)
# ekf = IteratedKalmanFilter(process_model) # or try the IEKF!

meas_idx = 0
start_time = time.time()
y = meas_data[meas_idx]
results_list = []
for k in range(len(input_data) - 1):

    u = input_data[k]
    
    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

        x = ekf.correct(x, y, u)

        # Load the next measurement
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]


    dt = input_data[k + 1].stamp - x.state.stamp
    x = ekf.predict(x, u, dt)
    
    results_list.append(GaussianResult(x, gt_data[k+1]))

print("Average filter computation frequency (Hz):")
print(1 / ((time.time() - start_time) / len(input_data)))


# ##############################################################################
# Post processing
results = GaussianResultList(results_list)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
fig, ax = plt.subplots(1, 1)
ax.plot(results.value[:, 0], results.value[:, 1], label="Estimate")
ax.plot(
    results.value_true[:, 0], results.value_true[:, 1], label="Ground truth"
)
ax.set_title("Trajectory")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend()

fig, axs = plt.subplots(2, 1)
axs: List[plt.Axes] = axs
for i in range(len(axs)):
    axs[i].fill_between(
        results.stamp,
        results.three_sigma[:, i],
        -results.three_sigma[:, i],
        alpha=0.5,
    )
    axs[i].plot(results.stamp, results.error[:, i])
axs[0].set_title("Estimation error")
axs[1].set_xlabel("Time (s)")
plt.show()
