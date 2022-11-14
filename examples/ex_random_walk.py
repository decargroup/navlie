from pynav.filters import ExtendedKalmanFilter
from pynav.lib.states import VectorState
from pynav.datagen import DataGenerator
from pynav.types import StateWithCovariance, StampedValue
from pynav.utils import GaussianResult, GaussianResultList, randvec, plot_error
from pynav.lib.models import DoubleIntegrator, RangePointToAnchor
import numpy as np

# ##############################################################################
# Problem Setup

x0 = VectorState(np.array([1, 0, 0, 0]))
P0 = 0.1**2*np.identity(x0.dof)
R = 0.1**2
Q = 0.1 * np.identity(2)
range_models = [
    RangePointToAnchor([0, 4], R),
    RangePointToAnchor([-2, 0], R),
    RangePointToAnchor([2, 0], R),
]
range_freqs = [1, 1, 1]
process_model = DoubleIntegrator(Q)

# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# The TRUE input profile is zero-mean random signal.
input_profile = lambda t, x: randvec(Q).ravel() # random walk.

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

input_covariance = Q
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

gt_data, input_data, meas_data = dg.generate(x0, 0, 50, noise=True)

x = x0.copy()
x.plus(randvec(P0))
x = StateWithCovariance(x, P0)

ekf = ExtendedKalmanFilter(process_model)

meas_idx = 0
y = meas_data[meas_idx]
results_list = []
for k in range(len(input_data) - 1):

    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # The data generator will add noise on top of the already random signal if 
    # `input_covariance` is not zero. So here we remove this.
    
    u: StampedValue = input_data[k]
    u.value = np.zeros(u.value.shape) # Zero-out the input

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

        x = ekf.correct(x, y, u)

        # Load the next measurement
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]

    x = ekf.predict(x, u)
    results_list.append(GaussianResult(x, gt_data[k]))

results = GaussianResultList(results_list)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2,2, sharex=True)
plot_error(results, axs)
axs[0,0].set_title("Position Error")
axs[0,1].set_title("Velocity Error")
axs[1,0].set_xlabel("Time (s)")
axs[1,1].set_xlabel("Time (s)")
axs[0,0].set_ylabel("x (m)")
axs[1,0].set_ylabel("y (m)")
axs[0,1].set_ylabel("x (m/s)")
axs[1,1].set_ylabel("y (m/s)")
plt.tight_layout()
plt.show()
