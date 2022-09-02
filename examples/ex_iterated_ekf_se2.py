from pynav.lib.states import SE3State
from pynav.lib.models import BodyFrameVelocity, RangePoseToAnchor
from pynav.datagen import DataGenerator
from pynav.filters import ExtendedKalmanFilter, IteratedKalmanFilter
from pynav.utils import GaussianResult, GaussianResultList, plot_error
from pynav.types import StateWithCovariance
import time
from pylie import SE3
import numpy as np
from typing import List

# ##############################################################################
# Problem Setup
x0 = SE3State(SE3.Exp([0, 0, 0, 0, 0, 0]), stamp=0.0)
P0 = 1 * np.identity(6)
Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.1, 0.1, 0.1])
process_model = BodyFrameVelocity(Q)


def input_profile(t, x):
    return np.array(
        [np.sin(0.1 * t), np.cos(0.1 * t), np.sin(0.1 * t), 1, 0, 0]
    )


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

# ##############################################################################
# Data Generation
dg = DataGenerator(process_model, input_profile, Q, 200, range_models, 10)
state_true, input_data, meas_data = dg.generate(x0, 0, 10, noise=True)

# %% ###########################################################################
# Run Filter
x = StateWithCovariance(x0, P0)

# Try an EKF or an IterEKF
# ekf = ExtendedKalmanFilter(process_model)
ekf = IteratedKalmanFilter(process_model)

meas_idx = 0
start_time = time.time()
y = meas_data[meas_idx]
results_list = []
for k in range(len(input_data) - 1):
    u = input_data[k]

    # Fuse any measurements that have occurred.
    while y.stamp < input_data[k + 1].stamp and meas_idx < len(meas_data):

        x = ekf.correct(x, y)
        meas_idx += 1
        if meas_idx < len(meas_data):
            y = meas_data[meas_idx]

    x = ekf.predict(x, u)
    results_list.append(GaussianResult(x, state_true[k]))


print("Average filter computation frequency (Hz):")
print(1 / ((time.time() - start_time) / len(input_data)))

results = GaussianResultList(results_list)

# ##############################################################################
# Plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
fig, axs = plot_error(results)
axs[-1][0].set_xlabel("Time (s)")
axs[-1][1].set_xlabel("Time (s)")
axs[0][0].set_title("Rotation Error")
axs[0][1].set_title("Translation Error")
plt.show()
